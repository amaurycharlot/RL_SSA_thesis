"""Strict schema-v1 loading and preflight/training validation.

No environments, workers, packages or output directories are initialized here.
Preference, seed and completed-update budget generation are pure operations.
"""
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
import json
from pathlib import Path
import re

from .types import (
    EnvConfig, PPOConfig, ExecutionConfig, NormalizationConfig, LoggingConfig,
    ExperimentConfig, SeedPlan, BudgetPlan, MAX_SEED, REWARD_ORDER,
)


class ConfigError(ValueError):
    """Invalid configuration with an actionable field/context explanation."""


_NESTED = {
    "env": EnvConfig, "ppo": PPOConfig, "execution": ExecutionConfig,
    "normalization": NormalizationConfig, "logging": LoggingConfig,
}
_ROOT_KEYS = frozenset(f.name for f in fields(ExperimentConfig) if f.name != "output_dir")
# These names are implemented safely for arbitrary combined-grid decisions.
# R4/R5 tracking require an observed selected target and index prior catalogues
# as dense target-ID lists; they fail on no-measurement/new-discovery steps.
SUPPORTED_TRACKING_REWARDS = frozenset(("R1", "R2", "R3", "R6", "R7"))
SUPPORTED_SEARCH_REWARDS = frozenset(("R1", "R2", "R3", "R4", "R5"))
_SCALE_STATUSES = frozenset(("identity_baseline_frozen", "frozen", "calibrated_frozen", "unselected"))
_DEVICE = re.compile(r"(?:auto|cpu|cuda(?::[0-9]+)?|mps(?::[0-9]+)?|xpu(?::[0-9]+)?)\Z")
WEIGHT_SUM_ATOL = 1e-6
# Algorithm v1: reserve disjoint uint32 domains, not just different RNG tags.
_SEED_VERSION = 1
_TRAIN_NAMESPACE = 0x54524149
_EVAL_NAMESPACE = 0x4556414C
_FINAL_NAMESPACE = 0x54455354
_TRAIN_WIDTH = 2 ** 31
_HOLDOUT_WIDTH = 2 ** 30


def make_budget_plan(config: ExperimentConfig) -> BudgetPlan:
    """Plan actual per-policy transitions at completed PPO update boundaries.

    Preserve every requested interval threshold and the final requested budget.
    Deduplicate only actual event steps; never interpolate an unattained step.
    Evaluation and periodic checkpoint schedules stay distinct. The later
    orchestrator also saves recovery checkpoints at each evaluation event.
    Integer arithmetic avoids float rounding even above 2**53 transitions.
    """
    validate_config(config, require_frozen_objectives=False)
    rollout = config.ppo.rollout_size
    requested = config.requested_timesteps_per_policy

    # Params: step; returns: (step + rollout - 1) // rollout * rollout.
    def completed_update(step):
        return ((step + rollout - 1) // rollout) * rollout

    # Params: interval; returns: (actual, mapping).
    def schedule(interval):
        thresholds = list(range(interval, requested + 1, interval))
        if not thresholds or thresholds[-1] != requested:
            thresholds.append(requested)
        mapping = tuple((step, completed_update(step)) for step in thresholds)
        actual = tuple(sorted({step for _, step in mapping}))
        return actual, mapping

    evaluations, eval_mapping = schedule(config.eval_interval)
    checkpoints, checkpoint_mapping = schedule(config.checkpoint_interval)
    return BudgetPlan(rollout_size=rollout, requested_total=requested,
                      effective_total=completed_update(requested),
                      eval_steps=evaluations, checkpoint_steps=checkpoints,
                      eval_schedule=eval_mapping, checkpoint_schedule=checkpoint_mapping)


def next_scheduled_step(steps: Sequence[int], current_step: int):
    """Return the first planned step strictly after current_step, or None.

    Require positive, sorted unique Python-integer schedule entries. Empty
    schedules are valid; the current step need not itself be a scheduled event.
    """
    if type(current_step) is not int:
        raise TypeError("current_step must be an integer, not a bool")
    if current_step < 0:
        raise ConfigError("current_step must be nonnegative")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise TypeError("steps must be a sequence of scheduled integers")
    previous = 0
    for step in steps:
        if type(step) is not int:
            raise TypeError("Scheduled steps must be Python integers, not bools")
        if step <= previous:
            raise ConfigError("Scheduled steps must be positive, sorted and unique")
        previous = step
    index = bisect_right(steps, current_step)
    return steps[index] if index < len(steps) else None


def training_transition_cost(config: ExperimentConfig) -> dict:
    """Return requested/effective training cost; exclude evaluation transitions."""
    plan = make_budget_plan(config)
    n_policies = len(config.run_seeds) * config.n_weights
    return {
        "n_policies": n_policies,
        "requested_timesteps_per_policy": plan.requested_total,
        "effective_timesteps_per_policy": plan.effective_total,
        "updates_per_policy": plan.effective_total // plan.rollout_size,
        "requested_training_transitions": n_policies * plan.requested_total,
        "effective_training_transitions": n_policies * plan.effective_total,
    }


# Params: root, namespace, identifier, width; returns: seed candidate.
def _seed_candidate(root, namespace, identifier, width):
    import numpy as np
    state = np.random.SeedSequence((_SEED_VERSION, namespace, root, identifier)).generate_state(1)
    return int(state[0]) % width


def _unique_seed_slots(root, namespace, identifiers, width):
    """Resolve rare collisions deterministically, retaining identifier order."""
    if len(identifiers) > width:
        raise ConfigError("Requested seed slots exceed the available namespace capacity")
    used, slots = set(), {}
    for identifier in identifiers:
        slot = _seed_candidate(root, namespace, identifier, width)
        while slot in used:
            slot = (slot + 1) % width
        used.add(slot)
        slots[identifier] = slot
    return slots


def make_seed_plan(config: ExperimentConfig, run_index: int) -> SeedPlan:
    """Derive explicit, reproducible seeds without touching global RNG or files.

    Training uses [0,2**31), intermediate evaluation [2**31,3*2**30),
    final testing [3*2**30,2**32). Replicates own non-overlapping contiguous
    n_envs-sized blocks, allocated by sorted run seed (not weight or run order).
    Learner/base seeds match; each worker seed is base + rank, as in SB3.

    Holdouts depend only on their own roots/namespaces, not on the run/weight or
    training root. Adding holdout episodes preserves the existing prefix. Save
    these integer lists before training; resume must read the persisted plans.
    Identical initial scenarios do not imply identical action-dependent noise.
    """
    validate_config(config, require_frozen_objectives=False)
    if type(run_index) is not int:
        raise TypeError("run_index must be an integer, not a bool")
    if not 0 <= run_index < len(config.run_seeds):
        raise ConfigError("run_index must identify a configured training replicate")
    n_envs = config.ppo.n_envs
    block_count = _TRAIN_WIDTH // n_envs
    if len(config.run_seeds) > block_count:
        raise ConfigError("Training replicates/workers exceed the disjoint training seed-domain capacity")
    if config.eval_episodes > _HOLDOUT_WIDTH or config.final_episodes > _HOLDOUT_WIDTH:
        raise ConfigError("Evaluation episode count exceeds its seed-domain capacity")
    blocks = _unique_seed_slots(config.experiment_seed, _TRAIN_NAMESPACE,
                                sorted(config.run_seeds), block_count)
    run_seed = config.run_seeds[run_index]
    base = blocks[run_seed] * n_envs
    evaluation = _unique_seed_slots(config.eval_seed_root, _EVAL_NAMESPACE,
                                    range(config.eval_episodes), _HOLDOUT_WIDTH)
    final = _unique_seed_slots(config.final_seed_root, _FINAL_NAMESPACE,
                               range(config.final_episodes), _HOLDOUT_WIDTH)
    return SeedPlan(run_seed=run_seed, run_index=run_index, learner_seed=base,
                    base_worker_seed=base,
                    worker_seeds=tuple(base + rank for rank in range(n_envs)),
                    eval_seeds=tuple(_TRAIN_WIDTH + slot for slot in evaluation.values()),
                    final_seeds=tuple(_TRAIN_WIDTH + _HOLDOUT_WIDTH + slot for slot in final.values()))


def make_weights(n_weights: int = 11):
    """Generate float32 [alpha, 1-alpha] in ascending tracking preference.

    Includes both endpoints and returns an independently owned array. Generate
    once for a new experiment; resume must validate its saved array instead.
    NumPy is imported only when this numerical operation is requested.
    """
    if type(n_weights) is not int:
        raise TypeError("n_weights must be an integer, not a bool or numeric schedule")
    if n_weights < 2:
        raise ConfigError("n_weights must be at least two to include both extreme policies")
    import numpy as np
    alpha = np.linspace(0.0, 1.0, n_weights, dtype=np.float32)
    return validate_weights(np.column_stack((alpha, np.float32(1.0) - alpha)))


def validate_weights(weights):
    """Validate and copy a nonempty float32 (W,2) preference array unchanged.

    Row sums use float64 accumulation and absolute tolerance 1e-6 (rtol=0).
    Reject negative/nonfinite entries and exact duplicate rows. Do not sort,
    round, cast, clip or renormalize saved preferences: row index is identity.
    A single saved preference is valid; a generated grid needs two endpoints.
    """
    import numpy as np
    if not isinstance(weights, np.ndarray) or weights.dtype != np.float32:
        raise TypeError("weights must be a float32 NumPy array; no implicit conversion is performed")
    if weights.ndim != 2 or weights.shape[1] != 2 or weights.shape[0] < 1:
        raise ConfigError("weights must have nonempty shape (W,2)")
    if not np.all(np.isfinite(weights)):
        raise ConfigError("weights must be finite")
    if np.any(weights < 0):
        raise ConfigError("weights must be nonnegative")
    if not np.all(np.isclose(weights.sum(axis=1, dtype=np.float64), 1.0,
                             rtol=0.0, atol=WEIGHT_SUM_ATOL)):
        raise ConfigError("Every weight row must sum to one within absolute tolerance 1e-6")
    if len(np.unique(weights, axis=0)) != len(weights):
        raise ConfigError("Exact duplicate weight rows are not allowed")
    return weights.copy(order="C")


# Params: value; returns: never (raises).
def _reject_constant(value):
    raise ConfigError(f"Nonfinite JSON constant {value!r} is not allowed")


# Params: pairs; returns: dict.
def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


# Params: value, allowed, location; returns: None.
def _check_keys(value, allowed, location):
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{location} keys must be strings")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ConfigError(f"Unknown {location} key(s): {', '.join(sorted(unknown))}")


# Params: record_type, values, location; returns: record instance.
def _construct(record_type, values, location):
    try:
        return record_type(**values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigError(f"{location}: {error}") from error


def load_config(path: Path, overrides: Mapping = None) -> ExperimentConfig:
    """Read a raw schema-v1 JSON and resolve every record default.

    Permitted overrides are output_dir and ppo={device: ...}; scientific values
    must be edited in the configuration, rather than silently overridden by CLI.
    Output paths resolve relative to the calling directory, without creation.
    This is preflight validation: no reference/scales are invented or frozen.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a path or string")
    config_path = Path(path).expanduser().absolute().resolve()
    try:
        with config_path.open("r", encoding="utf-8-sig") as source:
            data = json.load(source, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ConfigError(f"Cannot read configuration {config_path}: {error}") from error
    _check_keys(data, _ROOT_KEYS, "configuration")
    # Check input before applying overrides; an override cannot hide a typo.
    for name, record_type in _NESTED.items():
        if name in data:
            _check_keys(data[name], (f.name for f in fields(record_type)), name)
    if overrides is None:
        overrides = {}
    _check_keys(overrides, ("output_dir", "ppo"), "override")
    if "ppo" in overrides:
        _check_keys(overrides["ppo"], ("device",), "ppo override")

    # Nested inputs are copied; do not mutate the caller's overrides.
    values = dict(data)
    for name, record_type in _NESTED.items():
        nested = dict(data.get(name, {}))
        if name == "ppo":
            nested.update(overrides.get("ppo", {}))
        values[name] = _construct(record_type, nested, name)
    output = overrides.get("output_dir")
    if output is not None:
        if not isinstance(output, (str, Path)) or (isinstance(output, str) and not output.strip()):
            raise ConfigError("output_dir override must be a nonempty path")
        # Python 3.9 on this Windows host can leave nonexistent relative paths
        # relative after resolve(); make the calling-directory anchor explicit.
        output = Path(output).expanduser().absolute().resolve()
    values["output_dir"] = output
    config = _construct(ExperimentConfig, values, "configuration")
    validate_config(config, require_frozen_objectives=False)
    return config


def validate_config(config: ExperimentConfig, require_frozen_objectives: bool = True) -> None:
    """Validate structural contracts and optionally scientific training readiness.

    Device availability and environment runtime preflight belong to the later
    CLI; this pure validator checks supported device syntax and reward paths.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    if type(require_frozen_objectives) is not bool:
        raise TypeError("require_frozen_objectives must be a bool")
    # Re-run record validation so a malformed record cannot bypass this boundary.
    nested = {}
    for name, expected_type in _NESTED.items():
        record = getattr(config, name)
        if not isinstance(record, expected_type):
            raise ConfigError(f"{name} must be {expected_type.__name__}")
        nested[name] = _construct(expected_type, {f.name: getattr(record, f.name) for f in fields(record)}, name)
    checked = _construct(ExperimentConfig,
                         {f.name: nested.get(f.name, getattr(config, f.name)) for f in fields(config)},
                         "configuration")
    if checked.num_objectives != 2 or checked.reward_order != REWARD_ORDER:
        raise ConfigError("Exactly two maximized objectives in tracking/search order are required")
    if checked.env.mode != "combined" or checked.env.simple_obs or checked.env.observation_channels != 3:
        raise ConfigError("Schema v1 requires combined full three-channel observations")
    if checked.env.reward_type not in SUPPORTED_TRACKING_REWARDS:
        raise ConfigError("env.reward_type is unsupported for arbitrary combined decisions; "
                          "use R1/R2/R3/R6/R7. Current R4/R5 paths require a selected observed target "
                          "and cannot safely handle empty measurements/new discoveries")
    if checked.env.search_reward not in SUPPORTED_SEARCH_REWARDS:
        raise ConfigError("env.search_reward must be one of the implemented R1/R2/R3/R4/R5 rewards")
    if checked.ppo.n_envs > MAX_SEED + 1:
        raise ConfigError("ppo.n_envs exceeds the available base-plus-rank seed range")
    if not _DEVICE.fullmatch(checked.ppo.device):
        raise ConfigError("ppo.device must be auto, cpu, cuda[:index], mps[:index] or xpu[:index]")
    if checked.scale_status not in _SCALE_STATUSES:
        raise ConfigError("scale_status must be identity_baseline_frozen, frozen, calibrated_frozen or unselected")
    if checked.scale_status == "identity_baseline_frozen" and checked.objective_scales != (1.0, 1.0):
        raise ConfigError("identity_baseline_frozen requires objective_scales=[1.0, 1.0]")
    if checked.eval_seed_root == checked.final_seed_root:
        raise ConfigError("Evaluation and final-test seed roots must be distinct")
    if checked.logging.mode != "off" and checked.logging.project is None:
        raise ConfigError("Enabled external logging requires an explicitly supplied project")
    if require_frozen_objectives:
        if checked.reference_status != "frozen" or checked.reference_scaled is None:
            raise ConfigError("Training requires a selected reference_scaled and reference_status='frozen'; "
                              "the unselected example is for preflight/pilot use only")
        if checked.scale_status == "unselected":
            raise ConfigError("Training requires frozen objective scales")
