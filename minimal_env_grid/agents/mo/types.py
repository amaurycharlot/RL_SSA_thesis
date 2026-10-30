"""Passive, immutable configuration/identity contracts for two-objective PPO.

Only standard-library imports at module load; NumPy is lazy for array records.
Lists supplied by callers are copied to tuples; nested settings are frozen records.
Construction validates local contracts but
never allocates an environment, derives random seeds, or executes training.
JSON loading, scientific validation and plan generation belong to config.py.
"""
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, ClassVar, Optional, Tuple

REWARD_ORDER = ("tracking", "search")
NUM_OBJECTIVES = 2
MAX_SEED = 2 ** 32 - 1


# Params: name, value, minimum, maximum; returns: None.
def _integer(name, value, minimum=0, maximum=None):
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its supported range")


# Params: name, value, minimum, maximum, strict_min, strict_max; returns: None.
def _number(name, value, minimum=None, maximum=None, strict_min=False, strict_max=False):
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a constant numeric value")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (value < minimum or (strict_min and value == minimum)):
        raise ValueError(f"{name} is below its supported range")
    if maximum is not None and (value > maximum or (strict_max and value == maximum)):
        raise ValueError(f"{name} is above its supported range")


# Params: name, value; returns: None.
def _boolean(name, value):
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


# Params: name, value; returns: None.
def _text(name, value):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")


# Params: record, name; returns: value.
def _tuple(record, name):
    value = getattr(record, name)
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    value = tuple(value)
    object.__setattr__(record, name, value)
    return value


# Params: record, name, nonempty; returns: values.
def _seeds(record, name, nonempty=True):
    values = _tuple(record, name)
    if nonempty and not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _integer(name, value, maximum=MAX_SEED)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain distinct seeds")
    return values


# Params: record, name, positive, simplex; returns: values.
def _pair(record, name, positive=False, simplex=False):
    values = _tuple(record, name)
    if len(values) != NUM_OBJECTIVES:
        raise ValueError(f"{name} must have exactly two components")
    for value in values:
        _number(name, value, minimum=0 if positive or simplex else None, strict_min=positive)
    if simplex and not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to one within 1e-6; no renormalization is performed")
    return values


# Params: value; returns: JSON-compatible value.
def _primitive(value):
    if is_dataclass(value):
        return {item.name: _primitive(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _primitive(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    # Lazy import: plain configuration serialization never loads NumPy.
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to primitive data")


class _PrimitiveRecord:
    def to_dict(self):
        """Return a fresh JSON-compatible mapping without changing the record."""
        return _primitive(self)


@dataclass(frozen=True)
class EnvConfig(_PrimitiveRecord):
    known_targets: int = 10
    unknown_targets: int = 10
    grid_shape: Tuple[int, int] = (10, 10)
    episode_steps: int = 200
    measurement_noise_std: float = 1.0
    zero_noise: bool = False
    reward_type: str = "R6"
    search_reward: str = "R2"
    dwell_steps: int = 1
    slew_alpha: float = 1.0
    uncertainty_norm_scale: float = 100.0

    # Schema v1 invariants, not constructor overrides or duplicate JSON settings.
    mode: ClassVar[str] = "combined"
    simple_obs: ClassVar[bool] = False
    num_objectives: ClassVar[int] = NUM_OBJECTIVES
    reward_order: ClassVar[Tuple[str, str]] = REWARD_ORDER
    observation_dtype: ClassVar[str] = "float32"
    observation_channels: ClassVar[int] = 3
    action_type: ClassVar[str] = "MultiDiscrete"

    # Params: self; returns: None.
    def __post_init__(self):
        for name in ("known_targets", "unknown_targets"):
            _integer(name, getattr(self, name))
        if self.known_targets + self.unknown_targets == 0:
            raise ValueError("At least one physical target is required")
        shape = _tuple(self, "grid_shape")
        if len(shape) != 2:
            raise ValueError("grid_shape must have exactly two dimensions")
        for dimension in shape:
            _integer("grid_shape", dimension, minimum=1)
        for name in ("episode_steps", "dwell_steps"):
            _integer(name, getattr(self, name), minimum=1)
        _number("measurement_noise_std", self.measurement_noise_std, minimum=0)
        _number("slew_alpha", self.slew_alpha, minimum=0)
        _number("uncertainty_norm_scale", self.uncertainty_norm_scale, minimum=0, strict_min=True)
        _boolean("zero_noise", self.zero_noise)
        _text("reward_type", self.reward_type)
        _text("search_reward", self.search_reward)

    # Params: self; returns: (self.reward_type, self.search_reward).
    @property
    def reward_pair(self):
        return (self.reward_type, self.search_reward)

    # Params: self; returns: self.grid_shape + (self.observation_channels,).
    @property
    def observation_shape(self):
        return self.grid_shape + (self.observation_channels,)

    # Params: self; returns: self.grid_shape.
    @property
    def action_nvec(self):
        return self.grid_shape


@dataclass(frozen=True)
class PPOConfig(_PrimitiveRecord):
    n_envs: int = 8
    n_steps: int = 512
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "auto"
    # Explicit SB3 defaults omitted by the legacy trainer.
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    stats_window_size: int = 100
    verbose: int = 1
    policy: ClassVar[str] = "MlpPolicy"
    features_extractor: ClassVar[str] = "agents.RL.grid_cnn_extractor.GridCNNExtractor"
    features_dim: ClassVar[int] = 128

    # Params: self; returns: None.
    def __post_init__(self):
        for name in ("n_envs", "n_steps", "batch_size", "n_epochs", "stats_window_size"):
            _integer(name, getattr(self, name), minimum=1)
        _number("learning_rate", self.learning_rate, minimum=0, strict_min=True)
        _number("gamma", self.gamma, minimum=0, maximum=1, strict_min=True)
        _number("gae_lambda", self.gae_lambda, minimum=0, maximum=1)
        _number("clip_range", self.clip_range, minimum=0, maximum=1, strict_min=True, strict_max=True)
        for name in ("ent_coef", "vf_coef"):
            _number(name, getattr(self, name), minimum=0)
        _number("max_grad_norm", self.max_grad_norm, minimum=0, strict_min=True)
        for name in ("clip_range_vf", "target_kl"):
            if getattr(self, name) is not None:
                _number(name, getattr(self, name), minimum=0, strict_min=True)
        _boolean("normalize_advantage", self.normalize_advantage)
        _boolean("use_sde", self.use_sde)
        if self.use_sde:
            raise ValueError("gSDE is incompatible with the fixed MultiDiscrete action space")
        _integer("sde_sample_freq", self.sde_sample_freq, minimum=-1)
        _integer("verbose", self.verbose, maximum=2)
        _text("device", self.device)
        if self.rollout_size % self.batch_size:
            raise ValueError("batch_size must divide n_envs * n_steps")
        if self.normalize_advantage and (self.batch_size < 2 or self.rollout_size < 2):
            raise ValueError("Advantage normalization requires batch and rollout sizes above one")

    # Params: self; returns: self.n_envs * self.n_steps.
    @property
    def rollout_size(self):
        return self.n_envs * self.n_steps


@dataclass(frozen=True)
class ExecutionConfig(_PrimitiveRecord):
    subproc: bool = True
    worker_threads: int = 1

    # Params: self; returns: None.
    def __post_init__(self):
        _boolean("subproc", self.subproc)
        _integer("worker_threads", self.worker_threads, minimum=1)


@dataclass(frozen=True)
class NormalizationConfig(_PrimitiveRecord):
    norm_obs: bool = False
    norm_reward: bool = False
    epsilon: float = 1e-8
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    gamma: Optional[float] = None

    # Params: self; returns: None.
    def __post_init__(self):
        for name in ("norm_obs", "norm_reward"):
            _boolean(name, getattr(self, name))
        for name in ("epsilon", "clip_obs", "clip_reward"):
            _number(name, getattr(self, name), minimum=0, strict_min=True)
        if self.gamma is not None:
            _number("normalization.gamma", self.gamma, minimum=0, maximum=1, strict_min=True)


@dataclass(frozen=True)
class LoggingConfig(_PrimitiveRecord):
    mode: str = "off"
    project: Optional[str] = None
    entity: Optional[str] = None

    # Params: self; returns: None.
    def __post_init__(self):
        _text("logging.mode", self.mode)
        if self.mode not in ("off", "offline", "online"):
            raise ValueError("logging.mode must be off, offline or online")
        for name in ("project", "entity"):
            if getattr(self, name) is not None:
                _text(name, getattr(self, name))


@dataclass(frozen=True)
class ExperimentConfig(_PrimitiveRecord):
    schema_version: int = 1
    experiment_id: str = "r6_r2_identity_v1"
    experiment_seed: int = 42
    run_seeds: Tuple[int, ...] = tuple(range(10))
    n_weights: int = 11
    reward_order: Tuple[str, str] = REWARD_ORDER
    objective_scales: Tuple[float, float] = (1.0, 1.0)
    scale_status: str = "identity_baseline_frozen"
    reference_scaled: Optional[Tuple[float, float]] = None
    reference_status: str = "unselected"
    requested_timesteps_per_policy: int = 500000
    eval_interval: int = 20000
    checkpoint_interval: int = 50000
    eval_episodes: int = 20
    final_episodes: int = 100
    eval_seed_root: int = 100000
    final_seed_root: int = 200000
    error_target_set: str = "catalogue"
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # CLI-supplied path: serialized but excluded from scientific hashing later.
    output_dir: Optional[Path] = None
    num_objectives: ClassVar[int] = NUM_OBJECTIVES

    # Params: self; returns: None.
    def __post_init__(self):
        _integer("schema_version", self.schema_version, minimum=1, maximum=1)
        _text("experiment_id", self.experiment_id)
        for name in ("experiment_seed", "eval_seed_root", "final_seed_root"):
            _integer(name, getattr(self, name), maximum=MAX_SEED)
        _seeds(self, "run_seeds")
        _integer("n_weights", self.n_weights, minimum=2)
        if _tuple(self, "reward_order") != REWARD_ORDER:
            raise ValueError("Reward order must be tracking, search")
        _pair(self, "objective_scales", positive=True)
        _text("scale_status", self.scale_status)
        if self.reference_scaled is not None:
            _pair(self, "reference_scaled")
        if self.reference_status not in ("unselected", "frozen"):
            raise ValueError("reference_status must be unselected or frozen")
        if (self.reference_scaled is None) != (self.reference_status == "unselected"):
            raise ValueError("Reference value and status must agree")
        for name in ("requested_timesteps_per_policy", "eval_interval", "checkpoint_interval",
                     "eval_episodes", "final_episodes"):
            _integer(name, getattr(self, name), minimum=1)
        if self.error_target_set not in ("catalogue", "initial"):
            raise ValueError("error_target_set must be catalogue or initial")
        for name, expected in (("env", EnvConfig), ("ppo", PPOConfig), ("execution", ExecutionConfig),
                               ("normalization", NormalizationConfig), ("logging", LoggingConfig)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.normalization.gamma is None:
            object.__setattr__(self, "normalization", replace(self.normalization, gamma=self.ppo.gamma))
        if self.normalization.gamma != self.ppo.gamma:
            raise ValueError("Normalization gamma must match PPO gamma")
        if self.output_dir is not None:
            if not isinstance(self.output_dir, (str, Path)):
                raise TypeError("output_dir must be a path or string")
            if isinstance(self.output_dir, str) and not self.output_dir.strip():
                raise ValueError("output_dir must be nonempty")
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    # Params: self; returns: self.env.reward_pair.
    @property
    def reward_pair(self):
        return self.env.reward_pair

    # Params: self; returns: '_'.join result.
    @property
    def reward_pair_id(self):
        return "_".join(self.reward_pair)


@dataclass(frozen=True)
class SeedPlan(_PrimitiveRecord):
    run_seed: int
    run_index: int
    learner_seed: int
    base_worker_seed: int
    worker_seeds: Tuple[int, ...]
    eval_seeds: Tuple[int, ...]
    final_seeds: Tuple[int, ...]

    # Params: self; returns: None.
    def __post_init__(self):
        _integer("run_index", self.run_index)
        for name in ("run_seed", "learner_seed", "base_worker_seed"):
            _integer(name, getattr(self, name), maximum=MAX_SEED)
        workers = _seeds(self, "worker_seeds")
        evaluation = _seeds(self, "eval_seeds")
        final = _seeds(self, "final_seeds")
        if workers != tuple(self.base_worker_seed + rank for rank in range(len(workers))):
            raise ValueError("Worker seeds must equal base_worker_seed plus rank")
        if self.learner_seed != self.base_worker_seed:
            raise ValueError("Learner and base worker seeds must match within a replicate")
        if set(evaluation) & set(final):
            raise ValueError("Intermediate and final evaluation seeds must be disjoint")
        if set(workers) & (set(evaluation) | set(final)):
            raise ValueError("Training and evaluation seed domains must be disjoint")


@dataclass(frozen=True)
class BudgetPlan(_PrimitiveRecord):
    rollout_size: int
    requested_total: int
    effective_total: int
    eval_steps: Tuple[int, ...]
    checkpoint_steps: Tuple[int, ...]
    # Separate pairs preserve thresholds that map to the same completed update.
    eval_schedule: Tuple[Tuple[int, int], ...]
    checkpoint_schedule: Tuple[Tuple[int, int], ...]

    # Params: self; returns: None.
    def __post_init__(self):
        for name in ("rollout_size", "requested_total", "effective_total"):
            _integer(name, getattr(self, name), minimum=1)
        expected = ((self.requested_total + self.rollout_size - 1) // self.rollout_size) * self.rollout_size
        if self.effective_total != expected:
            raise ValueError("effective_total must round requested_total up to a completed rollout")
        for name in ("eval_steps", "checkpoint_steps"):
            steps = _tuple(self, name)
            for step in steps:
                _integer(name, step, minimum=1)
                if step % self.rollout_size or step > self.effective_total:
                    raise ValueError(f"{name} must contain completed updates within the budget")
            if not steps or steps != tuple(sorted(set(steps))) or steps[-1] != self.effective_total:
                raise ValueError(f"{name} must be sorted, unique and include the final update")
        for name, steps in (("eval_schedule", self.eval_steps), ("checkpoint_schedule", self.checkpoint_steps)):
            pairs = _tuple(self, name)
            normalized = []
            for pair in pairs:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise TypeError(f"{name} entries must be requested/actual pairs")
                requested, actual = pair
                _integer(name, requested, minimum=1, maximum=self.requested_total)
                _integer(name, actual, minimum=1)
                if actual != ((requested + self.rollout_size - 1) // self.rollout_size) * self.rollout_size:
                    raise ValueError(f"{name} must map to the first completed update at or above its threshold")
                normalized.append((requested, actual))
            normalized = tuple(normalized)
            object.__setattr__(self, name, normalized)
            requested_steps = tuple(pair[0] for pair in normalized)
            if not normalized or requested_steps != tuple(sorted(set(requested_steps))):
                raise ValueError(f"{name} requested thresholds must be sorted and unique")
            if normalized[-1] != (self.requested_total, self.effective_total):
                raise ValueError(f"{name} must record the final requested/effective budget")
            if tuple(sorted(set(actual for _, actual in normalized))) != steps:
                raise ValueError(f"{name} must account for every actual scheduled step")


@dataclass(frozen=True)
class PolicyIdentity(_PrimitiveRecord):
    experiment_fingerprint: str
    run_seed: int
    run_index: int
    weight_index: int
    weight: Tuple[float, float]
    actual_step: int
    bundle_id: str

    # Params: self; returns: None.
    def __post_init__(self):
        _text("experiment_fingerprint", self.experiment_fingerprint)
        _text("bundle_id", self.bundle_id)
        _integer("run_seed", self.run_seed, maximum=MAX_SEED)
        for name in ("run_index", "weight_index", "actual_step"):
            _integer(name, getattr(self, name))
        _pair(self, "weight", simplex=True)


def _array(record, name, shape=None, dtype=None, kind=None, finite=True, readonly=True):
    """Copy arrays; frozen records use immutable bytes-backed storage.

    No silent precision changes: an explicitly required dtype is checked.
    Mutable result containers instead own independent writable copies.
    """
    import numpy as np
    value = np.asarray(getattr(record, name))
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {dtype}")
    if kind is not None and value.dtype.kind not in kind:
        raise TypeError(f"{name} has an unsupported dtype")
    if shape is not None and (value.ndim != len(shape) or any(
            expected is not None and actual != expected for actual, expected in zip(value.shape, shape))):
        raise ValueError(f"{name} must have shape {shape}, received {value.shape}")
    if value.dtype.kind not in "fibu":
        raise TypeError(f"{name} must be a numerical or boolean array")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if not finite and np.any(np.isinf(value)):
        raise ValueError(f"{name} cannot contain infinity")
    copied = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape) if readonly else value.copy()
    object.__setattr__(record, name, copied)
    return copied


# Params: value; returns: immutable value.
def _freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Metadata keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, Path) or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Metadata must use null, not nonfinite numbers")
        return value
    raise TypeError("Metadata must contain primitive data")


# Params: record, name; returns: None.
def _metadata(record, name):
    if not isinstance(getattr(record, name), Mapping):
        raise TypeError(f"{name} must be a mapping")
    object.__setattr__(record, name, _freeze(getattr(record, name)))


# Params: record, name, expected; returns: values.
def _records(record, name, expected):
    values = _tuple(record, name)
    if any(not isinstance(value, expected) for value in values):
        raise TypeError(f"{name} must contain {expected.__name__} records")
    return values


# Params: record, name; returns: values.
def _strings(record, name):
    values = _tuple(record, name)
    for value in values:
        _text(name, value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain distinct IDs")
    return values


@dataclass(frozen=True)
class TargetObservationMetrics(_PrimitiveRecord):
    target_id: int
    initially_known: bool
    observation_count: int
    counting_start_timestep: Optional[int]
    eligible_timesteps: Optional[int]
    observation_frequency: Optional[float]

    # Params: self; returns: None.
    def __post_init__(self):
        _integer("target_id", self.target_id)
        _boolean("initially_known", self.initially_known)
        _integer("observation_count", self.observation_count)
        missing = (self.counting_start_timestep, self.eligible_timesteps, self.observation_frequency)
        if not self.initially_known and self.observation_count == 0:
            if missing != (None, None, None):
                raise ValueError("Never-detected unknowns have undefined exposure and frequency")
            return
        if any(value is None for value in missing):
            raise ValueError("Known/detected targets require exposure and frequency")
        _integer("counting_start_timestep", self.counting_start_timestep)
        _integer("eligible_timesteps", self.eligible_timesteps, minimum=1)
        _number("observation_frequency", self.observation_frequency, minimum=0, maximum=1)
        if self.initially_known and self.counting_start_timestep != 0:
            raise ValueError("Known targets count from t0")
        if not self.initially_known and self.counting_start_timestep == 0:
            raise ValueError("Unknown discovery occurs at an action-end timestep after t0")
        if not math.isclose(self.observation_frequency, self.observation_count / self.eligible_timesteps,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Frequency must equal count divided by eligible timesteps")


@dataclass(frozen=True)
class EpisodeTrace(_PrimitiveRecord):
    initial_observation: Any
    observations: Any
    vector_rewards: Any
    true_states: Any
    estimated_states: Any
    actions: Any
    observed_targets: Any
    elapsed_steps: Any
    coverage_values: Any
    terminated: Any
    truncated: Any
    seed: int
    initial_known_count: int

    # Params: self; returns: None.
    def __post_init__(self):
        import numpy as np
        initial = _array(self, "initial_observation", (None, None, 3), "float32")
        if min(initial.shape[:2]) < 1:
            raise ValueError("Observation dimensions must be positive")
        obs = _array(self, "observations", (None,) + initial.shape, "float32")
        decisions = obs.shape[0]
        if decisions < 1:
            raise ValueError("An episode trace must contain a real decision")
        _array(self, "vector_rewards", (decisions, 2), "float32")
        states = _array(self, "true_states", (decisions, None, 4), kind="f")
        targets = states.shape[1]
        if targets < 1:
            raise ValueError("A trace needs at least one physical target")
        _array(self, "estimated_states", (decisions, targets, 9), kind="f", finite=False)
        actions = _array(self, "actions", (decisions, 2), kind="iu")
        if np.any(actions < 0) or np.any(actions >= np.asarray(initial.shape[:2])):
            raise ValueError("Actions must lie within the original grid")
        _array(self, "observed_targets", (decisions, targets), "bool")
        durations = _array(self, "elapsed_steps", (decisions,), kind="iu")
        if np.any(durations < 1):
            raise ValueError("Elapsed durations must be positive")
        _array(self, "coverage_values", (decisions,), kind="f")
        done = _array(self, "terminated", (decisions,), "bool")
        truncated = _array(self, "truncated", (decisions,), "bool")
        if np.any(done[:-1] | truncated[:-1]) or not (done[-1] or truncated[-1]):
            raise ValueError("A completed episode ends exactly on its final recorded decision")
        _integer("seed", self.seed, maximum=MAX_SEED)
        _integer("initial_known_count", self.initial_known_count, maximum=targets)

    # Params: self; returns: self.observations.shape[0].
    @property
    def decisions(self):
        return self.observations.shape[0]


@dataclass(frozen=True)
class EpisodeMetrics(_PrimitiveRecord):
    mean_positional_error: Optional[float]
    mean_final_positional_error: Optional[float]
    mean_covariance_trace: Optional[float]
    final_coverage: Optional[float]
    mean_coverage: Optional[float]
    unique_new_detections: int
    mean_known_observation_frequency: Optional[float]
    mean_observation_frequency: Optional[float]
    decisions: int
    elapsed_timesteps: int
    search_time_percentage: Optional[float] = None
    target_observation_metrics: Tuple[TargetObservationMetrics, ...] = ()
    unavailable_reasons: Mapping = field(default_factory=dict)

    # Params: self; returns: None.
    def __post_init__(self):
        _metadata(self, "unavailable_reasons")
        nullable = ("mean_positional_error", "mean_final_positional_error", "mean_covariance_trace",
                    "final_coverage", "mean_coverage", "mean_known_observation_frequency",
                    "mean_observation_frequency", "search_time_percentage")
        for name in nullable:
            value = getattr(self, name)
            if value is None:
                _text("unavailable_reasons." + name, self.unavailable_reasons.get(name))
            else:
                _number(name, value, minimum=0)
                if name in self.unavailable_reasons:
                    raise ValueError("Available metrics cannot have missing reasons")
                if "frequency" in name and value > 1:
                    raise ValueError("Observation frequencies must be at most one")
                if name == "search_time_percentage" and value > 100:
                    raise ValueError("Search percentage must be at most 100")
        if set(self.unavailable_reasons) - set(nullable):
            raise ValueError("Unknown unavailable metric name")
        _integer("unique_new_detections", self.unique_new_detections)
        _integer("decisions", self.decisions, minimum=1)
        _integer("elapsed_timesteps", self.elapsed_timesteps, minimum=self.decisions)
        targets = _records(self, "target_observation_metrics", TargetObservationMetrics)
        if len({target.target_id for target in targets}) != len(targets):
            raise ValueError("Target metric IDs must be unique")


@dataclass(frozen=True)
class EpisodeEvaluation(_PrimitiveRecord):
    return_raw: Any
    return_scaled: Any
    metrics: EpisodeMetrics
    trace: Optional[EpisodeTrace] = None

    # Params: self; returns: None.
    def __post_init__(self):
        _array(self, "return_raw", (2,), "float64")
        _array(self, "return_scaled", (2,), "float64")
        if not isinstance(self.metrics, EpisodeMetrics):
            raise TypeError("metrics must be EpisodeMetrics")
        if self.trace is not None:
            if not isinstance(self.trace, EpisodeTrace):
                raise TypeError("trace must be EpisodeTrace")
            if self.metrics.decisions != self.trace.decisions or self.metrics.elapsed_timesteps != int(self.trace.elapsed_steps.sum()):
                raise ValueError("Trace and metric time axes must agree")


@dataclass(frozen=True)
class EvaluationResult(_PrimitiveRecord):
    episode_seeds: Tuple[int, ...]
    returns_raw: Any
    returns_scaled: Any
    mean_raw: Any
    mean_scaled: Any
    sd_raw: Any
    sd_scaled: Any
    se_raw: Any
    se_scaled: Any
    episode_metrics: Tuple[EpisodeMetrics, ...]
    deterministic: bool
    actual_step: int
    identity: PolicyIdentity
    traces: Tuple[EpisodeTrace, ...] = ()

    # Params: self; returns: None.
    def __post_init__(self):
        import numpy as np
        episodes = len(_seeds(self, "episode_seeds"))
        raw = _array(self, "returns_raw", (episodes, 2), "float64")
        scaled = _array(self, "returns_scaled", (episodes, 2), "float64")
        for suffix, values in (("raw", raw), ("scaled", scaled)):
            expected_sd = values.std(axis=0, ddof=1) if episodes > 1 else np.zeros(2)
            for prefix, expected in (("mean", values.mean(axis=0)), ("sd", expected_sd),
                                     ("se", expected_sd / math.sqrt(episodes))):
                actual = _array(self, prefix + "_" + suffix, (2,), "float64")
                if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{prefix}_{suffix} must summarize the episode vectors")
        if len(_records(self, "episode_metrics", EpisodeMetrics)) != episodes:
            raise ValueError("One metric record is required per evaluation episode")
        _boolean("deterministic", self.deterministic)
        _integer("actual_step", self.actual_step)
        if not isinstance(self.identity, PolicyIdentity) or self.identity.actual_step != self.actual_step:
            raise ValueError("Evaluation step must match its policy identity")
        traces = _records(self, "traces", EpisodeTrace)
        if traces and (len(traces) != episodes or tuple(trace.seed for trace in traces) != self.episode_seeds):
            raise ValueError("Optional traces must match the ordered episode seeds")
        for trace, metric in zip(traces, self.episode_metrics):
            if trace.decisions != metric.decisions or int(trace.elapsed_steps.sum()) != metric.elapsed_timesteps:
                raise ValueError("Each retained trace must match its episode metric time axes")


# Params: record, name; returns: budgets.
def _budgets(record, name="actual_budgets"):
    import numpy as np
    budgets = _array(record, name, (None,), kind="iu")
    if not len(budgets) or np.any(budgets <= 0) or np.any(budgets[1:] <= budgets[:-1]):
        raise ValueError(f"{name} must be positive, strictly increasing steps")
    return budgets


# Params: raw, scaled, valid, episodes; returns: None.
def _missing_slots(raw, scaled, valid, episodes=None):
    import numpy as np
    if not np.all(np.isfinite(raw[valid])) or not np.all(np.isfinite(scaled[valid])):
        raise ValueError("Valid policy slots must contain finite objective means")
    if not np.all(np.isnan(raw[~valid])) or not np.all(np.isnan(scaled[~valid])):
        raise ValueError("Missing policy slots must remain NaN")
    if episodes is not None and (not np.all(np.isfinite(episodes[valid])) or not np.all(np.isnan(episodes[~valid]))):
        raise ValueError("Episode returns must agree with slot validity")


@dataclass
class ExperimentResults:
    """Exclusively owned, writable arrays; call validate() after population.

    Object tables are also copied at construction. NPZ persistence, rather than
    primitive JSON, should store these NaN-bearing numerical arrays.
    """
    means_raw: Any
    means_scaled: Any
    validity: Any
    actual_budgets: Any
    hypervolume: Any
    episode_returns: Optional[Any] = None
    identities: dict = field(default_factory=dict)
    episode_metric_tables: dict = field(default_factory=dict)

    # Params: self; returns: None.
    def __post_init__(self):
        raw = _array(self, "means_raw", (None, None, None, 2), "float64", finite=False, readonly=False)
        if min(raw.shape[:3]) < 1:
            raise ValueError("Result dimensions must be positive")
        _array(self, "means_scaled", raw.shape, "float64", finite=False, readonly=False)
        _array(self, "validity", raw.shape[:3], "bool", readonly=False)
        budgets = _budgets(self)
        if len(budgets) != raw.shape[1]:
            raise ValueError("Budget axis must match result slots")
        _array(self, "hypervolume", raw.shape[:2], "float64", finite=False, readonly=False)
        if self.episode_returns is not None:
            episodes = _array(self, "episode_returns", raw.shape[:3] + (None, 2), "float64", finite=False, readonly=False)
            if episodes.shape[-2] < 1:
                raise ValueError("Episode count must be positive")
        if not isinstance(self.identities, dict) or not isinstance(self.episode_metric_tables, dict):
            raise TypeError("Result tables must be dictionaries")
        self.identities = self.identities.copy()
        self.episode_metric_tables = {key: tuple(value) for key, value in self.episode_metric_tables.items()}
        self.validate()

    # Params: cls, n_seeds, actual_budgets, n_weights, n_episodes; returns: result container.
    @classmethod
    def empty(cls, n_seeds, actual_budgets, n_weights, n_episodes=None):
        import numpy as np
        _integer("n_seeds", n_seeds, minimum=1)
        _integer("n_weights", n_weights, minimum=1)
        budgets = np.asarray(actual_budgets)
        if budgets.ndim != 1:
            raise ValueError("actual_budgets must be one-dimensional")
        shape = (n_seeds, len(budgets), n_weights)
        episodes = None
        if n_episodes is not None:
            _integer("n_episodes", n_episodes, minimum=1)
            episodes = np.full(shape + (n_episodes, 2), np.nan, dtype=np.float64)
        return cls(np.full(shape + (2,), np.nan), np.full(shape + (2,), np.nan),
                   np.zeros(shape, dtype=bool), budgets, np.full(shape[:2], np.nan), episodes)

    # Params: self; returns: None.
    def validate(self):
        import numpy as np
        shape = self.means_raw.shape
        if len(shape) != 4 or shape[-1] != 2 or self.means_scaled.shape != shape or self.validity.shape != shape[:3]:
            raise ValueError("Result axes must match (S,K,W,2)/(S,K,W)")
        if self.means_raw.dtype != np.float64 or self.means_scaled.dtype != np.float64 or self.validity.dtype != np.bool_:
            raise TypeError("Result means must remain float64 and validity boolean")
        if self.actual_budgets.shape != (shape[1],) or self.hypervolume.shape != shape[:2]:
            raise ValueError("Budget/hypervolume axes must match result slots")
        if self.episode_returns is not None and (self.episode_returns.ndim != 5 or
                self.episode_returns.shape[:3] != shape[:3] or self.episode_returns.shape[-1] != 2):
            raise ValueError("Episode return axes must be (S,K,W,E,2)")
        _missing_slots(self.means_raw, self.means_scaled, self.validity, self.episode_returns)
        complete = self.validity.all(axis=2)
        if np.any(np.isinf(self.hypervolume)) or np.any(np.isfinite(self.hypervolume) & ~complete):
            raise ValueError("Hypervolume is unavailable for an incomplete policy set")
        if np.any(self.hypervolume[np.isfinite(self.hypervolume)] < 0):
            raise ValueError("Hypervolume must be nonnegative")
        for table in (self.identities, self.episode_metric_tables):
            for key in table:
                if not isinstance(key, tuple) or len(key) != 3 or any(type(v) is not int or v < 0 or v >= shape[i] for i, v in enumerate(key)):
                    raise ValueError("Result table keys must be in-range (seed,budget,weight) slots")
                if not self.validity[key]:
                    raise ValueError("Missing slots cannot have populated record tables")
        for (seed, budget, weight), identity in self.identities.items():
            if not isinstance(identity, PolicyIdentity) or (identity.run_index, identity.weight_index, identity.actual_step) != (seed, weight, int(self.actual_budgets[budget])):
                raise ValueError("Policy identity must match its result slot")
        for values in self.episode_metric_tables.values():
            if any(not isinstance(value, EpisodeMetrics) for value in values):
                raise TypeError("Episode tables must contain EpisodeMetrics")
            if not values or (self.episode_returns is not None and len(values) != self.episode_returns.shape[-2]):
                raise ValueError("Episode metric tables must match the stored episode count")


@dataclass(frozen=True)
class ObsNormalizer:
    """Immutable observation-only RMS snapshot; no reward or running statistics."""
    mean: Any
    variance: Any
    count: float
    enabled: bool = True
    epsilon: float = 1e-8
    clip_obs: float = 10.0

    # Params: self; returns: None.
    def __post_init__(self):
        import numpy as np
        mean = _array(self, "mean", (None, None, 3), kind="f")
        variance = _array(self, "variance", mean.shape, kind="f")
        if min(mean.shape[:2]) < 1 or np.any(variance < 0):
            raise ValueError("RMS shape must be positive and variance nonnegative")
        _number("count", self.count, minimum=0)
        _boolean("enabled", self.enabled)
        _number("epsilon", self.epsilon, minimum=0, strict_min=True)
        _number("clip_obs", self.clip_obs, minimum=0, strict_min=True)

    # Params: self, raw_obs; returns: normalized float32 array.
    def normalize(self, raw_obs):
        import numpy as np
        obs = np.asarray(raw_obs)
        if obs.shape != self.mean.shape or obs.dtype != np.float32 or not np.all(np.isfinite(obs)):
            raise ValueError("Expected a finite float32 raw observation matching RMS shape")
        if not self.enabled:
            return obs.copy()
        return np.clip((obs - self.mean) / np.sqrt(self.variance + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)


class ResourceCleanupError(RuntimeError):
    """All children were attempted; retains individual cleanup errors on Python 3.9."""
    # Params: self, errors; returns: None.
    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__(f"{len(self.errors)} resource cleanup operation(s) failed")


def _close_resources(roots, workers=()):
    # SB3 wrappers own their .venv chain. Avoid closing successful chains twice;
    # if an outer close fails, still attempt every partially constructed child.
    errors, covered = [], set()
    for root in roots:
        child, chain = root, []
        while child is not None and id(child) not in {id(value) for value in chain}:
            chain.append(child)
            child = getattr(child, "venv", None)
        for position, resource in enumerate(chain):
            if id(resource) in covered:
                continue
            covered.add(id(resource))
            try:
                resource.close()
            except Exception as error:
                errors.append(error)
            else:
                covered.update(id(value) for value in chain[position:])
                break
    for worker in workers:
        if id(worker) in covered:
            continue
        covered.add(id(worker))
        try:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
            if worker.is_alive():
                raise RuntimeError("Worker remained alive after termination")
        except Exception as error:
            errors.append(error)
    if errors:
        raise ResourceCleanupError(errors)


class _OwnedResources:
    _closed = False

    # Params: self; returns: self._closed.
    @property
    def closed(self):
        return self._closed

    # Params: self; returns: self.
    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot enter a closed resource owner")
        return self

    # Params: self, exc_type, exc, traceback; returns: bool.
    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


@dataclass
class TrainingEnvHandle(_OwnedResources):
    """Incremental owner: construct empty before allocating any child resources.

    Register each child as it is created inside a with block. Wrapper chains are
    owned once; explicit workers are fallback cleanup for failed/partial startup.
    """
    vec_env: Any = None
    vec_normalize: Any = None
    raw_vec_env: Any = None
    workers: Tuple[Any, ...] = ()
    _closed: bool = field(default=False, init=False, repr=False)

    # Params: self; returns: None.
    def __post_init__(self):
        self.workers = tuple(self.workers)

    # Params: self; returns: None.
    def close(self):
        if self._closed:
            return
        self._closed = True
        _close_resources((self.vec_env, self.vec_normalize, self.raw_vec_env), self.workers)


@dataclass
class PolicyBundle(_OwnedResources):
    """Loaded policy plus borrowed-or-owned matching observation normalizer.

    Only normalization_env/owned_resources are closed. A normalizer borrowed
    from a live training handle is not an owned resource and is never closed.
    Construct the owner first and register children during partial loading.
    """
    model: Any = None
    vec_normalize: Any = None
    identity: Optional[PolicyIdentity] = None
    config_fingerprint: Optional[str] = None
    metadata: Mapping = field(default_factory=dict)
    normalization_env: Any = None
    owned_resources: Tuple[Any, ...] = ()
    _closed: bool = field(default=False, init=False, repr=False)

    # Params: self; returns: None.
    def __post_init__(self):
        if self.identity is not None and not isinstance(self.identity, PolicyIdentity):
            raise TypeError("identity must be PolicyIdentity")
        if self.config_fingerprint is not None:
            _text("config_fingerprint", self.config_fingerprint)
            if self.identity is not None and self.config_fingerprint != self.identity.experiment_fingerprint:
                raise ValueError("Bundle fingerprint must match its identity")
        _metadata(self, "metadata")
        self.owned_resources = tuple(self.owned_resources)

    # Params: self; returns: None.
    def close(self):
        if self._closed:
            return
        self._closed = True
        _close_resources((self.normalization_env,) + self.owned_resources)


@dataclass(frozen=True)
class ResumeState(_PrimitiveRecord):
    complete: bool
    current_step: int
    effective_target: int
    selected_bundle_path: Optional[Path] = None
    last_committed_checkpoint: Optional[Path] = None
    completed_evaluation_ids: Tuple[str, ...] = ()
    next_evaluation_budgets: Tuple[int, ...] = ()
    next_checkpoint_budgets: Tuple[int, ...] = ()
    resume_attempt_id: str = "initial"
    canonical_attempt_ids: Tuple[str, ...] = ()
    superseded_attempt_ids: Tuple[str, ...] = ()

    # Params: self; returns: None.
    def __post_init__(self):
        _boolean("complete", self.complete)
        _integer("effective_target", self.effective_target, minimum=1)
        _integer("current_step", self.current_step, maximum=self.effective_target)
        _text("resume_attempt_id", self.resume_attempt_id)
        for name in ("selected_bundle_path", "last_committed_checkpoint"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, Path)):
                    raise TypeError(f"{name} must be a path")
                object.__setattr__(self, name, Path(value))
        for name in ("completed_evaluation_ids", "canonical_attempt_ids", "superseded_attempt_ids"):
            _strings(self, name)
        if set(self.canonical_attempt_ids) & set(self.superseded_attempt_ids):
            raise ValueError("Canonical and superseded attempts must be disjoint")
        for name in ("next_evaluation_budgets", "next_checkpoint_budgets"):
            values = _tuple(self, name)
            for value in values:
                _integer(name, value, minimum=self.current_step + 1, maximum=self.effective_target)
            if values != tuple(sorted(set(values))):
                raise ValueError("Pending budgets must be sorted and unique")
            if self.complete and values:
                raise ValueError("Completed policies cannot have pending budgets")
        if self.complete and self.current_step != self.effective_target:
            raise ValueError("Completion requires the effective target")


@dataclass(frozen=True)
class PolicyTrainingSummary(_PrimitiveRecord):
    identity: PolicyIdentity
    requested_transitions: int
    effective_transitions: int
    actual_transitions: int
    complete: bool
    final_bundle_path: Optional[Path] = None
    committed_evaluation_ids: Tuple[str, ...] = ()
    failure_metadata: Mapping = field(default_factory=dict)
    resume_metadata: Mapping = field(default_factory=dict)

    # Params: self; returns: None.
    def __post_init__(self):
        if not isinstance(self.identity, PolicyIdentity):
            raise TypeError("identity must be PolicyIdentity")
        _integer("requested_transitions", self.requested_transitions, minimum=1)
        _integer("effective_transitions", self.effective_transitions, minimum=self.requested_transitions)
        _integer("actual_transitions", self.actual_transitions, maximum=self.effective_transitions)
        _boolean("complete", self.complete)
        if self.actual_transitions != self.identity.actual_step:
            raise ValueError("Summary actual transitions must match policy identity")
        if self.complete and (self.actual_transitions != self.effective_transitions or self.final_bundle_path is None):
            raise ValueError("Completed policy requires the final step and bundle path")
        if self.final_bundle_path is not None:
            if not isinstance(self.final_bundle_path, (str, Path)):
                raise TypeError("final_bundle_path must be a path")
            object.__setattr__(self, "final_bundle_path", Path(self.final_bundle_path))
        _strings(self, "committed_evaluation_ids")
        _metadata(self, "failure_metadata")
        _metadata(self, "resume_metadata")


@dataclass(frozen=True)
class ReportManifest(_PrimitiveRecord):
    relative_paths: Tuple[str, ...]
    file_hashes: Mapping
    experiment_fingerprint: str
    represented_budgets: Tuple[int, ...]
    seed_counts: Tuple[int, ...]
    generated_at: str

    # Params: self; returns: None.
    def __post_init__(self):
        paths = _strings(self, "relative_paths")
        for value in paths:
            path = Path(value)
            if path.is_absolute() or path.drive or ".." in path.parts:
                raise ValueError("Report paths must stay relative to the experiment")
        _metadata(self, "file_hashes")
        if set(paths) != set(self.file_hashes):
            raise ValueError("Each report requires exactly one file hash")
        for digest in self.file_hashes.values():
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("File hashes must be lowercase SHA-256 hex strings")
        _text("experiment_fingerprint", self.experiment_fingerprint)
        _text("generated_at", self.generated_at)
        budgets = _tuple(self, "represented_budgets")
        counts = _tuple(self, "seed_counts")
        if len(budgets) != len(counts):
            raise ValueError("Report budget and seed-count axes must match")
        for value in budgets:
            _integer("represented_budgets", value, minimum=1)
        if budgets != tuple(sorted(set(budgets))):
            raise ValueError("Report budgets must be sorted and unique")
        for count in counts:
            _integer("seed_counts", count)


@dataclass(frozen=True)
class ExperimentSummary(_PrimitiveRecord):
    experiment_fingerprint: str
    experiment_path: Path
    policies: Tuple[PolicyTrainingSummary, ...]
    completed_policies: int
    total_policies: int
    effective_total_transitions: int
    report_manifest: Optional[ReportManifest] = None

    # Params: self; returns: None.
    def __post_init__(self):
        _text("experiment_fingerprint", self.experiment_fingerprint)
        if not isinstance(self.experiment_path, (str, Path)):
            raise TypeError("experiment_path must be a path")
        object.__setattr__(self, "experiment_path", Path(self.experiment_path))
        policies = _records(self, "policies", PolicyTrainingSummary)
        _integer("total_policies", self.total_policies, minimum=1)
        _integer("completed_policies", self.completed_policies, maximum=self.total_policies)
        _integer("effective_total_transitions", self.effective_total_transitions, minimum=1)
        if len(policies) != self.total_policies or self.completed_policies != sum(p.complete for p in policies):
            raise ValueError("Completion counts must match policy summaries")
        if self.effective_total_transitions != sum(p.effective_transitions for p in policies):
            raise ValueError("Effective experiment budget must sum per-policy budgets")
        if any(p.identity.experiment_fingerprint != self.experiment_fingerprint for p in policies):
            raise ValueError("All policies must belong to the same experiment")
        if len({(p.identity.run_index, p.identity.weight_index) for p in policies}) != len(policies):
            raise ValueError("Policy slots must be unique")
        if self.report_manifest is not None and (not isinstance(self.report_manifest, ReportManifest) or
                self.report_manifest.experiment_fingerprint != self.experiment_fingerprint):
            raise ValueError("Report manifest must match the experiment")


@dataclass(frozen=True)
class FinalResults:
    means_raw: Any
    means_scaled: Any
    episode_returns_raw: Any
    episode_returns_scaled: Any
    validity: Any
    test_seeds: Tuple[int, ...]
    front_indices: Tuple[Tuple[int, ...], ...]
    hypervolume: Any
    operational_metrics: Tuple[Tuple[Tuple[EpisodeMetrics, ...], ...], ...] = ()

    # Params: self; returns: None.
    def __post_init__(self):
        import numpy as np
        episodes = len(_seeds(self, "test_seeds"))
        raw = _array(self, "means_raw", (None, None, 2), "float64", finite=False)
        seeds, weights = raw.shape[:2]
        if min(seeds, weights) < 1:
            raise ValueError("Final result axes must be positive")
        scaled = _array(self, "means_scaled", raw.shape, "float64", finite=False)
        valid = _array(self, "validity", (seeds, weights), "bool")
        for name, means in (("episode_returns_raw", raw), ("episode_returns_scaled", scaled)):
            values = _array(self, name, (seeds, weights, episodes, 2), "float64", finite=False)
            _missing_slots(means, means, valid, values)
            if not np.allclose(values[valid].mean(axis=1), means[valid], rtol=1e-12, atol=1e-12):
                raise ValueError("Final means must average each policy's episode vectors")
        hv = _array(self, "hypervolume", (seeds,), "float64", finite=False)
        fronts = _tuple(self, "front_indices")
        if len(fronts) != seeds:
            raise ValueError("One front is required per training seed")
        normalized = []
        for index, front in enumerate(fronts):
            if not isinstance(front, (list, tuple)):
                raise TypeError("Front indices must be sequences")
            front = tuple(front)
            for weight in front:
                _integer("front_indices", weight, maximum=weights - 1)
                if not valid[index, weight]:
                    raise ValueError("Front members must have valid final results")
            if len(set(front)) != len(front):
                raise ValueError("Front indices must be unique")
            if not valid[index].all() and (front or not np.isnan(hv[index])):
                raise ValueError("An incomplete final policy set has no front or hypervolume")
            normalized.append(front)
        if np.any(hv[np.isfinite(hv)] < 0):
            raise ValueError("Hypervolume must be nonnegative")
        object.__setattr__(self, "front_indices", tuple(normalized))
        tables = _tuple(self, "operational_metrics")
        if tables:
            if len(tables) != seeds or any(len(row) != weights for row in tables):
                raise ValueError("Operational metric axes must match final policy slots")
            copied = []
            for seed, row in enumerate(tables):
                copied_row = []
                for weight, entries in enumerate(row):
                    entries = tuple(entries)
                    if len(entries) != (episodes if valid[seed, weight] else 0) or any(not isinstance(item, EpisodeMetrics) for item in entries):
                        raise ValueError("Final operational metrics must match valid episode slots")
                    copied_row.append(entries)
                copied.append(tuple(copied_row))
            object.__setattr__(self, "operational_metrics", tuple(copied))

    # Params: cls, n_seeds, n_weights, test_seeds; returns: result container.
    @classmethod
    def empty(cls, n_seeds, n_weights, test_seeds):
        import numpy as np
        _integer("n_seeds", n_seeds, minimum=1)
        _integer("n_weights", n_weights, minimum=1)
        test_seeds = tuple(test_seeds)
        shape = (n_seeds, n_weights)
        episodes = shape + (len(test_seeds), 2)
        return cls(np.full(shape + (2,), np.nan), np.full(shape + (2,), np.nan),
                   np.full(episodes, np.nan), np.full(episodes, np.nan), np.zeros(shape, dtype=bool),
                   test_seeds, tuple(() for _ in range(n_seeds)), np.full(n_seeds, np.nan))


@dataclass(frozen=True)
class HVSummary(_PrimitiveRecord):
    actual_steps: Any
    mean: Any
    sd: Any
    complete_seed_counts: Any
    declared_seed_count: int
    completeness: Any

    # Params: self; returns: None.
    def __post_init__(self):
        import numpy as np
        steps = _budgets(self, "actual_steps")
        shape = steps.shape
        mean = _array(self, "mean", shape, "float64", finite=False)
        sd = _array(self, "sd", shape, "float64", finite=False)
        counts = _array(self, "complete_seed_counts", shape, kind="iu")
        complete = _array(self, "completeness", shape, "bool")
        _integer("declared_seed_count", self.declared_seed_count, minimum=1)
        if np.any(counts > self.declared_seed_count) or np.any(complete != (counts == self.declared_seed_count)):
            raise ValueError("Completeness must reflect all declared training seeds")
        if not np.all(np.isfinite(mean[complete])) or not np.all(np.isnan(mean[~complete])):
            raise ValueError("Incomplete seed sets must not have a reported mean HV")
        if np.any(mean[complete] < 0):
            raise ValueError("Mean hypervolume must be nonnegative")
        if not np.all(np.isfinite(sd[complete])) or not np.all(np.isnan(sd[~complete])) or np.any(sd[complete] < 0):
            raise ValueError("Sample SD must be available only for complete seed sets")
        if np.any(sd[complete & (counts == 1)] != 0):
            raise ValueError("Sample SD is zero for one seed")
