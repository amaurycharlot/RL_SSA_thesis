"""Shared experiment initialization/loading, without learners or global results.

Initialization is exclusive and explicit. Loading is read-only and validates the
same immutable files and source/package provenance as sequential resume.
Neither helper reconciles policies, opens a logger or runs evaluations.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Tuple

from .config import validate_config
from .types import BudgetPlan, ExperimentConfig, SeedPlan


@dataclass(frozen=True)
class InitializedExperiment:
    """The complete saved policy grid; preference storage is immutable."""

    config: ExperimentConfig
    root: Path
    weights: Any
    seed_plans: Tuple[SeedPlan, ...]
    budget_plan: BudgetPlan
    fingerprint: str


# Params: config, root, weights, plans, budget; returns: InitializedExperiment.
def _context(config, root, weights, plans, budget):
    import numpy as np
    from . import storage
    # A bytes-backed array cannot be made writable by a policy caller.
    weights = np.frombuffer(weights.tobytes(), dtype=weights.dtype).reshape(weights.shape)
    return InitializedExperiment(config, root, weights, tuple(plans), budget,
        storage._read_json(root / 'manifest.json')['fingerprint'])


def initialize_experiment(config, experiment_dir=None):
    """Commit one complete grid using the established initialization format.

    Refuse nonempty destinations and unfrozen scientific inputs before training.
    An explicit destination overrides output_dir without changing the caller's
    config or the identity algorithm. Source/version provenance is unchanged.
    """
    from . import config as planning, storage
    validate_config(config)
    if experiment_dir is not None:
        config = replace(config, output_dir=Path(experiment_dir))
    weights = planning.make_weights(config.n_weights)
    plans = tuple(planning.make_seed_plan(config, i) for i in range(len(config.run_seeds)))
    budget = planning.make_budget_plan(config)
    root = storage.create_experiment_dir(config, weights, plans, budget)
    return _context(config, root, weights, plans, budget)


def load_experiment(experiment_dir, config=None):
    """Read and validate a committed experiment without regenerating its plans.

    With no config, restore every saved setting including logging destinations;
    root is the current experiment directory, so relocated copies are supported.
    A supplied config is checked against saved science, allowing the established
    output/logging destination overrides. Runtime source/package changes remain
    incompatible. No policy recovery or shared-result writes occur here.
    """
    import numpy as np
    from . import config as planning, storage
    root = Path(experiment_dir).expanduser().absolute().resolve()
    if config is not None:
        validate_config(config)
    if not root.is_dir() or (root / '.INITIALIZING').exists():
        raise storage.StorageError('Loading requires a committed experiment initialization')
    try:
        root, payload, _ = storage._evaluation_context(root, config)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise storage.StorageError(f'Cannot load experiment {root}: {error}') from error
    if config is None:
        values = dict(payload['config'])
        values['logging'] = storage._read_json(root / 'config.json')['logging']
        for name, record_type in planning._NESTED.items():
            values[name] = record_type(**values[name])
        config = ExperimentConfig(**values, output_dir=root)
    validate_config(config)
    # Read the actual saved arrays/plans, preserving custom order and exact bits.
    weights = planning.validate_weights(np.load(root / 'weights.npy', allow_pickle=False))
    plans = tuple(SeedPlan(**p) for p in storage._read_json(root / 'seed_plan.json'))
    budget = BudgetPlan(**storage._read_json(root / 'budget_plan.json'))
    storage.create_experiment_dir(replace(config, output_dir=root), weights, plans, budget, resume=True)
    return _context(config, root, weights, plans, budget)


def task_to_policy(task_id, config):
    """Map a zero-based array task ID to its saved seed/preference indices.

    Tasks enumerate the complete grid in seed-major, weight-minor order:
    ``seed_index, weight_index = divmod(task_id, config.n_weights)``.
    Use the returned indices with the full saved weight matrix and seed plans;
    run-seed values and preference order are never changed. Mapping is pure and
    also supports structurally valid preflight configs with unfrozen objectives.
    """
    from numbers import Integral
    validate_config(config, require_frozen_objectives=False)
    if isinstance(task_id, bool) or not isinstance(task_id, Integral):
        raise TypeError('task_id must be an integer, not bool')
    task_id = int(task_id)
    task_count = len(config.run_seeds) * config.n_weights
    if not 0 <= task_id < task_count:
        raise ValueError(f'task_id must be in [0, {task_count}); received {task_id}')
    return divmod(task_id, config.n_weights)
