"""Pure position/covariance diagnostics, independent of plotting and PPO.

Position helpers preserve the saved-rollout definitions, including float32
arithmetic, valid-pair weighting and the legacy exception for no valid pairs.
Episode assembly translates that exception into a nullable metric with a reason.
Inputs are borrowed read-only; no helper mutates them or allocates resources.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
from .types import TargetObservationMetrics
from numbers import Integral
import numpy as np

IDX_X = 0
IDX_Y = 1
IDX_VAR_X = 4
IDX_VAR_Y = 5

INITIAL_CATALOGUE_META_KEYS = (
    "initial_catalogue_size",
    "initial_catalog_size",
    "n_known_targets",
    "num_known_targets",
    "initial_known_targets",
    "known_targets",
)


def initial_catalogue_size(meta: Dict) -> int | None:
    """Read the initial known-catalogue size from common metadata keys."""
    for key in INITIAL_CATALOGUE_META_KEYS:
        if key not in meta:
            continue

        value = meta[key]
        if isinstance(value, (list, tuple)):
            return len(value)
        return int(value)

    return None


def catalogue_mask(est_states: np.ndarray) -> np.ndarray:
    """Return a (T, N) mask indicating which estimated tracks are valid."""
    if est_states.ndim != 3:
        raise ValueError(
            "Expected est_states with shape (T, N, D), "
            f"got {est_states.shape}"
        )

    if est_states.shape[-1] > IDX_VAR_Y:
        covariance_is_valid = (
            (est_states[:, :, IDX_VAR_X] > 0.0)
            | (est_states[:, :, IDX_VAR_Y] > 0.0)
        )
        covariance_is_finite = np.isfinite(
            est_states[:, :, [IDX_VAR_X, IDX_VAR_Y]]
        ).all(axis=-1)
        return covariance_is_valid & covariance_is_finite

    positions = est_states[:, :, [IDX_X, IDX_Y]]
    return np.isfinite(positions).all(axis=-1) & np.any(
        positions != 0.0,
        axis=-1,
    )


def wrapped_difference(
    estimate: np.ndarray,
    truth: np.ndarray,
    world_size: float,
) -> np.ndarray:
    """Compute the shortest signed difference in a wrapped square world."""
    return (estimate - truth + world_size / 2.0) % world_size - world_size / 2.0


def compute_mean_positional_error(
    true_states: np.ndarray,
    est_states: np.ndarray,
    world_size: float,
    target_set: str,
    meta: Dict,
) -> float:
    """Compute mean Euclidean position error over valid target-time pairs.

    ``target_set='catalogue'`` includes every target while it has a valid
    catalogue estimate. ``target_set='initial'`` restricts the calculation to
    the first K targets, where K is read from meta.json. The latter is useful if
    discovery-dependent catalogue growth should not affect the tracking metric.
    """
    if true_states.ndim != 3 or est_states.ndim != 3:
        raise ValueError(
            "Expected true_states and est_states with shape (T, N, D), "
            f"got {true_states.shape} and {est_states.shape}"
        )

    if true_states.shape[-1] <= IDX_Y or est_states.shape[-1] <= IDX_Y:
        raise ValueError("State arrays must contain x and y at indices 0 and 1")

    T = min(true_states.shape[0], est_states.shape[0])
    N = min(true_states.shape[1], est_states.shape[1])

    if target_set == "initial":
        known_count = initial_catalogue_size(meta)
        if known_count is None:
            raise KeyError(
                "--error-target-set initial requires the initial catalogue "
                f"size in meta.json under one of {INITIAL_CATALOGUE_META_KEYS}"
            )
        N = min(N, known_count)

    truth_xy = true_states[:T, :N, [IDX_X, IDX_Y]]
    estimate_xy = est_states[:T, :N, [IDX_X, IDX_Y]]
    valid = catalogue_mask(est_states[:T, :N])
    valid &= np.isfinite(truth_xy).all(axis=-1)
    valid &= np.isfinite(estimate_xy).all(axis=-1)

    if not np.any(valid):
        raise ValueError("No valid catalogue target-time pairs found")

    position_difference = wrapped_difference(
        estimate=estimate_xy,
        truth=truth_xy,
        world_size=world_size,
    )
    positional_error = np.linalg.norm(position_difference, axis=-1)
    return float(np.mean(positional_error[valid]))


def compute_mean_final_positional_error(
    true_states: np.ndarray,
    est_states: np.ndarray,
    world_size: float,
    target_set: str,
    meta: Dict,
) -> float:
    """Mean wrapped error across valid target estimates at the final time.

    Catalogue mode includes initially known and subsequently discovered targets.
    Undetected targets have no estimate and are excluded, not treated as zeros.
    Initial mode retains the restriction to initially known targets.
    """
    if true_states.ndim != 3 or est_states.ndim != 3:
        raise ValueError("Expected state arrays with shape (T, N, D)")
    if true_states.shape[:2] != est_states.shape[:2]:
        raise ValueError("True and estimated states must align in time and target ID")
    if true_states.shape[0] == 0:
        raise ValueError("State arrays contain no final timestep")
    return compute_mean_positional_error(
        true_states[-1:], est_states[-1:], world_size, target_set, meta
    )


def mean_covariance_trace(est_states: np.ndarray, target_set: str,
                          known_count: int) -> float | None:
    """Mean positional covariance trace over valid post-action target-time pairs.

Accepts the trace's (D,N,9) estimates directly. ``catalogue`` includes valid
discovered tracks; ``initial`` restricts to the first known_count IDs. Position
variance entries 4 and 5 determine validity even for a track at the origin.
Empty/no-valid selections return None (episode assembly supplies the reason).
Float32 inputs retain the existing numerical precision and averaging convention.
"""
    if not isinstance(est_states, np.ndarray) or est_states.dtype.kind != 'f':
        raise TypeError('est_states must be a floating NumPy array')
    if est_states.ndim != 3 or est_states.shape[-1] != 9:
        raise ValueError('Expected est_states with shape (D,N,9)')
    if target_set not in ('catalogue', 'initial'):
        raise ValueError("target_set must be 'catalogue' or 'initial'")
    if isinstance(known_count, bool) or not isinstance(known_count, Integral):
        raise TypeError('known_count must be an integer')
    if not 0 <= known_count <= est_states.shape[1]:
        raise ValueError('known_count must be within the physical target count')
    selected = est_states if target_set == 'catalogue' else est_states[:, :known_count]
    valid = catalogue_mask(selected)
    if not np.any(valid):
        return None
    trace = selected[:, :, IDX_VAR_X] + selected[:, :, IDX_VAR_Y]
    value = float(np.mean(trace[valid]))
    if not np.isfinite(value):
        raise ValueError('Mean covariance trace is nonfinite')
    return value


def compute_observation_frequency_per_target(
    observed_targets: np.ndarray, elapsed_steps: np.ndarray, known_count: int
) -> List[TargetObservationMetrics]:
    """Observations per eligible simulated timestep, with equal weight per target.

    Count action-end measurement events used to update the catalogue, including
    the discovery measurement. Initial catalogue creation is not an observation.
    Initially known targets use the full elapsed episode duration from t0.
    New targets use end_time - detection_time + 1: only the detection timestep
    and subsequent timesteps count, not the preceding slew interval. A target
    first detected at the last timestep has one observation in one eligible
    timestep. Never-detected unknown targets have no frequency (None).
    """
    observed_targets = np.asarray(observed_targets)
    if observed_targets.ndim != 2 or observed_targets.shape[0] == 0:
        raise ValueError("Expected a nonempty (T, N) observation mask")
    if not np.isin(observed_targets, [0, 1]).all():
        raise ValueError("Observation mask must contain only booleans/zero/one")
    observed_targets = observed_targets.astype(bool)
    elapsed = np.asarray(elapsed_steps, dtype=float)
    if (elapsed.shape != (observed_targets.shape[0],)
            or not np.isfinite(elapsed).all() or np.any(elapsed <= 0)
            or np.any(elapsed != np.floor(elapsed))):
        raise ValueError("elapsed_steps must contain positive integer simulated durations")
    _known_count(known_count, observed_targets.shape[1])
    known_count = int(known_count)
    times = np.cumsum(elapsed)
    end_time = int(times[-1])
    result = []
    for target_id in range(observed_targets.shape[1]):
        indices = np.flatnonzero(observed_targets[:, target_id])
        known = target_id < known_count
        if known:
            start, duration = 0, end_time
        elif len(indices):
            start = int(times[indices[0]])
            duration = end_time - start + 1
        else:
            start, duration = None, None
        frequency = len(indices) / duration if duration is not None else None
        result.append(TargetObservationMetrics(
            target_id, known, len(indices), start, duration, frequency
        ))
    return result


def sample_std(values: Sequence[float]) -> float:
    """Return sample standard deviation, or zero for a single seed."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


# Params: values; returns: tuple[float, float].
def finite_mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.mean(values)), sample_std(values)

# Params: known_count, n_targets; returns: None.
def _known_count(known_count, n_targets):
    if isinstance(known_count, bool) or not isinstance(known_count, Integral):
        raise TypeError('known_count must be an integer')
    if not 0 <= known_count <= n_targets:
        raise ValueError('known_count must identify an initial prefix of physical IDs')


def count_unique_new_detections(observed_targets, known_count):
    """Count unknown physical IDs measured at least once, including action one."""
    observed = np.asarray(observed_targets)
    if observed.ndim != 2 or not np.isin(observed, [0, 1]).all():
        raise ValueError('Expected a binary (D,N) observation mask')
    _known_count(known_count, observed.shape[1])
    return int(np.count_nonzero(np.any(observed[:, known_count:], axis=0)))


def summarize_coverage(coverage_values):
    """Summarize per-decision means of the raw recency grid supplied by the env.

    These values are not means of the three-channel policy observation. The
    episode mean equally weights post-action samples, not elapsed durations.
    """
    values = np.asarray(coverage_values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in 'fiu':
        raise ValueError('coverage_values must be a nonempty numeric (D,) array')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('coverage_values must be finite and nonnegative')
    return {'mean_coverage': float(np.mean(values)), 'final_coverage': float(values[-1])}


def summarize_observation_frequencies(target_metrics):
    """Equal-target means; never-detected unknowns are excluded, known zeros kept."""
    items = tuple(target_metrics)
    if not all(isinstance(item, TargetObservationMetrics) for item in items):
        raise TypeError('Expected TargetObservationMetrics records')
    if len({item.target_id for item in items}) != len(items):
        raise ValueError('Target IDs must be unique')
    all_values = [item.observation_frequency for item in items
                  if item.observation_frequency is not None]
    known_values = [item.observation_frequency for item in items if item.initially_known]
    return {'mean_observation_frequency': float(np.mean(all_values)) if all_values else None,
            'mean_known_observation_frequency': float(np.mean(known_values)) if known_values else None}


def search_time_percentage(policy_modes, elapsed_steps):
    """Duration-weighted search percentage for explicit search/track labels only.

    None indicates a policy without an explicit controller mode (including PPO).
    Neither preference, reward nor the environment's combined mode classifies
    an action. No reconstruction or default action classification is performed.
    """
    if policy_modes is None:
        return None
    modes = np.asarray(policy_modes)
    elapsed = np.asarray(elapsed_steps, dtype=np.float64)
    if modes.ndim != 1 or modes.size == 0 or not np.isin(modes, ['search', 'track']).all():
        raise ValueError('policy_modes must contain one search/track label per decision')
    if elapsed.shape != modes.shape or not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError('elapsed_steps must contain one positive duration per decision')
    return float(100.0 * np.sum(elapsed[modes == 'search']) / np.sum(elapsed))


def summarize_episode(trace, env_config, metric_config):
    """Assemble diagnostics from aligned post-action EpisodeTrace arrays.

    metric_config is an ExperimentConfig or a mapping with error_target_set
    (catalogue/initial) and optional world_size. The unchanged SSA world is 100.
    Missing errors/frequencies use None plus an explicit reason. PPO traces
    contain no controller-mode labels, so their search time is always missing.
    """
    from collections.abc import Mapping
    from .types import EpisodeTrace, EpisodeMetrics, EnvConfig, ExperimentConfig
    if not isinstance(trace, EpisodeTrace) or not isinstance(env_config, EnvConfig):
        raise TypeError('Expected EpisodeTrace and EnvConfig')
    if isinstance(metric_config, ExperimentConfig):
        target_set, world_size = metric_config.error_target_set, 100.0
    elif isinstance(metric_config, Mapping):
        if set(metric_config) - {'error_target_set', 'world_size'}:
            raise ValueError('Unknown metric configuration keys')
        target_set = metric_config['error_target_set']
        world_size = metric_config.get('world_size', 100.0)
    else:
        raise TypeError('metric_config must be an ExperimentConfig or mapping')
    if target_set not in ('catalogue', 'initial'):
        raise ValueError('error_target_set must be catalogue or initial')
    if isinstance(world_size, bool) or not isinstance(world_size, (int, float, np.number)):
        raise TypeError('world_size must be numeric')
    if not np.isfinite(world_size) or world_size <= 0:
        raise ValueError('world_size must be finite and positive')
    if (trace.true_states.shape[1] != env_config.known_targets + env_config.unknown_targets
            or trace.initial_known_count != env_config.known_targets
            or trace.initial_observation.shape != env_config.observation_shape):
        raise ValueError('Trace target IDs, known count and grid must match environment configuration')
    reasons = {}
    meta = {'known_targets': trace.initial_known_count}
    errors = {}
    for name, function in (
            ('mean_positional_error', compute_mean_positional_error),
            ('mean_final_positional_error', compute_mean_final_positional_error)):
        try:
            errors[name] = function(trace.true_states, trace.estimated_states,
                                    world_size, target_set, meta)
        except ValueError as error:
            if str(error) != 'No valid catalogue target-time pairs found':
                raise
            errors[name] = None
            reasons[name] = 'No valid positional pairs in the selected target set/time samples'
    covariance = mean_covariance_trace(trace.estimated_states, target_set, trace.initial_known_count)
    if covariance is None:
        reasons['mean_covariance_trace'] = 'No valid covariance pairs in the selected target set'
    target_metrics = compute_observation_frequency_per_target(
        trace.observed_targets, trace.elapsed_steps, trace.initial_known_count)
    frequencies = summarize_observation_frequencies(target_metrics)
    for name, value in frequencies.items():
        if value is None:
            reasons[name] = ('No initially known targets' if 'known' in name
                             else 'No known or detected targets with defined observation frequency')
    reasons['search_time_percentage'] = 'PPO has no explicit search/track action mode'
    return EpisodeMetrics(**errors, mean_covariance_trace=covariance,
        **summarize_coverage(trace.coverage_values), **frequencies,
        unique_new_detections=count_unique_new_detections(trace.observed_targets, trace.initial_known_count),
        decisions=trace.decisions, elapsed_timesteps=sum(int(t) for t in trace.elapsed_steps),
        search_time_percentage=search_time_percentage(None, trace.elapsed_steps),
        target_observation_metrics=tuple(target_metrics), unavailable_reasons=reasons)
