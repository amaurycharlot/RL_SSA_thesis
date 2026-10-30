"""Plot the search--tracking trade-off of combined heuristic policies.

Each rollout directory must follow the naming convention

    heuristic_combined_threshold{THRESHOLD}_seed{SEED}

The script computes one point per rollout:

* search: final cumulative number of uniquely detected targets;
* tracking: mean wrapped positional error over valid catalogue target-time pairs.

It then averages both measures across seeds for every threshold and plots mean
positional error (lower is better) against detections (higher is better).
The second plot relates elapsed search-time percentage to final mean positional
error. A third plot relates initially known-target observation frequency to
search time. Per-target observation frequencies start at t0 for known targets
and at first detection for new targets. All metrics are included in per-seed and aggregated CSV files.
Per-seed values and aggregated values are saved as CSV files so that additional
metrics, Pareto plots, and hypervolume calculations can be added later.

Six mode-independent surveillance metrics are also exported: sliding-window
coverage, grid age, initially known-target frequency and age, discrete-time
known-target position error, and restricted mean discovery time. Their common
horizon defaults to 200 simulated timesteps (--metric-horizon), and the coverage
window defaults to 20 (--coverage-window). Legacy metrics/plots keep their
original full-rollout scope. In particular, mean_known_observation_frequency is
the legacy full-run value; known_target_observation_frequency uses the common H.
Sparse state logs cannot determine exact time-averaged error: this is NaN,
with an explicitly named endpoint diagnostic alongside it. Undefined metrics
are NaN, aggregated over finite runs with sample SD and exported valid counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# Keep direct script execution working without requiring an editable install.
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.mo.metrics import (
    IDX_X, IDX_Y, IDX_VAR_X, IDX_VAR_Y, INITIAL_CATALOGUE_META_KEYS,
    initial_catalogue_size, catalogue_mask, wrapped_difference,
    compute_mean_positional_error, compute_mean_final_positional_error,
    compute_observation_frequency_per_target, finite_mean_and_std, sample_std,
    summarize_observation_frequencies, search_time_percentage,
)
from agents.mo.types import TargetObservationMetrics

KNOWN_TARGETS = 2
UNKNOWN_TARGETS = 10
DEFAULT_ROLL_ROOT = Path(f"rollouts/combined_{KNOWN_TARGETS}kn_{UNKNOWN_TARGETS}uk")
DEFAULT_OUT_DIR = Path(f"plots/combined_threshold_tradeoff_{KNOWN_TARGETS}kn_{UNKNOWN_TARGETS}uk")
DEFAULT_WORLD_SIZE = 100.0


RUN_PATTERN = re.compile(
    r"^heuristic_combined_threshold"
    r"(?P<threshold>[0-9]+(?:\.[0-9]+)?)"
    r"_seed(?P<seed>[0-9]+)$"
)


@dataclass(frozen=True)
class RunSpec:
    """Identity and location of one seeded threshold rollout."""

    threshold: float
    seed: int
    run_dir: Path


@dataclass(frozen=True)
class RunMetrics:
    """Scalar evaluation metrics for one rollout."""

    threshold: float
    seed: int
    detections: float
    mean_positional_error: float
    mean_final_positional_error: float
    search_time: float
    mean_observation_frequency: float = float("nan")
    mean_known_observation_frequency: float = float("nan")
    target_observation_metrics: tuple[TargetObservationMetrics, ...] = ()
    mean_window_coverage: float = float("nan")
    mean_grid_observation_age: float = float("nan")
    known_target_observation_frequency: float = float("nan")
    mean_known_observation_age: float = float("nan")
    time_averaged_known_positional_error: float = float("nan")
    endpoint_mean_known_positional_error: float = float("nan")
    restricted_mean_discovery_time: float = float("nan")
    coverage_window: int = 20
    metric_horizon: int = 0


@dataclass(frozen=True)
class ThresholdMetrics:
    """Metrics aggregated across all available seeds for one threshold."""

    threshold: float
    n_seeds: int
    detections_mean: float
    detections_std: float
    positional_error_mean: float
    positional_error_std: float
    final_positional_error_mean: float
    final_positional_error_std: float
    search_time_mean: float
    search_time_std: float
    observation_frequency_mean: float = float("nan")
    observation_frequency_std: float = float("nan")
    known_observation_frequency_mean: float = float("nan")
    known_observation_frequency_std: float = float("nan")
    mean_window_coverage_mean: float = float("nan")
    mean_window_coverage_std: float = float("nan")
    mean_window_coverage_n_valid: int = 0
    mean_grid_observation_age_mean: float = float("nan")
    mean_grid_observation_age_std: float = float("nan")
    mean_grid_observation_age_n_valid: int = 0
    known_target_observation_frequency_mean: float = float("nan")
    known_target_observation_frequency_std: float = float("nan")
    known_target_observation_frequency_n_valid: int = 0
    mean_known_observation_age_mean: float = float("nan")
    mean_known_observation_age_std: float = float("nan")
    mean_known_observation_age_n_valid: int = 0
    time_averaged_known_positional_error_mean: float = float("nan")
    time_averaged_known_positional_error_std: float = float("nan")
    time_averaged_known_positional_error_n_valid: int = 0
    endpoint_mean_known_positional_error_mean: float = float("nan")
    endpoint_mean_known_positional_error_std: float = float("nan")
    endpoint_mean_known_positional_error_n_valid: int = 0
    restricted_mean_discovery_time_mean: float = float("nan")
    restricted_mean_discovery_time_std: float = float("nan")
    restricted_mean_discovery_time_n_valid: int = 0
    known_observation_frequency_n_valid: int = 0
    coverage_window: int = 20
    metric_horizon: int = 0


def load_meta(run_dir: Path) -> Dict:
    """Load rollout metadata when a meta.json file is available."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}

    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_runs(roll_root: Path) -> List[RunSpec]:
    """Find and numerically sort all matching threshold/seed directories."""
    if not roll_root.exists():
        raise FileNotFoundError(f"Rollout root does not exist: {roll_root}")

    runs: List[RunSpec] = []

    for path in roll_root.iterdir():
        if not path.is_dir():
            continue

        match = RUN_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        runs.append(
            RunSpec(
                threshold=float(match.group("threshold")),
                seed=int(match.group("seed")),
                run_dir=path,
            )
        )

    if not runs:
        raise FileNotFoundError(
            "No rollout directories matching "
            "'heuristic_combined_threshold{THRESHOLD}_seed{SEED}' were "
            f"found in {roll_root}"
        )

    return sorted(runs, key=lambda run: (run.threshold, run.seed))


def cumulative_unique_detections_from_array(array: np.ndarray) -> float:
    """Extract the final cumulative unique-target count from a saved array.

    This follows the convention used by the supplied search-only script:
    one-dimensional arrays contain a cumulative count, while two-dimensional
    arrays indicate which targets have been detected at each timestep.
    """
    array = np.asarray(array).squeeze()

    if array.ndim == 0:
        return float(array)

    if array.ndim == 1:
        return float(np.max(array))

    if array.ndim == 2:
        detected_at_any_time = np.any(array > 0, axis=0)
        return float(np.count_nonzero(detected_at_any_time))

    raise ValueError(f"Unsupported detection array shape: {array.shape}")


def load_total_detections(
    run_dir: Path,
    est_states: np.ndarray,
    meta: Dict,
) -> float:
    """Load the final number of unique search targets detected.

    Explicit detection files are preferred. As a fallback, acquisition is
    inferred from valid estimated-track covariances, and the initial known
    catalogue is subtracted.
    """
    candidate_files = (
        "detections.npy",
        "num_detections.npy",
        "discovered_targets.npy",
    )

    for filename in candidate_files:
        path = run_dir / filename
        if path.exists():
            return cumulative_unique_detections_from_array(
                np.load(path, allow_pickle=False)
            )

    acquired = catalogue_mask(est_states)
    acquired_ever = np.maximum.accumulate(acquired, axis=0)
    final_catalogue_size = int(np.count_nonzero(acquired_ever[-1]))

    known_count = initial_catalogue_size(meta)
    if known_count is None:
        # This assumes the first saved state precedes any new acquisition.
        known_count = int(np.count_nonzero(acquired_ever[0]))

    return float(max(0, final_catalogue_size - known_count))


def compute_search_time(
    run_dir: Path,
    threshold: float,
    est_states: np.ndarray,
    meta: Dict,
) -> float:
    """Percentage of elapsed simulated timesteps spent in search mode.

    Prefer logged action modes. Legacy observations are post-action: action t
    uses observation t-1. Reconstruct the initial observation using metadata or
    the original writer defaults (position variance 1, normalization scale 100).
    Missing elapsed durations imply one timestep per decision.
    """
    n_steps = est_states.shape[0]
    if n_steps == 0:
        raise ValueError("Cannot compute search time for an empty rollout")
    mode_path = run_dir / "policy_modes.npy"
    if mode_path.exists():
        modes = np.load(mode_path, allow_pickle=False)
        if modes.shape != (n_steps,) or not np.isin(modes, ["search", "track"]).all():
            raise ValueError("policy_modes must contain one search/track label per decision")
        search = modes == "search"
    else:
        observations = np.load(run_dir / "observations.npy", allow_pickle=False)
        if observations.ndim != 4 or observations.shape[0] != n_steps:
            raise ValueError("Expected post-action observations with shape (T, H, W, C)")
        scale = float(meta.get("uncertainty_norm_scale", 100.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("uncertainty_norm_scale must be finite and positive")
        post_max = np.max(observations[:, :, :, 0], axis=(1, 2))
        traces = est_states[:, :, IDX_VAR_X] + est_states[:, :, IDX_VAR_Y]
        expected = np.log1p(np.max(traces, axis=1).astype(np.float32))
        expected /= np.log1p(scale)
        expected = np.clip(expected, 0.0, 1.0)
        if not np.allclose(post_max, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("Saved observations do not match uncertainty_norm_scale; "
                             "provide the rollout's actual scale in meta.json")
        initial_path = run_dir / "initial_observation.npy"
        if initial_path.exists():
            initial = np.load(initial_path, allow_pickle=False)
            if initial.shape != observations.shape[1:]:
                raise ValueError("Initial and post-action observation shapes differ")
            initial_max = np.max(initial[:, :, 0])
        else:
            known = initial_catalogue_size(meta)
            if known is None:
                raise KeyError("Legacy search-time reconstruction requires initial known targets")
            variances = meta.get("initial_target_vars")
            initial_trace = 0.0 if known == 0 else 2.0 * (
                max(variances[:known]) if variances is not None
                else float(meta.get("init_pos_var", 1.0))
            )
            initial_value = np.log1p(np.array([initial_trace], dtype=np.float32))
            initial_value /= np.log1p(scale)
            initial_max = np.clip(initial_value, 0.0, 1.0)[0]
        pre_max = np.concatenate(([initial_max], post_max[:-1]))
        # Match HeuristicAgent's scalar inverse-normalization and comparison.
        uncertainties = np.array([
            np.expm1(value * np.log1p(scale)) for value in pre_max
        ])
        if not np.isfinite(uncertainties).all() or not np.isfinite(threshold):
            raise ValueError("Search-mode reconstruction requires finite uncertainty/threshold")
        search = uncertainties <= threshold
    elapsed_path = run_dir / "elapsed_steps.npy"
    elapsed = (np.load(elapsed_path, allow_pickle=False) if elapsed_path.exists()
               else np.ones(n_steps))
    if elapsed.shape != (n_steps,) or not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
        raise ValueError("elapsed_steps must contain one positive duration per decision")
    elapsed = elapsed.astype(np.float64)
    return search_time_percentage(np.where(search, "search", "track"), elapsed)


def load_target_observations(
    run_dir: Path, true_states: np.ndarray, meta: Dict, world_size: float
) -> np.ndarray:
    """Load measurements or recover them using the current guaranteed-detection FoV.

    Each row describes the measurement at the END of an action, using the true
    positions saved at that same time. A covariance decrease is not a reliable
    indication of observation and is deliberately not used.
    """
    expected_shape = true_states.shape[:2]
    path = run_dir / "observed_targets.npy"
    if path.exists():
        observed = np.load(path, allow_pickle=False)
        if observed.shape != expected_shape or not np.isin(observed, [0, 1]).all():
            raise ValueError("observed_targets must be a binary (T, N) measurement mask")
        return observed.astype(bool)
    pointings = np.load(run_dir / "sensor_pointings.npy", allow_pickle=False)
    if pointings.shape != (expected_shape[0], 2):
        raise ValueError("Sensor pointings must align with saved true states")
    grid = meta.get("grid_shape")
    if grid is None or len(grid) != 2 or any(int(n) != n or n < 1 for n in grid):
        raise ValueError("Legacy measurement recovery requires a valid grid_shape")
    if world_size <= 0 or not np.isfinite(world_size):
        raise ValueError("world_size must be finite and positive")
    if not np.isfinite(pointings).all() or not np.isfinite(true_states[:, :, :2]).all():
        raise ValueError("Cannot recover observations from nonfinite positions")
    observed = np.ones(expected_shape, dtype=bool)
    for axis, size in enumerate(grid):
        size = int(size)
        width = world_size / size
        # Match Sensor.is_in_fov's scalar floor division, including float32 edges.
        sensor_cells = np.array([int(v // width) for v in pointings[:, axis]])
        target_cells = np.array([
            [int(v // width) for v in row] for row in true_states[:, :, axis]
        ])
        distance = np.abs(target_cells - sensor_cells[:, None])
        distance = np.where(distance > size // 2, size - distance, distance)
        observed &= distance <= 1
    return observed


def _measurement_timeline(elapsed_steps, n_events, horizon=None):
    """Validate integer simulated durations; optionally truncate at a common H."""
    elapsed = np.asarray(elapsed_steps, dtype=float)
    if (elapsed.shape != (n_events,) or n_events == 0
            or not np.isfinite(elapsed).all() or np.any(elapsed <= 0)
            or np.any(elapsed != np.floor(elapsed))):
        raise ValueError("elapsed_steps must contain one positive integer per event")
    times = np.cumsum(elapsed).astype(np.int64)
    if horizon is None:
        horizon = int(times[-1])
    if (not np.isfinite(horizon) or horizon != int(horizon)
            or horizon < 1 or horizon > times[-1]):
        raise ValueError("horizon must be a positive integer within the recorded duration")
    return times, int(horizon)


# Params: observed_targets, known_count; returns: boolean observation mask.
def _observation_mask(observed_targets, known_count):
    observed = np.asarray(observed_targets)
    if observed.ndim != 2 or not np.isin(observed, [0, 1]).all():
        raise ValueError("Expected a binary (events, targets) observation mask")
    if (known_count != int(known_count) or known_count < 0
            or known_count > observed.shape[1]):
        raise ValueError("known_count must identify an initial prefix of target IDs")
    return observed.astype(bool)


def _fov_masks(sensor_pointings, grid_shape, world_size):
    """Binary wrapped 3x3 footprints, matching the current sensor geometry."""
    pointings = np.asarray(sensor_pointings)
    grid = np.asarray(grid_shape, dtype=float)
    if (grid.shape != (2,) or not np.isfinite(grid).all()
            or np.any(grid < 1) or np.any(grid != np.floor(grid))):
        raise ValueError("grid_shape must contain two positive integers")
    if not np.isfinite(world_size) or world_size <= 0:
        raise ValueError("world_size must be finite and positive")
    if (pointings.ndim != 2 or pointings.shape[1] != 2
            or not np.isfinite(pointings).all()):
        raise ValueError("sensor_pointings must be a finite (events, 2) array")
    grid = grid.astype(int)
    cells = np.column_stack([
        [int(v // (world_size / size)) % size for v in pointings[:, axis]]
        for axis, size in enumerate(grid)
    ])
    masks = np.zeros((len(pointings), *grid), dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            masks[np.arange(len(pointings)), (cells[:, 0] + dx) % grid[0],
                  (cells[:, 1] + dy) % grid[1]] = True
    return masks.reshape(len(pointings), -1)


def compute_mean_window_coverage(
    sensor_pointings, elapsed_steps, grid_shape, world_size, window=20, horizon=None,
) -> float:
    """Mean unique FoV-covered grid fraction in complete windows (t-W, t].

    Average at every integer simulated timestep W..H. Measurements occur only
    at action endpoints: no coverage is credited during slew. Return NaN if
    H < W. No initial FoV observation is invented at reset.
    """
    if not np.isfinite(window) or window < 1 or window != int(window):
        raise ValueError("window must be a positive integer simulated duration")
    footprints = _fov_masks(sensor_pointings, grid_shape, world_size)
    times, end = _measurement_timeline(elapsed_steps, len(footprints), horizon)
    if end < window:
        return float("nan")
    last = np.full(footprints.shape[1], -int(window), dtype=np.int64)
    total, event = 0.0, 0
    for t in range(1, end + 1):
        if event < len(times) and times[event] == t:
            last[footprints[event]] = t
            event += 1
        if t >= window:
            total += np.mean(last > t - window)
    return float(total / (end - int(window) + 1))


def _mean_observation_age(observed, times, end):
    """Post-measurement integer-time age, including all unobserved intervals."""
    if observed.shape[1] == 0:
        return float("nan")
    last = np.zeros(observed.shape[1], dtype=np.int64)
    total, event = 0.0, 0
    for t in range(1, end + 1):
        if event < len(times) and times[event] == t:
            last[observed[event]] = t
            event += 1
        total += np.mean(t - last)
    return float(total / end)


def compute_mean_grid_observation_age(
    sensor_pointings, elapsed_steps, grid_shape, world_size, horizon=None,
) -> float:
    """Mean cell-time age in simulated timesteps; never-observed cells have age t.

    Age is sampled AFTER any measurement at integer t, over t=1..H. All cells
    start at age zero at reset as an evaluation convention, not an observation.
    """
    footprints = _fov_masks(sensor_pointings, grid_shape, world_size)
    times, end = _measurement_timeline(elapsed_steps, len(footprints), horizon)
    return _mean_observation_age(footprints, times, end)


def compute_mean_known_observation_frequency(
    observed_targets, elapsed_steps, known_count, horizon=None,
) -> float:
    """Equal-target mean measurement events/H for the initially known catalogue.

    Initialization is not counted; zero known targets gives NaN. Includes known
    targets with zero observations and excludes measurements after horizon.
    """
    observed = _observation_mask(observed_targets, known_count)
    times, end = _measurement_timeline(elapsed_steps, len(observed), horizon)
    if known_count == 0:
        return float("nan")
    return float(observed[times <= end, :known_count].sum() / (known_count * end))


def compute_mean_known_observation_age(
    observed_targets, elapsed_steps, known_count, horizon=None,
) -> float:
    """Equal-target/time mean age since a known target's last measurement.

    Uses t0 as the initial information reference; includes the terminal gap
    and never-reobserved targets. Integer-time samples are post-measurement.
    """
    observed = _observation_mask(observed_targets, known_count)
    times, end = _measurement_timeline(elapsed_steps, len(observed), horizon)
    return _mean_observation_age(observed[:, :known_count], times, end)


def compute_time_averaged_known_positional_error(
    true_states, est_states, elapsed_steps, known_count, world_size, horizon=None,
) -> float:
    """Exact discrete-time wrapped position error over known targets at t=1..H.

    Every integer-time state must be recorded (including slew interiors).
    Sparse action-end logs return NaN, never a duration-weighted post-update
    approximation. Dense arrays can be supplied with all-one elapsed_steps.
    Zero known targets gives NaN. Invalid/missing known estimates also give NaN
    rather than silently dropping difficult target-time pairs.
    """
    truth, estimate = np.asarray(true_states), np.asarray(est_states)
    if (truth.ndim != 3 or estimate.ndim != 3
            or truth.shape[:2] != estimate.shape[:2]
            or truth.shape[-1] < 2 or estimate.shape[-1] <= IDX_VAR_Y):
        raise ValueError("Expected aligned (events, targets, state) arrays")
    _observation_mask(np.zeros(truth.shape[:2], dtype=bool), known_count)
    if not np.isfinite(world_size) or world_size <= 0:
        raise ValueError("world_size must be finite and positive")
    times, end = _measurement_timeline(elapsed_steps, len(truth), horizon)
    keep = times <= end
    if known_count == 0 or not np.array_equal(times[keep], np.arange(1, end + 1)):
        return float("nan")
    truth = truth[keep, :known_count]
    estimate = estimate[keep, :known_count]
    if (not catalogue_mask(estimate).all()
            or not np.isfinite(truth[:, :, :2]).all()
            or not np.isfinite(estimate[:, :, :2]).all()):
        return float("nan")
    return float(np.linalg.norm(wrapped_difference(
        estimate[:, :, :2], truth[:, :, :2], world_size), axis=-1).mean())


def compute_restricted_mean_discovery_time(
    observed_targets, elapsed_steps, known_count, horizon=None,
) -> float:
    """Mean min(first detection time, H) over initially unknown targets.

    Undetected targets contribute H; no unknown targets gives NaN. Equivalent
    to the area under the pre-measurement undiscovered-fraction curve on [0,H].
    Incidental detections count, as they do in the environment.
    """
    observed = _observation_mask(observed_targets, known_count)
    times, end = _measurement_timeline(elapsed_steps, len(observed), horizon)
    unknown = observed[times <= end, known_count:]
    if unknown.shape[1] == 0:
        return float("nan")
    first = np.full(unknown.shape[1], float(end))
    for row, t in zip(unknown, times[times <= end]):
        first[row] = np.minimum(first[row], t)
    return float(first.mean())


# Legacy frequency fields keep their full-run scope; the new field uses H.
EXTRA_SCALAR_METRICS = (
    "mean_window_coverage", "mean_grid_observation_age",
    "known_target_observation_frequency",
    "mean_known_observation_age", "time_averaged_known_positional_error",
    "endpoint_mean_known_positional_error", "restricted_mean_discovery_time",
)


def load_run_metrics(
    run: RunSpec,
    world_size: float,
    target_set: str,
    coverage_window: int = 20,
    metric_horizon: int | None = 200,
) -> RunMetrics:
    """Compute legacy full-run metrics plus surveillance metrics on [0, H].

    Requires elapsed_steps, sensor pointings, and grid metadata. State arrays
    are action-end samples; exact time-averaged error is unavailable when they
    skip integer timesteps. Explicit horizon=None uses actual duration; callers
    must then ensure equal durations before aggregating.
    """
    true_path = run.run_dir / "true_states.npy"
    est_path = run.run_dir / "est_states.npy"

    if not true_path.exists():
        raise FileNotFoundError(f"Missing true state file: {true_path}")
    if not est_path.exists():
        raise FileNotFoundError(f"Missing estimated state file: {est_path}")

    true_states = np.load(true_path, allow_pickle=False)
    est_states = np.load(est_path, allow_pickle=False)
    meta = load_meta(run.run_dir)
    if true_states.shape[:2] != est_states.shape[:2]:
        raise ValueError("True and estimated states must align in time and target ID")
    known_count = initial_catalogue_size(meta)
    if known_count is None:
        raise KeyError("Observation frequency requires the initial known-target count")
    observed = load_target_observations(run.run_dir, true_states, meta, world_size)
    elapsed_path = run.run_dir / "elapsed_steps.npy"
    if not elapsed_path.exists():
        raise FileNotFoundError("Simulated-time metrics require elapsed_steps.npy; "
                                "action counts cannot substitute for elapsed time")
    elapsed = np.load(elapsed_path, allow_pickle=False)
    times, horizon = _measurement_timeline(elapsed, len(true_states), metric_horizon)
    pointings = np.load(run.run_dir / "sensor_pointings.npy", allow_pickle=False)
    grid_shape = meta.get("grid_shape")
    if grid_shape is None:
        raise KeyError("Spatial metrics require grid_shape in meta.json")
    target_metrics = compute_observation_frequency_per_target(observed, elapsed, known_count)
    frequency_summary = summarize_observation_frequencies(target_metrics)
    # Plot/CSV adapters retain NaN for unavailable means; episode records use None.
    all_frequency = frequency_summary["mean_observation_frequency"]
    known_frequency = frequency_summary["mean_known_observation_frequency"]
    all_frequency = float("nan") if all_frequency is None else all_frequency
    known_frequency = float("nan") if known_frequency is None else known_frequency
    # Endpoint diagnostic uses the same evaluation horizon as the new metrics.
    # Missing known estimates must not silently improve the diagnostic.
    keep = times <= horizon
    endpoint_known_error = float("nan")
    if (known_count and np.any(keep)
            and catalogue_mask(est_states[keep, :known_count]).all()
            and np.isfinite(true_states[keep, :known_count, :2]).all()
            and np.isfinite(est_states[keep, :known_count, :2]).all()):
        endpoint_known_error = compute_mean_positional_error(
            true_states[keep], est_states[keep], world_size, "initial", meta)

    return RunMetrics(
        threshold=run.threshold,
        seed=run.seed,
        detections=load_total_detections(run.run_dir, est_states, meta),
        mean_observation_frequency=all_frequency,
        mean_known_observation_frequency=known_frequency,
        target_observation_metrics=tuple(target_metrics),
        mean_window_coverage=compute_mean_window_coverage(
            pointings, elapsed, grid_shape, world_size, coverage_window, horizon),
        mean_grid_observation_age=compute_mean_grid_observation_age(
            pointings, elapsed, grid_shape, world_size, horizon),
        known_target_observation_frequency=compute_mean_known_observation_frequency(
            observed, elapsed, known_count, horizon),
        mean_known_observation_age=compute_mean_known_observation_age(
            observed, elapsed, known_count, horizon),
        time_averaged_known_positional_error=compute_time_averaged_known_positional_error(
            true_states, est_states, elapsed, known_count, world_size, horizon),
        endpoint_mean_known_positional_error=endpoint_known_error,
        restricted_mean_discovery_time=compute_restricted_mean_discovery_time(
            observed, elapsed, known_count, horizon),
        coverage_window=coverage_window,
        metric_horizon=horizon,
        mean_positional_error=compute_mean_positional_error(
            true_states=true_states,
            est_states=est_states,
            world_size=world_size,
            target_set=target_set,
            meta=meta,
        ),
        search_time=compute_search_time(run.run_dir, run.threshold, est_states, meta),
        mean_final_positional_error=compute_mean_final_positional_error(
            true_states=true_states,
            est_states=est_states,
            world_size=world_size,
            target_set=target_set,
            meta=meta,
        ),
    )


def aggregate_by_threshold(
    run_metrics: Iterable[RunMetrics],
) -> List[ThresholdMetrics]:
    """Average seeded rollout metrics independently for every threshold."""
    grouped: Dict[float, List[RunMetrics]] = {}
    for metrics in run_metrics:
        grouped.setdefault(metrics.threshold, []).append(metrics)

    aggregated: List[ThresholdMetrics] = []
    for threshold in sorted(grouped):
        group = grouped[threshold]
        if len({(item.coverage_window, item.metric_horizon) for item in group}) != 1:
            raise ValueError("Cannot aggregate different coverage windows/horizons; "
                             "set a common metric_horizon when durations differ")
        extra = {}
        for name in EXTRA_SCALAR_METRICS:
            values = [getattr(item, name) for item in group]
            extra[name + "_mean"], extra[name + "_std"] = finite_mean_and_std(values)
            extra[name + "_n_valid"] = int(np.isfinite(values).sum())
        detections = [item.detections for item in group]
        errors = [item.mean_positional_error for item in group]
        final_errors = [item.mean_final_positional_error for item in group]
        search_times = [item.search_time for item in group]
        freq_mean, freq_std = finite_mean_and_std([item.mean_observation_frequency for item in group])
        known_freq_mean, known_freq_std = finite_mean_and_std([
            item.mean_known_observation_frequency for item in group
        ])

        aggregated.append(
            ThresholdMetrics(
                threshold=threshold,
                n_seeds=len(group),
                detections_mean=float(np.mean(detections)),
                detections_std=sample_std(detections),
                positional_error_mean=float(np.mean(errors)),
                positional_error_std=sample_std(errors),
                final_positional_error_mean=float(np.mean(final_errors)),
                final_positional_error_std=sample_std(final_errors),
                search_time_mean=float(np.mean(search_times)),
                search_time_std=sample_std(search_times),
                observation_frequency_mean=freq_mean,
                observation_frequency_std=freq_std,
                known_observation_frequency_mean=known_freq_mean,
                known_observation_frequency_std=known_freq_std,
                known_observation_frequency_n_valid=int(np.isfinite([
                    item.mean_known_observation_frequency for item in group]).sum()),
                coverage_window=group[0].coverage_window,
                metric_horizon=group[0].metric_horizon,
                **extra,
            )
        )

    return aggregated


def nondominated_mask(
    positional_error: np.ndarray,
    detections: np.ndarray,
) -> np.ndarray:
    """Identify points nondominated under error minimization/detection maximization."""
    n_points = len(positional_error)
    keep = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        no_worse = (
            (positional_error <= positional_error[i])
            & (detections >= detections[i])
        )
        strictly_better = (
            (positional_error < positional_error[i])
            | (detections > detections[i])
        )
        if np.any(no_worse & strictly_better):
            keep[i] = False

    return keep


def save_csv_files(
    run_metrics: Sequence[RunMetrics],
    aggregated: Sequence[ThresholdMetrics],
    out_dir: Path,
) -> None:
    """Save raw per-seed metrics and threshold aggregates."""
    per_seed_path = out_dir / "per_seed_metrics.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["threshold", "seed", "detections", "mean_positional_error",
             "mean_final_positional_error", "search_time_percent",
             "mean_observation_frequency", "mean_known_observation_frequency"]
            + list(EXTRA_SCALAR_METRICS) + ["coverage_window", "metric_horizon"]
        )
        for item in run_metrics:
            writer.writerow(
                [
                    item.threshold,
                    item.seed,
                    item.detections,
                    item.mean_positional_error,
                    item.mean_final_positional_error,
                    item.search_time,
                    item.mean_observation_frequency,
                    item.mean_known_observation_frequency,
                ]
                + [getattr(item, name) for name in EXTRA_SCALAR_METRICS]
                + [item.coverage_window, item.metric_horizon]
            )

    aggregate_path = out_dir / "threshold_metrics.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "threshold",
                "n_seeds",
                "detections_mean",
                "detections_std",
                "mean_positional_error_mean",
                "mean_positional_error_std",
                "mean_final_positional_error_mean",
                "mean_final_positional_error_std",
                "search_time_percent_mean",
                "search_time_percent_std",
                "observation_frequency_mean",
                "observation_frequency_std",
                "known_observation_frequency_mean",
                "known_observation_frequency_std",
                "nondominated",
            ]
            + [name + suffix for name in EXTRA_SCALAR_METRICS
               for suffix in ("_mean", "_std", "_n_valid")]
            + ["known_observation_frequency_n_valid", "coverage_window", "metric_horizon"]
        )

        errors = np.array(
            [item.positional_error_mean for item in aggregated], dtype=float
        )
        detections = np.array(
            [item.detections_mean for item in aggregated], dtype=float
        )
        pareto = nondominated_mask(errors, detections)

        for item, is_nondominated in zip(aggregated, pareto):
            writer.writerow(
                [
                    item.threshold,
                    item.n_seeds,
                    item.detections_mean,
                    item.detections_std,
                    item.positional_error_mean,
                    item.positional_error_std,
                    item.final_positional_error_mean,
                    item.final_positional_error_std,
                    item.search_time_mean,
                    item.search_time_std,
                    item.observation_frequency_mean,
                    item.observation_frequency_std,
                    item.known_observation_frequency_mean,
                    item.known_observation_frequency_std,
                    bool(is_nondominated),
                ]
                + [getattr(item, name + suffix) for name in EXTRA_SCALAR_METRICS
                   for suffix in ("_mean", "_std", "_n_valid")]
                + [item.known_observation_frequency_n_valid,
                   item.coverage_window, item.metric_horizon]
            )


    with (out_dir / "per_target_observation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["threshold", "seed", "target_id", "initially_known",
                         "observation_count", "counting_start_timestep",
                         "eligible_timesteps", "observation_frequency"])
        for run in run_metrics:
            for item in run.target_observation_metrics:
                writer.writerow([run.threshold, run.seed, item.target_id,
                                 item.initially_known, item.observation_count,
                                 item.counting_start_timestep, item.eligible_timesteps,
                                 item.observation_frequency])


def plot_tradeoff(
    aggregated: Sequence[ThresholdMetrics],
    save_path: Path,
    annotate: bool,
) -> None:
    """Plot one seed-averaged search--tracking point per threshold."""
    thresholds = np.array([item.threshold for item in aggregated], dtype=float)
    errors = np.array(
        [item.positional_error_mean for item in aggregated], dtype=float
    )
    errors_std = np.array(
        [item.positional_error_std for item in aggregated], dtype=float
    )
    detections = np.array(
        [item.detections_mean for item in aggregated], dtype=float
    )
    detections_std = np.array(
        [item.detections_std for item in aggregated], dtype=float
    )
    pareto = nondominated_mask(errors, detections)

    fig, ax = plt.subplots(figsize=(9, 6))

    # The dashed path shows how behavior changes as the threshold increases.
    ax.plot(
        errors,
        detections,
        linestyle="--",
        linewidth=1.2,
        color="0.65",
        zorder=1,
        label="Threshold sequence",
    )

    ax.errorbar(
        errors,
        detections,
        xerr=errors_std,
        yerr=detections_std,
        fmt="none",
        ecolor="0.55",
        elinewidth=1.0,
        capsize=3,
        alpha=0.8,
        zorder=2,
        label="±1 SD across seeds",
    )

    normalizer = Normalize(vmin=float(np.min(thresholds)), vmax=float(np.max(thresholds)))
    points = ax.scatter(
        errors,
        detections,
        c=thresholds,
        cmap="viridis",
        norm=normalizer,
        s=75,
        edgecolors=np.where(pareto, "black", "none"),
        linewidths=np.where(pareto, 1.5, 0.0),
        zorder=3,
    )

    # Draw the mean nondominated frontier without implying that intermediate
    # policies exist between the evaluated thresholds.
    pareto_indices = np.flatnonzero(pareto)
    pareto_indices = pareto_indices[np.argsort(errors[pareto_indices])]
    if len(pareto_indices) > 1:
        ax.plot(
            errors[pareto_indices],
            detections[pareto_indices],
            color="black",
            linewidth=2.0,
            zorder=2,
            label="Nondominated heuristic set",
        )

    if annotate:
        for threshold, x_value, y_value in zip(thresholds, errors, detections):
            ax.annotate(
                f"{threshold:g}",
                xy=(x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    colorbar = fig.colorbar(points, ax=ax, pad=0.02)
    colorbar.set_label("Uncertainty threshold")

    ax.set_xlabel("Mean positional error")
    ax.set_ylabel("Total detected targets")
    ax.set_title("Combined heuristic: search–tracking trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Better policies lie toward the upper-left: lower error, more detections.
    ax.text(
        0.02,
        0.98,
        "Better ↑ detections\nBetter ← error",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_search_time_tradeoff(
    aggregated: Sequence[ThresholdMetrics],
    save_path: Path,
    annotate: bool,
) -> None:
    """Plot final positional error against percentage of time spent searching."""
    thresholds = np.array([item.threshold for item in aggregated])
    errors = np.array([item.final_positional_error_mean for item in aggregated])
    search_times = np.array([item.search_time_mean for item in aggregated])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(errors, search_times, "--", color="0.65", linewidth=1.2,
            label="Threshold sequence", zorder=1)
    ax.errorbar(errors, search_times,
                xerr=[item.final_positional_error_std for item in aggregated],
                yerr=[item.search_time_std for item in aggregated],
                fmt="none", ecolor="0.55", capsize=3, alpha=0.8,
                label="±1 SD across seeds", zorder=2)
    points = ax.scatter(errors, search_times, c=thresholds, cmap="viridis",
                        s=75, zorder=3)
    if annotate:
        labels = {}
        for threshold, error, search_time in zip(thresholds, errors, search_times):
            labels.setdefault((error, search_time), []).append(f"{threshold:g}")
        for point, threshold_labels in labels.items():
            ax.annotate(", ".join(threshold_labels), point, xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
    fig.colorbar(points, ax=ax, pad=0.02).set_label("Uncertainty threshold")
    ax.set_xlabel("Final mean positional error")
    ax.set_ylabel("Time spent in search mode (%)")
    ax.set_title("Combined heuristic: search time vs final positional error")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_observation_frequency_tradeoff(
    aggregated: Sequence[ThresholdMetrics], save_path: Path, annotate: bool
) -> None:
    """Initially known-target observation frequency (y) vs search time (x)."""
    valid = [item for item in aggregated if np.isfinite(item.known_observation_frequency_mean)]
    fig, ax = plt.subplots(figsize=(9, 6))
    if valid:
        thresholds = np.array([item.threshold for item in valid])
        search_times = np.array([item.search_time_mean for item in valid])
        frequencies = np.array([item.known_observation_frequency_mean for item in valid])
        ax.plot(search_times, frequencies, "--", color="0.65", linewidth=1.2,
                label="Threshold sequence", zorder=1)
        ax.errorbar(search_times, frequencies,
                    xerr=[item.search_time_std for item in valid],
                    yerr=[item.known_observation_frequency_std for item in valid],
                    fmt="none", ecolor="0.55", capsize=3, alpha=0.8,
                    label="±1 SD across seeds", zorder=2)
        points = ax.scatter(search_times, frequencies, c=thresholds,
                            cmap="viridis", s=75, zorder=3)
        if annotate:
            labels = {}
            for threshold, search, frequency in zip(thresholds, search_times, frequencies):
                labels.setdefault((search, frequency), []).append(f"{threshold:g}")
            for point, values in labels.items():
                ax.annotate(", ".join(values), point, xytext=(5, 5),
                            textcoords="offset points", fontsize=8)
        fig.colorbar(points, ax=ax, pad=0.02).set_label("Uncertainty threshold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No initially known targets in these rollouts",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Time spent in search mode (%)")
    ax.set_ylabel("Mean initially known-target observation frequency\n(observations per simulated timestep)")
    ax.set_title("Combined heuristic: observation frequency vs search time")
    ax.set_xlim(-5, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_surveillance_tradeoffs(
    aggregated: Sequence[ThresholdMetrics], out_dir: Path, annotate: bool,
) -> None:
    """Plot three spatial/discovery measures (y) against three tracking measures (x)."""
    search_metrics = {
        "mean_window_coverage": "Mean sliding-window coverage (fraction; higher is better)",
        "mean_grid_observation_age": "Mean grid observation age (timesteps; lower is better)",
        "restricted_mean_discovery_time": "Restricted mean discovery time (timesteps; lower is better)",
    }
    tracking_metrics = {
        "final_positional_error": "Final mean positional error (averaged across seeds; lower is better)",
        "mean_known_observation_age": "Mean known-target observation age (timesteps; lower is better)",
        "known_target_observation_frequency": "Known-target observation frequency (observations/timestep; higher is better)",
    }
    for y_metric, y_label in search_metrics.items():
        for x_metric, x_label in tracking_metrics.items():
            valid = [item for item in aggregated
                     if np.isfinite(getattr(item, x_metric + "_mean"))
                     and np.isfinite(getattr(item, y_metric + "_mean"))]
            fig, ax = plt.subplots(figsize=(10, 6))
            if valid:
                x = np.array([getattr(item, x_metric + "_mean") for item in valid])
                y = np.array([getattr(item, y_metric + "_mean") for item in valid])
                thresholds = np.array([item.threshold for item in valid])
                ax.plot(x, y, "--", color="0.65", linewidth=1.2,
                        label="Threshold sequence", zorder=1)
                ax.errorbar(x, y,
                            xerr=[getattr(item, x_metric + "_std") for item in valid],
                            yerr=[getattr(item, y_metric + "_std") for item in valid],
                            fmt="none", ecolor="0.55", capsize=3, alpha=0.8,
                            label="±1 SD across seeds", zorder=2)
                points = ax.scatter(x, y, c=thresholds, cmap="viridis", s=75, zorder=3)
                if annotate:
                    labels = {}
                    for threshold, xv, yv in zip(thresholds, x, y):
                        labels.setdefault((xv, yv), []).append(f"{threshold:g}")
                    placed = []
                    xscale = max(float(np.ptp(x)), 1e-12)
                    yscale = max(float(np.ptp(y)), 1e-12)
                    for point, values in labels.items():
                        nearby = sum(abs(point[0] - prior[0]) / xscale < 0.08
                                     and abs(point[1] - prior[1]) / yscale < 0.08
                                     for prior in placed)
                        ax.annotate(", ".join(values), point,
                                    xytext=(5, 5 + 14 * nearby),
                                    textcoords="offset points", fontsize=8)
                        placed.append(point)
                fig.colorbar(points, ax=ax, pad=0.02).set_label("Uncertainty threshold")
                ax.legend()
            else:
                reason = ("No initially unknown targets: discovery time is undefined"
                          if y_metric == "restricted_mean_discovery_time"
                          else "No finite metric pairs available")
                ax.text(0.5, 0.5, reason, transform=ax.transAxes,
                        ha="center", va="center", wrap=True)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            settings = aggregated[0]
            ax.set_title(f"{out_dir.name.removeprefix('combined_threshold_tradeoff_')}: "
                         f"surveillance trade-off\nH = {settings.metric_horizon} timesteps; "
                         f"coverage window = {settings.coverage_window} timesteps")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"{y_metric}_vs_{x_metric}.png",
                        dpi=250, bbox_inches="tight")
            plt.close(fig)


def print_summary(aggregated: Sequence[ThresholdMetrics]) -> None:
    """Print a compact threshold-by-threshold summary."""
    print("\n=== Combined heuristic threshold comparison ===\n")
    print(
        f"{'Threshold':>10}  {'Seeds':>5}  "
        f"{'Detections (mean ± SD)':>24}  "
        f"{'Position error (mean ± SD)':>28}  "
        f"{'Final error (mean ± SD)':>26}  {'Search time % (mean ± SD)':>28}  "
        f"{'Known observation freq (mean ± SD)':>34}"
    )

    for item in aggregated:
        print(
            f"{item.threshold:>10g}  {item.n_seeds:>5d}  "
            f"{item.detections_mean:>9.3f} ± {item.detections_std:<9.3f}  "
            f"{item.positional_error_mean:>10.4f} ± "
            f"{item.positional_error_std:<10.4f}  "
            f"{item.final_positional_error_mean:.4f} ± {item.final_positional_error_std:.4f}  "
            f"{item.search_time_mean:.2f} ± {item.search_time_std:.2f}  "
            f"{item.known_observation_frequency_mean:.4f} ± "
            f"{item.known_observation_frequency_std:.4f}"
        )


# Params: none; returns: argparse.Namespace.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot heuristic combined-policy search/tracking trade-offs."
    )
    parser.add_argument(
        "--roll-root",
        type=Path,
        default=DEFAULT_ROLL_ROOT,
        help=f"Rollout directory (default: {DEFAULT_ROLL_ROOT})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--world-size",
        type=float,
        default=DEFAULT_WORLD_SIZE,
        help=f"Wrapped world width/height (default: {DEFAULT_WORLD_SIZE})",
    )
    parser.add_argument(
        "--error-target-set",
        choices=("catalogue", "initial"),
        default="catalogue",
        help=(
            "Targets included in positional error: all valid catalogue tracks "
            "or only the initially known catalogue (default: catalogue)"
        ),
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Do not annotate points with their threshold values",
    )
    parser.add_argument(
        "--coverage-window", type=int, default=20,
        help="Sliding coverage window in simulated timesteps (default: 20)",
    )
    parser.add_argument(
        "--metric-horizon", type=int, default=200,
        help="Common horizon for the six new metrics (default: 200); "
             "must fit every rollout. Existing metrics retain full-run scope.",
    )
    return parser.parse_args()


# Params: none; returns: None.
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.roll_root)
    run_metrics: List[RunMetrics] = []

    for run in runs:
        print(f"Loading {run.run_dir.name}")
        run_metrics.append(
            load_run_metrics(
                run=run,
                world_size=args.world_size,
                target_set=args.error_target_set,
                coverage_window=args.coverage_window,
                metric_horizon=args.metric_horizon,
            )
        )

    aggregated = aggregate_by_threshold(run_metrics)
    print_summary(aggregated)
    save_csv_files(run_metrics, aggregated, args.out_dir)
    plot_tradeoff(
        aggregated=aggregated,
        save_path=args.out_dir / "detections_vs_mean_positional_error.png",
        annotate=not args.no_annotations,
    )

    plot_search_time_tradeoff(
        aggregated=aggregated,
        save_path=args.out_dir / "search_time_vs_final_mean_positional_error.png",
        annotate=not args.no_annotations,
    )
    plot_observation_frequency_tradeoff(
        aggregated=aggregated,
        save_path=args.out_dir / "known_observation_frequency_vs_search_time.png",
        annotate=not args.no_annotations,
    )
    plot_surveillance_tradeoffs(aggregated, args.out_dir, not args.no_annotations)
    print(f"\nSaved plots and metric tables to: {args.out_dir}")


if __name__ == "__main__":
    main()
