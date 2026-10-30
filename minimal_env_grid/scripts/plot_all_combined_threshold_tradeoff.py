"""Plot the search--tracking trade-off of combined heuristic and PPO policies.

Each rollout directory must follow the naming convention

    heuristic_combined_threshold{THRESHOLD}_seed{SEED}
    ppo_combined_{AGENT}_seed{SEED}

The script computes one point per rollout:

* search: final cumulative number of uniquely detected targets;
* tracking: mean wrapped positional error over valid catalogue target-time pairs.

It then averages both measures across seeds for every heuristic threshold and
PPO agent, and plots mean positional error (lower is better) against detections
(higher is better).
Per-seed values and aggregated values are saved as CSV files so that additional
metrics, Pareto plots, and hypervolume calculations can be added later.
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


DEFAULT_ROLL_ROOT = Path("rollouts/combined")
DEFAULT_OUT_DIR = Path("plots/combined_threshold_tradeoff")
DEFAULT_WORLD_SIZE = 100.0

IDX_X = 0
IDX_Y = 1
IDX_VAR_X = 4
IDX_VAR_Y = 5

HEURISTIC_RUN_PATTERN = re.compile(
    r"^heuristic_combined_threshold"
    r"(?P<threshold>[0-9]+(?:\.[0-9]+)?)"
    r"_seed(?P<seed>[0-9]+)$"
)

PPO_RUN_PATTERN = re.compile(
    r"^ppo_combined_(?P<agent>[0-9]+)_seed(?P<seed>[0-9]+)$"
)

INITIAL_CATALOGUE_META_KEYS = (
    "initial_catalogue_size",
    "initial_catalog_size",
    "n_known_targets",
    "num_known_targets",
    "initial_known_targets",
)


@dataclass(frozen=True)
class RunSpec:
    """Identity and location of one seeded policy rollout."""

    policy_type: str
    policy_id: str
    order_value: float
    seed: int
    run_dir: Path

    # Params: self; returns: str.
    @property
    def label(self) -> str:
        if self.policy_type == "heuristic":
            return f"Heuristic threshold {self.policy_id}"
        return f"PPO combined {self.policy_id}"


@dataclass(frozen=True)
class RunMetrics:
    """Scalar evaluation metrics for one rollout."""

    policy_type: str
    policy_id: str
    order_value: float
    seed: int
    detections: float
    mean_positional_error: float


@dataclass(frozen=True)
class PolicyMetrics:
    """Metrics aggregated across all available seeds for one policy."""

    policy_type: str
    policy_id: str
    label: str
    order_value: float
    n_seeds: int
    detections_mean: float
    detections_std: float
    positional_error_mean: float
    positional_error_std: float


def load_meta(run_dir: Path) -> Dict:
    """Load rollout metadata when a meta.json file is available."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}

    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def discover_runs(roll_root: Path) -> List[RunSpec]:
    """Find and sort all matching heuristic and PPO rollout directories."""
    if not roll_root.exists():
        raise FileNotFoundError(f"Rollout root does not exist: {roll_root}")

    runs: List[RunSpec] = []

    for path in roll_root.iterdir():
        if not path.is_dir():
            continue

        heuristic_match = HEURISTIC_RUN_PATTERN.fullmatch(path.name)
        if heuristic_match is not None:
            threshold = float(heuristic_match.group("threshold"))
            runs.append(
                RunSpec(
                    policy_type="heuristic",
                    policy_id=f"{threshold:g}",
                    order_value=threshold,
                    seed=int(heuristic_match.group("seed")),
                    run_dir=path,
                )
            )
            continue

        ppo_match = PPO_RUN_PATTERN.fullmatch(path.name)
        if ppo_match is not None:
            agent = int(ppo_match.group("agent"))
            runs.append(
                RunSpec(
                    policy_type="ppo",
                    policy_id=str(agent),
                    order_value=float(agent),
                    seed=int(ppo_match.group("seed")),
                    run_dir=path,
                )
            )

    if not runs:
        raise FileNotFoundError(
            "No rollout directories matching either supported naming "
            "convention were "
            f"found in {roll_root}"
        )

    policy_order = {"heuristic": 0, "ppo": 1}
    return sorted(
        runs,
        key=lambda run: (
            policy_order[run.policy_type],
            run.order_value,
            run.seed,
        ),
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


def load_run_metrics(
    run: RunSpec,
    world_size: float,
    target_set: str,
) -> RunMetrics:
    """Compute both scalar metrics for one seeded rollout."""
    true_path = run.run_dir / "true_states.npy"
    est_path = run.run_dir / "est_states.npy"

    if not true_path.exists():
        raise FileNotFoundError(f"Missing true state file: {true_path}")
    if not est_path.exists():
        raise FileNotFoundError(f"Missing estimated state file: {est_path}")

    true_states = np.load(true_path, allow_pickle=False)
    est_states = np.load(est_path, allow_pickle=False)
    meta = load_meta(run.run_dir)

    return RunMetrics(
        policy_type=run.policy_type,
        policy_id=run.policy_id,
        order_value=run.order_value,
        seed=run.seed,
        detections=load_total_detections(run.run_dir, est_states, meta),
        mean_positional_error=compute_mean_positional_error(
            true_states=true_states,
            est_states=est_states,
            world_size=world_size,
            target_set=target_set,
            meta=meta,
        ),
    )


def sample_std(values: Sequence[float]) -> float:
    """Return sample standard deviation, or zero for a single seed."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def aggregate_by_policy(
    run_metrics: Iterable[RunMetrics],
) -> List[PolicyMetrics]:
    """Average seeded rollout metrics independently for every policy."""
    grouped: Dict[tuple[str, str], List[RunMetrics]] = {}
    for metrics in run_metrics:
        key = (metrics.policy_type, metrics.policy_id)
        grouped.setdefault(key, []).append(metrics)

    policy_order = {"heuristic": 0, "ppo": 1}
    groups = sorted(
        grouped.values(),
        key=lambda group: (
            policy_order[group[0].policy_type],
            group[0].order_value,
        ),
    )

    aggregated: List[PolicyMetrics] = []
    for group in groups:
        representative = group[0]
        detections = [item.detections for item in group]
        errors = [item.mean_positional_error for item in group]

        if representative.policy_type == "heuristic":
            label = f"Heuristic threshold {representative.policy_id}"
        else:
            label = f"PPO combined {representative.policy_id}"

        aggregated.append(
            PolicyMetrics(
                policy_type=representative.policy_type,
                policy_id=representative.policy_id,
                label=label,
                order_value=representative.order_value,
                n_seeds=len(group),
                detections_mean=float(np.mean(detections)),
                detections_std=sample_std(detections),
                positional_error_mean=float(np.mean(errors)),
                positional_error_std=sample_std(errors),
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
    aggregated: Sequence[PolicyMetrics],
    out_dir: Path,
) -> None:
    """Save raw per-seed metrics and policy aggregates."""
    per_seed_path = out_dir / "per_seed_metrics.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "policy_type",
                "policy_id",
                "seed",
                "detections",
                "mean_positional_error",
            ]
        )
        for item in run_metrics:
            writer.writerow(
                [
                    item.policy_type,
                    item.policy_id,
                    item.seed,
                    item.detections,
                    item.mean_positional_error,
                ]
            )

    aggregate_path = out_dir / "aggregated_policy_metrics.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "policy_type",
                "policy_id",
                "policy_label",
                "n_seeds",
                "detections_mean",
                "detections_std",
                "mean_positional_error_mean",
                "mean_positional_error_std",
                "nondominated",
            ]
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
                    item.policy_type,
                    item.policy_id,
                    item.label,
                    item.n_seeds,
                    item.detections_mean,
                    item.detections_std,
                    item.positional_error_mean,
                    item.positional_error_std,
                    bool(is_nondominated),
                ]
            )


def plot_tradeoff(
    aggregated: Sequence[PolicyMetrics],
    save_path: Path,
    annotate: bool,
) -> None:
    """Plot heuristic thresholds and PPO agents with distinct visual styles."""
    heuristic = [item for item in aggregated if item.policy_type == "heuristic"]
    ppo_agents = [item for item in aggregated if item.policy_type == "ppo"]

    fig, ax = plt.subplots(figsize=(9, 6))

    if heuristic:
        thresholds = np.array(
            [item.order_value for item in heuristic], dtype=float
        )
        heuristic_errors = np.array(
            [item.positional_error_mean for item in heuristic], dtype=float
        )
        heuristic_errors_std = np.array(
            [item.positional_error_std for item in heuristic], dtype=float
        )
        heuristic_detections = np.array(
            [item.detections_mean for item in heuristic], dtype=float
        )
        heuristic_detections_std = np.array(
            [item.detections_std for item in heuristic], dtype=float
        )
        heuristic_pareto = nondominated_mask(
            heuristic_errors,
            heuristic_detections,
        )

        # The dashed path shows how behavior changes as the threshold increases.
        ax.plot(
            heuristic_errors,
            heuristic_detections,
            linestyle="--",
            linewidth=1.2,
            color="0.65",
            zorder=1,
            label="Heuristic threshold sequence",
        )

        ax.errorbar(
            heuristic_errors,
            heuristic_detections,
            xerr=heuristic_errors_std,
            yerr=heuristic_detections_std,
            fmt="none",
            ecolor="0.55",
            elinewidth=1.0,
            capsize=3,
            alpha=0.8,
            zorder=2,
            label="Heuristic ±1 SD",
        )

        normalizer = Normalize(
            vmin=float(np.min(thresholds)),
            vmax=float(np.max(thresholds)),
        )
        heuristic_points = ax.scatter(
            heuristic_errors,
            heuristic_detections,
            c=thresholds,
            cmap="viridis",
            norm=normalizer,
            marker="o",
            s=75,
            edgecolors=np.where(heuristic_pareto, "black", "none"),
            linewidths=np.where(heuristic_pareto, 1.5, 0.0),
            zorder=3,
        )

        # Draw the heuristic baseline front only; PPO agents remain individual
        # learned-policy points rather than being connected into this sweep.
        pareto_indices = np.flatnonzero(heuristic_pareto)
        pareto_indices = pareto_indices[
            np.argsort(heuristic_errors[pareto_indices])
        ]
        if len(pareto_indices) > 1:
            ax.plot(
                heuristic_errors[pareto_indices],
                heuristic_detections[pareto_indices],
                color="black",
                linewidth=2.0,
                zorder=2,
                label="Nondominated heuristic set",
            )

        if annotate:
            for threshold, x_value, y_value in zip(
                thresholds,
                heuristic_errors,
                heuristic_detections,
            ):
                ax.annotate(
                    f"{threshold:g}",
                    xy=(x_value, y_value),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        colorbar = fig.colorbar(heuristic_points, ax=ax, pad=0.02)
        colorbar.set_label("Heuristic uncertainty threshold")

    ppo_markers = ("X", "P", "D", "*", "s", "^")
    ppo_colors = plt.get_cmap("tab10").colors

    for index, item in enumerate(ppo_agents):
        color = ppo_colors[index % len(ppo_colors)]
        marker = ppo_markers[index % len(ppo_markers)]

        ax.errorbar(
            item.positional_error_mean,
            item.detections_mean,
            xerr=item.positional_error_std,
            yerr=item.detections_std,
            fmt=marker,
            markersize=12,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=1.0,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            linestyle="none",
            zorder=5,
            label=f"{item.label} (±1 SD)",
        )

    ax.set_xlabel("Mean positional error")
    ax.set_ylabel("Total detected targets")
    ax.set_title("Combined policies: search–tracking trade-off")
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


def print_summary(aggregated: Sequence[PolicyMetrics]) -> None:
    """Print a compact policy-by-policy summary."""
    print("\n=== Combined policy comparison ===\n")
    print(
        f"{'Policy':>26}  {'Seeds':>5}  "
        f"{'Detections (mean ± SD)':>24}  "
        f"{'Position error (mean ± SD)':>28}"
    )

    for item in aggregated:
        print(
            f"{item.label:>26}  {item.n_seeds:>5d}  "
            f"{item.detections_mean:>9.3f} ± {item.detections_std:<9.3f}  "
            f"{item.positional_error_mean:>10.4f} ± "
            f"{item.positional_error_std:<10.4f}"
        )


# Params: none; returns: argparse.Namespace.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot heuristic and PPO combined-policy trade-offs."
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
        help="Do not annotate heuristic points with their threshold values",
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
            )
        )

    aggregated = aggregate_by_policy(run_metrics)
    print_summary(aggregated)
    save_csv_files(run_metrics, aggregated, args.out_dir)
    plot_tradeoff(
        aggregated=aggregated,
        save_path=args.out_dir / "detections_vs_mean_positional_error.png",
        annotate=not args.no_annotations,
    )

    print(f"\nSaved plot and metric tables to: {args.out_dir}")


if __name__ == "__main__":
    main()
