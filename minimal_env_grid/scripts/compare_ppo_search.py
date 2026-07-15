import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


ROLL_ROOT = Path("rollouts/search")
OUT_DIR = Path("plots/search_reward_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10

# PPO rollout folders:
# ppo_search_S1_seed0, ..., ppo_search_S1_seed9
# ...
# ppo_search_S4_seed0, ..., ppo_search_S4_seed9
PPO_RUNS = {
    f"PPO S{i}": [f"ppo_search_S{i}_seed{seed}" for seed in range(N_SEEDS)]
    for i in range(1, 5)
}

# Single heuristic reference rollout.
HEURISTIC_RUN = {
    "Heuristic": "heuristic_search_seed1_100targets_200steps",
}


def load_meta(run_dir: Path) -> Dict:
    meta_path = run_dir / "meta.json"

    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_time_axis(run_dir: Path, T: int) -> np.ndarray:
    """
    Load the physical/environment time axis when available.

    If total_steps.npy is unavailable, use rollout decision indices:
    0, 1, ..., T-1.
    """
    total_steps_path = run_dir / "total_steps.npy"

    if total_steps_path.exists():
        time_axis = np.asarray(
            np.load(total_steps_path),
            dtype=float,
        ).squeeze()

        if time_axis.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional time axis in "
                f"{total_steps_path}, got shape {time_axis.shape}"
            )

        return time_axis[:T]

    return np.arange(T, dtype=float)


def compute_coverage_over_time(
    observations: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Compute cumulative percentage of grid cells observed over time.

    Expected observation shape:
        (T, grid_rows, grid_cols, obs_layers)
    """
    if observations.ndim != 4:
        raise ValueError(
            f"Expected observations with four dimensions, "
            f"got shape {observations.shape}"
        )

    T, grid_rows, grid_cols, _ = observations.shape

    if mode == "search":
        observed_mask = observations[:, :, :, 0] > 0

    elif mode == "combined":
        observed_mask = observations[:, :, :, 1] > 0

    else:
        raise ValueError(
            f"Unsupported mode for coverage computation: {mode}"
        )

    cumulative_observed = np.zeros(
        (grid_rows, grid_cols),
        dtype=bool,
    )

    coverage_t = np.zeros(T, dtype=float)

    for timestep in range(T):
        cumulative_observed |= observed_mask[timestep]

        coverage_t[timestep] = (
            100.0
            * np.count_nonzero(cumulative_observed)
            / (grid_rows * grid_cols)
        )

    return coverage_t


def compute_detections_from_est_states(
    est_states: np.ndarray,
) -> np.ndarray:
    """
    Infer cumulative detections from estimated target states.

    Assumed layout:
        [x, y, vx, vy, var_x, var_y, ...]

    A target is considered acquired once either positional covariance
    entry is positive.
    """
    if est_states.ndim != 3:
        raise ValueError(
            f"Expected est_states shape (T, N, D), "
            f"got {est_states.shape}"
        )

    state_dimension = est_states.shape[-1]

    if state_dimension < 6:
        raise ValueError(
            f"Cannot infer detections from est_states shape "
            f"{est_states.shape}"
        )

    idx_var_x = 4
    idx_var_y = 5

    acquired = (
        (est_states[:, :, idx_var_x] > 0.0)
        | (est_states[:, :, idx_var_y] > 0.0)
    )

    detections_t = np.sum(
        acquired,
        axis=1,
    ).astype(float)

    return np.maximum.accumulate(detections_t)


def load_detections(run_dir: Path) -> np.ndarray:
    """
    Load or infer cumulative detections over time.

    Preferred files, in order:
    1. detections.npy
    2. num_detections.npy
    3. discovered_targets.npy
    4. est_states.npy
    """
    candidate_files = [
        "detections.npy",
        "num_detections.npy",
        "discovered_targets.npy",
    ]

    for filename in candidate_files:
        path = run_dir / filename

        if not path.exists():
            continue

        array = np.asarray(np.load(path)).squeeze()

        if array.ndim == 1:
            return np.maximum.accumulate(
                array.astype(float)
            )

        if array.ndim == 2:
            detections_t = np.sum(
                array > 0,
                axis=1,
            ).astype(float)

            return np.maximum.accumulate(detections_t)

        raise ValueError(
            f"Unsupported detection array shape "
            f"{array.shape} in {path}"
        )

    est_path = run_dir / "est_states.npy"

    if not est_path.exists():
        raise FileNotFoundError(
            f"No detection file or est_states.npy found in {run_dir}"
        )

    est_states = np.load(est_path)

    return compute_detections_from_est_states(est_states)


def load_run(run_name: str) -> Dict[str, np.ndarray]:
    """Load the coverage and detection curves for one rollout."""
    run_dir = ROLL_ROOT / run_name

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Rollout directory does not exist: {run_dir}"
        )

    observations_path = run_dir / "observations.npy"

    if not observations_path.exists():
        raise FileNotFoundError(
            f"Missing observation file: {observations_path}"
        )

    observations = np.load(observations_path)

    meta = load_meta(run_dir)
    mode = meta.get("mode", "search")

    coverage_t = compute_coverage_over_time(
        observations,
        mode,
    )

    detections_t = load_detections(run_dir)

    T = min(
        len(coverage_t),
        len(detections_t),
    )

    time_axis = load_time_axis(
        run_dir,
        T,
    )

    T = min(T, len(time_axis))

    time_axis = np.asarray(
        time_axis[:T],
        dtype=float,
    )

    coverage_t = np.asarray(
        coverage_t[:T],
        dtype=float,
    )

    detections_t = np.asarray(
        detections_t[:T],
        dtype=float,
    )

    # Interpolation requires an increasing time axis.
    order = np.argsort(time_axis)

    time_axis = time_axis[order]
    coverage_t = coverage_t[order]
    detections_t = detections_t[order]

    # Remove repeated time points, retaining the last value at each time.
    unique_times, reverse_indices = np.unique(
        time_axis[::-1],
        return_index=True,
    )

    selected_indices = (
        len(time_axis) - 1 - reverse_indices
    )

    selected_indices = np.sort(selected_indices)

    return {
        "t": time_axis[selected_indices],
        "coverage_t": coverage_t[selected_indices],
        "detections_t": detections_t[selected_indices],
    }


def construct_common_time_axis(
    runs: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Construct a common integer time axis shared by all seeded runs.

    The interval ends at the earliest final time among the runs, avoiding
    extrapolation beyond any rollout.
    """
    start_time = max(
        float(run["t"][0])
        for run in runs
    )

    end_time = min(
        float(run["t"][-1])
        for run in runs
    )

    common_start = int(np.ceil(start_time))
    common_end = int(np.floor(end_time))

    if common_end < common_start:
        raise ValueError(
            "The rollout time axes do not have a shared interval."
        )

    return np.arange(
        common_start,
        common_end + 1,
        dtype=float,
    )


def interpolate_run(
    run: Dict[str, np.ndarray],
    common_t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Interpolate one rollout onto a shared time axis."""
    return {
        "t": common_t,
        "coverage_t": np.interp(
            common_t,
            run["t"],
            run["coverage_t"],
        ),
        "detections_t": np.interp(
            common_t,
            run["t"],
            run["detections_t"],
        ),
    }


def aggregate_runs(
    run_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load and aggregate multiple seeded rollout folders.

    Returns means, standard deviations, and individual seeded curves.
    """
    loaded_runs = []

    for run_name in run_names:
        print(f"Loading {run_name}")
        loaded_runs.append(load_run(run_name))

    common_t = construct_common_time_axis(loaded_runs)

    aligned_runs = [
        interpolate_run(run, common_t)
        for run in loaded_runs
    ]

    coverage_all = np.stack(
        [run["coverage_t"] for run in aligned_runs],
        axis=0,
    )

    detections_all = np.stack(
        [run["detections_t"] for run in aligned_runs],
        axis=0,
    )

    return {
        "t": common_t,
        "coverage_mean": np.mean(coverage_all, axis=0),
        "coverage_std": np.std(
            coverage_all,
            axis=0,
            ddof=1,
        ),
        "detections_mean": np.mean(detections_all, axis=0),
        "detections_std": np.std(
            detections_all,
            axis=0,
            ddof=1,
        ),
        "coverage_all": coverage_all,
        "detections_all": detections_all,
    }


def load_all_results() -> Dict[str, Dict[str, np.ndarray]]:
    """Load the heuristic and aggregate all four PPO reward variants."""
    results = {}

    for label, run_name in HEURISTIC_RUN.items():
        heuristic_data = load_run(run_name)

        results[label] = {
            "t": heuristic_data["t"],
            "coverage_mean": heuristic_data["coverage_t"],
            "coverage_std": np.zeros_like(
                heuristic_data["coverage_t"]
            ),
            "detections_mean": heuristic_data["detections_t"],
            "detections_std": np.zeros_like(
                heuristic_data["detections_t"]
            ),
            "n_runs": 1,
        }

    for label, run_names in PPO_RUNS.items():
        aggregate = aggregate_runs(run_names)
        aggregate["n_runs"] = len(run_names)
        results[label] = aggregate

    return results


def plot_metric(
    results: Dict[str, Dict[str, np.ndarray]],
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """
    Plot mean performance and ±1 standard deviation.

    The heuristic is shown as a single line because only one heuristic
    rollout is currently configured.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in results.items():
        time_axis = data["t"]
        mean_values = data[mean_key]
        std_values = data[std_key]

        line, = ax.plot(
            time_axis,
            mean_values,
            label=label,
            linewidth=2,
        )

        if data["n_runs"] > 1:
            lower = mean_values - std_values
            upper = mean_values + std_values

            # Coverage and detections cannot be negative.
            lower = np.maximum(lower, 0.0)

            ax.fill_between(
                time_axis,
                lower,
                upper,
                alpha=0.2,
                color=line.get_color(),
            )

    ax.set_xlabel("Environment timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_summary(
    results: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Print the final mean and standard deviation for every method."""
    print("\n=== Search rollout comparison ===\n")

    for label, data in results.items():
        final_coverage_mean = data["coverage_mean"][-1]
        final_coverage_std = data["coverage_std"][-1]

        final_detections_mean = data["detections_mean"][-1]
        final_detections_std = data["detections_std"][-1]

        print(f"{label} ({data['n_runs']} run(s))")

        if data["n_runs"] > 1:
            print(
                "  Final grid coverage: "
                f"{final_coverage_mean:.2f} "
                f"± {final_coverage_std:.2f}%"
            )
            print(
                "  Final detections:    "
                f"{final_detections_mean:.2f} "
                f"± {final_detections_std:.2f}"
            )
        else:
            print(
                "  Final grid coverage: "
                f"{final_coverage_mean:.2f}%"
            )
            print(
                "  Final detections:    "
                f"{final_detections_mean:.0f}"
            )

        print()


def save_aggregated_results(
    results: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Save aggregated curves for later thesis plots or analysis."""
    aggregate_dir = OUT_DIR / "aggregated_data"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    for label, data in results.items():
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        save_data = {
            "t": data["t"],
            "coverage_mean": data["coverage_mean"],
            "coverage_std": data["coverage_std"],
            "detections_mean": data["detections_mean"],
            "detections_std": data["detections_std"],
        }

        if "coverage_all" in data:
            save_data["coverage_all"] = data["coverage_all"]

        if "detections_all" in data:
            save_data["detections_all"] = data["detections_all"]

        np.savez(
            aggregate_dir / f"{safe_label}.npz",
            **save_data,
        )


def main() -> None:
    results = load_all_results()

    print_summary(results)
    save_aggregated_results(results)

    plot_metric(
        results=results,
        mean_key="coverage_mean",
        std_key="coverage_std",
        ylabel="Grid coverage (%)",
        title="Search: mean grid coverage over 10 seeds",
        save_path=OUT_DIR / "grid_coverage_over_time_mean_std.png",
    )

    plot_metric(
        results=results,
        mean_key="detections_mean",
        std_key="detections_std",
        ylabel="Cumulative number of detected targets",
        title="Search: mean detections over 10 seeds",
        save_path=OUT_DIR / "detections_over_time_mean_std.png",
    )

    print(f"Saved plots and aggregated data to: {OUT_DIR}")


if __name__ == "__main__":
    main()