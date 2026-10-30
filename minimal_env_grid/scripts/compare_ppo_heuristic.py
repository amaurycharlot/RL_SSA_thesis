import json
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from scripts.plot_xy_errors import wrap_diff


ROLL_ROOT = Path("rollouts")   # change to Path("rollouts") if needed
OUT_DIR = Path("plots/comparison_tracking_grid_agents")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORLD_SIZE_DEFAULT = 100.0
IDX_X = 0
IDX_Y = 1


RUNS = {
    "Heuristic baseline": "heuristic_track_seed1_20targets_R6_200steps",
    "PPO proximity bonus": "ppo_model5_track_seed1_20targets_200steps",
    "PPO probability bonus": "ppo_model3_track_seed1_20targets_200steps",
}


# Params: est; returns: (int, int).
def infer_variance_indices(est: np.ndarray):
    D = est.shape[-1]

    # Current 4D state + diag(P) layout: [x, y, vx, vy, var_x, var_y, ...]
    if D >= 6:
        return 4, 5

    raise ValueError(f"Cannot infer variance indices from est_states shape {est.shape}")


# Params: run_name; returns: (est, true, world_size).
def load_rollout(run_name: str):
    run_dir = ROLL_ROOT / run_name

    est = np.load(run_dir / "est_states.npy")
    true = np.load(run_dir / "true_states.npy")

    meta_path = run_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    world_size = float(meta.get("world_size", WORLD_SIZE_DEFAULT))

    return est, true, world_size


# Params: est, true, world_size; returns: Dict[str, np.ndarray].
def compute_metrics(est: np.ndarray, true: np.ndarray, world_size: float) -> Dict[str, np.ndarray]:
    idx_varx, idx_vary = infer_variance_indices(est)

    T = min(est.shape[0], true.shape[0])
    est = est[:T]
    true = true[:T]

    err_x = wrap_diff(est[:, :, IDX_X], true[:, :, 0], world_size)
    err_y = wrap_diff(est[:, :, IDX_Y], true[:, :, 1], world_size)
    pos_err = np.sqrt(err_x**2 + err_y**2)

    var_x = np.maximum(est[:, :, idx_varx], 0.0)
    var_y = np.maximum(est[:, :, idx_vary], 0.0)
    cov_trace = var_x + var_y

    return {
        "mean_pos_error_t": np.mean(pos_err, axis=1),
        "median_pos_error_t": np.median(pos_err, axis=1),
        "mean_cov_trace_t": np.mean(cov_trace, axis=1),
        "median_cov_trace_t": np.median(cov_trace, axis=1),
        "final_mean_pos_error": float(np.mean(pos_err[-1])),
        "final_median_pos_error": float(np.median(pos_err[-1])),
        "final_mean_cov_trace": float(np.mean(cov_trace[-1])),
        "final_median_cov_trace": float(np.median(cov_trace[-1])),
        "episode_mean_pos_error": float(np.mean(pos_err)),
        "episode_mean_cov_trace": float(np.mean(cov_trace)),
    }


# Params: none; returns: results.
def load_all_results():
    results = {}

    for label, run_name in RUNS.items():
        est, true, world_size = load_rollout(run_name)
        results[label] = compute_metrics(est, true, world_size)

    return results


# Params: results, key, ylabel, title, filename; returns: None.
def plot_time_series(results: Dict[str, Dict[str, np.ndarray]], key: str, ylabel: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, metrics in results.items():
        y = metrics[key]
        t = np.arange(len(y))
        ax.plot(t, y, label=label)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


# Params: results, key, ylabel, title, filename; returns: None.
def plot_final_bar(results: Dict[str, Dict[str, np.ndarray]], key: str, ylabel: str, title: str, filename: str):
    labels = list(results.keys())
    values = [results[label][key] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


# Params: results, key, ylabel, title, filename; returns: None.
def plot_relative_improvement(results: Dict[str, Dict[str, np.ndarray]], key: str, ylabel: str, title: str, filename: str):
    baseline = results["Heuristic baseline"][key]

    labels = []
    improvements = []

    for label, metrics in results.items():
        if label == "Heuristic baseline":
            continue

        value = metrics[key]
        improvement = 100.0 * (baseline - value) / (baseline + 1e-12)

        labels.append(label)
        improvements.append(improvement)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, improvements)
    ax.axhline(0.0, linestyle="--")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


# Params: results; returns: None.
def print_summary(results: Dict[str, Dict[str, np.ndarray]]):
    print("\n=== Tracking comparison summary ===\n")

    for label, m in results.items():
        print(label)
        print(f"  Final mean position error:   {m['final_mean_pos_error']:.4f}")
        print(f"  Final median position error: {m['final_median_pos_error']:.4f}")
        print(f"  Final mean covariance trace: {m['final_mean_cov_trace']:.4f}")
        print(f"  Final median cov trace:      {m['final_median_cov_trace']:.4f}")
        print(f"  Episode mean position error: {m['episode_mean_pos_error']:.4f}")
        print(f"  Episode mean cov trace:      {m['episode_mean_cov_trace']:.4f}")
        print()

    h = results["Heuristic baseline"]

    for label in ["PPO proximity bonus", "PPO probability bonus"]:
        m = results[label]
        pos_gain = 100.0 * (h["episode_mean_pos_error"] - m["episode_mean_pos_error"]) / (h["episode_mean_pos_error"] + 1e-12)
        cov_gain = 100.0 * (h["episode_mean_cov_trace"] - m["episode_mean_cov_trace"]) / (h["episode_mean_cov_trace"] + 1e-12)

        print(f"{label} vs heuristic:")
        print(f"  Episode mean position error improvement: {pos_gain:.2f}%")
        print(f"  Episode mean covariance trace improvement: {cov_gain:.2f}%")
        print()


# Params: none; returns: None.
def main():
    results = load_all_results()
    print_summary(results)

    plot_time_series(
        results,
        key="mean_pos_error_t",
        ylabel="Mean position error",
        title="Tracking: mean position error over time",
        filename="timeseries_mean_pos_error.png",
    )

    plot_time_series(
        results,
        key="mean_cov_trace_t",
        ylabel="Mean covariance trace",
        title="Tracking: mean covariance trace over time",
        filename="timeseries_mean_cov_trace.png",
    )

    plot_time_series(
        results,
        key="median_pos_error_t",
        ylabel="Median position error",
        title="Tracking: median position error over time",
        filename="timeseries_median_pos_error.png",
    )

    plot_time_series(
        results,
        key="median_cov_trace_t",
        ylabel="Median covariance trace",
        title="Tracking: median covariance trace over time",
        filename="timeseries_median_cov_trace.png",
    )

    plot_final_bar(
        results,
        key="episode_mean_pos_error",
        ylabel="Episode mean position error",
        title="Tracking: average position error over episode",
        filename="bar_episode_mean_pos_error.png",
    )

    plot_final_bar(
        results,
        key="episode_mean_cov_trace",
        ylabel="Episode mean covariance trace",
        title="Tracking: average covariance trace over episode",
        filename="bar_episode_mean_cov_trace.png",
    )

    plot_relative_improvement(
        results,
        key="episode_mean_pos_error",
        ylabel="Improvement over heuristic (%)",
        title="PPO improvement in episode mean position error",
        filename="relative_improvement_pos_error.png",
    )

    plot_relative_improvement(
        results,
        key="episode_mean_cov_trace",
        ylabel="Improvement over heuristic (%)",
        title="PPO improvement in episode mean covariance trace",
        filename="relative_improvement_cov_trace.png",
    )

    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()