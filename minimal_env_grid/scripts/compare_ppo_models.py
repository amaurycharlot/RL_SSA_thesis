import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


WORLD_SIZE = 100.0

IDX_X = 0
IDX_Y = 1
IDX_VARX = 4
IDX_VARY = 5

MODEL_IDS = [1, 2, 3, 4, 5]

ROLL_ROOT = Path("rollouts")
OUT_DIR = Path("plots/comparison_ppo_models_20targets_200steps")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wrap_diff(a: np.ndarray, b: np.ndarray, world_size: float) -> np.ndarray:
    """Shortest signed difference on periodic world."""
    return (a - b + world_size / 2.0) % world_size - world_size / 2.0


# Params: model_id; returns: (est, true).
def load_rollout(model_id: int):
    run_dir = ROLL_ROOT / f"ppo_model{model_id}_track_seed1_20targets_200steps"

    est = np.load(run_dir / "est_states.npy")
    true = np.load(run_dir / "true_states.npy")

    return est, true


# Params: est, true; returns: Dict[str, np.ndarray].
def compute_time_series_metrics(est: np.ndarray, true: np.ndarray) -> Dict[str, np.ndarray]:
    err_x = wrap_diff(est[:, :, IDX_X], true[:, :, 0], WORLD_SIZE)
    err_y = wrap_diff(est[:, :, IDX_Y], true[:, :, 1], WORLD_SIZE)

    pos_err = np.sqrt(err_x**2 + err_y**2)

    var_x = np.maximum(est[:, :, IDX_VARX], 0.0)
    var_y = np.maximum(est[:, :, IDX_VARY], 0.0)

    cov_trace = var_x + var_y

    return {
        "mean_pos_error_t": np.mean(pos_err, axis=1),
        "mean_cov_trace_t": np.mean(cov_trace, axis=1),
        "final_mean_pos_error": float(np.mean(pos_err[-1])),
        "final_mean_cov_trace": float(np.mean(cov_trace[-1])),
    }


# Params: none; returns: Dict[int, Dict[str, np.ndarray]].
def load_all_results() -> Dict[int, Dict[str, np.ndarray]]:
    results = {}

    for model_id in MODEL_IDS:
        est, true = load_rollout(model_id)
        results[model_id] = compute_time_series_metrics(est, true)

    return results


# Params: results, metric_key, ylabel, title, save_path; returns: None.
def plot_time_series_all(
    results: Dict[int, Dict[str, np.ndarray]],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_id, metrics in results.items():
        y = metrics[metric_key]
        t = np.arange(len(y))

        ax.plot(t, y, label=f"PPO model {model_id}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# Params: results, metric_key, ylabel, title, save_path; returns: None.
def plot_final_bar(
    results: Dict[int, Dict[str, np.ndarray]],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    labels = [f"model {i}" for i in MODEL_IDS]
    values = [results[i][metric_key] for i in MODEL_IDS]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# Params: results, x_model, y_model, metric_key, xlabel, ylabel, title, save_path; returns: None.
def plot_paired_scatter(
    results: Dict[int, Dict[str, np.ndarray]],
    x_model: int,
    y_model: int,
    metric_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    xvals = results[x_model][metric_key]
    yvals = results[y_model][metric_key]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(xvals, yvals)

    vmin = min(np.min(xvals), np.min(yvals))
    vmax = max(np.max(xvals), np.max(yvals))
    margin = 0.05 * (vmax - vmin + 1e-12)

    ax.plot(
        [vmin - margin, vmax + margin],
        [vmin - margin, vmax + margin],
        "--",
    )

    ax.set_xlim(vmin - margin, vmax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# Params: none; returns: None.
def main():
    results = load_all_results()

    print("\n=== Final metrics ===")
    for model_id, metrics in results.items():
        print(
            f"PPO model {model_id}: "
            f"final mean position error = {metrics['final_mean_pos_error']:.4f}, "
            f"final mean covariance trace = {metrics['final_mean_cov_trace']:.4f}"
        )

    plot_time_series_all(
        results,
        metric_key="mean_pos_error_t",
        ylabel="Mean position error",
        title="PPO models: mean position error over time",
        save_path=OUT_DIR / "timeseries_mean_pos_error.png",
    )

    plot_time_series_all(
        results,
        metric_key="mean_cov_trace_t",
        ylabel="Mean covariance trace",
        title="PPO models: mean covariance trace over time",
        save_path=OUT_DIR / "timeseries_mean_cov_trace.png",
    )

    plot_final_bar(
        results,
        metric_key="final_mean_pos_error",
        ylabel="Final mean position error",
        title="PPO models: final mean position error",
        save_path=OUT_DIR / "bar_final_mean_pos_error.png",
    )

    plot_final_bar(
        results,
        metric_key="final_mean_cov_trace",
        ylabel="Final mean covariance trace",
        title="PPO models: final mean covariance trace",
        save_path=OUT_DIR / "bar_final_mean_cov_trace.png",
    )

    for x_model, y_model in [(2, 3), (2, 4), (3, 4)]:
        plot_paired_scatter(
            results,
            x_model=x_model,
            y_model=y_model,
            metric_key="mean_pos_error_t",
            xlabel=f"Model {x_model} mean position error",
            ylabel=f"Model {y_model} mean position error",
            title=f"Mean position error: model {x_model} vs model {y_model}",
            save_path=OUT_DIR / f"scatter_pos_error_model{x_model}_vs_model{y_model}.png",
        )

        plot_paired_scatter(
            results,
            x_model=x_model,
            y_model=y_model,
            metric_key="mean_cov_trace_t",
            xlabel=f"Model {x_model} mean covariance trace",
            ylabel=f"Model {y_model} mean covariance trace",
            title=f"Mean covariance trace: model {x_model} vs model {y_model}",
            save_path=OUT_DIR / f"scatter_cov_trace_model{x_model}_vs_model{y_model}.png",
        )


if __name__ == "__main__":
    main()