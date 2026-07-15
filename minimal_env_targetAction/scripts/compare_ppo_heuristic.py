import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from scripts.plot_xy_errors import wrap_diff


WORLD_SIZE = 100.0
IDX_X = 0
IDX_Y = 1
IDX_VARX = 4
IDX_VARY = 5

REWARDS = range(1, 6)
SEEDS = range(1, 6)

ROLL_ROOT = Path("rollouts/13_04_5rewards_5seeds")
OUT_DIR = Path("plots/comparison_reward_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rollout(agent_name: str, reward_id: int, seed: int):
    run_dir = ROLL_ROOT / f"{agent_name}_5targets_reward_R{reward_id}_seed{seed}"
    est = np.load(run_dir / "est_states.npy")
    true = np.load(run_dir / "true_states.npy")
    return est, true


def compute_time_series_metrics(est: np.ndarray, true: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute per-timestep metrics for one rollout.
    """
    err_x = wrap_diff(est[:, :, IDX_X], true[:, :, 0], WORLD_SIZE)
    err_y = wrap_diff(est[:, :, IDX_Y], true[:, :, 1], WORLD_SIZE)
    pos_err = np.sqrt(err_x**2 + err_y**2)  # shape (T, N)

    var_x = np.maximum(est[:, :, IDX_VARX], 0.0)
    var_y = np.maximum(est[:, :, IDX_VARY], 0.0)
    cov_trace = var_x + var_y  # shape (T, N)

    return {
        "mean_pos_error_t": np.mean(pos_err, axis=1),   # shape (T,)
        "mean_cov_trace_t": np.mean(cov_trace, axis=1), # shape (T,)
        "final_mean_pos_error": float(np.mean(pos_err[-1])),
        "final_mean_cov_trace": float(np.mean(cov_trace[-1])),
    }


def aggregate_agent_reward(agent_name: str, reward_id: int, seeds=SEEDS) -> Dict[str, np.ndarray]:
    """
    Aggregate metrics over seeds for one agent and one reward.
    """
    final_mean_pos_error = []
    final_mean_cov_trace = []
    mean_pos_error_t = []
    mean_cov_trace_t = []

    for seed in seeds:
        est, true = load_rollout(agent_name, reward_id, seed)
        metrics = compute_time_series_metrics(est, true)

        final_mean_pos_error.append(metrics["final_mean_pos_error"])
        final_mean_cov_trace.append(metrics["final_mean_cov_trace"])
        mean_pos_error_t.append(metrics["mean_pos_error_t"])
        mean_cov_trace_t.append(metrics["mean_cov_trace_t"])

    return {
        "final_mean_pos_error": np.array(final_mean_pos_error),
        "final_mean_cov_trace": np.array(final_mean_cov_trace),
        "mean_pos_error_t": np.stack(mean_pos_error_t, axis=0),   # shape (S, T)
        "mean_cov_trace_t": np.stack(mean_cov_trace_t, axis=0),   # shape (S, T)
    }


def plot_paired_scatter(
    xvals: np.ndarray,
    yvals: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(xvals, yvals)

    vmin = min(np.min(xvals), np.min(yvals))
    vmax = max(np.max(xvals), np.max(yvals))
    margin = 0.05 * (vmax - vmin + 1e-12)

    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin], "--")
    ax.set_xlim(vmin - margin, vmax + margin)
    ax.set_ylim(vmin - margin, vmax + margin)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_time_series_comparison(
    heuristic_ts: np.ndarray,
    ppo_ts: np.ndarray,
    ylabel: str,
    title: str,
    save_path: Path,
):
    """
    heuristic_ts, ppo_ts: shape (S, T)
    """
    h_mean = np.mean(heuristic_ts, axis=0)
    h_std = np.std(heuristic_ts, axis=0)

    p_mean = np.mean(ppo_ts, axis=0)
    p_std = np.std(ppo_ts, axis=0)

    t = np.arange(h_mean.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, h_mean, label="Heuristic")
    ax.fill_between(t, h_mean - h_std, h_mean + h_std, alpha=0.25)

    ax.plot(t, p_mean, label="PPO")
    ax.fill_between(t, p_mean - p_std, p_mean + p_std, alpha=0.25)

    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_delta_boxplot(
    deltas_by_reward: List[np.ndarray],
    reward_labels: List[str],
    ylabel: str,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(deltas_by_reward, labels=reward_labels, showfliers=True)

    for i, vals in enumerate(deltas_by_reward, start=1):
        x = np.full_like(vals, i, dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(x + jitter, vals, alpha=0.8)

    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Reward type")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    all_results = {}

    pos_error_deltas = []
    cov_trace_deltas = []
    reward_labels = []

    for R in REWARDS:
        heuristic = aggregate_agent_reward("heuristicRL", R)
        ppo = aggregate_agent_reward("ppo", R)

        all_results[R] = {
            "heuristic": heuristic,
            "ppo": ppo,
        }

        # Console summary
        print(
            f"Reward R{R}: Heuristic - "
            f"Mean Position Error = {np.mean(heuristic['final_mean_pos_error']):.4f}, "
            f"Mean Covariance Trace = {np.mean(heuristic['final_mean_cov_trace']):.4f}"
        )
        print(
            f"Reward R{R}: PPO       - "
            f"Mean Position Error = {np.mean(ppo['final_mean_pos_error']):.4f}, "
            f"Mean Covariance Trace = {np.mean(ppo['final_mean_cov_trace']):.4f}"
        )

        # Deltas: positive means PPO is better here because lower metric is better
        delta_pos = heuristic["final_mean_pos_error"] - ppo["final_mean_pos_error"]
        delta_cov = heuristic["final_mean_cov_trace"] - ppo["final_mean_cov_trace"]

        pos_error_deltas.append(delta_pos)
        cov_trace_deltas.append(delta_cov)
        reward_labels.append(f"R{R}")

        # 1) Paired scatter: final mean position error
        plot_paired_scatter(
            xvals=heuristic["final_mean_pos_error"],
            yvals=ppo["final_mean_pos_error"],
            xlabel="Heuristic final mean position error",
            ylabel="PPO final mean position error",
            title=f"Reward R{R}: Final mean position error",
            save_path=OUT_DIR / f"R{R}_paired_scatter_final_pos_error.png",
        )

        # 2) Paired scatter: final mean covariance trace
        plot_paired_scatter(
            xvals=heuristic["final_mean_cov_trace"],
            yvals=ppo["final_mean_cov_trace"],
            xlabel="Heuristic final mean covariance trace",
            ylabel="PPO final mean covariance trace",
            title=f"Reward R{R}: Final mean covariance trace",
            save_path=OUT_DIR / f"R{R}_paired_scatter_final_cov_trace.png",
        )

        # 3) Time series: mean position error
        plot_time_series_comparison(
            heuristic_ts=heuristic["mean_pos_error_t"],
            ppo_ts=ppo["mean_pos_error_t"],
            ylabel="Mean position error",
            title=f"Reward R{R}: Mean position error over time",
            save_path=OUT_DIR / f"R{R}_timeseries_mean_pos_error.png",
        )

        # 4) Time series: mean covariance trace
        plot_time_series_comparison(
            heuristic_ts=heuristic["mean_cov_trace_t"],
            ppo_ts=ppo["mean_cov_trace_t"],
            ylabel="Mean covariance trace",
            title=f"Reward R{R}: Mean covariance trace over time",
            save_path=OUT_DIR / f"R{R}_timeseries_mean_cov_trace.png",
        )

    # 5) Delta boxplot across rewards
    plot_delta_boxplot(
        deltas_by_reward=pos_error_deltas,
        reward_labels=reward_labels,
        ylabel="Heuristic - PPO final mean position error",
        title="Final mean position error gain by PPO across rewards",
        save_path=OUT_DIR / "delta_boxplot_final_pos_error.png",
    )

    plot_delta_boxplot(
        deltas_by_reward=cov_trace_deltas,
        reward_labels=reward_labels,
        ylabel="Heuristic - PPO final mean covariance trace",
        title="Final mean covariance trace gain by PPO across rewards",
        save_path=OUT_DIR / "delta_boxplot_final_cov_trace.png",
    )


if __name__ == "__main__":
    main()