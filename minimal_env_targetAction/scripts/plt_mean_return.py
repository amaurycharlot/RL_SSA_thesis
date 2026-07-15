# scripts/plot_mean_return.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_mean_return(run_dir: Path, out_file: Path):
    """
    Plot mean episodic return vs timesteps from SB3 EvalCallback logs.

    Requires:
        run_dir / "eval_logs" / "evaluations.npz"
    """

    eval_file = run_dir / "eval_logs" / "evaluations.npz"

    if not eval_file.exists():
        raise FileNotFoundError(
            f"Could not find {eval_file}\n"
            f"Did you train with eval_env_fn enabled?"
        )

    data = np.load(eval_file)

    timesteps = data["timesteps"]            # shape (n_evals,)
    results = data["results"]                # shape (n_evals, n_eval_episodes)

    mean_returns = results.mean(axis=1)
    std_returns = results.std(axis=1)

    # ---- Plot ----
    plt.figure(figsize=(7, 4.5))

    plt.plot(timesteps, mean_returns, label="Mean return")
    plt.fill_between(
        timesteps,
        mean_returns - std_returns,
        mean_returns + std_returns,
        alpha=0.25,
        label="±1 std",
    )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean episodic return")
    plt.title("Mean return vs timesteps")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

    print(f"Saved plot to: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/ppo_baseline",
        help="Directory containing eval_logs/",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: run_dir/mean_return.png)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_file = (
        Path(args.out)
        if args.out is not None
        else run_dir / "mean_return_vs_timesteps.png"
    )

    plot_mean_return(run_dir, out_file)


if __name__ == "__main__":
    main()