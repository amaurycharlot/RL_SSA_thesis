import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_meta(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_random_runs(indir, reward_index):
    random_rewards = []
    random_errors = []

    for name in os.listdir(indir):
        if not name.startswith(f"R{reward_index}_random_"):
            continue

        meta_path = os.path.join(indir, name, "meta.json")
        if not os.path.exists(meta_path):
            continue

        meta = load_meta(meta_path)

        random_rewards.append(meta["mean_reward"])
        random_errors.append(meta["mean_pos_error"])

    return np.array(random_errors), np.array(random_rewards)


def load_greedy(indir):
    path = os.path.join(indir, "greedy_kl", "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing greedy_kl/meta.json")

    meta = load_meta(path)
    return meta["mean_pos_error"], meta["mean_reward"]


def main(args):
    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_scatter")
    os.makedirs(outdir, exist_ok=True)
    for i in range(1, 6):
        # Load data
        rand_err, rand_rew = collect_random_runs(indir, i)
        if args.greedyKL:
            greedy_err, greedy_rew = load_greedy(indir)

        # Plot
        plt.figure(figsize=(6, 5))

        # Random runs
        plt.scatter(
            rand_err,
            rand_rew,
            alpha=0.6,
            label="Random agents",
        )

        if args.greedyKL:
            # Greedy KL point
            plt.scatter(
                [greedy_err],
                [greedy_rew],
                marker="x",
                s=100,
                linewidths=2,
                label="Greedy KL",
            )

        plt.xlabel("Mean positional error")
        plt.ylabel("Mean reward")
        plt.title("Reward vs Positional Error")

        plt.legend()
        plt.grid(True)

        # Save
        out_path = os.path.join(outdir, f"R{i}_scatter_reward_vs_error.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="rollouts/random_agents", help="Directory containing random agent runs")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--greedyKL", action="store_true", default=False, help="If set, plot greedy KL point")
    args = parser.parse_args()
    main(args)