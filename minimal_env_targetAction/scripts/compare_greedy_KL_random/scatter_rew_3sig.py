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
    random_sum_3sigma = []

    for name in os.listdir(indir):
        if not name.startswith(f"R{reward_index}_random_"):
            continue

        meta_path = os.path.join(indir, name, "meta.json")
        if not os.path.exists(meta_path):
            continue

        meta = load_meta(meta_path)

        random_rewards.append(meta["mean_reward"])
        random_sum_3sigma.append(meta["sum_3sigma"])

    return (
        np.array(random_sum_3sigma, dtype=np.float64),
        np.array(random_rewards, dtype=np.float64),
    )


def load_greedy(indir):
    path = os.path.join(indir, "greedy_kl", "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing greedy_kl/meta.json")

    meta = load_meta(path)
    return float(meta["sum_3sigma"]), float(meta["mean_reward"])


def main(args):

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_scatter")
    os.makedirs(outdir, exist_ok=True)

    for i in range(1, 6):
        rand_sum3, rand_rew = collect_random_runs(indir, i)
        if args.greedyKL:
            greedy_sum3, greedy_rew = load_greedy(indir)

        plt.figure(figsize=(6, 5))

        plt.scatter(
            rand_sum3,
            rand_rew,
            alpha=0.7,
            label="Random agents",
        )

        if args.greedyKL:
            plt.scatter(
                [greedy_sum3],
                [greedy_rew],
                marker="x",
                s=120,
                linewidths=2,
                label="Greedy KL",
            )

        plt.xlabel("Sum 3-sigma")
        plt.ylabel("Mean reward")
        plt.title("Reward vs Total Uncertainty")
        plt.grid(True)
        plt.legend()

        out_path = os.path.join(outdir, f"R{i}_scatter_reward_vs_sum_3sigma.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="rollouts/random_agents", help="Directory containing random agent runs")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--greedyKL", action="store_true", help="Include Greedy KL point in the plot")
    args = parser.parse_args()
    main(args)