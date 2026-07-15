import os
import json
import argparse
import numpy as np


def load_meta(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_random_runs(indir, reward_index):
    rewards = []
    sum_3sigma = []

    for name in os.listdir(indir):
        if not name.startswith(f"R{reward_index}_random_"):
            continue

        meta_path = os.path.join(indir, name, "meta.json")
        if not os.path.exists(meta_path):
            continue

        meta = load_meta(meta_path)

        rewards.append(meta["mean_reward"])
        sum_3sigma.append(meta["sum_3sigma"])

    rewards = np.array(rewards, dtype=np.float64)
    sum_3sigma = np.array(sum_3sigma, dtype=np.float64)

    return rewards, sum_3sigma


def spearman_corr(x, y):
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def main(args):
    for i in range(1, 6):
        rewards, sum_3sigma = collect_random_runs(args.indir, i)

        if len(rewards) == 0:
            raise ValueError("No random runs found.")
        if len(rewards) < 2:
            raise ValueError("Need at least 2 random runs to compute a correlation.")

        pearson = float(np.corrcoef(rewards, sum_3sigma)[0, 1])
        spearman = spearman_corr(rewards, sum_3sigma)

        print(f"==== Correlation Analysis (R{i}) ====")
        print(f"Number of runs: {len(rewards)}")
        print()
        print(f"Pearson corr(reward, sum_3sigma): {pearson:.4f}")
        print(f"Spearman corr(reward, sum_3sigma): {spearman:.4f}")
        print()

        if pearson < 0:
            print("✔ Negative correlation: higher reward -> lower total uncertainty (good)")
        else:
            print("✘ Positive correlation: higher reward -> higher total uncertainty (bad / misaligned reward)")

        out = {
            "num_runs": int(len(rewards)),
            "pearson_reward_sum_3sigma": pearson,
            "spearman_reward_sum_3sigma": spearman,
        }

        out_path = os.path.join(args.indir, f"R{i}_correlation_reward_sum_3sigma.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="rollouts/random_agents", help="Directory containing random agent runs")
    args = parser.parse_args()
    main(args)