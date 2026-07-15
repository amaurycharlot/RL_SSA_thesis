import os
import json
import argparse
import numpy as np


def load_meta(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_random_runs(indir, reward_index):
    rewards = []
    errors = []

    for name in os.listdir(indir):
        if not name.startswith(f"R{reward_index}_random_"):
            continue

        meta_path = os.path.join(indir, name, "meta.json")
        if not os.path.exists(meta_path):
            continue

        meta = load_meta(meta_path)

        rewards.append(meta["mean_reward"])
        errors.append(meta["mean_pos_error"])

    rewards = np.array(rewards, dtype=np.float32)
    errors = np.array(errors, dtype=np.float32)

    return rewards, errors


def main(args):
    for i in range(1, 6):
        rewards, errors = collect_random_runs(args.indir, i)

        if len(rewards) == 0:
            raise ValueError("No random runs found.")

        # Pearson correlation
        corr = np.corrcoef(rewards, errors)[0, 1]

        # Also compute Spearman (rank correlation)
        rank_rewards = np.argsort(np.argsort(rewards))
        rank_errors = np.argsort(np.argsort(errors))
        spearman = np.corrcoef(rank_rewards, rank_errors)[0, 1]

        print(f"==== Correlation Analysis (R{i}) ====")
        print(f"Number of runs: {len(rewards)}")
        print()
        print(f"Pearson corr(reward, error): {corr:.4f}")
        print(f"Spearman corr(reward, error): {spearman:.4f}")
        print()

        if corr < 0:
            print("✔ Negative correlation: higher reward → lower error (good)")
        else:
            print("✘ Positive correlation: higher reward → higher error (bad / misaligned reward)")

        # Save results
        out = {
            "num_runs": int(len(rewards)),
            "pearson_reward_error": float(corr),
            "spearman_reward_error": float(spearman),
        }

        out_path = os.path.join(args.indir, f"R{i}_correlation_reward_error.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="rollouts/random_agents", help="Directory containing random agent runs")
    args = parser.parse_args()
    main(args)