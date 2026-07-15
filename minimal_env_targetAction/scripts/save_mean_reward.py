"""
Loads the rewards from a rollout and saves the mean reward to meta.json. This is used to compute the correlation between reward and error or reward and sum_3sigma.
"""
import os
import json
import argparse
import numpy as np

def main(args):
    # Load rewards from the rollout
    rewards_path = os.path.join(args.indir, "rewards.npy")
    if not os.path.exists(rewards_path):
        raise FileNotFoundError(f"Missing rewards.npy in {args.indir}")

    rewards = np.load(rewards_path)
    mean_reward = float(np.mean(rewards))

    # Save mean reward to meta.json
    meta_path = os.path.join(args.indir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    meta["mean_reward"] = mean_reward

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved mean reward {mean_reward:.4f} to {meta_path}")

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="rollouts/random_agents/R1_random_0", help="Directory containing the rollout with rewards.npy")
    args = parser.parse_args()
    for R in range(1, 6):
        for seed in range(100):
            args.indir = os.path.join(f"rollouts/random_agents_std0.1/R{R}_random_{seed}")
            main(args)

    