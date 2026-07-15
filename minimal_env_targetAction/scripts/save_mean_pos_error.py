"""
Loads the rewards from a rollout and saves the mean positional error and sum of 3-sigma to meta.json. This is used to compute the correlation between reward and error
"""
import os
import json
import argparse
import numpy as np

def main(args):
    # Load estimated states and true states from the rollout
    estimated_states_path = os.path.join(args.indir, "est_states.npy")
    true_states_path = os.path.join(args.indir, "true_states.npy")

    if not os.path.exists(estimated_states_path):
        raise FileNotFoundError(f"Missing estimated_states.npy in {args.indir}")
    if not os.path.exists(true_states_path):
        raise FileNotFoundError(f"Missing true_states.npy in {args.indir}")

    estimated_states = np.load(estimated_states_path)
    true_states = np.load(true_states_path)

    # Compute positional errors
    pos_errors = wrap_diff(estimated_states[:, :, 0:2], true_states[:, :, 0:2], L=100.0)
    pos_errors_euclidean = np.linalg.norm(pos_errors, axis=2)
    mean_pos_error = float(np.mean(pos_errors_euclidean))

    # Compute sum of 3-sigma
    var_x = np.maximum(estimated_states[:, :, 4], 0.0)
    var_y = np.maximum(estimated_states[:, :, 5], 0.0)
    var_xy = np.linalg.norm(np.stack([var_x, var_y], axis=2), axis=2)
    sum_3sigma = float(np.sum(3.0 * np.sqrt(var_xy)))

    # Save mean positional error and sum of 3-sigma to meta.json
    meta_path = os.path.join(args.indir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    meta["mean_pos_error"] = mean_pos_error
    meta["sum_3sigma"] = sum_3sigma

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved mean positional error {mean_pos_error:.4f} and sum of 3-sigma {sum_3sigma:.4f} to {meta_path}")

def wrap_diff(a, b, L):
    d = a - b
    return (d + L/2.0) % L - L/2.0

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="rollouts/random_agents/R1_random_0", help="Directory containing the rollout with rewards.npy")
    args = parser.parse_args()
    for R in range(1, 6):
        for seed in range(100):
            args.indir = os.path.join(f"rollouts/random_agents_std0.1/R{R}_random_{seed}")
            main(args)

    