import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Params: args; returns: None.
def plot_rewards(args):
    run_dir = args.run_folder
    meta_path = os.path.join(run_dir, "meta.json")
    rew_path = os.path.join(run_dir, "rewards.npy")

    if not os.path.exists(rew_path):
        raise FileNotFoundError(f"Missing {rew_path}")

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    rewards = np.load(rew_path)  # (T,) in search/track, (T, 2) in combined

    T = rewards.shape[0]
    total_steps_path = os.path.join(run_dir, "total_steps.npy")
    if os.path.exists(total_steps_path):
        t = np.load(total_steps_path).astype(float)
    else:
        print("Warning: total_steps.npy not found, using action indices.")
        t = np.arange(T, dtype=float)


    if meta is not None and meta.get("mode") == "combined":
        #make two subplots for tracking and search rewards
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, rewards[:, 0], label="Tracking Reward", color="tab:blue")
        plt.title("Tracking Reward over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, rewards[:, 1], label="Search Reward", color="tab:orange")
        plt.title("Search Reward over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
    else:
        # Plot rewards    
        plt.title("Reward over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    out_path = os.path.join(run_dir, f"rewards.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved reward plot to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()

# Params: none; returns: None.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_folder",
        type=str,
        help="Path to a saved rollout folder, e.g. rollouts/heuristic_seed1",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = parser.parse_args()

    plot_rewards(args)

if __name__ == "__main__":
    main()