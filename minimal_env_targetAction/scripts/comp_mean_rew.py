import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def load_mean_reward(run_dir: str) -> float:
    rew_path = os.path.join(run_dir, "rewards.npy")
    rewards = np.load(rew_path)
    return float(np.mean(rewards))


def plot_random_vs_heuristic_box(random_means, heuristic_mean, reward_id, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Boxplot for random schedules
    ax.boxplot([random_means], positions=[1], widths=0.5, patch_artist=False)

    # Jittered raw points
    jitter = np.linspace(-0.08, 0.08, len(random_means))
    ax.scatter(1 + jitter, random_means, alpha=0.8, label="Random schedules")

    # Highlight points above heuristic
    better_mask = np.array(random_means) > heuristic_mean
    if np.any(better_mask):
        ax.scatter(
            1 + jitter[better_mask],
            np.array(random_means)[better_mask],
            marker="o",
            s=60,
            label="Random > heuristic",
        )

    # Heuristic baseline
    ax.axhline(heuristic_mean, linestyle="--", label=f"Heuristic = {heuristic_mean:.3f}")

    ax.set_xticks([1])
    ax.set_xticklabels([f"R{reward_id} random schedules"])
    ax.set_ylabel("Mean reward")
    ax.set_title(f"Reward R{reward_id}: random schedules vs heuristic baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_random_vs_heuristic_scatter(random_means, heuristic_mean, reward_id, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    seeds = np.arange(len(random_means))
    random_means = np.array(random_means)

    ax.scatter(seeds, random_means, label="Random schedules")

    better_mask = random_means > heuristic_mean
    if np.any(better_mask):
        ax.scatter(
            seeds[better_mask],
            random_means[better_mask],
            s=60,
            label="Random > heuristic",
        )

    ax.axhline(heuristic_mean, linestyle="--", label=f"Heuristic = {heuristic_mean:.3f}")

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Mean reward")
    ax.set_title(f"Reward R{reward_id}: mean reward by random schedule")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    out_dir = Path("plots/random_vs_heuristic")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose which reward types to analyze (currently just R5, but can expand to R1-R4)
    for reward_id in range(5, 6):
        print("-------------------------------")
        print(f"Reward Type: R{reward_id}")
        print("-------------------------------")

        random_means = []
        for rnd_seed in range(25):
            run_dir = f"rollouts/random_agents/R{reward_id}_random_{rnd_seed}"
            mean_reward = load_mean_reward(run_dir)
            random_means.append(mean_reward)

        heuristic_dir = f"rollouts/random_agents/R{reward_id}_heuristic"
        heuristic_mean = load_mean_reward(heuristic_dir)

        print(f"Mean Reward = {heuristic_mean:.6f} (heuristic)")

        for rnd_seed, mr in enumerate(random_means):
            if mr > heuristic_mean:
                print(f"Mean Reward = {mr:.6f} (random seed {rnd_seed})")

        num_better = sum(mr > heuristic_mean for mr in random_means)
        print(f"Random schedules beating heuristic: {num_better}/{len(random_means)}")
        print(f"Random mean ± std: {np.mean(random_means):.6f} ± {np.std(random_means):.6f}")

        """
        plot_random_vs_heuristic_box(
            random_means=random_means,
            heuristic_mean=heuristic_mean,
            reward_id=reward_id,
            save_path=out_dir / f"R{reward_id}_boxplot.png",
        )

        plot_random_vs_heuristic_scatter(
            random_means=random_means,
            heuristic_mean=heuristic_mean,
            reward_id=reward_id,
            save_path=out_dir / f"R{reward_id}_scatter.png",
        )
        """

if __name__ == "__main__":
    main()

    """
    for i in range(1,5) :
        #compute mean reward for rigged heuristic agent
        run_dir = f"rollouts/cheats/heuristicRL_5targets_R{i}"
        rew_path = os.path.join(run_dir, "rewards.npy")
        rewards = np.load(rew_path)
        mean_reward_heuristic = np.mean(rewards)
        print(f"Mean Reward = {mean_reward_heuristic} (rigged heuristic)")

        #compute mean reward for normal heuristic agent
        run_dir = f"rollouts/cheats/heuristicRL_5targets_R{i}_nrml"
        rew_path = os.path.join(run_dir, "rewards.npy")
        rewards = np.load(rew_path)
        mean_reward_heuristic = np.mean(rewards)
        print(f"Mean Reward = {mean_reward_heuristic} (normal heuristic)")
    """