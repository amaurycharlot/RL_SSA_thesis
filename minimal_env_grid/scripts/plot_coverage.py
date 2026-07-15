"""
Script to plot the % of the grid that has been observed (i.e. "coverage") over time during a rollout.
"""

import argparse
from html import parser
import json
import os

from flask import json
import numpy as np
import matplotlib.pyplot as plt

def compute_coverage_over_time(observations: np.ndarray, mode: str) -> np.ndarray:
    """
    Computes the percentage of the grid that has been observed at each timestep.

    Args:
        observations: Array of shape (T, grid_rows, grid_cols, obs_layers) containing the observations at each timestep.

    Returns:
        coverage_t: Array of shape (T,) containing the percentage of the grid observed at each timestep.
    """
    T, grid_rows, grid_cols, obs_layers = observations.shape

    if mode == "search":
        # For the search mode,an observed cell has a non-zero value in observations[:,:,0]
        observed_mask = observations[:, :, :, 0] > 0
    elif mode == "combined":
        # For combined mode,an observed cell has a non-zero value in observations[:,:,1]
        observed_mask = observations[:, :, :, 1] > 0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Count how many unique cells have been observed up to each timestep
    cumulative_observed = np.zeros((grid_rows, grid_cols), dtype=bool)
    coverage_t = np.zeros(T)

    for t in range(T):
        cumulative_observed |= observed_mask[t]  # Update cumulative mask
        coverage_t[t] = np.sum(cumulative_observed) / (grid_rows * grid_cols) * 100  # Percentage

    return coverage_t

def plot_coverage_over_time(coverage_t: np.ndarray, t: np.ndarray, save_path: str):
    """
    Plots the coverage over time.

    Args:
        coverage_t: Array of shape (T,) containing the percentage of the grid observed at each timestep.
        t: Array of shape (T,) containing the timesteps.
        save_path: Path to save the plot image.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(t, coverage_t, label="Grid Coverage (%)")
    plt.xlabel("Timestep")
    plt.ylabel("Coverage (%)")
    plt.title("Grid Coverage Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main(args):

    # Load observations from the rollout
    obs_path = os.path.join(args.run_folder, "observations.npy")
    observations = np.load(obs_path)  # Shape: (T, grid_rows, grid_cols, obs_layers)
    # Extract mode from meta.json
    meta_path = os.path.join(args.run_folder, "meta.json")
    with open(meta_path, "r") as f:        
        meta = json.load(f)
        mode = meta.get("mode", "unknown")

    # Compute coverage over time
    coverage_t = compute_coverage_over_time(observations, mode)
    total_steps_path = os.path.join(args.run_folder, "total_steps.npy")
    if os.path.exists(total_steps_path):
        t = np.load(total_steps_path).astype(float)
    else:
        print("Warning: total_steps.npy not found, using action indices.")
        t = np.arange(observations.shape[0], dtype=float)

    # Plot and save the coverage over time
    save_path = os.path.join(args.run_folder, "coverage_over_time.png")
    plot_coverage_over_time(coverage_t, t, save_path)
    print(f"Saved coverage plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot grid coverage over time.")
    parser.add_argument("run_folder", type=str, help="Path to a saved rollout folder, e.g. rollouts/heuristic_seed1")

    args = parser.parse_args()
    main(args)
    