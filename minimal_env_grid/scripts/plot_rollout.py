# Script to run plot_episode, plot_xy_errors and plot_rewards on a rollout folder
import argparse
import os

from flask import json
from scripts.plot_episode import plot_episode
from scripts.plot_xy_errors import plot_xy_errors
from scripts.plot_rewards import plot_rewards
from scripts.plot_coverage import main as plot_coverage

def main():

    parser = argparse.ArgumentParser(description="Plot rollout results")
    parser.add_argument("run_folder", help="Path to the rollout folder")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print("Plotting rollout results from: ", args.run_folder)
    print(args.run_folder)
    plot_episode(args)
    plot_xy_errors(args)
    plot_rewards(args)
    # load meta.json to check if mode is track, search or combined and pass to plot_coverage
    meta_path = os.path.join(args.run_folder, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            mode = meta.get("mode", "unknown")
        if mode in ["search", "combined"]:
            plot_coverage(args)
    else :
        print("Warning: meta.json not found")


if __name__ == "__main__":
    main()