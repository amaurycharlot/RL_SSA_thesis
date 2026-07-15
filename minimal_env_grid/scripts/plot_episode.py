# scripts/plot_episode.py
import os
import json
import argparse
from matplotlib.patches import Circle, Rectangle
import numpy as np
import matplotlib.pyplot as plt

def plot_episode(args, time_step: int=None, out_path: str=None, legend: bool=True):
    
    run_dir = args.run_folder
    meta_path = os.path.join(run_dir, "meta.json")
    true_path = os.path.join(run_dir, "true_states.npy")
    est_path = os.path.join(run_dir, "est_states.npy")
    actions_path = os.path.join(run_dir, "sensor_pointings.npy")

    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing {true_path}")
    if not os.path.exists(est_path):
        raise FileNotFoundError(f"Missing {est_path}")
    if not os.path.exists(actions_path):
        actions_path = os.path.join(run_dir, "actions.npy")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Missing {actions_path}")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    world_size = float(meta.get("world_size", 100.0))

    true = np.load(true_path)      # (T, N, true_dim)
    est = np.load(est_path)        # (T, N, per_target_dim)
    actions = np.load(actions_path)  # (T, 2) ; sensor pointing per step

    T, N, _ = true.shape
    _, _, D = est.shape

    # We consider a target "acquired" once any variance entry becomes > 0.
    # For 5D belief: var_x index = 5, var_y = 6
    # For 4D belief: var_x index = 4, var_y = 5
    if D >= 7:
        IDX_VARX, IDX_VARY = 5, 6   # old 5D layout
    elif D >= 6:
        IDX_VARX, IDX_VARY = 4, 5   # 4D layout with diag(P) right after state
    else:
        raise ValueError(f"Unexpected per_target_dim={D}; cannot infer variance indices.")

    # True positions
    true_xy = true[:, :, 0:2].astype(np.float32)

    # Estimated positions (assume state starts at [x,y,...])
    est_xy = est[:, :, 0:2].astype(np.float32)

    # First acquisition per target
    first_acq = []
    for tid in range(N):
        acquired_mask = (est[:, tid, IDX_VARX] > 0.0) | (est[:, tid, IDX_VARY] > 0.0)
        if not np.any(acquired_mask):
            first_acq.append(None)
        else:
            first_acq.append(int(np.argmax(acquired_mask)))

    plt.figure(figsize=(7, 7))

    # Plot targets. If time_step is given, plot :
    #   - True positions only AT that time step (to see where targets actually were at that moment)
    #   - Estimated positions only up to that time step
    # Otherwise plot whole episode.
    for tid in range(N):
        tp = true_xy[time_step, tid, :] if time_step is not None else true_xy[:, tid, :]
        label=f"Target {tid} true"
        if time_step is not None:
            plt.scatter(tp[0], tp[1], s=80, label=label)
        else:
            plt.scatter(tp[:, 0], tp[:, 1], s=18, label=label)

        k0 = first_acq[tid]
        if k0 is not None:
            ep = est_xy[k0:time_step, tid, :] if time_step is not None else est_xy[k0:, tid, :]
            plt.scatter(ep[:, 0], ep[:, 1], s=28, marker="x", label=f"Target {tid} est (from t={k0})")
        else:
            print(f"Target {tid}: never acquired; skipping estimated positions.")

    # Plot sensor positions as red squares.
    # We assume the sensor pointing equals the action.
    # If time_step is given, plot sensor :
    #   - Only AT that time step (to see where sensor pointed at that moment)
    #   - as a light circle with radius equal to FoV (=10 for now)
    if time_step is not None:
        x = actions[time_step, 0]
        y = actions[time_step, 1]

        # Plot the sensor point
        plt.scatter(x, y, s=28, label="Sensor pointing", color="red", alpha=0.8)

        # Plot FoV square of size 30x30 centered at (x, y)
        square_size = 30

        square = Rectangle(
            (x - square_size / 2, y - square_size / 2),  # bottom-left corner
            square_size,
            square_size,
            edgecolor='red',
            facecolor='red',
            alpha=0.2
        )

        plt.gca().add_patch(square)
    else:
        plt.scatter(actions[:, 0], actions[:, 1], s=28, marker="s", label="Sensor pointing", color="red")

    plt.xlim(0.0, world_size)
    plt.ylim(0.0, world_size)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Episode Plot: True vs Estimated Target Positions + Sensor Pointing")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    if legend:
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    if out_path is None:
        out_path = os.path.join(run_dir, "episode_plot.png") if time_step is None else os.path.join(run_dir, f"episode_plot_t{time_step}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved:", out_path)

    if args.show:
        plt.show()
    else:
        plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_folder",
        type=str,
        help="Path to a saved rollout folder, e.g. rollouts/heuristic_seed1",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=None,
        help="If given, plot only the specified time step (for true positions and sensor pointing). Otherwise plot whole episode.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively.")
    parser.add_argument("--no_legend", action="store_true", help="Whether to hide legend (useful for single-target plots).")
    args = parser.parse_args()
    
    plot_episode(args, time_step=args.time_step, legend=not args.no_legend)



if __name__ == "__main__":
    main()