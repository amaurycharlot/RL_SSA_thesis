import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_xy_errors(args):
    run_dir = args.run_folder
    meta_path = os.path.join(run_dir, "meta.json")
    est_path = os.path.join(run_dir, "est_states.npy")
    true_path = os.path.join(run_dir, "true_states.npy")

    if not os.path.exists(est_path):
        raise FileNotFoundError(f"Missing {est_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing {true_path}")

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    est = np.load(est_path)     # (T, N, per_target_dim)
    true = np.load(true_path)   # (T, N, true_dim)

    T, N, D = est.shape

    total_steps_path = os.path.join(run_dir, "total_steps.npy")
    if os.path.exists(total_steps_path):
        t = np.load(total_steps_path).astype(float)
    else:
        print("Warning: total_steps.npy not found, using action indices.")
        t = np.arange(T, dtype=float)

    # Per-target indices
    IDX_X = 0
    IDX_Y = 1
    IDX_VARX = 4
    IDX_VARY = 5

    for tid in range(N):
        WORLD_SIZE = 100.0  # or load from meta

        err_x = wrap_diff(est[:, tid, IDX_X], true[:, tid, 0], WORLD_SIZE)
        err_y = wrap_diff(est[:, tid, IDX_Y], true[:, tid, 1], WORLD_SIZE)

        var_x = np.maximum(est[:, tid, IDX_VARX], 0.0)
        var_y = np.maximum(est[:, tid, IDX_VARY], 0.0)

        sig3_x = 3.0 * np.sqrt(var_x)
        sig3_y = 3.0 * np.sqrt(var_y)

        # --- find first acquisition/observation time ---
        acquired_mask = (var_x > 0.0) | (var_y > 0.0)
        if not np.any(acquired_mask):
            print(f"Target {tid}: never acquired (skipping plots).")
            continue

        k0 = int(np.argmax(acquired_mask))  # first True index
        t_plot = t[k0:]

        err_x_plot = err_x[k0:]
        err_y_plot = err_y[k0:]
        sig3_x_plot = sig3_x[k0:]
        sig3_y_plot = sig3_y[k0:]

        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        # --- X subplot ---
        axes[0].scatter(t_plot, err_x_plot, s=18, label="error x")
        axes[0].plot(t_plot, +sig3_x_plot, linestyle=":", label="+3σ x")
        axes[0].plot(t_plot, -sig3_x_plot, linestyle=":", label="-3σ x")
        axes[0].set_ylabel("x error")
        axes[0].set_title(f"Target {tid} — X Position Error (from first obs at t={t[k0]})")
        axes[0].grid(True)
        axes[0].legend()

        # --- Y subplot ---
        axes[1].scatter(t_plot, err_y_plot, s=18, label="error y")
        axes[1].plot(t_plot, +sig3_y_plot, linestyle=":", label="+3σ y")
        axes[1].plot(t_plot, -sig3_y_plot, linestyle=":", label="-3σ y")
        axes[1].set_xlabel("t (step)")
        axes[1].set_ylabel("y error")
        axes[1].set_title(f"Target {tid} — Y Position Error (from first obs at t={t[k0]})")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()

        out_path = os.path.join(run_dir, f"target_{tid}_xy_error_3sigma.png")
        plt.savefig(out_path, dpi=150)

        if args.show:
            plt.show()
        else:
            plt.close()

        
        print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_folder",
        type=str,
        help="Path to a saved rollout folder, e.g. rollouts/heuristic_seed1",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = parser.parse_args()

    plot_xy_errors(args)

    
def wrap_diff(a, b, L):
    d = a - b
    return (d + L/2.0) % L - L/2.0


if __name__ == "__main__":
    main()