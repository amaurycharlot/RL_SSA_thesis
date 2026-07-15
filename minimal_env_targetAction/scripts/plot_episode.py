# scripts/plot_episode.py
import os
import json
import argparse
from matplotlib.patches import Circle
import numpy as np
import matplotlib.pyplot as plt

def plot_episode(args, time_step: int=None, out_path: str=None):
    
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

        # Plot FoV circle with radius = 10 in data units
        circle = Circle((x, y), radius=10,
                        edgecolor='red',
                        facecolor='red',
                        alpha=0.2)  # light fill

        plt.gca().add_patch(circle)
    else:
        plt.scatter(actions[:, 0], actions[:, 1], s=28, marker="s", label="Sensor pointing", color="red")

    plt.xlim(0.0, world_size)
    plt.ylim(0.0, world_size)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Episode Plot: True vs Estimated Target Positions + Sensor Pointing")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
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

def _load_rollout(run_dir):
    with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f) if os.path.exists(os.path.join(run_dir, "meta.json")) else {}

    true = np.load(os.path.join(run_dir, "true_states.npy"))
    est = np.load(os.path.join(run_dir, "est_states.npy"))

    actions_path = os.path.join(run_dir, "sensor_pointings.npy")
    if not os.path.exists(actions_path):
        actions_path = os.path.join(run_dir, "actions.npy")
    actions = np.load(actions_path)

    return meta, true, est, actions


def _variance_indices(est):
    D = est.shape[-1]
    if D >= 7:
        return 5, 6
    if D >= 6:
        return 4, 5
    raise ValueError(f"Unexpected est dimension {D}")


def _first_acquisitions(est):
    idx_varx, idx_vary = _variance_indices(est)
    first_acq = []
    for tid in range(est.shape[1]):
        mask = (est[:, tid, idx_varx] > 0.0) | (est[:, tid, idx_vary] > 0.0)
        first_acq.append(int(np.argmax(mask)) if np.any(mask) else None)
    return first_acq


def _plot_base_step(
    true_xy,
    est_xy,
    first_acq,
    time_step,
    world_size,
    title,
    show_est_history=True,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    N = true_xy.shape[1]

    for tid in range(N):
        # true position at displayed time
        ax.scatter(
            true_xy[time_step, tid, 0],
            true_xy[time_step, tid, 1],
            s=80,
            label=f"Target {tid} true",
        )

        # estimated trajectory up to displayed time
        k0 = first_acq[tid]
        if show_est_history and k0 is not None and k0 <= time_step:
            ep = est_xy[k0:time_step + 1, tid, :]
            ax.scatter(
                ep[:, 0],
                ep[:, 1],
                s=28,
                marker="x",
                label=f"Target {tid} est",
            )

    ax.set_xlim(0.0, world_size)
    ax.set_ylim(0.0, world_size)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    return fig, ax


def _add_fov(ax, action_xy, radius=10.0):
    x, y = float(action_xy[0]), float(action_xy[1])
    ax.scatter(x, y, s=40, color="red", label="Sensor pointing", zorder=5)
    ax.add_patch(
        Circle(
            (x, y),
            radius=radius,
            edgecolor="red",
            facecolor="red",
            alpha=0.2,
        )
    )


def _save_or_show(fig, out_path, show=False):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved:", out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_environment_step_illustration(
    run_folder: str,
    time_step: int,
    phase: str,
    out_path: str = None,
    fov_radius: float = 10.0,
    show: bool = False,
):
    """
    Generate one slide-style illustration for a given environment phase.

    Convention:
    - time_step t represents the state before action t.
    - action[t] points the sensor.
    - true[t+1] approximates the state after target propagation.
    - est[t+1] approximates the catalogue after measurement update.

    Valid phases:
    - "initial"
    - "select_action"
    - "propagate"
    - "update"
    """

    meta, true, est, actions = _load_rollout(run_folder)
    world_size = float(meta.get("world_size", 100.0))

    true_xy = true[:, :, 0:2].astype(np.float32)
    est_xy = est[:, :, 0:2].astype(np.float32)
    first_acq = _first_acquisitions(est)

    T = true.shape[0]
    if time_step < 0 or time_step >= T:
        raise ValueError(f"time_step must be in [0, {T - 1}]")

    if phase in {"propagate", "update"} and time_step + 1 >= T:
        raise ValueError(f"phase='{phase}' needs time_step + 1, but t={time_step} is last frame.")

    if phase == "initial":
        fig, ax = _plot_base_step(
            true_xy,
            est_xy,
            first_acq,
            time_step,
            world_size,
            title=f"1) Initial state at t={time_step}",
        )

    elif phase == "select_action":
        fig, ax = _plot_base_step(
            true_xy,
            est_xy,
            first_acq,
            time_step,
            world_size,
            title=f"2) Select action: point sensor at t={time_step}",
        )
        _add_fov(ax, actions[time_step], radius=fov_radius)

    elif phase == "propagate":
        fig, ax = _plot_base_step(
            true_xy,
            est_xy,
            first_acq,
            time_step + 1,
            world_size,
            title=f"3) Propagate targets: true state t={time_step} → t={time_step + 1}",
        )
        _add_fov(ax, actions[time_step], radius=fov_radius)

        # draw arrows from old true positions to propagated true positions
        for tid in range(true_xy.shape[1]):
            x0, y0 = true_xy[time_step, tid]
            x1, y1 = true_xy[time_step + 1, tid]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    elif phase == "update":
        fig, ax = _plot_base_step(
            true_xy,
            est_xy,
            first_acq,
            time_step + 1,
            world_size,
            title=f"4) Measurements + EKF catalogue update at t={time_step + 1}",
        )
        _add_fov(ax, actions[time_step], radius=fov_radius)

    else:
        raise ValueError(
            "phase must be one of: initial, select_action, propagate, update"
        )

    if out_path is None:
        out_path = os.path.join(run_folder, f"env_step_t{time_step}_{phase}.png")

    _save_or_show(fig, out_path, show=show)


def plot_environment_step_sequence(
    run_folder: str,
    time_step: int,
    out_dir: str = None,
    fov_radius: float = 10.0,
    show: bool = False,
):
    """
    Generate all slide illustrations for one environment step.
    """
    if out_dir is None:
        out_dir = os.path.join(run_folder, f"step_illustration_t{time_step}")
    os.makedirs(out_dir, exist_ok=True)

    phases = ["initial", "select_action", "propagate", "update"]

    for i, phase in enumerate(phases, start=1):
        out_path = os.path.join(out_dir, f"{i}_{phase}.png")
        plot_environment_step_illustration(
            run_folder=run_folder,
            time_step=time_step,
            phase=phase,
            out_path=out_path,
            fov_radius=fov_radius,
            show=show,
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder", type=str)
    parser.add_argument("--time_step", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--illustrate_step",
        action="store_true",
        help="Generate the step-by-step environment illustrations for a given timestep.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="Optional single phase: initial | select_action | propagate | update",
    )
    parser.add_argument("--fov_radius", type=float, default=10.0)

    args = parser.parse_args()

    if args.illustrate_step:
        if args.time_step is None:
            raise ValueError("--illustrate_step requires --time_step")

        if args.phase is None:
            plot_environment_step_sequence(
                run_folder=args.run_folder,
                time_step=args.time_step,
                fov_radius=args.fov_radius,
                show=args.show,
            )
        else:
            plot_environment_step_illustration(
                run_folder=args.run_folder,
                time_step=args.time_step,
                phase=args.phase,
                fov_radius=args.fov_radius,
                show=args.show,
            )
    else:
        plot_episode(args, time_step=args.time_step)



if __name__ == "__main__":
    main()