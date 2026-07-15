import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from agents.ckf.kf_predict_update import make_F


def wrap_diff(a, b, L):
    d = a - b
    return (d + L/2.0) % L - L/2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder", type=str)
    parser.add_argument("--world_size", type=float, default=None)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--per_target_dim", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_folder
    meta_path = os.path.join(run_dir, "meta.json")
    true_path = os.path.join(run_dir, "true_states.npy")
    est_path = os.path.join(run_dir, "est_states.npy")
    act_path = os.path.join(run_dir, "actions.npy")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    true = np.load(true_path)  # (T,N,true_dim)
    est = np.load(est_path)    # (T,N,per_dim)
    actions = np.load(act_path)

    T, N, true_dim = true.shape
    _, _, per_dim = est.shape

    world_size = args.world_size
    if world_size is None:
        world_size = float(meta.get("world_size", 100.0))

    if args.per_target_dim is not None:
        per_dim = args.per_target_dim

    # ---- interpret est layout ----
    # After your 4D change: per target likely [x,y,vx,vy, diagP(4), traceP] => 9 dims
    # We'll try to infer.
    if per_dim == 9:
        IDX_X, IDX_Y, IDX_VX, IDX_VY = 0, 1, 2, 3
        IDX_VARX, IDX_VARY = 4, 5
        IDX_TRACE = 8
    elif per_dim == 8:
        # (rare) [x,y,vx,vy, diagP(4)]
        IDX_X, IDX_Y, IDX_VX, IDX_VY = 0, 1, 2, 3
        IDX_VARX, IDX_VARY = 4, 5
        IDX_TRACE = None
    else:
        print(f"WARNING: unexpected per_target_dim={per_dim}.")
        print("Edit indices in script if needed.")
        return

    omega = args.omega
    if omega is None:
        omega = meta.get("omega_const", None)
    if omega is None:
        # fallback (you must set it)
        omega = 0.05
    omega = float(omega)

    F = make_F(args.dt, omega).astype(np.float32)

    # ---- 1) Check acquisition / whether tracks exist ----
    # We assume "unknown" targets have var_x==0 and var_y==0 (no track)
    acquired = (est[:, :, IDX_VARX] > 0.0) | (est[:, :, IDX_VARY] > 0.0)
    first_acq = []
    for tid in range(N):
        if not np.any(acquired[:, tid]):
            first_acq.append(None)
        else:
            first_acq.append(int(np.argmax(acquired[:, tid])))
    print("First acquisition step per target:", first_acq)

    # ---- 2) Check predict consistency ----
    # Compare predicted state from previous posterior to current posterior.
    # If filter works, current state should be near F * previous state when no measurement update,
    # and should deviate when update occurs.
    pred_err = {tid: [] for tid in range(N)}
    update_flags = {tid: [] for tid in range(N)}

    for tid in range(N):
        for k in range(1, T):
            if not acquired[k-1, tid]:
                pred_err[tid].append(np.nan)
                update_flags[tid].append(False)
                continue

            x_prev = est[k-1, tid, 0:4].astype(np.float32)
            x_pred = F @ x_prev

            x_now = est[k, tid, 0:4].astype(np.float32)

            # torus distance on position, euclidean on velocity
            dx = wrap_diff(x_now[0], x_pred[0], world_size)
            dy = wrap_diff(x_now[1], x_pred[1], world_size)
            dvx = float(x_now[2] - x_pred[2])
            dvy = float(x_now[3] - x_pred[3])

            e = float(np.linalg.norm([dx, dy, dvx, dvy]))
            pred_err[tid].append(e)

            # crude update detection: covariance trace drop from k-1 to k
            tr_prev = float(est[k-1, tid, IDX_VARX] + est[k-1, tid, IDX_VARY])
            tr_now = float(est[k, tid, IDX_VARX] + est[k, tid, IDX_VARY])
            update_flags[tid].append(tr_now < tr_prev * 0.9)  # big shrink implies update

    for tid in range(N):
        arr = np.array(pred_err[tid], dtype=float)
        print(f"Target {tid}: predict error stats (nan ignored)")
        print("  mean:", np.nanmean(arr), "max:", np.nanmax(arr))

    # ---- 3) Plot predict error and covariance proxy ----
    t = np.arange(T, dtype=float)

    for tid in range(N):
        if first_acq[tid] is None:
            continue
        k0 = first_acq[tid]
        t2 = np.arange(1, T, dtype=float)

        pe = np.array(pred_err[tid], dtype=float)
        covproxy = est[1:, tid, IDX_VARX] + est[1:, tid, IDX_VARY]

        plt.figure(figsize=(10, 4))
        plt.plot(t2[k0:], pe[k0:], label="||x_k - F x_{k-1}|| (torus on pos)")
        plt.xlabel("t")
        plt.ylabel("predict mismatch")
        plt.title(f"Target {tid}: Predict-step Consistency")
        plt.grid(True)
        plt.legend()
        out = os.path.join(run_dir, f"debug_target_{tid}_predict_mismatch.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        if args.show:
            plt.show()
        else:
            plt.close()
        print("Saved:", out)

        plt.figure(figsize=(10, 4))
        plt.plot(t[1:][k0:], covproxy[k0:], label="var_x + var_y (proxy)")
        plt.xlabel("t")
        plt.ylabel("uncertainty proxy")
        plt.title(f"Target {tid}: Uncertainty Proxy Over Time")
        plt.grid(True)
        plt.legend()
        out2 = os.path.join(run_dir, f"debug_target_{tid}_uncertainty_proxy.png")
        plt.tight_layout()
        plt.savefig(out2, dpi=150)
        if args.show:
            plt.show()
        else:
            plt.close()
        print("Saved:", out2)

    # ---- 4) Plot actual position errors (wrapped) ----
    for tid in range(N):
        if first_acq[tid] is None:
            continue
        k0 = first_acq[tid]

        ex = wrap_diff(est[:, tid, IDX_X], true[:, tid, 0], world_size)
        ey = wrap_diff(est[:, tid, IDX_Y], true[:, tid, 1], world_size)

        plt.figure(figsize=(10, 4))
        plt.plot(t[k0:], ex[k0:], label="x error (wrapped)")
        plt.plot(t[k0:], ey[k0:], label="y error (wrapped)")
        plt.xlabel("t")
        plt.ylabel("error")
        plt.title(f"Target {tid}: Wrapped Position Errors")
        plt.grid(True)
        plt.legend()
        out3 = os.path.join(run_dir, f"debug_target_{tid}_wrapped_pos_error.png")
        plt.tight_layout()
        plt.savefig(out3, dpi=150)
        if args.show:
            plt.show()
        else:
            plt.close()
        print("Saved:", out3)


if __name__ == "__main__":
    main()