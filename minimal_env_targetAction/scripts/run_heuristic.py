import numpy as np
import matplotlib.pyplot as plt

# import env
from ssa_env.envs.minimal_ssa_env import MinimalSSAEnv
from agents.heuristic_agent import HeuristicAgent


def _fmt_vec(v):
    return np.array(v, dtype=float).round(3)


def main():
    env = MinimalSSAEnv(num_targets=1)
    agent = HeuristicAgent(num_targets=3, world_size=env.world_size)

    obs, info = env.reset(seed=1)
    agent.reset()
    agent.observe_and_update(obs)
    # --- logging ---
    true_positions = {tid: [] for tid in range(env.num_targets)}
    est_positions = {tid: [] for tid in range(env.num_targets)}

    traceP = {tid: [] for tid in range(env.num_targets)}
    t_hist = []
    
    pos_error = {tid: [] for tid in range(env.num_targets)}
    sigma_pos = {tid: [] for tid in range(env.num_targets)}

    print("=== RESET ===")
    print("t:", float(obs["t"][0]))
    print("sensor_pos:", _fmt_vec(obs["sensor_pos"]))
    print("meas_mask:", obs["meas_mask"])
    print("meas_target_ids:", obs["meas_target_ids"])
    print("measurements:\n", _fmt_vec(obs["measurements"]))
    print()

    done = False
    step = 0

    while not done and step < 50:
        action = agent.select_action()

        # Safety check: action in bounds
        if np.any(action < 0.0) or np.any(action > env.world_size):
            print("WARNING: action out of bounds:", action)

        obs, reward, terminated, truncated, info = env.step(action)

        # Agent uses measurement-based obs to update belief
        agent.observe_and_update(obs)

        # --- save P trace ---
        t_now = float(obs["t"][0])
        t_hist.append(t_now)

        for tid in range(env.num_targets):
            if tid in agent.belief.tracks:
                trk = agent.belief.tracks[tid]
                traceP[tid].append(float(np.trace(trk.P)))
            else:
                traceP[tid].append(np.nan)

        # --- log true positions ---
        true_states = info["true_states"]  # shape (num_targets, state_dim)
        for tid in range(env.num_targets):
            true_positions[tid].append(true_states[tid, 0:2].copy())

        # --- log estimated positions ---
        for tid, trk in agent.belief.tracks.items():
            est_positions[tid].append(trk.x[0:2].copy())

        print(f"=== STEP {step:02d} ===")
        print("t:", float(obs["t"][0]), "reward:", reward)
        print("action:", _fmt_vec(action), "sensor_pos:", _fmt_vec(obs["sensor_pos"]))

        # --- log error + sigma bound (position) ---
        for tid in range(env.num_targets):
            true_xy = true_states[tid, 0:2].astype(np.float32)

            if tid in agent.belief.tracks:
                trk = agent.belief.tracks[tid]
                est_xy = trk.x[0:2].astype(np.float32)

                err = float(np.linalg.norm(est_xy - true_xy))
                Pxy = trk.P[0:2, 0:2]
                sig = float(np.sqrt(np.trace(Pxy)))  # 1-sigma scalar bound for position

                pos_error[tid].append(err)
                sigma_pos[tid].append(sig)
            else:
                pos_error[tid].append(np.nan)
                sigma_pos[tid].append(np.nan)

        # Print measurements received this step
        print("meas_mask:", obs["meas_mask"])
        print("meas_target_ids:", obs["meas_target_ids"])
        print("measurements:\n", _fmt_vec(obs["measurements"]))

        # Print belief state (xhat) for each known track
        mdp_state = agent.belief.as_mdp_state(include_cov_diag=False)
        print("belief xhat (rows=target_id, cols=[x,y,vx,vy]):\n", _fmt_vec(mdp_state))

        print()

        done = terminated or truncated
        step += 1
    # --- plotting ---
    plt.figure(figsize=(6, 6))

    for tid in range(env.num_targets):
        tp = np.array(true_positions[tid])
        if len(tp) > 0:
            plt.scatter(tp[:, 0], tp[:, 1], s=20, label=f"Target {tid} (true)")

        ep = np.array(est_positions[tid])
        if len(ep) > 0:
            plt.scatter(ep[:, 0], ep[:, 1], s=20, marker="x", label=f"Target {tid} (est)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True vs Estimated Target Positions")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # --- trace(P) plot ---
    plt.figure(figsize=(8, 4))

    t_arr = np.array(t_hist, dtype=float)

    for tid in range(env.num_targets):
        tr = np.array(traceP[tid], dtype=float)
        plt.plot(t_arr, tr, label=f"Target {tid} trace(P)")

    plt.xlabel("t")
    plt.ylabel("trace(P)")
    plt.title("Uncertainty Over Time (trace of covariance)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- per-target plots: position error vs sigma bound ---
    k_sigma = 3.0  # change to 1.0 if you want 1-sigma bounds

    t_arr = np.array(t_hist, dtype=float)

    for tid in range(env.num_targets):
        err = np.array(pos_error[tid], dtype=float)
        sig = np.array(sigma_pos[tid], dtype=float)

        plt.figure(figsize=(8, 4))
        plt.plot(t_arr, err, label="pos error ||x_hat - x||")
        plt.plot(t_arr, k_sigma * sig, label=f"{int(k_sigma)}σ bound (sqrt(trace(Pxy)))")
        plt.xlabel("t")
        plt.ylabel("distance")
        plt.title(f"Target {tid}: Position Error vs {int(k_sigma)}σ Uncertainty Bound")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()