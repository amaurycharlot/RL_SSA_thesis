import os
import json
import argparse
import numpy as np

from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
from agents.greedy_KL_agent import GreedyKLAgent
from agents.random_pointing_agent_RL import RandomPointingAgent as RandomPointingAgent_RL


def parse_initial_target_vars(s: str, num_targets: int):
    vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) != num_targets:
        raise ValueError(
            f"initial_target_vars must have length {num_targets}, got {len(vals)}: {vals}"
        )
    return vals


def wrap_diff_1d(a: float, b: float, world_size: float) -> float:
    d = float(a - b)
    half = world_size / 2.0
    if d > half:
        d -= world_size
    elif d < -half:
        d += world_size
    return d


def wrapped_position_error(pos_hat: np.ndarray, pos_true: np.ndarray, world_size: float) -> float:
    dx = wrap_diff_1d(pos_hat[0], pos_true[0], world_size)
    dy = wrap_diff_1d(pos_hat[1], pos_true[1], world_size)
    return float(np.sqrt(dx * dx + dy * dy))


def make_env(args):
    return MinimalSSAEnv(
        targets_known=True,
        num_targets=args.num_targets,
        measurement_noise_std=args.measurement_noise_std,
        zero_noise=args.zero_noise,
        reward_type="R5",   # KL-divergence reward
        timesteps=args.timesteps,
        initial_target_vars=args.initial_target_vars,
    )


def rollout_episode(env, agent, agent_name: str, seed: int, timesteps: int, random_seed: int = 42):
    obs, info = env.reset(seed=seed)
    agent.reset()

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)

    T = int(timesteps)
    N = int(env.num_targets)

    actions = np.zeros((T,), dtype=np.int32)
    rewards = np.zeros((T,), dtype=np.float32)
    sensor_pointings = np.zeros((T, 2), dtype=np.float32)
    est_states = np.zeros((T, N, 9), dtype=np.float32)
    true_states = np.zeros((T, N, 4), dtype=np.float32)
    pos_errors = np.zeros((T, N), dtype=np.float32)
    pos_3sigma = np.zeros((T, N), dtype=np.float32)
    terminated_flags = np.zeros((T,), dtype=np.int8)
    truncated_flags = np.zeros((T,), dtype=np.int8)

    for t in range(T):
        if agent_name == "greedy_kl":
            action = int(agent.select_action(env))
        else:
            action = int(agent.select_action(obs))

        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        actions[t] = action
        rewards[t] = float(reward)
        terminated_flags[t] = int(bool(terminated))
        truncated_flags[t] = int(bool(truncated))

        if "estimated_states" not in info:
            raise KeyError("info['estimated_states'] is missing from env.step(...)")
        if "true_states" not in info:
            raise KeyError("info['true_states'] is missing from env.step(...)")
        if "sensor_pointing" not in info:
            raise KeyError("info['sensor_pointing'] is missing from env.step(...)")

        est_t = np.asarray(info["estimated_states"], dtype=np.float32).reshape(N, 9)
        true_t = np.asarray(info["true_states"], dtype=np.float32).reshape(N, 4)

        est_states[t] = est_t
        true_states[t] = true_t
        sensor_pointings[t] = np.asarray(info["sensor_pointing"], dtype=np.float32).reshape(2,)

        # Positional error + scalar 3-sigma metric
        for i in range(N):
            pos_hat = est_t[i, 0:2]
            pos_true = true_t[i, 0:2]

            pos_errors[t, i] = wrapped_position_error(pos_hat, pos_true, env.world_size)

            # estimated_states[:,4:8] are diag(P), so [4], [5] = position variances
            pxx = max(float(est_t[i, 4]), 0.0)
            pyy = max(float(est_t[i, 5]), 0.0)
            pos_3sigma[t, i] = float(3.0 * np.sqrt(pxx + pyy))

        if terminated or truncated:
            end = t + 1
            actions = actions[:end]
            rewards = rewards[:end]
            sensor_pointings = sensor_pointings[:end]
            est_states = est_states[:end]
            true_states = true_states[:end]
            pos_errors = pos_errors[:end]
            pos_3sigma = pos_3sigma[:end]
            terminated_flags = terminated_flags[:end]
            truncated_flags = truncated_flags[:end]
            break

    metrics = {
        "steps_saved": int(actions.shape[0]),
        "mean_reward": float(np.mean(rewards)),
        "sum_reward": float(np.sum(rewards)),
        "mean_pos_error": float(np.mean(pos_errors)),
        "sum_pos_error": float(np.sum(pos_errors)),
        "mean_3sigma": float(np.mean(pos_3sigma)),
        "sum_3sigma": float(np.sum(pos_3sigma)),
    }

    return {
        "actions": actions,
        "rewards": rewards,
        "sensor_pointings": sensor_pointings,
        "est_states": est_states,
        "true_states": true_states,
        "pos_errors": pos_errors,
        "pos_3sigma": pos_3sigma,
        "terminated": terminated_flags,
        "truncated": truncated_flags,
        "metrics": metrics,
    }


def save_rollout(outdir: str, name: str, data: dict, meta: dict):
    run_dir = os.path.join(outdir, name)
    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, "actions.npy"), data["actions"])
    np.save(os.path.join(run_dir, "rewards.npy"), data["rewards"])
    np.save(os.path.join(run_dir, "sensor_pointings.npy"), data["sensor_pointings"])
    np.save(os.path.join(run_dir, "est_states.npy"), data["est_states"])
    np.save(os.path.join(run_dir, "true_states.npy"), data["true_states"])
    np.save(os.path.join(run_dir, "pos_errors.npy"), data["pos_errors"])
    np.save(os.path.join(run_dir, "pos_3sigma.npy"), data["pos_3sigma"])
    np.save(os.path.join(run_dir, "terminated.npy"), data["terminated"])
    np.save(os.path.join(run_dir, "truncated.npy"), data["truncated"])

    meta_out = dict(meta)
    meta_out.update(data["metrics"])

    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="comparison_rollouts")
    parser.add_argument("--seed", type=int, default=1, help="Environment seed")
    parser.add_argument("--num_targets", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--measurement_noise_std", type=float, default=1.0)
    parser.add_argument("--zero_noise", action="store_true")
    parser.add_argument(
        "--initial_target_vars",
        type=str,
        required=True,
        help='Comma-separated per-target initial variances, e.g. "25,9,4,1,1"',
    )
    parser.add_argument(
        "--num_random_runs",
        type=int,
        default=25,
        help="Number of random agent runs with different seeds",
    )
    args = parser.parse_args()

    args.initial_target_vars = parse_initial_target_vars(
        args.initial_target_vars, args.num_targets
    )

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # 1. GREEDY KL (single run)
    # -----------------------------
    env_greedy = make_env(args)
    greedy_agent = GreedyKLAgent(
        num_targets=args.num_targets, world_size=env_greedy.world_size
    )

    greedy_data = rollout_episode(
        env=env_greedy,
        agent=greedy_agent,
        agent_name="greedy_kl",
        seed=args.seed,
        timesteps=args.timesteps,
    )

    common_meta = {
        "seed": int(args.seed),
        "num_targets": int(args.num_targets),
        "timesteps_requested": int(args.timesteps),
        "measurement_noise_std": float(args.measurement_noise_std),
        "zero_noise": bool(args.zero_noise),
        "reward_type": "R5",
        "initial_target_vars": list(args.initial_target_vars),
    }

    save_rollout(
        args.outdir,
        "greedy_kl",
        greedy_data,
        {**common_meta, "agent": "greedy_kl"},
    )

    # -----------------------------
    # 2. RANDOM AGENTS (multiple runs)
    # -----------------------------
    random_metrics = []

    for i in range(args.num_random_runs):
        random_seed = i

        env_random = make_env(args)
        random_agent = RandomPointingAgent_RL(
            num_targets=args.num_targets,
            world_size=env_random.world_size,
            seed=random_seed,
        )

        run_name = f"random_seed_{random_seed}"

        random_data = rollout_episode(
            env=env_random,
            agent=random_agent,
            agent_name="random",
            seed=args.seed,   # SAME ENV SEED
            timesteps=args.timesteps,
            random_seed=random_seed,
        )

        save_rollout(
            args.outdir,
            run_name,
            random_data,
            {**common_meta, "agent": "random", "random_seed": random_seed},
        )

        metrics = dict(random_data["metrics"])
        metrics["random_seed"] = random_seed
        random_metrics.append(metrics)

    # -----------------------------
    # 3. AGGREGATE RANDOM METRICS
    # -----------------------------
    def extract(metric_name):
        return np.array([m[metric_name] for m in random_metrics], dtype=np.float32)

    random_summary = {
        "num_runs": args.num_random_runs,
        "sum_3sigma_mean": float(np.mean(extract("sum_3sigma"))),
        "sum_3sigma_std": float(np.std(extract("sum_3sigma"))),
        "mean_3sigma_mean": float(np.mean(extract("mean_3sigma"))),
        "mean_3sigma_std": float(np.std(extract("mean_3sigma"))),
        "sum_pos_error_mean": float(np.mean(extract("sum_pos_error"))),
        "sum_pos_error_std": float(np.std(extract("sum_pos_error"))),
        "mean_pos_error_mean": float(np.mean(extract("mean_pos_error"))),
        "mean_pos_error_std": float(np.std(extract("mean_pos_error"))),
        "sum_reward_mean": float(np.mean(extract("sum_reward"))),
        "sum_reward_std": float(np.std(extract("sum_reward"))),
    }

    # -----------------------------
    # 4. FINAL SUMMARY
    # -----------------------------
    summary = {
        "greedy_kl": greedy_data["metrics"],
        "random_stats": random_summary,
        "difference_mean_random_minus_greedy": {
            "sum_3sigma": float(random_summary["sum_3sigma_mean"] - greedy_data["metrics"]["sum_3sigma"]),
            "mean_3sigma": float(random_summary["mean_3sigma_mean"] - greedy_data["metrics"]["mean_3sigma"]),
            "sum_pos_error": float(random_summary["sum_pos_error_mean"] - greedy_data["metrics"]["sum_pos_error"]),
            "mean_pos_error": float(random_summary["mean_pos_error_mean"] - greedy_data["metrics"]["mean_pos_error"]),
        },
    }

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()