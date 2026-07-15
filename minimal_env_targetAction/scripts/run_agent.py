import os
import json
import argparse
import numpy as np

from agents.greedy_KL_agent import GreedyKLAgent
from ssa_env.envs.minimal_ssa_env import MinimalSSAEnv
from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv as MinimalSSAEnv_RL
from agents.heuristic_agent import HeuristicAgent
from agents.heuristic_agent_RL import HeuristicAgent as HeuristicAgent_RL
from agents.random_pointing_agent import RandomPointingAgent
from agents.random_pointing_agent_RL import RandomPointingAgent as RandomPointingAgent_RL
from agents.omniscient_agent import OmniscientAgent
from agents.ppo_agent import PPOAgent


def make_agent(name: str, env: MinimalSSAEnv):
    name = name.lower().strip()

    if name in {"heuristic", "heuristicagent"}:
        return HeuristicAgent(num_targets=env.num_targets, world_size=env.world_size)

    if name in {"random", "randompointing", "randompointingagent"}:
        return RandomPointingAgent(num_targets=env.num_targets, world_size=env.world_size)

    if name in {"omniscient", "omniscientagent"}:
        return OmniscientAgent(num_targets=env.num_targets, world_size=env.world_size)

    raise ValueError(f"Unknown agent '{name}'. Use 'heuristic', 'random', 'omniscient', or 'ppo'.")

def make_RL_agent(name: str, env: MinimalSSAEnv_RL, ppodir: str, random_seed: int) -> PPOAgent:
    name = name.lower().strip()

    if name in {"heuristic", "heuristicagent"}:
        return HeuristicAgent_RL(num_targets=env.num_targets, world_size=env.world_size)
    
    if name in {"greedy_kl", "greedykl", "kl"}:
        return GreedyKLAgent(num_targets=env.num_targets, world_size=env.world_size)
    
    if name in {"random", "randompointing", "randompointingagent"}:
        return RandomPointingAgent_RL(num_targets=env.num_targets, world_size=env.world_size, seed=random_seed)

    if name in {"ppo", "ppoagent"}:
        ppopath = os.path.join(ppodir, "final_model.zip")
        return PPOAgent(num_targets=env.num_targets, world_size=env.world_size, model_path=ppopath)

    raise ValueError(f"Unknown agent '{name}'. Use 'heuristic' or 'ppo'.")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="heuristic", help="heuristic | random | omniscient | ppo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="rollouts")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_targets", type=int, default=2)
    parser.add_argument("--measurement_noise_std", type=float, default=1.0)
    parser.add_argument("--targets_known", action="store_true", default=True, help="Whether to use the known-targets version of the environment.")
    parser.add_argument("--targets_unknown", action="store_true", default=False, help="Whether to use the unknown-targets version of the environment (overrides --targets_known).")
    parser.add_argument("--RL", action="store_true", help="Whether to use the RL version of the environment and agent.")
    parser.add_argument("--ppodir", type=str, default=None, help="Directory where PPO models are saved.")
    parser.add_argument("--zero_noise", action="store_true", help="Whether to set all noise to zero.")
    parser.add_argument("--reward", type=str, default="R1", help="Reward type to use in the environment (e.g., R1, R2, R3, R4).")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for the random agent.")
    parser.add_argument("--timesteps", type=int, default=50, help="Number of timesteps in the environment episode.")
    args = parser.parse_args(argv)

    if args.targets_unknown:
        args.targets_known = False

    if args.agent.lower() in {"ppo", "ppoagent"} or args.RL:
        env = MinimalSSAEnv_RL(
            targets_known=args.targets_known,
            num_targets=args.num_targets,
            measurement_noise_std=args.measurement_noise_std,
            zero_noise=args.zero_noise,
            reward_type=args.reward,
            timesteps=args.timesteps
        )
        if args.ppodir is None:
            ppodir = f"runs/ppo_baseline_{args.num_targets}targets"
        else:
            ppodir = args.ppodir
        agent = make_RL_agent(args.agent, env=env, ppodir=ppodir, random_seed=args.random_seed)
    else:
        env = MinimalSSAEnv(
            targets_known=args.targets_known,
            num_targets=args.num_targets,
            measurement_noise_std=args.measurement_noise_std,
            zero_noise=args.zero_noise,
        )
        agent = make_agent(args.agent, env=env)

    run_name = args.run_name or f"{args.agent}_seed{args.seed}"
    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)

    obs, info = env.reset(seed=args.seed)
    agent.reset()

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    per_target_dim = obs.size // env.num_targets

    T = int(args.timesteps)
    actions = np.zeros((T,), dtype=np.int32)
    sensor_pointings = np.zeros((T, 2), dtype=np.float32)
    est_states = np.zeros((T, env.num_targets, 9), dtype=np.float32)
    true_states = np.zeros((T, env.num_targets, info["true_states"].shape[1]), dtype=np.float32)

    rewards = np.zeros((T,), dtype=np.float32)
    terminated_flags = np.zeros((T,), dtype=np.int8)
    truncated_flags = np.zeros((T,), dtype=np.int8)

    for t in range(T):
        if args.agent.lower() == "omniscient":
            action = agent.select_action(info["true_states"])
            print(f"Step {t}: OmniscientAgent action: {action}")
        elif args.agent.lower() in {"greedy_kl", "greedykl", "kl"}:
            action = agent.select_action(env)
        else:
            action = agent.select_action(obs)

        #action = np.asarray(action, dtype=np.float32).reshape(2,)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        actions[t] = int(action)
        if "estimated_states" in info:
            #print(f"Step {t}: Estimated states:\n{info['estimated_states']}")
            est_states[t] = info["estimated_states"].reshape(env.num_targets, 9)
        else:
            est_states[t] = obs.reshape(env.num_targets, per_target_dim)

        if "true_states" not in info:
            raise KeyError(
                "info['true_states'] is missing. "
                "Please keep true_states in info for logging."
            )
        true_states[t] = info["true_states"].astype(np.float32)
        if "sensor_pointing" in info:
            sensor_pointings[t] = np.asarray(info["sensor_pointing"], dtype=np.float32).reshape(2,)

        rewards[t] = float(reward)
        terminated_flags[t] = int(bool(terminated))
        truncated_flags[t] = int(bool(truncated))

        if terminated or truncated:
            actions = actions[: t + 1]
            sensor_pointings = sensor_pointings[: t + 1]
            est_states = est_states[: t + 1]
            true_states = true_states[: t + 1]
            rewards = rewards[: t + 1]
            terminated_flags = terminated_flags[: t + 1]
            truncated_flags = truncated_flags[: t + 1]
            break

    np.save(os.path.join(outdir, "actions.npy"), actions)
    np.save(os.path.join(outdir, "sensor_pointings.npy"), sensor_pointings)
    np.save(os.path.join(outdir, "est_states.npy"), est_states)
    np.save(os.path.join(outdir, "true_states.npy"), true_states)
    np.save(os.path.join(outdir, "rewards.npy"), rewards)
    np.save(os.path.join(outdir, "terminated.npy"), terminated_flags)
    np.save(os.path.join(outdir, "truncated.npy"), truncated_flags)

    meta = {
        "agent": args.agent,
        "seed": args.seed,
        "steps_requested": args.timesteps,
        "steps_saved": int(actions.shape[0]),
        "num_targets": int(env.num_targets),
        "world_size": float(env.world_size),
        "obs_dim": int(obs.size),
        "per_target_dim": int(per_target_dim),
        "true_state_dim": int(true_states.shape[2]),
        "measurement_noise_std": float(env.sensor.measurement_noise_std),
        "targets_known": bool(env.targets_known),
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved rollout to: {outdir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()