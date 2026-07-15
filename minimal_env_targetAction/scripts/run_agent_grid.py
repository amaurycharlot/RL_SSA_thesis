import os
import json
import argparse
import numpy as np

from agents.base_agent import BaseAgent
from ssa_env.envs.minimal_ssa_env_RL_grid import MinimalSSAEnv
from agents.heuristic_agent_RL_grid import HeuristicAgent
#from agents.random_pointing_agent import RandomPointingAgent
#from agents.omniscient_agent import OmniscientAgent
from agents.ppo_agent import PPOAgent


def make_agent(name: str, env: MinimalSSAEnv, ppodir: str) -> BaseAgent:
    name = name.lower().strip()

    if name in {"heuristic", "heuristicagent"}:
        return HeuristicAgent(num_targets=env.num_targets, world_size=env.world_size)
    elif name in {"ppo", "ppoagent"}:
        return PPOAgent(num_targets=env.num_targets, world_size=env.world_size, model_path=os.path.join(ppodir, "final_model.zip"))

    raise ValueError(f"Unknown agent '{name}'. Use 'heuristic', 'random', 'omniscient', or 'ppo'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="heuristic", help="heuristic | ppo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="grid_rollouts")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_targets", type=int, default=2)
    parser.add_argument("--measurement_noise_std", type=float, default=1.0)
    parser.add_argument("--ppodir", type=str, default=None, help="Directory where PPO models are saved.")
    args = parser.parse_args()

    env = MinimalSSAEnv(
        targets_known=True,
        num_targets=args.num_targets,
        measurement_noise_std=args.measurement_noise_std,
    )
    if args.ppodir is None:
        ppodir = f"grid_runs/ppo_baseline_{args.num_targets}targets"
    else:
        ppodir = args.ppodir
    agent = make_agent(args.agent, env=env, ppodir=ppodir)

    run_name = args.run_name or f"{args.agent}_seed{args.seed}"
    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)

    obs, info = env.reset(seed=args.seed)
    agent.reset()

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    per_target_dim = obs.size // env.num_targets

    T = int(args.steps)
    actions = np.zeros((T, 2), dtype=np.float32)
    sensor_pointings = np.zeros((T, 2), dtype=np.float32)
    est_states = np.zeros((T, env.num_targets, 9), dtype=np.float32)
    true_states = np.zeros((T, env.num_targets, info["true_states"].shape[1]), dtype=np.float32)

    rewards = np.zeros((T,), dtype=np.float32)
    terminated_flags = np.zeros((T,), dtype=np.int8)
    truncated_flags = np.zeros((T,), dtype=np.int8)

    for t in range(T):
        
        action = agent.select_action(obs)

        #action = np.asarray(action, dtype=np.float32).reshape(2,)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        actions[t] = action
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
        "steps_requested": args.steps,
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