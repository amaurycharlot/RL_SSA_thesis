import os
import json
import argparse
import numpy as np

from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.ppo_agent import PPOAgent


def make_agent(name: str, env: MinimalSSAEnv, ppodir: str = None, random_seed: int = 42):
    name = name.lower().strip()

    if name in {"heuristic", "heuristicagent"}:
        return HeuristicAgent(num_targets=env.num_targets, world_size=env.world_size, grid_shape=env.grid_shape, mode=env.mode)

    if name in {"random", "randompointing", "randompointingagent"}:
        return RandomAgent(num_targets=env.num_targets, world_size=env.world_size, grid_shape=env.grid_shape, 
                           random_seed=random_seed, mode=env.mode)
    
    if name in {"ppo", "ppoagent"}:
        if ppodir is None:
            raise ValueError("ppodir must be provided for PPO agent.")
        model_path = os.path.join(ppodir, "final_model.zip")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"PPO model not found at: {model_path}")
        return PPOAgent(num_targets=env.num_targets, world_size=env.world_size, grid_shape=env.grid_shape, mode=env.mode, model_path=model_path)

    raise ValueError(f"Unknown agent '{name}'. Use 'heuristic', 'random', 'omniscient', or 'ppo'.")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="heuristic", help="heuristic | random | omniscient | ppo")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outdir", type=str, default="rollouts")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--known_targets", type=int, default=None, help="Number of targets that are known at the start of the episode.")
    parser.add_argument("--unknown_targets", type=int, default=None, help="Number of targets that are unknown at the start of the episode.")
    parser.add_argument("--num_targets", type=int, default=20, help="Total number of targets (overrides known_targets + unknown_targets if set).")
    parser.add_argument("--grid_shape", nargs=2, type=int, default=(10, 10), help="Shape of the grid (rows, columns).")
    parser.add_argument("--measurement_noise_std", type=float, default=1.0)
    parser.add_argument("--ppodir", type=str, default=None, help="Directory where PPO models are saved.")
    parser.add_argument("--zero_noise", action="store_true", help="Whether to set all noise to zero.")
    parser.add_argument("--reward", type=str, default="R6", help="Reward type to use in the environment (e.g., R1, R2, R3, R4).")
    parser.add_argument("--search_reward", type=str, default="R1", help="Reward type to use for search reward in combined mode (e.g., R1, R2).")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for the random agent.")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of timesteps in the environment episode.")
    parser.add_argument("--mode", type=str, default="track", help="Environment mode: track | search | combined")
    parser.add_argument("--dwell_steps", type=int, default=1, help="Number of steps required to dwell on a cell for a successful observation.")
    parser.add_argument("--slew", type=float, default=0.0, help="Alpha parameter for slew time.")
    parser.add_argument("--simple_obs", action="store_true", help="Whether to use simple observations without sensor pointing information.")
    args = parser.parse_args(argv)
    if args.num_targets is not None:
        if args.mode == "track" :
            if args.known_targets is not None :
                args.num_targets = args.known_targets
            else :
                args.known_targets = args.num_targets
            args.unknown_targets = 0
        elif args.mode == "search" :
            if args.unknown_targets is not None :
                args.num_targets = args.unknown_targets
            else :
                args.unknown_targets = args.num_targets
            args.known_targets = 0
        elif args.mode == "combined":
            if args.known_targets is not None and args.unknown_targets is not None:
                args.num_targets = args.known_targets + args.unknown_targets
            else:
                if args.known_targets is not None and args.unknown_targets is None:
                    args.unknown_targets = args.num_targets - args.known_targets
                elif args.unknown_targets is not None and args.known_targets is None:
                    args.known_targets = args.num_targets - args.unknown_targets
                else:
                    # If neither known nor unknown targets are specified, default to half known, half unknown
                    args.known_targets = args.num_targets // 2
                    args.unknown_targets = args.num_targets - args.known_targets
        else:
            raise ValueError(f"Unknown mode '{args.mode}'. Use 'track', 'search', or 'combined'.")

    env = MinimalSSAEnv(
        known_targets=args.known_targets,
        unknown_targets=args.unknown_targets,
        grid_shape=args.grid_shape,
        measurement_noise_std=args.measurement_noise_std,
        reward_type=args.reward,
        search_reward=args.search_reward,
        zero_noise=args.zero_noise,
        mode=args.mode,
        timesteps=args.timesteps,
        dwell_steps=args.dwell_steps,
        slew_alpha=args.slew,
        simple_obs=args.simple_obs
    )
    if args.ppodir is None:
        ppodir = f"runs/ppo_baseline_{args.known_targets + args.unknown_targets}targets"
    else:
        ppodir = args.ppodir
    agent = make_agent(args.agent, env=env, ppodir=ppodir, random_seed=args.random_seed)

    num_target_str = ""
    if args.mode == "track" :
        num_target_str = f"_{args.known_targets}targets"
    elif args.mode == "search" :
        num_target_str = f"_{args.unknown_targets}targets"
    else : #combined
        num_target_str = f"_{args.known_targets}known_{args.unknown_targets}unknown"
    reward_str = ""
    if args.reward != "R6":
        reward_str = f"_{args.reward}"
    steps_str = ""
    if args.timesteps != 50:
        steps_str = f"_{args.timesteps}steps"
    slew_str = ""
    if args.slew != 0.0:
        slew_str = f"_slew{args.slew}"
    run_name = args.run_name or f"{args.agent}_{args.mode}_seed{args.seed}" + num_target_str + reward_str + steps_str + slew_str
    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)

    obs, info = env.reset(seed=args.seed)
    agent.reset()

    obs = np.asarray(obs, dtype=np.float32)
    per_target_dim = obs.size // env.num_targets

    T = int(args.timesteps)
    actions = np.zeros((T,2), dtype=np.int32)
    sensor_pointings = np.zeros((T, 2), dtype=np.float32)
    est_states = np.zeros((T, env.num_targets, 9), dtype=np.float32)
    true_states = np.zeros((T, env.num_targets, info["true_states"].shape[1]), dtype=np.float32)
    total_steps = np.zeros((T,), dtype=np.int32)
    elapsed_steps = np.zeros((T,), dtype=np.int32)


    # compute shape of observations to be saved : (concatenate T and obs.shape)
    obsevations = np.zeros((T, *obs.shape), dtype=np.float32)
    rewards = np.zeros((T,), dtype=np.float32)
    terminated_flags = np.zeros((T,), dtype=np.int8)
    truncated_flags = np.zeros((T,), dtype=np.int8)

    for t in range(T):
        action = agent.select_action(obs)

        #action = np.asarray(action, dtype=np.float32).reshape(2,)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        obsevations[t] = obs
        actions[t] = action
        if "estimated_states" in info:
            #print(f"Step {t}: Estimated states:\n{info['estimated_states']}")
            est_states[t] = info["estimated_states"]
        else:
            '''DEPRECATED'''
            est_states[t] = obs.reshape(env.num_targets, per_target_dim)

        if "true_states" not in info:
            raise KeyError(
                "info['true_states'] is missing. "
                "Please keep true_states in info for logging."
            )
        true_states[t] = info["true_states"].astype(np.float32)
        if "sensor_pointing" in info:
            sensor_pointings[t] = np.asarray(info["sensor_pointing"], dtype=np.float32).reshape(2,)
        if "total_steps" in info:
            total_steps[t] = info["total_steps"]
        if "elapsed_steps" in info:
            elapsed_steps[t] = info["elapsed_steps"]

        rewards[t] = float(reward)
        terminated_flags[t] = int(bool(terminated))
        truncated_flags[t] = int(bool(truncated))

        if terminated or truncated:
            actions = actions[: t + 1]
            sensor_pointings = sensor_pointings[: t + 1]
            est_states = est_states[: t + 1]
            true_states = true_states[: t + 1]
            obsevations = obsevations[: t + 1]
            rewards = rewards[: t + 1]
            total_steps = total_steps[: t + 1]
            elapsed_steps = elapsed_steps[: t + 1]
            terminated_flags = terminated_flags[: t + 1]
            truncated_flags = truncated_flags[: t + 1]
            break

    np.save(os.path.join(outdir, "actions.npy"), actions)
    np.save(os.path.join(outdir, "sensor_pointings.npy"), sensor_pointings)
    np.save(os.path.join(outdir, "est_states.npy"), est_states)
    np.save(os.path.join(outdir, "true_states.npy"), true_states)
    np.save(os.path.join(outdir, "observations.npy"), obsevations)
    np.save(os.path.join(outdir, "rewards.npy"), rewards)
    np.save(os.path.join(outdir, "terminated.npy"), terminated_flags)
    np.save(os.path.join(outdir, "truncated.npy"), truncated_flags)
    np.save(os.path.join(outdir, "total_steps.npy"), total_steps)
    np.save(os.path.join(outdir, "elapsed_steps.npy"), elapsed_steps)

    meta = {
        "agent": args.agent,
        "seed": args.seed,
        "steps_requested": args.timesteps,
        "steps_saved": int(actions.shape[0]),
        "known_targets": int(env.known_targets),
        "unknown_targets": int(env.unknown_targets),
        "world_size": float(env.world_size),
        "obs_dim": int(obs.size),
        "per_target_dim": int(per_target_dim),
        "true_state_dim": int(true_states.shape[2]),
        "measurement_noise_std": float(env.sensor.measurement_noise_std),
        "mode": env.mode,
        "grid_shape": env.grid_shape,
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved rollout to: {outdir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()