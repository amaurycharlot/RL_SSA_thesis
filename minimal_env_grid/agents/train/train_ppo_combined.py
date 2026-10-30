import os
import json
import argparse
import time
import multiprocessing as mp
from contextlib import ExitStack

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
)

from agents.mo.types import EnvConfig
from agents.mo.wrappers import LinearScalarizationWrapper
from agents.mo.envs import (
    TrainingSubprocVecEnv, worker_thread_environment, make_scalar_env, set_worker_threads,
)
from agents.RL.grid_cnn_extractor import GridCNNExtractor




class TimingCallback(BaseCallback):
    """
    Prints elapsed time and ETA during training.
    """

    # Params: self, total_timesteps, verbose; returns: None.
    def __init__(self, total_timesteps: int, verbose=1):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps

    # Params: self; returns: None.
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_print_step = 0

    # Params: self; returns: bool.
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        """
        Called after each PPO rollout collection.
        """

        elapsed = time.time() - self.start_time

        timesteps_done = self.num_timesteps
        fps = timesteps_done / max(elapsed, 1e-8)

        progress = timesteps_done / self.total_timesteps_target
        remaining_timesteps = (
            self.total_timesteps_target - timesteps_done
        )

        eta_seconds = remaining_timesteps / max(fps, 1e-8)
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60

        print(
            "\n"
            f"[Timing] "
            f"{timesteps_done:,}/{self.total_timesteps_target:,} timesteps "
            f"({100 * progress:.1f}%) | "
            f"Elapsed: {elapsed / 60:.1f} min | "
            f"FPS: {fps:.0f} | "
            f"ETA: {eta_minutes:.1f} min ({eta_hours:.2f} h)"
        )


def create_combined_env(args, seed):
    """Legacy args adapter; physical defaults and identity scales are unchanged."""
    env_config = EnvConfig(
        known_targets=args.known_targets, unknown_targets=args.unknown_targets,
        grid_shape=(args.grid_rows, args.grid_cols), episode_steps=args.episode_steps,
        measurement_noise_std=args.measurement_noise_std, zero_noise=args.zero_noise,
        reward_type=args.reward, search_reward=args.search_reward,
        dwell_steps=1, slew_alpha=1.0, uncertainty_norm_scale=100.0,
    )
    weights = np.array([args.track_weight, args.search_weight], dtype=np.float32)
    return make_scalar_env(env_config, weights, seed)




# Params: rank, args, worker_threads; returns: _init.
def make_env(rank: int, args, worker_threads=None):
    # Params: none; returns: Monitor.
    def _init():
        set_worker_threads(worker_threads)
        return Monitor(create_combined_env(args, args.seed + rank))
    return _init




def create_train_env(args):
    """Keep the same environment count and rank seeds in both execution modes."""
    if args.subproc:
        with worker_thread_environment(args.worker_threads):
            return TrainingSubprocVecEnv(
                [make_env(i, args, args.worker_threads) for i in range(args.n_envs)],
                start_method="spawn",
            )
    return DummyVecEnv([make_env(i, args) for i in range(args.n_envs)])


# Params: none; returns: parser.
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--known_targets", type=int, default=10)
    parser.add_argument("--unknown_targets", type=int, default=10)

    parser.add_argument("--grid_rows", type=int, default=10)
    parser.add_argument("--grid_cols", type=int, default=10)
    parser.add_argument("--episode_steps", type=int, default=200)
    parser.add_argument("--total_timesteps", type=int, default=500_000)

    parser.add_argument("--reward", type=str, default="R6")
    parser.add_argument("--search_reward", type=str, default="R2")
    parser.add_argument(
        "--measurement_noise_std",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--zero_noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # Reward order:
    # [tracking_reward, search_reward]
    parser.add_argument("--track_weight", type=float, default=0.85)
    parser.add_argument("--search_weight", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--check_env", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--subproc", action=argparse.BooleanOptionalAction, default=True,
        help="Run training environments in spawned processes (default). "
             "Use --no-subproc for sequential execution.",
    )
    execution.add_argument(
        "--sequential", dest="subproc", action="store_false",
        help="Alias for --no-subproc; use DummyVecEnv.",
    )
    parser.add_argument(
        "--worker_threads", type=int, default=1,
        help="Numerical-library threads per simulation worker; learner unchanged.",
    )
    return parser


# Params: none; returns: None.
def main():
    args = build_parser().parse_args()
    if args.n_envs < 1 or args.worker_threads < 1:
        raise ValueError("n_envs and worker_threads must be positive.")

    weights = np.array(
        [args.track_weight, args.search_weight],
        dtype=np.float32,
    )

    if np.any(weights < 0):
        raise ValueError("Scalarization weights cannot be negative.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            "The scalarization weights must sum to 1. "
            f"Received {weights.tolist()}, with sum {weights.sum():.6f}."
        )

    weight_label = (
        f"w{args.track_weight:g}_{args.search_weight:g}"
    )

    run_name = args.run_name or (
        f"ppo_combined_grid_{args.grid_rows}x{args.grid_cols}_"
        f"{args.known_targets}known_"
        f"{args.unknown_targets}unknown_"
        f"{args.episode_steps}steps_"
        f"{weight_label}_seed{args.seed}"
    )

    run_dir = os.path.join(args.outdir, run_name)
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    config = vars(args).copy()
    config["mode"] = "combined"
    config["measurement_rng"] = "environment_np_random"
    config["reward_order"] = [
        "tracking_reward",
        "search_reward",
    ]
    config["scalarization_weights"] = weights.tolist()

    with open(os.path.join(run_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2)

    print(
        "Training in combined mode with scalarization:\n"
        f"  tracking weight: {weights[0]:.3f}\n"
        f"  search weight:   {weights[1]:.3f}"
    )

    if args.check_env:
        # Check the wrapped environment because SB3 requires scalar rewards.
        test_env = create_combined_env(
            args=args,
            seed=args.seed,
        )
        try:
            check_env(test_env, warn=True)
        finally:
            test_env.close()

    print(f"Training execution: {'parallel (spawn)' if args.subproc else 'sequential'}; "
          f"{args.n_envs} environments")

    with ExitStack() as resources:
        train_env = create_train_env(args)
        resources.callback(train_env.close)
        train_env = VecMonitor(train_env)

        eval_env = DummyVecEnv(
            [make_env(10_000, args)]
        )
        resources.callback(eval_env.close)
        eval_env = VecMonitor(eval_env)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=model_dir,
            name_prefix="checkpoint",
        )

        timing_callback = TimingCallback(
            total_timesteps=args.total_timesteps
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs={
                "features_extractor_class": GridCNNExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 128,
                },
            },
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[
                eval_callback,
                checkpoint_callback,
                timing_callback,
            ],
            tb_log_name="ppo",
            progress_bar=True,
        )

        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)

    print(f"Saved final model to: {final_path}")
    print(f"Best model, checkpoints, and logs saved in: {run_dir}")


if __name__ == "__main__":
    mp.freeze_support()
    main()