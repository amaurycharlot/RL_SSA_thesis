import os
import json
import argparse
import numpy as np
import time

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv

from agents.RL.grid_cnn_extractor import GridCNNExtractor

class TimingCallback(BaseCallback):
    """
    Prints elapsed time and ETA during training.
    """

    def __init__(self, total_timesteps: int, verbose=1):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_print_step = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        """
        Called after each PPO rollout collection.
        This aligns nicely with SB3 summary printing.
        """

        elapsed = time.time() - self.start_time

        timesteps_done = self.num_timesteps
        fps = timesteps_done / max(elapsed, 1e-8)

        progress = timesteps_done / self.total_timesteps_target
        remaining_timesteps = self.total_timesteps_target - timesteps_done

        eta_seconds = remaining_timesteps / max(fps, 1e-8)

        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60

        print(
            "\n"
            f"[Timing] "
            f"{timesteps_done:,}/{self.total_timesteps_target:,} timesteps "
            f"({100*progress:.1f}%) | "
            f"Elapsed: {elapsed/60:.1f} min | "
            f"FPS: {fps:.0f} | "
            f"ETA: {eta_minutes:.1f} min ({eta_hours:.2f} h)"
        )

def make_env(rank: int, args):
    def _init():
        env = MinimalSSAEnv(
            known_targets=args.known_targets,
            unknown_targets=0,
            grid_shape=(args.grid_rows, args.grid_cols),
            timesteps=args.episode_steps,
            measurement_noise_std=args.measurement_noise_std,
            reward_type=args.reward,
            zero_noise=args.zero_noise,
            mode="track",
        )
        env.reset(seed=args.seed + rank)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--known_targets", type=int, default=20)
    parser.add_argument("--grid_rows", type=int, default=10)
    parser.add_argument("--grid_cols", type=int, default=10)
    parser.add_argument("--episode_steps", type=int, default=200)
    parser.add_argument("--total_timesteps", type=int, default=500_000)

    parser.add_argument("--reward", type=str, default="R6")
    parser.add_argument("--measurement_noise_std", type=float, default=1.0)
    parser.add_argument("--zero_noise", type=bool, default=True)

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
    parser.add_argument("--subproc", action="store_true")

    args = parser.parse_args()

    run_name = args.run_name or (
        f"ppo_track_grid_{args.grid_rows}x{args.grid_cols}_"
        f"{args.known_targets}targets_{args.episode_steps}steps_seed{args.seed}"
    )

    run_dir = os.path.join(args.outdir, run_name)
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.check_env:
        test_env = MinimalSSAEnv(
            known_targets=args.known_targets,
            unknown_targets=0,
            grid_shape=(args.grid_rows, args.grid_cols),
            timesteps=args.episode_steps,
            measurement_noise_std=args.measurement_noise_std,
            reward_type=args.reward,
            zero_noise=args.zero_noise,
            mode="track",
        )
        check_env(test_env, warn=True)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv

    train_env = vec_cls([make_env(i, args) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(10_000, args)])
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
        policy_kwargs=dict(
            features_extractor_class=GridCNNExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        ),
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
        callback=[eval_callback, checkpoint_callback, timing_callback],
        tb_log_name="ppo",
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)

    train_env.close()
    eval_env.close()

    print(f"Saved final model to: {final_path}")
    print(f"Best model, checkpoints, logs saved in: {run_dir}")


if __name__ == "__main__":
    main()