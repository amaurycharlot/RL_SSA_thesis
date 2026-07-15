"""
Optuna hyperparameter optimization for PPO in the MinimalSSAEnv search mode.

Example:
    python optimize_ppo_search.py \
        --unknown_targets 0 \
        --search_reward R2 \
        --n_trials 50 \
        --trial_timesteps 300000 \
        --n_envs 8 \
        --subproc

Resume an existing study:
    python optimize_ppo_search.py \
        --study_name ppo_search_R2 \
        --storage sqlite:///optuna_ppo_search.db \
        --n_trials 50
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import Trial

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
from agents.RL.grid_cnn_extractor import GridCNNExtractor


# ---------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------

def make_env(
    args: argparse.Namespace,
    rank: int,
    seed_offset: int = 0,
) -> Callable[[], gym.Env]:
    """
    Return an environment factory compatible with DummyVecEnv/SubprocVecEnv.

    The seed is set during reset rather than globally to ensure that each
    vectorized environment receives a distinct initial state.
    """

    def _init() -> gym.Env:
        env = MinimalSSAEnv(
            unknown_targets=args.unknown_targets,
            grid_shape=(args.grid_rows, args.grid_cols),
            timesteps=args.episode_steps,
            measurement_noise_std=args.measurement_noise_std,
            search_reward=args.search_reward,
            zero_noise=args.zero_noise,
            mode="search",
            simple_obs=args.simple_obs,
        )

        env.reset(seed=args.seed + seed_offset + rank)
        return Monitor(env)

    return _init


def create_train_env(
    args: argparse.Namespace,
    trial_number: int,
) -> VecEnv:
    """Create vectorized training environments for one Optuna trial."""

    vec_env_class = SubprocVecEnv if args.subproc else DummyVecEnv

    # Keep each trial's environments reproducible but distinct from other trials.
    trial_seed_offset = trial_number * 100_000

    env = vec_env_class(
        [
            make_env(
                args=args,
                rank=rank,
                seed_offset=trial_seed_offset,
            )
            for rank in range(args.n_envs)
        ]
    )

    return VecMonitor(env)


def create_eval_env(
    args: argparse.Namespace,
    trial_number: int,
) -> VecEnv:
    """
    Create an evaluation environment independent of the training environments.
    """

    trial_seed_offset = 10_000_000 + trial_number * 100_000

    env = DummyVecEnv(
        [
            make_env(
                args=args,
                rank=0,
                seed_offset=trial_seed_offset,
            )
        ]
    )

    return VecMonitor(env)


# ---------------------------------------------------------------------
# Optuna-aware evaluation callback
# ---------------------------------------------------------------------

class TrialEvalCallback(EvalCallback):
    """
    Evaluate a model periodically and report the score to Optuna.

    The value reported to Optuna is EvalCallback.best_mean_reward.
    A trial is marked for pruning when the configured Optuna pruner
    determines that its intermediate results are not competitive.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: Trial,
        n_eval_episodes: int,
        eval_freq: int,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=None,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
            warn=True,
        )

        self.trial = trial
        self.eval_index = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        # EvalCallback evaluates whenever n_calls % eval_freq == 0.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_index += 1

            self.trial.report(
                float(self.last_mean_reward),
                step=self.eval_index,
            )

            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return continue_training
    
def sample_ppo_params(
    trial: Trial,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Sample PPO hyperparameters.

    n_steps is defined per environment. Therefore the rollout buffer contains:

        rollout_size = n_steps * n_envs

    batch_size must divide rollout_size cleanly to avoid truncated minibatches.
    """

    n_steps = trial.suggest_categorical(
        "n_steps",
        [128, 256, 512, 1024, 2048],
    )

    rollout_size = n_steps * args.n_envs

    valid_batch_sizes = [
        batch_size
        for batch_size in [32, 64, 128, 256, 512, 1024]
        if batch_size <= rollout_size
        and rollout_size % batch_size == 0
    ]

    if not valid_batch_sizes:
        raise optuna.TrialPruned(
            f"No valid batch size for rollout size {rollout_size}."
        )

    batch_size = trial.suggest_categorical(
        "batch_size",
        valid_batch_sizes,
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True,
    )

    gamma = trial.suggest_float(
        "gamma",
        0.90,
        0.9999,
        log=True,
    )

    gae_lambda = trial.suggest_float(
        "gae_lambda",
        0.80,
        1.00,
    )

    clip_range = trial.suggest_float(
        "clip_range",
        0.10,
        0.40,
    )

    ent_coef = trial.suggest_float(
        "ent_coef",
        1e-6,
        0.05,
        log=True,
    )

    vf_coef = trial.suggest_float(
        "vf_coef",
        0.10,
        1.00,
    )

    max_grad_norm = trial.suggest_float(
        "max_grad_norm",
        0.30,
        2.00,
        log=True,
    )

    n_epochs = trial.suggest_categorical(
        "n_epochs",
        [3, 5, 10, 15, 20],
    )

    features_dim = trial.suggest_categorical(
        "features_dim",
        [64, 128, 256],
    )

    # SB3 also permits schedules, but fixed learning rates and clip ranges
    # make the first optimization study easier to interpret.
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "features_dim": features_dim,
    }

def objective(
    trial: Trial,
    args: argparse.Namespace,
) -> float:
    """
    Train and evaluate one PPO configuration.

    Returns:
        Best mean evaluation return obtained during the trial.
    """

    train_env: VecEnv | None = None
    eval_env: VecEnv | None = None
    model: PPO | None = None

    try:
        sampled_params = sample_ppo_params(trial, args)

        train_env = create_train_env(
            args=args,
            trial_number=trial.number,
        )

        eval_env = create_eval_env(
            args=args,
            trial_number=trial.number,
        )

        # EvalCallback frequencies count callback calls. With vectorized
        # environments, each call corresponds to n_envs environment steps.
        eval_freq = max(
            args.eval_freq // args.n_envs,
            1,
        )

        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=0,
        )

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs={
                "features_extractor_class": GridCNNExtractor,
                "features_extractor_kwargs": {
                    "features_dim": sampled_params["features_dim"],
                },
            },
            learning_rate=sampled_params["learning_rate"],
            n_steps=sampled_params["n_steps"],
            batch_size=sampled_params["batch_size"],
            n_epochs=sampled_params["n_epochs"],
            gamma=sampled_params["gamma"],
            gae_lambda=sampled_params["gae_lambda"],
            clip_range=sampled_params["clip_range"],
            ent_coef=sampled_params["ent_coef"],
            vf_coef=sampled_params["vf_coef"],
            max_grad_norm=sampled_params["max_grad_norm"],
            verbose=0,
            seed=args.seed + trial.number,
            device=args.device,
        )

        model.learn(
            total_timesteps=args.trial_timesteps,
            callback=eval_callback,
            progress_bar=False,
        )

        if eval_callback.is_pruned:
            raise optuna.TrialPruned()

        score = float(eval_callback.best_mean_reward)

        if not np.isfinite(score):
            raise optuna.TrialPruned(
                f"Trial produced a non-finite score: {score}"
            )

        # Useful metadata stored alongside the trial.
        trial.set_user_attr(
            "last_mean_reward",
            float(eval_callback.last_mean_reward),
        )
        trial.set_user_attr(
            "best_mean_reward",
            score,
        )
        trial.set_user_attr(
            "rollout_size",
            sampled_params["n_steps"] * args.n_envs,
        )

        return score

    except (AssertionError, ValueError, RuntimeError) as error:
        # Record the failure so it can be inspected in Optuna.
        trial.set_user_attr(
            "failure_reason",
            f"{type(error).__name__}: {error}",
        )

        # Treat numerical or incompatible configurations as pruned rather
        # than terminating the complete study.
        raise optuna.TrialPruned(str(error)) from error

    finally:
        if train_env is not None:
            train_env.close()

        if eval_env is not None:
            eval_env.close()

        del model
        gc.collect()

def save_study_results(
    study: optuna.Study,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_result = {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }

    with open(
        output_dir / "best_hyperparameters.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_result, file, indent=2)

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "params",
            "state",
            "user_attrs",
            "datetime_start",
            "datetime_complete",
            "duration",
        )
    )

    trials_df.to_csv(
        output_dir / "trials.csv",
        index=False,
    )


def retrain_best_model(
    study: optuna.Study,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """
    Retrain the best Optuna configuration using the full training budget.

    This is intentionally separate from the objective. Models from short
    optimization trials should generally not be treated as final models.
    """

    params = study.best_params.copy()

    features_dim = int(params.pop("features_dim"))

    train_env = create_train_env(
        args=args,
        trial_number=study.best_trial.number + 1_000_000,
    )

    eval_env = create_eval_env(
        args=args,
        trial_number=study.best_trial.number + 1_000_000,
    )

    model_dir = output_dir / "best_model"
    log_dir = output_dir / "best_model_logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs={
            "features_extractor_class": GridCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
            },
        },
        **params,
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
    )

    model.learn(
        total_timesteps=args.final_timesteps,
        callback=eval_callback,
        tb_log_name="ppo_best",
        progress_bar=True,
    )

    model.save(model_dir / "final_model")

    train_env.close()
    eval_env.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--unknown_targets", type=int, default=0)
    parser.add_argument("--grid_rows", type=int, default=10)
    parser.add_argument("--grid_cols", type=int, default=10)
    parser.add_argument("--episode_steps", type=int, default=200)
    parser.add_argument("--search_reward", type=str, default="R2")
    parser.add_argument("--measurement_noise_std", type=float, default=1.0)

    parser.add_argument(
        "--zero_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--simple_obs", action="store_true")

    # Optimization
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--trial_timesteps", type=int, default=500_000)
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)

    parser.add_argument("--n_startup_trials", type=int, default=8)
    parser.add_argument("--n_warmup_evaluations", type=int, default=3)

    # Parallel environments inside each PPO trial
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--subproc", action="store_true")

    # Reproducibility and execution
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', 'cuda', etc.",
    )

    # Study persistence
    parser.add_argument(
        "--study_name",
        type=str,
        default="ppo_search_R2",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_ppo_search.db",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="runs/optuna",
    )

    # Final model
    parser.add_argument(
        "--retrain_best",
        action="store_true",
    )
    parser.add_argument(
        "--final_timesteps",
        type=int,
        default=1_000_000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.unknown_targets < 0:
        raise ValueError(
            "Search-mode optimization requires unknown_targets >= 0. "
            "Otherwise there are no targets to discover."
        )

    output_dir = Path(args.outdir) / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "optimization_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(vars(args), file, indent=2)

    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
    )

    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_evaluations,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    print("\nOptimization complete")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best mean evaluation reward: {study.best_value:.6f}")
    print("Best parameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    save_study_results(
        study=study,
        output_dir=output_dir,
    )

    print(
        f"\nSaved study results to: {output_dir}"
    )

    if args.retrain_best:
        retrain_best_model(
            study=study,
            args=args,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()