from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import optuna
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from agents.train.train_ppo import train_ppo, make_train_env, make_eval_env


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
STUDY_NAME = "ppo_ssa_optuna"
STORAGE = "sqlite:///optuna_ppo_ssa.db"

N_TRIALS = 40
TOTAL_TIMESTEPS = 200_000   # cheap screening budget
N_EVAL_EPISODES = 10
SEED = 42

SAVE_ROOT = Path("runs/optuna_ppo")
DELETE_TRIAL_DIRS = False   # set True if you do not want to keep all model files

# Fixed training options we're NOT tuning here
N_EPOCHS = 5
NORMALIZE_OBS = True
NORMALIZE_REWARD = False
USE_TENSORBOARD = False

N_ENVS = 1

# Candidate batch sizes
BATCH_SIZE_CANDIDATES = [32, 64, 128, 256, 512, 1024, 2048]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def make_trial_dir(trial: optuna.Trial) -> Path:
    trial_dir = SAVE_ROOT / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def build_eval_vec_env(
    env_fn: Callable[[], gym.Env],
    vec_norm: Optional[VecNormalize],
    gamma: float,
) -> DummyVecEnv | VecNormalize:
    """
    Build an evaluation VecEnv consistent with the train_ppo setup.
    If VecNormalize was used during training, reuse its observation statistics.
    """
    eval_env = DummyVecEnv([env_fn])
    eval_env = VecMonitor(eval_env)

    if vec_norm is not None:
        wrapped_eval_env = VecNormalize(
            eval_env,
            norm_obs=vec_norm.norm_obs,
            norm_reward=False,   # keep reward unnormalized during evaluation
            clip_obs=10.0,
            gamma=gamma,
            training=False,
        )
        wrapped_eval_env.obs_rms = vec_norm.obs_rms
        if hasattr(vec_norm, "ret_rms") and vec_norm.ret_rms is not None:
            wrapped_eval_env.ret_rms = vec_norm.ret_rms
        return wrapped_eval_env

    return eval_env


def objective(trial: optuna.Trial) -> float:
    trial_dir = make_trial_dir(trial)

    # -------------------------------
    # Hyperparameter sampling
    # -------------------------------
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    if (n_steps * N_ENVS) % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.2, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    sampled_params = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "n_epochs": N_EPOCHS,
        "normalize_obs": NORMALIZE_OBS,
        "normalize_reward": NORMALIZE_REWARD,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
    }

    with open(trial_dir / "sampled_params.json", "w") as f:
        json.dump(sampled_params, f, indent=2)

    try:
        # -------------------------------
        # Train
        # -------------------------------
        model, vec_norm = train_ppo(
            env_fn=make_train_env,
            eval_env_fn=make_eval_env,
            total_timesteps=TOTAL_TIMESTEPS,
            save_dir=str(trial_dir),
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=N_EPOCHS,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_obs=NORMALIZE_OBS,
            normalize_reward=NORMALIZE_REWARD,
            seed=SEED,
            use_tensorboard=USE_TENSORBOARD,
        )

        # -------------------------------
        # Evaluate
        # -------------------------------

        # Evaluate final observation (covariance traces) over several episodes
        final_traces = []

        for i in range(N_EVAL_EPISODES):
            env = make_eval_env()
            obs, _ = env.reset(seed=SEED + i)
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            final_traces.append(float(np.sum(obs)))  # obs shape (N,)
            env.close()

        mean_final_trace = float(np.mean(final_traces))

        trial.set_user_attr("mean_final_trace", mean_final_trace)

        if DELETE_TRIAL_DIRS:
            shutil.rmtree(trial_dir, ignore_errors=True)

        return mean_final_trace  # study direction="minimize"

    except Exception as e:
        # Penalize failed trials heavily instead of crashing the whole study
        trial.set_user_attr("failed", True)
        trial.set_user_attr("exception", repr(e))

        with open(trial_dir / "error.txt", "w") as f:
            f.write(repr(e))

        if DELETE_TRIAL_DIRS:
            shutil.rmtree(trial_dir, ignore_errors=True)

        return -1e12


def main() -> None:
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=SEED, multivariate=True)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    print("\nStudy complete.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }

    with open(SAVE_ROOT / "best_result.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()