import argparse
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv


def make_env(env_fn: Callable[[], gym.Env]) -> Callable[[], gym.Env]:
    """
    Wrap an env constructor for SB3 vectorized training.
    """
    def _init() -> gym.Env:
        return env_fn()

    return _init

"""
First optuna run gave best hyperparameters:
  learning_rate: 3.1208880418345674e-05
  n_steps: 1024
  batch_size: 128
  gamma: 0.9890380853416586
  gae_lambda: 0.9628935945750411
  clip_range: 0.1092203613478283
  ent_coef: 0.02796305121985432
  vf_coef: 0.6432933066304029
  max_grad_norm: 0.5652321957863001
"""
def train_ppo(
    env_fn: Callable[[], gym.Env],
    total_timesteps: int,
    save_dir: str,
    timesteps: int = 50,
    eval_env_fn: Optional[Callable[[], gym.Env]] = None,
    learning_rate: float = 3.1208880418345674e-05,
    n_steps: int = 1024, # PPO default is 2048
    batch_size: int = 128, # def 256
    n_epochs: int = 5, # def 10
    gamma: float = 0.9890380853416586,
    gae_lambda: float = 0.9628935945750411,
    clip_range: float = 0.1092203613478283,
    ent_coef: float = 0.02796305121985432,
    vf_coef: float = 0.6432933066304029,
    max_grad_norm: float = 0.5652321957863001,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    seed: int = 0,
    use_tensorboard: bool = False,
) -> tuple[PPO, Optional[VecNormalize]]:
    """
    Train a PPO agent for the SSA environment.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(env_fn)])
    train_env = VecMonitor(train_env)

    vec_norm = None
    if normalize_obs or normalize_reward:
        train_env = VecNormalize(
            train_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
            gamma=gamma,
        )
        vec_norm = train_env

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256],
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(save_path / "tb") if use_tensorboard else None,
    )

    callbacks = []

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(save_path / "checkpoints"),
        name_prefix="ppo_ssa",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_cb)

    if eval_env_fn is not None:
        eval_env = DummyVecEnv([make_env(eval_env_fn)])
        eval_env = VecMonitor(eval_env)

        if vec_norm is not None:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=normalize_obs,
                norm_reward=False,
                clip_obs=10.0,
                gamma=gamma,
                training=False,
            )
            eval_env.obs_rms = vec_norm.obs_rms
            if normalize_reward:
                eval_env.ret_rms = vec_norm.ret_rms

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path / "best_model"),
            log_path=str(save_path / "eval_logs"),
            eval_freq=10_000,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model.save(str(save_path / "final_model"))

    if vec_norm is not None:
        vec_norm.save(str(save_path / "vecnormalize.pkl"))

    return model, vec_norm


def make_train_env(num_targets: int = 2) -> gym.Env:
    return MinimalSSAEnv(targets_known=True, num_targets=num_targets)


def make_eval_env(num_targets: int = 2) -> gym.Env:
    return MinimalSSAEnv(targets_known=True, num_targets=num_targets)

def main(num_targets: int, total_timesteps: int, reward_type: str = "R1", learning_rate: float = 3e-3, save_dir: str = None, train_zero_noise: bool = False, timesteps: int = 50):
    env_fn=lambda: MinimalSSAEnv(targets_known=True, num_targets=num_targets, reward_type=reward_type, zero_noise=train_zero_noise, timesteps=timesteps)
    eval_env_fn=lambda: MinimalSSAEnv(targets_known=True, num_targets=num_targets, reward_type=reward_type, timesteps=timesteps)
    if save_dir is None:
        save_dir = f"runs/ppo_baseline_{num_targets}targets_{reward_type}_{train_zero_noise}"

    train_ppo(
        env_fn=env_fn,
        eval_env_fn=eval_env_fn,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        save_dir=save_dir,
        normalize_obs=True,
        normalize_reward=False,
        seed=42,
        use_tensorboard=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_targets", type=int, default=2)
    parser.add_argument("--reward_type", type=str, default="R1")
    args = parser.parse_args()
    main(num_targets=args.num_targets, total_timesteps=250_000, reward_type=args.reward_type, train_zero_noise=True)
    main(num_targets=args.num_targets, total_timesteps=250_000, reward_type=args.reward_type, train_zero_noise=False)