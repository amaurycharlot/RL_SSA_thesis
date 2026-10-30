"""Shared SSA factories and existing Windows-safe spawn/cleanup helpers.

Importing creates no environments/workers/models and changes no thread settings.
Optional normalization statistics belong exclusively to each policy's handle.
"""
from contextlib import contextmanager
import multiprocessing as mp
import os

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
from .config import validate_config
from .types import EnvConfig, SeedPlan, TrainingEnvHandle, MAX_SEED, _close_resources
from .wrappers import LinearScalarizationWrapper, _fixed_pair


# Params: seed; returns: None.
def _seed(seed):
    if type(seed) is not int:
        raise TypeError("seed must be a Python integer, not a bool")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must fit the uint32 range")


def make_raw_env(env_config: EnvConfig, seed=None):
    """Create unscalarized combined env; reset only when a seed is supplied."""
    if not isinstance(env_config, EnvConfig):
        raise TypeError("env_config must be EnvConfig")
    if seed is not None:
        _seed(seed)
    env = MinimalSSAEnv(
        known_targets=env_config.known_targets, unknown_targets=env_config.unknown_targets,
        grid_shape=env_config.grid_shape, timesteps=env_config.episode_steps,
        measurement_noise_std=env_config.measurement_noise_std, zero_noise=env_config.zero_noise,
        reward_type=env_config.reward_type, search_reward=env_config.search_reward,
        dwell_steps=env_config.dwell_steps, slew_alpha=env_config.slew_alpha,
        uncertainty_norm_scale=env_config.uncertainty_norm_scale,
        mode="combined", simple_obs=False,
    )
    try:
        if seed is not None:
            env.reset(seed=seed)
        return env
    except BaseException:
        env.close()
        raise


def make_scalar_env(env_config, weights, seed, scales=(1., 1.)):
    """Shared scalar factory; close the raw child if wrapping fails."""
    weights = _fixed_pair(weights, "weights")
    scales = _fixed_pair(scales, "scales", positive=True)
    env = make_raw_env(env_config, seed)
    try:
        return LinearScalarizationWrapper(env, weights, scales)
    except BaseException:
        env.close()
        raise


def set_worker_threads(threads):
    """Apply the existing Torch limit in child processes only."""
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("worker_threads must be a positive Python integer")
        if mp.current_process().name != "MainProcess":
            import torch
            torch.set_num_threads(threads)


@contextmanager
def worker_thread_environment(threads):
    """Only spawned workers inherit these limits; restore the parent's settings."""
    if type(threads) is not int or threads < 1:
        raise ValueError("worker_threads must be a positive Python integer")
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = str(threads)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TrainingSubprocVecEnv(SubprocVecEnv):
    """Existing startup/broken-worker cleanup, shared by both trainers."""

    # Params: self, env_fns, start_method; returns: None.
    def __init__(self, env_fns, start_method="spawn"):
        try:
            super().__init__(env_fns, start_method=start_method)
        except BaseException:
            self._release_workers()
            raise

    # Params: self; returns: None.
    def _release_workers(self):
        processes = getattr(self, "processes", [])
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
        for remote in (*getattr(self, "remotes", ()),
                       *getattr(self, "work_remotes", ())):
            remote.close()
        self.closed = True

    # Params: self; returns: (observations, rewards, dones, infos).
    def step_wait(self):
        try:
            return super().step_wait()
        except BaseException:
            # Some earlier replies may already have been consumed. SB3 leaves
            # waiting=True on recv failure, so close() would read them a second
            # time and block indefinitely on an otherwise healthy worker.
            self.waiting = False
            self._release_workers()
            raise

    # Params: self; returns: None.
    def close(self):
        try:
            super().close()
        except (EOFError, OSError):
            pass
        finally:
            self._release_workers()


def make_env_factory(config, weights, rank, base_seed, worker_threads=None):
    """Return a cloudpickle-compatible rank factory without allocating an env."""
    validate_config(config, require_frozen_objectives=False)
    if type(rank) is not int or not 0 <= rank < config.ppo.n_envs:
        raise ValueError("rank must identify a configured worker")
    _seed(base_seed)
    _seed(base_seed + config.ppo.n_envs - 1)
    if worker_threads is not None and (type(worker_threads) is not int or worker_threads < 1):
        raise ValueError("worker_threads must be a positive Python integer")
    fixed_weights = _fixed_pair(weights, "weights")
    fixed_scales = _fixed_pair(config.objective_scales, "scales", positive=True)

    # Params: none; returns: environment.
    def _init():
        set_worker_threads(worker_threads)
        return make_scalar_env(config.env, fixed_weights, base_seed + rank, fixed_scales)

    return _init


def _make_dummy_vec_env(factories):
    """Track children before DummyVecEnv construction finishes."""
    children = []

    # Params: factory; returns: _init.
    def tracked(factory):
        # Params: none; returns: environment.
        def _init():
            env = factory()
            children.append(env)
            return env
        return _init

    try:
        return DummyVecEnv([tracked(factory) for factory in factories])
    except BaseException:
        _close_resources(tuple(children))
        raise


def make_train_vec_env(config, weights, seed_plan):
    """Own raw->scalar->VecEnv->one VecMonitor->optional policy VecNormalize."""
    validate_config(config, require_frozen_objectives=False)
    if not isinstance(seed_plan, SeedPlan):
        raise TypeError("seed_plan must be SeedPlan")
    if (not 0 <= seed_plan.run_index < len(config.run_seeds)
            or config.run_seeds[seed_plan.run_index] != seed_plan.run_seed
            or len(seed_plan.worker_seeds) != config.ppo.n_envs
            or len(seed_plan.eval_seeds) != config.eval_episodes
            or len(seed_plan.final_seeds) != config.final_episodes):
        raise ValueError("seed_plan dimensions/replicate must match the experiment")
    limits = config.execution.worker_threads if config.execution.subproc else None
    factories = [make_env_factory(config, weights, rank, seed_plan.base_worker_seed, limits)
                 for rank in range(config.ppo.n_envs)]
    handle = TrainingEnvHandle()
    try:
        if config.execution.subproc:
            with worker_thread_environment(config.execution.worker_threads):
                handle.raw_vec_env = TrainingSubprocVecEnv(factories, start_method="spawn")
            handle.workers = tuple(handle.raw_vec_env.processes)
        else:
            handle.raw_vec_env = _make_dummy_vec_env(factories)
        handle.vec_env = VecMonitor(handle.raw_vec_env)
        if config.normalization.norm_obs or config.normalization.norm_reward:
            # Monitor/raw workers are already registered before construction.
            # A new normalizer is created for each handle; statistics are never
            # global or shared between preference policies.
            settings = config.normalization
            handle.vec_normalize = VecNormalize(
                handle.vec_env, training=True,
                norm_obs=settings.norm_obs, norm_reward=settings.norm_reward,
                clip_obs=settings.clip_obs, clip_reward=settings.clip_reward,
                gamma=settings.gamma, epsilon=settings.epsilon, norm_obs_keys=None,
            )
            handle.vec_env = handle.vec_normalize
        handle.vec_env.seed(seed_plan.base_worker_seed)
        return handle
    except BaseException:
        handle.close()
        raise
