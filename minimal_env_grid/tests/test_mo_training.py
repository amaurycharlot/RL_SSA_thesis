"""Block 25 local diagnostics checks; no workers or PPO updates required."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo.types import ExperimentConfig, LoggingConfig
from agents.mo.reporting import LocalLogger, make_logger
from agents.mo.callbacks import EpisodeDiagnosticsCallback


class JournalTests(unittest.TestCase):
    # Params: self; returns: None.
    def test_roundtrip_reopen_nulls_and_owned_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig()
            payload = {'raw': np.array([1., np.nan]), 'zero': 0., 'reason': None}
            with make_logger(config, tmp) as logger:
                logger.record('test', {'seed_index': 0}, payload)
                payload['raw'][0] = 99
                logger.flush()
            logger.close()
            with make_logger(config, tmp) as reopened:
                reopened.record('second', {}, {})
            files = sorted((Path(tmp) / 'logs/events').glob('*.json'))
            self.assertEqual(len(files), 2)
            first = json.loads(files[0].read_text())
            self.assertEqual(first['payload'], {'raw': [1., None], 'zero': 0., 'reason': None})
            self.assertEqual(first['identity'], {'seed_index': 0})
            self.assertEqual(json.loads(files[1].read_text())['event_index'], 1)
            with self.assertRaises(RuntimeError):
                logger.record('late', {}, {})

    # Params: self; returns: None.
    def test_failed_write_preserves_records_and_retries_same_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = LocalLogger(tmp)
            logger.record('first', {}, {})
            original = (Path(tmp) / '000000000000.json').read_bytes()
            with patch('agents.mo.storage.os.replace', side_effect=OSError('disk')):
                with self.assertRaises(OSError):
                    logger.record('failed', {}, {})
            self.assertEqual((Path(tmp) / '000000000000.json').read_bytes(), original)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            logger.record('retry', {}, {})
            self.assertTrue((Path(tmp) / '000000000001.json').exists())

    # Params: self; returns: None.
    def test_rejects_missing_external_owner_settings_before_allocating(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('offline', 'online'):
                config = replace(ExperimentConfig(), logging=LoggingConfig(mode=mode))
                with self.assertRaisesRegex(ValueError, 'project and entity'):
                    make_logger(config, Path(tmp) / mode)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    # Params: self; returns: None.
    def test_passive_import_and_empty_logger(self):
        code = ('import sys; from agents.mo.reporting import LocalLogger, make_logger; '
                'from agents.mo.trainer import build_model, learn_one_update; '
                'assert not any(m in sys.modules for m in '
                '["torch", "stable_baselines3", "gymnasium", "wandb", "matplotlib"])')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=PROJECT_ROOT, check=True)
        with tempfile.TemporaryDirectory() as tmp:
            with LocalLogger(tmp):
                pass
            self.assertEqual(list(Path(tmp).iterdir()), [])


class MemoryLogger:
    # Params: self; returns: None.
    def __init__(self):
        self.events = []

    # Params: self, *args; returns: None.
    def record(self, *args):
        self.events.append(deepcopy(args))


# Params: raw, scalar, coverage, elapsed, detections; returns: dict.
def info(raw=(1., 2.), scalar=1.5, coverage=.25, elapsed=1, detections=0):
    return {'vector_reward_raw': np.asarray(raw, np.float32),
            'scalarized_reward': scalar, 'coverage_mean': coverage,
            'elapsed_steps': elapsed, 'new_detections': detections,
            'num_measurements': 2}


class EpisodeDiagnosticsTests(unittest.TestCase):
    # Params: self, count; returns: (callback, logger).
    def callback(self, count=2):
        logger = MemoryLogger()
        callback = EpisodeDiagnosticsCallback(logger, 3, 4, count)
        callback.init_callback(SimpleNamespace(num_timesteps=0, get_env=lambda: SimpleNamespace(num_envs=count)))
        return callback, logger

    # Params: self, cb, infos, dones, rewards; returns: None.
    def step(self, cb, infos, dones, rewards=None):
        cb.num_timesteps += cb.n_envs
        cb.model.num_timesteps = cb.num_timesteps
        cb.update_locals({'infos': infos, 'dones': np.asarray(dones),
                          'rewards': np.asarray(rewards if rewards is not None else [9.] * cb.n_envs)})
        self.assertIs(cb._on_step(), True)

    # Params: self; returns: None.
    def test_terminal_steps_workers_and_partial_chunks(self):
        cb, logger = self.callback()
        cb.on_training_start({}, {})
        self.step(cb, [info(), info(raw=(10., 20.))], [False, True])
        cb.on_training_end()
        cb.on_training_start({}, {})
        cb.on_training_start({}, {})
        self.step(cb, [info(raw=(3., 4.), scalar=3.5, coverage=.75, elapsed=4, detections=1), info()], [True, False])
        self.assertEqual(len(logger.events), 2)
        event, identity, payload = logger.events[1]
        self.assertEqual(event, 'training_episode')
        self.assertEqual(identity, {'seed_index': 3, 'weight_index': 4, 'worker_index': 0,
                                   'episode_index': 0, 'collection_step': 4})
        np.testing.assert_array_equal(payload['raw_reward_sum'], [4., 6.])
        self.assertEqual(payload['scalarized_reward_sum'], 5.)
        self.assertEqual(payload['learner_reward_sum'], 18.)
        self.assertEqual(payload['decisions'], 2)
        self.assertEqual(payload['elapsed_timesteps'], 5)
        self.assertEqual(payload['coverage_mean'], .5)
        self.assertEqual(payload['coverage_final'], .75)
        self.assertEqual(payload['new_detections'], 1)
        self.assertEqual(payload['measurement_count'], 4)
        self.assertEqual(payload['diagnostic_kind'], 'undiscounted_training_episode')
        self.step(cb, [info(raw=(0., 0.), scalar=0.), info()], [True, True])
        self.assertEqual(logger.events[2][1]['episode_index'], 1)
        np.testing.assert_array_equal(logger.events[2][2]['raw_reward_sum'], [0., 0.])
        self.assertEqual(logger.events[3][2]['decisions'], 2)

    # Params: self; returns: None.
    def test_inputs_rng_and_normalizer_unchanged_with_real_auto_reset(self):
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
        class Grid(gym.Env):
            observation_space = gym.spaces.Box(0., 100., (2, 2, 3), dtype=np.float32)
            action_space = gym.spaces.MultiDiscrete([2, 2])
            # Params: self, seed, options; returns: (array, dict).
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros((2, 2, 3), np.float32), {'coverage_mean': 999.}
            # Params: self, action; returns: (array, float, bool, bool, info result).
            def step(self, action):
                return np.full((2, 2, 3), 5., np.float32), 1.5, False, True, info(coverage=.75)
        env = VecNormalize(VecMonitor(DummyVecEnv([Grid])), norm_obs=True, norm_reward=True)
        self.addCleanup(env.close)
        cb, logger = self.callback(1)
        cb.init_callback(SimpleNamespace(num_timesteps=0, get_env=lambda: env))
        env.reset()
        observations, rewards, dones, infos = env.step(np.array([[0, 0]]))
        cb.on_training_start({}, {})
        data = {'infos': infos, 'dones': dones, 'rewards': rewards, 'new_obs': observations}
        before_data = pickle.dumps(data)
        before_env = pickle.dumps((env.obs_rms, env.ret_rms, env.returns, env.training, env.norm_reward))
        before_rng = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
        cb.update_locals(data)
        self.assertTrue(cb._on_step())
        self.assertEqual(pickle.dumps(data), before_data)
        self.assertEqual(pickle.dumps((env.obs_rms, env.ret_rms, env.returns, env.training, env.norm_reward)), before_env)
        self.assertEqual(random.getstate(), before_rng[0])
        self.assertEqual(pickle.dumps(np.random.get_state()), before_rng[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng[2]))
        payload = logger.events[0][2]
        self.assertEqual(payload['coverage_final'], .75)
        self.assertEqual(payload['scalarized_reward_sum'], 1.5)
        self.assertEqual(payload['learner_reward_sum'], float(rewards[0]))
        self.assertNotEqual(payload['learner_reward_sum'], 1.5)
        np.testing.assert_array_equal(observations, 0.)
        self.assertIn('terminal_observation', infos[0])

    # Params: self; returns: None.
    def test_malformed_step_does_not_partially_accumulate(self):
        cb, logger = self.callback()
        broken = info(raw=(np.nan, 1.))
        with self.assertRaises(ValueError):
            self.step(cb, [info(), broken], [False, True])
        self.assertEqual(logger.events, [])
        self.step(cb, [info(), info()], [True, True])
        self.assertEqual(logger.events[0][2]['decisions'], 1)

    # Params: self; returns: None.
    def test_invalid_identity_and_worker_counts(self):
        for args in ((-1, 0, 1), (0, True, 1), (0, 0, 0)):
            with self.assertRaises(ValueError):
                EpisodeDiagnosticsCallback(MemoryLogger(), *args)
        cb, _ = self.callback()
        cb.init_callback(SimpleNamespace(num_timesteps=0, get_env=lambda: SimpleNamespace(num_envs=1)))
        with self.assertRaises(ValueError):
            cb.on_training_start({}, {})


class BoundedPPOTrainingTests(unittest.TestCase):
    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    # Params: cls; returns: None.
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    # Params: self, **ppo_overrides; returns: replace result.
    def config(self, **ppo_overrides):
        from agents.mo.types import PPOConfig, EnvConfig, ExecutionConfig, NormalizationConfig
        ppo = replace(PPOConfig(), n_envs=2, n_steps=4, batch_size=4,
                      n_epochs=2, verbose=0, device='cpu')
        ppo = replace(ppo, **ppo_overrides)
        return replace(ExperimentConfig(), run_seeds=(0,), n_weights=2,
                       env=replace(EnvConfig(), known_targets=1, unknown_targets=1,
                                   grid_shape=(3, 3), episode_steps=7),
                       execution=ExecutionConfig(subproc=False),
                       reference_scaled=(-10., -10.), reference_status='frozen',
                       requested_timesteps_per_policy=24,
                       normalization=NormalizationConfig(gamma=ppo.gamma), ppo=ppo)

    # Params: self, config; returns: handle.
    def synthetic_handle(self, config):
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
        from agents.mo.types import TrainingEnvHandle
        class Grid(gym.Env):
            observation_space = gym.spaces.Box(0., 100., (3, 3, 3), dtype=np.float32)
            action_space = gym.spaces.MultiDiscrete([3, 3])
            # Params: self; returns: None.
            def __init__(self):
                self.resets = 0
                self.actions = []
                self.step_count = 0
            # Params: self; returns: float32 observation array.
            def observation(self):
                return self.np_random.uniform(0., 1., (3, 3, 3)).astype(np.float32)
            # Params: self, seed, options; returns: (self.observation result, dict).
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.resets += 1
                self.step_count = 0
                return self.observation(), {}
            # Params: self, action; returns: (self.observation result, scalar, bool, bool, info result).
            def step(self, action):
                self.step_count += 1
                self.actions.append(np.asarray(action).copy())
                raw = (float(action[0] + self.np_random.uniform()), float(action[1]))
                scalar = float(np.mean(raw))
                return self.observation(), scalar, self.step_count == 7, False, info(raw=raw, scalar=scalar)
        raw = DummyVecEnv([Grid for _ in range(config.ppo.n_envs)])
        monitor = VecMonitor(raw)
        handle = TrainingEnvHandle(vec_env=monitor, raw_vec_env=raw)
        if config.normalization.norm_obs or config.normalization.norm_reward:
            n = config.normalization
            handle.vec_normalize = VecNormalize(monitor, norm_obs=n.norm_obs,
                                                norm_reward=n.norm_reward, gamma=config.ppo.gamma,
                                                epsilon=n.epsilon, clip_obs=n.clip_obs,
                                                clip_reward=n.clip_reward)
            handle.vec_env = handle.vec_normalize
        self.addCleanup(handle.close)
        return handle

    # Params: self, config, handle; returns: (build_model result, handle).
    def model(self, config, handle=None):
        from agents.mo.config import make_seed_plan
        from agents.mo.trainer import build_model
        if handle is None:
            handle = self.synthetic_handle(config)
        return build_model(config, handle, make_seed_plan(config, 0)), handle

    # Params: self, first, second; returns: None.
    def assert_nested_equal(self, first, second):
        if isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        elif isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        else:
            self.assertEqual(first, second)

    # Params: self; returns: None.
    def test_all_configured_parameters_and_learner_seed_are_forwarded(self):
        from dataclasses import fields
        from agents.mo.config import make_seed_plan
        from agents.mo.trainer import build_model
        from agents.RL.grid_cnn_extractor import GridCNNExtractor
        config = self.config(learning_rate=.0007, gamma=.91, gae_lambda=.8,
                             clip_range=.15, clip_range_vf=.25, ent_coef=.03,
                             vf_coef=.6, max_grad_norm=.7, normalize_advantage=False,
                             sde_sample_freq=5, target_kl=.2, stats_window_size=17)
        handle = self.synthetic_handle(config)
        seed_plan = make_seed_plan(config, 0)
        with patch('stable_baselines3.PPO') as constructor:
            build_model(config, handle, seed_plan)
        kwargs = constructor.call_args.kwargs
        for field in fields(config.ppo):
            if field.name != 'n_envs':
                self.assertEqual(kwargs[field.name], getattr(config.ppo, field.name))
        self.assertIs(kwargs['env'], handle.vec_env)
        self.assertEqual(kwargs['policy'], 'MlpPolicy')
        self.assertEqual(kwargs['seed'], seed_plan.learner_seed)
        self.assertIsNone(kwargs['tensorboard_log'])
        self.assertEqual(kwargs['policy_kwargs'], {'features_extractor_class': GridCNNExtractor,
                                                 'features_extractor_kwargs': {'features_dim': 128}})

    # Params: self; returns: None.
    def test_two_real_ssa_preferences_have_independent_networks_and_optimizers(self):
        from agents.mo.config import make_seed_plan
        from agents.mo.envs import make_train_vec_env
        from agents.mo.trainer import build_model, learn_one_update
        from agents.RL.grid_cnn_extractor import GridCNNExtractor
        config = self.config()
        seed = make_seed_plan(config, 0)
        handles = [make_train_vec_env(config, weight, seed) for weight in ((1., 0.), (0., 1.))]
        for handle in handles:
            self.addCleanup(handle.close)
        first, second = [build_model(config, handle, seed) for handle in handles]
        self.assertIsInstance(first.policy.features_extractor, GridCNNExtractor)
        self.assertEqual(first.policy.features_extractor.features_dim, 128)
        self.assertIsNot(first.policy.optimizer, second.policy.optimizer)
        for a, b in zip(first.policy.parameters(), second.policy.parameters()):
            self.assertIsNot(a, b)
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        untouched = deepcopy(second.policy.state_dict())
        optimizer = first.policy.optimizer
        self.assertEqual(learn_one_update(first, config), config.ppo.rollout_size)
        self.assertIs(first.policy.optimizer, optimizer)
        self.assertTrue(optimizer.state)
        self.assertFalse(second.policy.optimizer.state)
        self.assert_nested_equal(second.policy.state_dict(), untouched)
        self.assertEqual(second.num_timesteps, 0)

    # Params: self; returns: None.
    def test_three_chunks_match_continuous_cpu_training_exactly(self):
        from agents.mo.trainer import learn_one_update
        config = self.config()
        chunked, handle = self.model(config)
        logger = MemoryLogger()
        diagnostics = EpisodeDiagnosticsCallback(logger, 0, 0, config.ppo.n_envs)
        env = chunked.get_env()
        optimizer = chunked.policy.optimizer
        collected_epochs = []
        diagnostics._on_rollout_end = lambda: collected_epochs.append(chunked._n_updates)
        for update in range(1, 4):
            self.assertEqual(learn_one_update(chunked, config, diagnostics), update * config.ppo.rollout_size)
            self.assertEqual(chunked._n_updates, update * config.ppo.n_epochs)
            self.assertIs(chunked.get_env(), env)
            self.assertIs(chunked.policy.optimizer, optimizer)
        self.assertEqual(collected_epochs, [0, 2, 4])
        self.assertEqual([env.resets for env in handle.raw_vec_env.envs], [2, 2])
        continuous, continuous_handle = self.model(config)
        continuous_logger = MemoryLogger()
        continuous.learn(total_timesteps=3 * config.ppo.rollout_size,
                         reset_num_timesteps=False, log_interval=None,
                         callback=EpisodeDiagnosticsCallback(continuous_logger, 0, 0, config.ppo.n_envs))
        self.assert_nested_equal(chunked.policy.state_dict(), continuous.policy.state_dict())
        self.assert_nested_equal(optimizer.state_dict(), continuous.policy.optimizer.state_dict())
        self.assert_nested_equal(chunked._last_obs, continuous._last_obs)
        self.assert_nested_equal(chunked._last_episode_starts, continuous._last_episode_starts)
        for a, b in zip(handle.raw_vec_env.envs, continuous_handle.raw_vec_env.envs):
            self.assertEqual(a.resets, b.resets)
            self.assert_nested_equal(a.actions, b.actions)
        self.assert_nested_equal(logger.events, continuous_logger.events)
        self.assertEqual(logger.events[0][2]['decisions'], 7)
        self.assertEqual(chunked._n_updates, continuous._n_updates)
        self.assertEqual(chunked.num_timesteps, continuous.num_timesteps)

    # Params: self; returns: None.
    def test_enabled_normalization_and_partial_episodes_survive_chunks(self):
        from agents.mo.types import NormalizationConfig
        from agents.mo.trainer import learn_one_update
        config = replace(self.config(), normalization=NormalizationConfig(norm_obs=True, norm_reward=True, gamma=.99))
        model, handle = self.model(config)
        normalizer = handle.vec_normalize
        logger = MemoryLogger()
        callback = EpisodeDiagnosticsCallback(logger, 0, 0, config.ppo.n_envs)
        learn_one_update(model, config, callback)
        first_count = normalizer.obs_rms.count
        self.assertEqual([e.resets for e in handle.raw_vec_env.envs], [1, 1])
        self.assertEqual(logger.events, [])
        learn_one_update(model, config, callback)
        self.assertIs(model.get_env(), normalizer)
        self.assertGreater(normalizer.obs_rms.count, first_count)
        self.assertEqual([e.resets for e in handle.raw_vec_env.envs], [2, 2])
        self.assertEqual([event[2]['decisions'] for event in logger.events], [7, 7])
        self.assertIsNot(handle.vec_normalize.obs_rms, self.synthetic_handle(config).vec_normalize.obs_rms)

    # Params: self; returns: None.
    def test_construction_rejects_bad_inputs_before_ppo_or_training(self):
        from agents.mo.config import make_seed_plan
        from agents.mo.trainer import build_model
        config = self.config()
        handle = self.synthetic_handle(config)
        seed = make_seed_plan(config, 0)
        with patch('stable_baselines3.PPO') as constructor:
            with self.assertRaises(ValueError):
                build_model(replace(config, reference_status='unselected', reference_scaled=None), handle, seed)
            with self.assertRaises(TypeError):
                build_model(config, handle, None)
            with self.assertRaises(ValueError):
                build_model(config, handle, replace(seed, run_seed=1))
            with self.assertRaises(ValueError):
                build_model(replace(config, ppo=replace(config.ppo, n_envs=1)), handle, seed)
            handle.close()
            with self.assertRaises(ValueError):
                build_model(config, handle, seed)
            constructor.assert_not_called()

    # Params: self; returns: None.
    def test_update_boundaries_and_failed_collection_are_not_success(self):
        from agents.mo.trainer import learn_one_update
        config = self.config()
        model, _ = self.model(config)
        with self.assertRaises(TypeError):
            learn_one_update(model, config, lambda *_: True)
        model.num_timesteps = 1
        with self.assertRaises(ValueError):
            learn_one_update(model, config)
        model.num_timesteps = 0
        with self.assertRaises(ValueError):
            learn_one_update(model, replace(config, ppo=replace(config.ppo, n_steps=8)))
        with patch.object(model, 'learn', return_value=model):
            with self.assertRaisesRegex(RuntimeError, 'full rollout'):
                learn_one_update(model, config)
        callback = EpisodeDiagnosticsCallback(MemoryLogger(), 0, 0, config.ppo.n_envs)
        callback._on_step = lambda: False
        with self.assertRaisesRegex(RuntimeError, 'full rollout'):
            learn_one_update(model, config, callback)
        self.assertEqual(model._n_updates, 0)
        self.assertEqual(model.num_timesteps, config.ppo.n_envs)


class PolicyOrchestrationTests(unittest.TestCase):
    setUpClass = classmethod(BoundedPPOTrainingTests.setUpClass.__func__)
    tearDownClass = classmethod(BoundedPPOTrainingTests.tearDownClass.__func__)

    # Params: self; returns: None.
    def setUp(self):
        from agents.mo import storage
        from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
        from agents.mo.types import EnvConfig, PPOConfig, ExecutionConfig
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ExperimentConfig(output_dir=Path(self.temporary.name) / 'experiment',
            run_seeds=(7,), n_weights=2, eval_episodes=1, final_episodes=2,
            reference_scaled=(-10., -10.), reference_status='frozen',
            requested_timesteps_per_policy=5, eval_interval=1, checkpoint_interval=3,
            env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3), episode_steps=2, slew_alpha=0.),
            ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0),
            execution=ExecutionConfig(subproc=False))
        self.weights = make_weights(2)
        self.seed = make_seed_plan(self.config, 0)
        self.budget = make_budget_plan(self.config)
        self.root = storage.create_experiment_dir(self.config, self.weights, (self.seed,), self.budget)
        self.results = storage.load_evaluation_records(self.root, self.config)
        self.logger = LocalLogger(self.root / 'logs/test-events')
        self.addCleanup(self.logger.close)
        self.policy = self.root / 'models/seed_0/w_0'

    # Params: self, **overrides; returns: train_one_policy result.
    def run_policy(self, **overrides):
        from agents.mo.trainer import train_one_policy
        args = dict(config=self.config, experiment_dir=self.root, seed_index=0, weight_index=0,
                    weights=self.weights, seed_plan=self.seed, budget_plan=self.budget,
                    result_store=self.results, logger=self.logger)
        args.update(overrides)
        return train_one_policy(**args)

    # Params: self; returns: (handles, factory).
    def track_handles(self):
        from agents.mo.envs import make_train_vec_env
        handles = []
        # Params: *args, **kwargs; returns: handle.
        def factory(*args, **kwargs):
            handle = make_train_vec_env(*args, **kwargs)
            handles.append(handle)
            return handle
        return handles, factory

    # Params: self; returns: list.
    def events(self):
        return [json.loads(path.read_text()) for path in sorted(self.logger.directory.glob('*.json'))]

    # Params: self; returns: None.
    def test_post_update_events_final_once_and_exact_checkpoint_snapshot(self):
        from agents.mo import trainer, storage, evaluation
        timeline, models, initial = [], [], []
        build, save, evaluate = trainer.build_model, storage.save_policy_bundle, evaluation.evaluate_policy
        # Params: *args; returns: model.
        def build_model(*args):
            model = build(*args); models.append(model)
            initial.extend(p.detach().clone() for p in model.policy.parameters())
            return model
        # Params: model, *args; returns: save result.
        def checkpoint(model, *args):
            timeline.append(('save', model.num_timesteps, model._n_updates))
            self.assertTrue(any(not torch.equal(p, before) for p, before in zip(model.policy.parameters(), initial)))
            return save(model, *args)
        # Params: model, *args, **kwargs; returns: EvaluationResult.
        def evaluate_policy(model, *args, **kwargs):
            timeline.append(('evaluate', model.num_timesteps, model._n_updates))
            self.assertEqual(args[1], self.seed.eval_seeds)
            return evaluate(model, *args, **kwargs)
        handles, factory = self.track_handles()
        with patch.object(trainer, 'build_model', side_effect=build_model), \
             patch('agents.mo.envs.make_train_vec_env', side_effect=factory), \
             patch.object(storage, 'save_policy_bundle', side_effect=checkpoint), \
             patch.object(evaluation, 'evaluate_policy', side_effect=evaluate_policy):
            summary = self.run_policy()
        self.assertEqual(timeline, [('save', 2, 1), ('evaluate', 2, 1), ('save', 4, 2),
                                    ('evaluate', 4, 2), ('save', 6, 3), ('evaluate', 6, 3)])
        self.assertEqual(len(models), 1); self.assertEqual(len(handles), 1)
        self.assertTrue(handles[0].closed); self.assertFalse(self.logger._closed)
        self.assertTrue(summary.complete)
        self.assertEqual((summary.requested_transitions, summary.effective_transitions, summary.actual_transitions), (5, 6, 6))
        self.assertEqual(summary.committed_evaluation_ids, ('seed_0/w_0/step_2', 'seed_0/w_0/step_4', 'seed_0/w_0/step_6'))
        self.assertTrue(self.results.validity[0, :, 0].all())
        self.assertFalse(self.results.validity[0, :, 1].any())
        self.assertTrue(np.isnan(self.results.hypervolume).all())
        self.assertEqual((self.policy / 'final/final_model.zip').read_bytes(),
                         (self.policy / 'checkpoints/step_6/model.zip').read_bytes())
        self.assertTrue((self.policy / 'COMPLETE.json').exists())
        self.assertEqual([e['identity']['actual_step'] for e in self.events() if e['event'] == 'policy_evaluation'], [2, 4, 6])

    # Params: self; returns: None.
    def test_valid_complete_policy_skips_all_model_worker_and_evaluation_allocation(self):
        self.run_policy()
        with patch('agents.mo.envs.make_train_vec_env') as create, \
             patch('agents.mo.trainer.build_model') as build, \
             patch('agents.mo.evaluation.evaluate_policy') as evaluate:
            summary = self.run_policy()
        create.assert_not_called(); build.assert_not_called(); evaluate.assert_not_called()
        self.assertTrue(summary.complete)
        self.assertTrue(summary.resume_metadata['skipped_complete'])
        self.assertEqual(summary.resume_metadata['additional_training_transitions'], 0)

    # Params: self; returns: None.
    def test_evaluation_failure_leaves_recoverable_checkpoint_and_missing_slot(self):
        handles, factory = self.track_handles()
        with patch('agents.mo.envs.make_train_vec_env', side_effect=factory), \
             patch('agents.mo.evaluation.evaluate_policy', side_effect=RuntimeError('evaluation failed')):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'): self.run_policy()
        from agents.mo import storage
        bundle, _, _ = storage._committed_bundle(self.policy / 'checkpoints/step_2')
        self.assertEqual(bundle.actual_step, 2)
        self.assertFalse(self.results.validity.any())
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        self.assertTrue(handles[0].closed)
        self.assertEqual(self.events()[-1]['payload']['phase'], 'evaluation')

    # Params: self; returns: None.
    def test_resume_repairs_missing_current_evaluation_before_learning(self):
        with patch('agents.mo.evaluation.evaluate_policy', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError): self.run_policy()
        from agents.mo import trainer, evaluation
        learn, evaluate = trainer.learn_one_update, evaluation.evaluate_policy
        timeline = []
        # Params: model, *args; returns: learn result.
        def learn_update(model, *args):
            timeline.append(('learn', model.num_timesteps)); return learn(model, *args)
        # Params: model, *args, **kwargs; returns: EvaluationResult.
        def evaluate_policy(model, *args, **kwargs):
            timeline.append(('evaluate', model.num_timesteps)); return evaluate(model, *args, **kwargs)
        with patch.object(trainer, 'learn_one_update', side_effect=learn_update), \
             patch.object(evaluation, 'evaluate_policy', side_effect=evaluate_policy):
            summary = self.run_policy()
        self.assertEqual(timeline, [('evaluate', 2), ('learn', 2), ('evaluate', 4), ('learn', 4), ('evaluate', 6)])
        self.assertEqual(summary.resume_metadata['restored_step'], 2)
        self.assertEqual(summary.resume_metadata['additional_training_transitions'], 4)
        self.assertTrue(summary.complete)

    # Params: self; returns: None.
    def test_repeated_recovery_evaluation_failure_cannot_advance_learning(self):
        from agents.mo.trainer import learn_one_update
        for attempt in range(2):
            with patch('agents.mo.evaluation.evaluate_policy', side_effect=RuntimeError('still interrupted')), \
                 patch('agents.mo.trainer.learn_one_update', wraps=learn_one_update) as learn:
                with self.assertRaises(RuntimeError): self.run_policy()
            self.assertEqual(learn.call_count, 1 if attempt == 0 else 0)
        self.assertFalse((self.policy / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_checkpoint_failure_prevents_evaluation_and_completion(self):
        handles, factory = self.track_handles()
        with patch('agents.mo.envs.make_train_vec_env', side_effect=factory), \
             patch('agents.mo.storage.save_policy_bundle', side_effect=OSError('checkpoint failed')), \
             patch('agents.mo.evaluation.evaluate_policy') as evaluate:
            with self.assertRaisesRegex(OSError, 'checkpoint failed'): self.run_policy()
        evaluate.assert_not_called()
        self.assertTrue(handles[0].closed)
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        self.assertEqual(self.events()[-1]['payload']['phase'], 'checkpoint')

    # Params: self; returns: None.
    def test_evaluation_write_failure_does_not_fill_a_slot(self):
        with patch('agents.mo.storage.save_evaluation_record', side_effect=OSError('record failed')):
            with self.assertRaisesRegex(OSError, 'record failed'): self.run_policy()
        self.assertTrue((self.policy / 'checkpoints/step_2/COMMITTED').exists())
        self.assertFalse(self.results.validity.any())
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        self.assertEqual(self.events()[-1]['payload']['phase'], 'evaluation_write')

    # Params: self; returns: None.
    def test_final_budget_checkpoint_is_promoted_without_another_update(self):
        with patch('agents.mo.trainer._final_bundle_from_checkpoint', side_effect=OSError('final publication failed')):
            with self.assertRaises(OSError): self.run_policy()
        checkpoint = self.policy / 'checkpoints/step_6'
        self.assertTrue((checkpoint / 'COMMITTED').exists())
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        from agents.mo import storage
        loaded, load = [], storage.load_policy_bundle
        # Params: *args; returns: bundle.
        def load_bundle(*args):
            bundle = load(*args); loaded.append(bundle)
            self.assertFalse(bundle.model.rollout_buffer.full)
            return bundle
        with patch.object(storage, 'load_policy_bundle', side_effect=load_bundle), \
             patch('agents.mo.trainer.learn_one_update') as learn, \
             patch('agents.mo.evaluation.evaluate_policy') as evaluate:
            summary = self.run_policy()
        learn.assert_not_called(); evaluate.assert_not_called()
        self.assertTrue(loaded[0].closed)
        self.assertEqual(summary.resume_metadata['additional_training_transitions'], 0)
        self.assertEqual((self.policy / 'final/final_model.zip').read_bytes(), (checkpoint / 'model.zip').read_bytes())

    # Params: self; returns: None.
    def test_final_budget_missing_evaluation_is_repaired_without_training(self):
        from agents.mo import evaluation
        evaluate = evaluation.evaluate_policy
        # Params: model, *args, **kwargs; returns: evaluate result.
        def fail_last(model, *args, **kwargs):
            if model.num_timesteps == 6: raise RuntimeError('last evaluation failed')
            return evaluate(model, *args, **kwargs)
        with patch.object(evaluation, 'evaluate_policy', side_effect=fail_last):
            with self.assertRaises(RuntimeError): self.run_policy()
        self.assertTrue(self.results.validity[0, :2, 0].all())
        self.assertFalse(self.results.validity[0, 2, 0])
        with patch('agents.mo.trainer.learn_one_update') as learn:
            summary = self.run_policy()
        learn.assert_not_called()
        self.assertTrue(summary.complete)
        self.assertTrue(self.results.validity[0, 2, 0])

    # Params: self; returns: None.
    def test_normalized_evaluation_preserves_live_statistics_and_final_snapshot(self):
        from agents.mo import storage, evaluation
        from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
        from agents.mo.types import NormalizationConfig
        self.config = replace(self.config, output_dir=Path(self.temporary.name) / 'normalized',
                              normalization=NormalizationConfig(norm_obs=True, norm_reward=True))
        self.seed = make_seed_plan(self.config, 0); self.budget = make_budget_plan(self.config)
        self.root = storage.create_experiment_dir(self.config, self.weights, (self.seed,), self.budget)
        self.results = storage.load_evaluation_records(self.root, self.config)
        self.policy = self.root / 'models/seed_0/w_0'
        evaluate = evaluation.evaluate_policy
        counts = []
        # Params: model, *args, **kwargs; returns: EvaluationResult.
        def evaluate_policy(model, *args, **kwargs):
            normalizer = model.get_vec_normalize_env()
            before = pickle.dumps(normalizer)
            self.assertEqual(kwargs['frozen_normalizer'].count, normalizer.obs_rms.count)
            result = evaluate(model, *args, **kwargs)
            self.assertEqual(pickle.dumps(normalizer), before)
            counts.append(normalizer.obs_rms.count)
            return result
        with patch.object(evaluation, 'evaluate_policy', side_effect=evaluate_policy):
            self.run_policy()
        np.testing.assert_allclose(counts, np.array([3., 5., 7.]) + 1e-4, rtol=0, atol=1e-12)
        self.assertEqual((self.policy / 'final/vecnormalize.pkl').read_bytes(),
                         (self.policy / 'checkpoints/step_6/vecnormalize.pkl').read_bytes())

    # Params: self; returns: None.
    def test_model_construction_failure_closes_the_owned_training_handle(self):
        handles, factory = self.track_handles()
        with patch('agents.mo.envs.make_train_vec_env', side_effect=factory), \
             patch('agents.mo.trainer.build_model', side_effect=RuntimeError('model failed')):
            with self.assertRaisesRegex(RuntimeError, 'model failed'): self.run_policy()
        self.assertTrue(handles[0].closed)
        self.assertFalse(self.logger._closed)
        self.assertFalse((self.policy / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_cleanup_failure_does_not_mark_complete(self):
        handles, factory = self.track_handles()
        # Params: *args; returns: handle.
        def create(*args):
            handle = factory(*args)
            close = handle.close
            # Params: none; returns: never (raises).
            def fail_close():
                close(); raise RuntimeError('close failed')
            handle.close = fail_close
            return handle
        with patch('agents.mo.envs.make_train_vec_env', side_effect=create):
            with self.assertRaisesRegex(RuntimeError, 'close failed'): self.run_policy()
        self.assertTrue(handles[0].closed)
        self.assertTrue((self.policy / 'final/COMMITTED').exists())
        self.assertFalse((self.policy / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_logger_flush_failure_preserves_artifacts_but_not_completion(self):
        with patch.object(self.logger, 'flush', side_effect=OSError('flush failed')):
            with self.assertRaisesRegex(OSError, 'flush failed'): self.run_policy()
        self.assertTrue((self.policy / 'final/COMMITTED').exists())
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        with patch('agents.mo.trainer.learn_one_update') as learn:
            self.assertTrue(self.run_policy().complete)
        learn.assert_not_called()

    # Params: self; returns: None.
    def test_invalid_inputs_fail_before_allocating_workers(self):
        from agents.mo.types import ExperimentResults
        bad_table = ExperimentResults.empty(1, [2, 6], 2, 1)
        for kwargs in ({'weights': self.weights[::-1].copy()}, {'weight_index': True},
                {'seed_plan': replace(self.seed, run_seed=8)}, {'budget_plan': None},
                {'result_store': bad_table}, {'logger': object()}):
            with patch('agents.mo.envs.make_train_vec_env') as create:
                with self.assertRaises((TypeError, ValueError)): self.run_policy(**kwargs)
                create.assert_not_called()

    # Params: self; returns: None.
    def test_learning_failure_closes_owned_resources_before_any_event_save(self):
        handles, factory = self.track_handles()
        with patch('agents.mo.envs.make_train_vec_env', side_effect=factory), \
             patch('agents.mo.trainer.learn_one_update', side_effect=RuntimeError('learning failed')), \
             patch('agents.mo.storage.save_policy_bundle') as save:
            with self.assertRaisesRegex(RuntimeError, 'learning failed'): self.run_policy()
        save.assert_not_called()
        self.assertTrue(handles[0].closed)
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        self.assertEqual(self.events()[-1]['payload']['phase'], 'training')

    # Params: self; returns: None.
    def test_partial_final_copy_is_not_a_final_bundle_or_completion(self):
        import shutil
        original = shutil.copyfile
        # Params: source, destination, *args, **kwargs; returns: original result.
        def copy(source, destination, *args, **kwargs):
            if Path(destination).name == 'final_model.zip':
                Path(destination).write_bytes(b'partial final snapshot')
                raise OSError('copy interrupted')
            return original(source, destination, *args, **kwargs)
        with patch.object(shutil, 'copyfile', side_effect=copy):
            with self.assertRaisesRegex(OSError, 'copy interrupted'): self.run_policy()
        self.assertFalse((self.policy / 'final').exists())
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        self.assertFalse(list(self.policy.glob('.final.*.tmp')))
        self.assertTrue((self.policy / 'checkpoints/step_6/COMMITTED').exists())

    # Params: self; returns: None.
    def test_completion_marker_failure_can_retry_without_learning(self):
        with patch('agents.mo.storage.mark_policy_complete', side_effect=OSError('marker failed')):
            with self.assertRaisesRegex(OSError, 'marker failed'): self.run_policy()
        self.assertFalse((self.policy / 'COMPLETE.json').exists())
        with patch('agents.mo.trainer.learn_one_update') as learn:
            self.assertTrue(self.run_policy().complete)
        learn.assert_not_called()

    # Params: self; returns: None.
    def test_resume_refreshes_a_stale_result_table_after_future_quarantine(self):
        from agents.mo import evaluation, storage
        evaluate = evaluation.evaluate_policy
        # Params: model, *args, **kwargs; returns: evaluate result.
        def fail_last(model, *args, **kwargs):
            if model.num_timesteps == 6: raise RuntimeError('last evaluation interrupted')
            return evaluate(model, *args, **kwargs)
        with patch.object(evaluation, 'evaluate_policy', side_effect=fail_last):
            with self.assertRaises(RuntimeError): self.run_policy()
        early = self.root / 'results/evaluations/seed_0/w_0/step_2.json'
        future = early.with_name('step_4.json')
        early_bytes, future_bytes = early.read_bytes(), future.read_bytes()
        old_identity = self.results.identities[(0, 1, 0)]
        for step in (4, 6):
            (self.policy / f'checkpoints/step_{step}/model.zip').write_bytes(b'injected corruption')
        summary = self.run_policy()
        self.assertTrue(summary.complete)
        self.assertEqual(summary.resume_metadata['restored_step'], 2)
        self.assertEqual(early.read_bytes(), early_bytes)
        self.assertNotEqual(self.results.identities[(0, 1, 0)].bundle_id, old_identity.bundle_id)
        archived = list((self.root / 'results/superseded').rglob('step_4.json'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), future_bytes)
        self.assertTrue(self.results.validity[0, :, 0].all())

    # Params: self; returns: None.
    def test_eight_environments_use_the_rounded_transition_budget_once(self):
        from agents.mo import storage
        from agents.mo.config import make_seed_plan, make_budget_plan
        self.config = replace(self.config, output_dir=Path(self.temporary.name) / 'eight',
                              ppo=replace(self.config.ppo, n_envs=8, batch_size=8))
        self.seed = make_seed_plan(self.config, 0); self.budget = make_budget_plan(self.config)
        self.root = storage.create_experiment_dir(self.config, self.weights, (self.seed,), self.budget)
        self.results = storage.load_evaluation_records(self.root, self.config)
        handles, factory = self.track_handles()
        with patch('agents.mo.envs.make_train_vec_env', side_effect=factory):
            summary = self.run_policy()
        self.assertEqual((summary.requested_transitions, summary.effective_transitions, summary.actual_transitions), (5, 16, 16))
        self.assertEqual(handles[0].vec_env.num_envs, 8)
        self.assertTrue(handles[0].closed)
        self.assertEqual(summary.committed_evaluation_ids, ('seed_0/w_0/step_16',))
        self.assertEqual(len([e for e in self.events() if e['event'] == 'policy_evaluation']), 1)


class ExperimentOrchestrationTests(unittest.TestCase):
    setUpClass = classmethod(BoundedPPOTrainingTests.setUpClass.__func__)
    tearDownClass = classmethod(BoundedPPOTrainingTests.tearDownClass.__func__)

    # Params: self; returns: None.
    def setUp(self):
        PolicyOrchestrationTests.setUp(self)
        self.config = replace(self.config, run_seeds=(7, 9), requested_timesteps_per_policy=2,
                              output_dir=Path(self.temporary.name) / 'outer')

    # Params: self, **kwargs; returns: run_experiment result.
    def run_experiment(self, **kwargs):
        import warnings
        from agents.mo.trainer import run_experiment
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return run_experiment(self.config, **kwargs)

    # Params: self; returns: None.
    def test_real_grid_slots_independent_models_sequential_cleanup_and_cost(self):
        from agents.mo import trainer, storage, reporting, envs
        models, initial, handles, loggers = [], [], [], []
        build, factory, make_logger = trainer.build_model, envs.make_train_vec_env, reporting.make_logger
        # Params: *args; returns: handle.
        def track_handle(*args):
            self.assertTrue(all(h.closed for h in handles))
            handle = factory(*args); handles.append(handle); return handle
        # Params: *args; returns: model.
        def track_build(*args):
            model = build(*args); models.append(model)
            initial.append([p.detach().clone() for p in model.policy.parameters()])
            self.assertEqual(model.num_timesteps, 0)
            self.assertEqual(len(model.policy.optimizer.state), 0)
            return model
        # Params: *args; returns: logger.
        def track_logger(*args):
            logger = make_logger(*args); loggers.append(logger); return logger
        with patch.object(trainer, 'build_model', side_effect=track_build), \
             patch.object(envs, 'make_train_vec_env', side_effect=track_handle), \
             patch.object(reporting, 'make_logger', side_effect=track_logger):
            summary = self.run_experiment()
        self.assertEqual((summary.completed_policies, summary.total_policies, summary.effective_total_transitions), (4, 4, 8))
        self.assertEqual([(p.identity.run_index, p.identity.weight_index) for p in summary.policies],
                         [(0, 0), (0, 1), (1, 0), (1, 1)])
        results = storage.load_evaluation_records(summary.experiment_path, self.config)
        self.assertEqual(results.means_raw.shape, (2, 1, 2, 2))
        self.assertTrue(results.validity.all())
        self.assertTrue(np.isfinite(results.hypervolume).all())
        self.assertEqual(len({id(m.policy.optimizer) for m in models}), 4)
        for a, b in ((0, 1), (2, 3)):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(initial[a], initial[b])))
            self.assertTrue(any(not torch.equal(x, y) for x, y in zip(models[a].policy.parameters(), initial[b])))
        self.assertTrue(all(h.closed for h in handles))
        self.assertTrue(loggers[0]._closed)
        events = [json.loads(p.read_text()) for p in loggers[0].directory.glob('*.json')]
        end = next(e['payload'] for e in events if e['event'] == 'experiment_training_complete')
        self.assertEqual(end['additional_training_transitions'], 8)
        self.assertEqual(end['intermediate_evaluation_episodes'], 4)
        self.assertEqual(len(list((summary.experiment_path / 'results/final').rglob('step_2.json'))), 4)
        self.assertTrue((summary.experiment_path / 'results/final_results.npz').is_file())

    # Params: self; returns: None.
    def test_resume_uses_saved_plans_and_validated_completion_without_workers(self):
        from agents.mo import config, trainer, envs
        first = self.run_experiment()
        with patch.object(config, 'make_weights', side_effect=AssertionError('regenerated weights')), \
             patch.object(config, 'make_seed_plan', side_effect=AssertionError('regenerated seeds')), \
             patch.object(config, 'make_budget_plan', wraps=config.make_budget_plan), \
             patch.object(trainer, 'build_model') as build, patch.object(envs, 'make_train_vec_env') as workers:
            second = self.run_experiment(resume=True)
        self.assertEqual(first.experiment_fingerprint, second.experiment_fingerprint)
        self.assertTrue(all(p.resume_metadata['skipped_complete'] for p in second.policies))
        self.assertEqual(sum(p.resume_metadata['additional_training_transitions'] for p in second.policies), 0)
        build.assert_not_called(); workers.assert_not_called()

    # Params: self; returns: None.
    def test_failure_keeps_incomplete_hv_missing_closes_logger_and_can_resume(self):
        from agents.mo import trainer, storage, reporting
        train, factory = trainer.train_one_policy, reporting.make_logger
        loggers = []
        # Params: *args; returns: value.
        def logger(*args):
            value = factory(*args); loggers.append(value); return value
        # Params: *args; returns: train result.
        def fail_second(*args):
            if args[3] == 1: raise RuntimeError('injected second-policy failure')
            return train(*args)
        with patch.object(trainer, 'train_one_policy', side_effect=fail_second), \
             patch.object(reporting, 'make_logger', side_effect=logger):
            with self.assertRaisesRegex(RuntimeError, 'second-policy'): self.run_experiment()
        self.assertTrue(loggers[0]._closed)
        results = storage.load_evaluation_records(self.config.output_dir, self.config)
        self.assertTrue(results.validity[0, :, 0].all())
        self.assertFalse(results.validity[:, :, 1].any())
        self.assertTrue(np.isnan(results.hypervolume).all())
        self.assertEqual(self.run_experiment(resume=True).completed_policies, 4)

    # Params: self; returns: None.
    def test_saved_reversed_weight_order_is_immutable_and_keeps_slot_identity(self):
        from agents.mo import trainer, config
        original = config.make_weights(2)[::-1].copy()
        train, seen = trainer.train_one_policy, []
        # Params: *args; returns: train result.
        def track(*args):
            weights = args[4]
            with self.assertRaises(ValueError): weights.setflags(write=True)
            seen.append(tuple(weights[args[3]].tolist()))
            return train(*args)
        with patch.object(config, 'make_weights', return_value=original), \
             patch.object(trainer, 'train_one_policy', side_effect=track):
            summary = self.run_experiment()
        self.assertEqual(seen, [tuple(w.tolist()) for w in original] * 2)
        resumed = self.run_experiment(resume=True)
        self.assertEqual([p.identity.weight for p in summary.policies], [p.identity.weight for p in resumed.policies])

    # Params: self; returns: None.
    def test_nonempty_destination_and_changed_science_refused_before_workers(self):
        from agents.mo import storage, envs
        self.run_experiment()
        with patch.object(envs, 'make_train_vec_env') as workers:
            with self.assertRaises(storage.StorageError): self.run_experiment()
            self.config = replace(self.config, objective_scales=(2., 1.), scale_status='frozen')
            with self.assertRaises(storage.StorageError): self.run_experiment(resume=True)
        workers.assert_not_called()

    # Params: self; returns: None.
    def test_explicit_destination_override_and_warning_counts(self):
        import warnings
        from agents.mo.trainer import run_experiment
        destination = Path(self.temporary.name) / 'explicit'
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', RuntimeWarning)
            result = run_experiment(self.config, experiment_dir=destination)
        self.assertEqual(result.experiment_path, destination.resolve())
        self.assertFalse(self.config.output_dir.exists())
        self.assertTrue(any('2 seeds * 2 weights * 2 = 8 training transitions plus 4' in str(w.message)
                            for w in captured))

    # Params: self; returns: None.
    def test_unfrozen_config_rejected_without_creating_output(self):
        from agents.mo.trainer import run_experiment
        with self.assertRaises(ValueError): run_experiment(replace(self.config, reference_status='unselected', reference_scaled=None))
        self.assertFalse(self.config.output_dir.exists())

    # Params: self; returns: None.
    def test_final_testing_failure_closes_logger_and_resume_reuses_trained_policies(self):
        from agents.mo import evaluation, reporting, trainer
        loggers, make_logger = [], reporting.make_logger
        # Params: *args; returns: logger.
        def track_logger(*args):
            logger = make_logger(*args); loggers.append(logger); return logger
        with patch.object(reporting, 'make_logger', side_effect=track_logger), \
             patch.object(evaluation, 'evaluate_final_policy_set', side_effect=RuntimeError('final phase failed')):
            with self.assertRaisesRegex(RuntimeError, 'final phase failed'): self.run_experiment()
        self.assertTrue(loggers[0]._closed)
        self.assertEqual(len(list(self.config.output_dir.glob('models/seed_*/w_*/COMPLETE.json'))), 4)
        events = [json.loads(p.read_text()) for p in loggers[0].directory.glob('*.json')]
        self.assertTrue(any(e['event'] == 'experiment_failure' for e in events))
        self.assertFalse(any(e['event'] == 'experiment_final_testing_complete' for e in events))
        with patch.object(trainer, 'learn_one_update') as learn:
            result = self.run_experiment(resume=True)
        learn.assert_not_called()
        self.assertTrue(all(p.resume_metadata['skipped_complete'] for p in result.policies))
        self.assertTrue((result.experiment_path / 'results/final_results.npz').is_file())


class LegacySSACompatibility(unittest.TestCase):
    """Lock the scientific example and the established training CLI defaults."""

    # Params: self; returns: None.
    def test_legacy_cli_defaults_and_execution_aliases(self):
        import inspect
        from stable_baselines3 import PPO
        from agents.train.train_ppo_combined import build_parser
        expected = dict(known_targets=10, unknown_targets=10, grid_rows=10, grid_cols=10,
            episode_steps=200, total_timesteps=500000, reward='R6', search_reward='R2',
            measurement_noise_std=1., zero_noise=False, track_weight=.85, search_weight=.15,
            seed=1, n_envs=8, n_steps=512, batch_size=256, learning_rate=.0003,
            gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.01, vf_coef=.5,
            max_grad_norm=.5, eval_freq=20000, n_eval_episodes=20, outdir='runs',
            run_name=None, check_env=False, subproc=True, worker_threads=1)
        parser = build_parser()
        self.assertEqual(vars(parser.parse_args([])), expected)
        for flag in ('--no-subproc', '--sequential'):
            self.assertEqual(vars(parser.parse_args([flag])), dict(expected, subproc=False))
        self.assertEqual(inspect.signature(PPO.__init__).parameters['n_epochs'].default, 10)

    # Params: self; returns: None.
    def test_production_scientific_defaults(self):
        from agents.mo.types import ExperimentConfig, EnvConfig, PPOConfig
        # Schema defaults are independent of the owner's selected run grid and reference.
        config = ExperimentConfig()
        self.assertEqual(config.env, EnvConfig())
        self.assertEqual(config.ppo, PPOConfig())
        self.assertEqual(config.run_seeds, tuple(range(10)))
        self.assertEqual(config.n_weights, 11)
        self.assertEqual(config.requested_timesteps_per_policy, 500000)
        self.assertEqual(config.objective_scales, (1., 1.))
        self.assertIsNone(config.reference_scaled)
        self.assertEqual(config.reference_status, 'unselected')
        self.assertFalse(config.normalization.norm_obs)
        self.assertFalse(config.normalization.norm_reward)


@unittest.skipUnless(__import__('os').environ.get('MO_RUN_STANDARD_SSA_SMOKE') == '1',
                     'Opt in with MO_RUN_STANDARD_SSA_SMOKE=1: real 8x512 standard SSA update')
class StandardSSACompatibility(unittest.TestCase):
    """Block 38: one production-sized update, not a seed/preference sweep."""

    # Params: self; returns: None.
    def test_one_standard_update_frozen_evaluation_reload_and_worker_cleanup(self):
        import ast
        import hashlib
        import inspect
        import multiprocessing as mp
        import os
        import time
        from agents.mo import config as planning, envs, evaluation, storage, trainer
        from agents.mo.types import EnvConfig, PPOConfig
        from test_mo_wrappers import numerical_thread_snapshot

        started = time.monotonic()
        evidence = {'kind': 'block38_standard_ssa_compatibility', 'status': 'running'}
        evidence_path = os.environ.get('MO_BLOCK38_EVIDENCE')
        # Params: phase, **values; returns: None.
        def progress(phase, **values):
            evidence.update(values)
            evidence['phase'] = phase
            evidence['wall_seconds'] = time.monotonic() - started
            if evidence_path:
                storage.atomic_write_json(Path(evidence_path), evidence)
            print('Block 38: ' + phase, flush=True)

        # Params: none; returns: dict.
        def source_hashes():
            paths = []
            for directory in ('agents', 'ssa_env', 'scripts'):
                paths.extend((PROJECT_ROOT / directory).rglob('*.py'))
            paths.extend((PROJECT_ROOT / 'configs').glob('*.json'))
            return {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(paths)}

        production = PROJECT_ROOT / 'configs/ppo_mo_r6_r2.json'
        before_sources = source_hashes()
        original_bytes = production.read_bytes()
        baseline = planning.load_config(production)
        self.assertIsNone(baseline.reference_scaled)
        self.assertEqual(baseline.env, EnvConfig())
        self.assertEqual(baseline.ppo, PPOConfig())
        self.assertEqual((baseline.ppo.n_envs, baseline.ppo.n_steps, baseline.ppo.batch_size), (8, 512, 256))
        self.assertTrue(baseline.execution.subproc)
        self.assertEqual(baseline.execution.worker_threads, 1)
        self.assertFalse(baseline.normalization.norm_obs)
        self.assertFalse(baseline.normalization.norm_reward)
        self.assertEqual(baseline.objective_scales, (1., 1.))
        parent_threads = torch.get_num_threads()
        parent_pools = numerical_thread_snapshot()
        thread_keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
        parent_environment = {key: os.environ.get(key) for key in thread_keys}
        initial_children = {p.pid for p in mp.active_children()}
        self.assertFalse(initial_children, 'Run this heavyweight test in an isolated process')

        # Assert that the real constructor explicitly passes every resolved PPO
        # setting rather than depending on potentially changing SDK defaults.
        constructor = ast.parse(inspect.getsource(trainer.build_model))
        ppo_call = next(n for n in ast.walk(constructor) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == 'PPO')
        explicit = {k.arg for k in ppo_call.keywords}
        self.assertTrue((set(baseline.ppo.to_dict()) - {'n_envs'}) <= explicit)
        self.assertTrue({'policy_kwargs', 'seed', 'tensorboard_log'} <= explicit)

        handles, models, initial_parameters, live_evaluations = [], [], [], []
        observations = []
        original_factory = envs.make_env_factory
        original_handle = envs.make_train_vec_env
        original_build = trainer.build_model
        original_learn = trainer.learn_one_update
        original_evaluate = evaluation.evaluate_policy
        # Params: *args, **kwargs; returns: construct.
        def probe_factory(*args, **kwargs):
            base = original_factory(*args, **kwargs)
            # Params: none; returns: env.
            def construct():
                import multiprocessing as worker_mp
                import os as worker_os
                import torch as worker_torch
                from test_mo_wrappers import numerical_thread_snapshot as worker_pools
                env = base()
                env._block38_runtime = {'pid': worker_os.getpid(),
                    'start_method': worker_mp.get_start_method(), 'torch_threads': worker_torch.get_num_threads(),
                    'numerical_threads': worker_pools(), 'seed': env.unwrapped.np_random_seed}
                counters = {'decisions': 0, 'elapsed_timesteps': 0, 'episodes': 0}
                env._block38_counters = counters
                step = env.step
                # Params: action; returns: (observation, reward, terminated, truncated, info).
                def tracked_step(action):
                    result = step(action)
                    counters['decisions'] += 1
                    counters['elapsed_timesteps'] += int(result[4]['elapsed_steps'])
                    counters['episodes'] += int(result[2] or result[3])
                    return result
                env.step = tracked_step
                return env
            return construct
        # Params: *args, **kwargs; returns: handle.
        def make_handle(*args, **kwargs):
            progress('starting_eight_workers')
            handle = original_handle(*args, **kwargs)
            handles.append(handle)
            self.addCleanup(handle.close)
            self.assertIsNone(handle.vec_normalize)
            self.assertEqual(len(handle.workers), 8)
            runtime = handle.vec_env.get_attr('_block38_runtime')
            self.assertEqual(len({item['pid'] for item in runtime}), 8)
            for rank, item in enumerate(runtime):
                self.assertEqual(item['start_method'], 'spawn')
                self.assertEqual(item['torch_threads'], 1)
                self.assertEqual(item['seed'], args[2].worker_seeds[rank])
                self.assertTrue(item['numerical_threads'])
                self.assertTrue(all(count == 1 for _, count in item['numerical_threads']))
            for name, expected in (('known_targets', 10), ('unknown_targets', 10), ('max_steps', 200),
                ('dwell_steps', 1), ('slew_alpha', 1.), ('uncertainty_norm_scale', 100.),
                ('reward_type', 'R6'), ('search_reward', 'R2'), ('zero_noise', False), ('simple_obs', False)):
                self.assertEqual(handle.vec_env.get_attr(name), [expected] * 8)
            progress('workers_started', workers=runtime)
            return handle
        # Params: *args, **kwargs; returns: model.
        def build(*args, **kwargs):
            model = original_build(*args, **kwargs)
            models.append(model)
            initial_parameters.extend(p.detach().clone() for p in model.policy.parameters())
            self.assertEqual(model.num_timesteps, 0)
            self.assertEqual(model._n_updates, 0)
            self.assertEqual(model.policy.features_extractor.features_dim, 128)
            self.assertEqual(model.policy.features_extractor.__class__.__name__, 'GridCNNExtractor')
            for name, expected in args[0].ppo.to_dict().items():
                if name in ('device', 'stats_window_size'):
                    continue
                value = getattr(model, name)
                if name in ('clip_range', 'clip_range_vf') and value is not None:
                    value = value(1.)
                self.assertEqual(value, expected, name)
            progress('learner_ready', learner_device=str(model.device),
                     ppo_settings=args[0].ppo.to_dict(), parent_torch_threads=parent_threads,
                     parent_numerical_threads=parent_pools)
            return model
        # Params: *args, **kwargs; returns: step.
        def learn(*args, **kwargs):
            progress('collecting_4096_transitions_and_optimizing')
            step = original_learn(*args, **kwargs)
            model = args[0]
            self.assertEqual(step, 4096)
            self.assertEqual(model._n_updates, 10)
            self.assertTrue(model.rollout_buffer.full)
            self.assertTrue(any(not torch.equal(p, before) for p, before in zip(model.policy.parameters(), initial_parameters)))
            optimizer_steps = [float(state['step']) for state in model.policy.optimizer.state.values()]
            self.assertTrue(optimizer_steps)
            self.assertEqual(set(optimizer_steps), {160.})
            counters = handles[0].vec_env.get_attr('_block38_counters')
            self.assertEqual([s['decisions'] for s in counters], [512] * 8)
            self.assertTrue(all(s['elapsed_timesteps'] >= 512 and s['episodes'] >= 2 for s in counters))
            observations.extend(counters)
            progress('one_update_complete', actual_step=step, optimization_epochs=model._n_updates,
                     optimizer_steps=160, worker_counters=counters)
            return step
        # Params: *args, **kwargs; returns: EvaluationResult.
        def evaluate(*args, **kwargs):
            handle, model = handles[0], args[0]
            self.assertEqual(model.num_timesteps, 4096)
            self.assertEqual(model._n_updates, 10)
            self.assertIsNone(kwargs['frozen_normalizer'])
            state = (deepcopy(handle.vec_env.get_attr('_block38_counters')),
                deepcopy(handle.vec_env.get_attr('np_random')), handle.vec_env.get_attr('steps'),
                handle.vec_env.get_attr('t'), tuple(module.training for module in model.policy.modules()))
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            progress('frozen_raw_vector_evaluation')
            result = original_evaluate(*args, **kwargs)
            self.assertEqual(handle.vec_env.get_attr('_block38_counters'), state[0])
            for before, after in zip(state[1], handle.vec_env.get_attr('np_random')):
                self.assertEqual(before.bit_generator.state, after.bit_generator.state)
            self.assertEqual(handle.vec_env.get_attr('steps'), state[2])
            self.assertEqual(handle.vec_env.get_attr('t'), state[3])
            self.assertEqual(tuple(module.training for module in model.policy.modules()), state[4])
            self.assertEqual(random.getstate(), rng[0])
            current = np.random.get_state()
            self.assertEqual(current[0], rng[1][0]); np.testing.assert_array_equal(current[1], rng[1][1])
            self.assertEqual(current[2:], rng[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))
            np.testing.assert_array_equal(result.returns_raw, result.returns_scaled)
            self.assertTrue(np.all(np.isfinite(result.returns_raw)))
            live_evaluations.append(result)
            return result

        try:
            with tempfile.TemporaryDirectory(prefix='mo_block38_') as temporary:
                # Only synthetic budget/holdout/reference/output identifiers vary.
                # All physical settings, PPO values, normalization and scales stay
                # identical to the checked-in production example.
                config = replace(baseline, experiment_id='block38_standard_compatibility',
                    output_dir=Path(temporary) / 'experiment', run_seeds=(0,),
                    requested_timesteps_per_policy=4096, eval_episodes=2, final_episodes=2,
                    reference_scaled=(-10., -10.), reference_status='frozen')
                weights = planning.make_weights(config.n_weights)
                plan, budget = planning.make_seed_plan(config, 0), planning.make_budget_plan(config)
                weight_index = 5
                np.testing.assert_array_equal(weights[weight_index], [.5, .5])
                self.assertEqual(budget.effective_total, 4096)
                root = storage.create_experiment_dir(config, weights, (plan,), budget)
                results = storage.load_evaluation_records(root, config)
                with LocalLogger(root / 'logs/compatibility') as logger, \
                     patch.object(envs, 'make_env_factory', side_effect=probe_factory), \
                     patch.object(envs, 'make_train_vec_env', side_effect=make_handle), \
                     patch.object(trainer, 'build_model', side_effect=build) as built, \
                     patch.object(trainer, 'learn_one_update', side_effect=learn) as learned, \
                     patch.object(evaluation, 'evaluate_policy', side_effect=evaluate) as evaluated:
                    summary = trainer.train_one_policy(config, root, 0, weight_index, weights,
                                                      plan, budget, results, logger)
                    built.assert_called_once(); learned.assert_called_once(); evaluated.assert_called_once()
                    self.assertTrue(summary.complete)
                    self.assertEqual(summary.actual_transitions, 4096)
                self.assertTrue(handles[0].closed)
                for worker in handles[0].workers:
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(worker.exitcode, 0)
                self.assertTrue(all(remote.closed for remote in handles[0].raw_vec_env.remotes))
                progress('training_workers_closed')
                policy = root / 'models/seed_0/w_5'
                self.assertTrue((policy / 'COMPLETE.json').is_file())
                self.assertFalse((policy / 'final/vecnormalize.pkl').exists())
                record = storage.discover_latest_checkpoint(policy, summary.identity.experiment_fingerprint)
                self.assertEqual(record.actual_step, 4096)
                before_reload_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
                with storage.load_policy_bundle(policy / 'final', config) as bundle:
                    self.assertEqual(bundle.model.num_timesteps, 4096)
                    self.assertEqual(bundle.model._n_updates, 10)
                    self.assertIsNone(bundle.vec_normalize)
                    self.assertIsNone(bundle.model.get_env())
                    for p, loaded in zip(models[0].policy.parameters(), bundle.model.policy.parameters()):
                        self.assertTrue(torch.equal(p, loaded))
                    reloaded = original_evaluate(bundle.model, lambda: envs.make_raw_env(config.env),
                        plan.eval_seeds, config, bundle.identity, collect_traces=True)
                    reference = live_evaluations[0]
                    np.testing.assert_array_equal(reloaded.returns_raw, reference.returns_raw)
                    np.testing.assert_array_equal(reloaded.returns_scaled, reference.returns_scaled)
                    self.assertEqual([m.to_dict() for m in reloaded.episode_metrics],
                                     [m.to_dict() for m in reference.episode_metrics])
                    for i, trace in enumerate(reloaded.traces):
                        self.assertEqual(trace.true_states.shape[1], 20)
                        self.assertEqual(trace.initial_known_count, 10)
                        accumulated = np.zeros(2, np.float64)
                        discount = 1.
                        for reward in trace.vector_rewards:
                            accumulated += discount * reward.astype(np.float64)
                            discount *= config.ppo.gamma
                        np.testing.assert_array_equal(accumulated, reloaded.returns_raw[i])
                        metric = reloaded.episode_metrics[i]
                        self.assertEqual(metric.unique_new_detections,
                            int(trace.observed_targets[:, 10:].any(axis=0).sum()))
                        self.assertEqual(metric.mean_coverage, float(trace.coverage_values.mean()))
                        self.assertEqual(metric.final_coverage, float(trace.coverage_values[-1]))
                        self.assertIsNone(metric.search_time_percentage)
                        expected_known_frequency = np.mean(trace.observed_targets[:, :10].sum(axis=0)
                                                          / int(trace.elapsed_steps.sum()))
                        self.assertEqual(metric.mean_known_observation_frequency, expected_known_frequency)
                self.assertTrue(bundle.closed)
                self.assertEqual(random.getstate(), before_reload_rng[0])
                current = np.random.get_state()
                np.testing.assert_array_equal(current[1], before_reload_rng[1][1])
                self.assertEqual(current[2:], before_reload_rng[1][2:])
                self.assertTrue(torch.equal(torch.get_rng_state(), before_reload_rng[2]))
                events = [json.loads(p.read_text()) for p in sorted((root / 'logs/compatibility').glob('*.json'))]
                episodes = [e for e in events if e['event'] == 'training_episode']
                self.assertEqual(len(episodes), sum(c['episodes'] for c in observations))
                self.assertEqual({e['identity']['worker_index'] for e in episodes}, set(range(8)))
                restored = storage.load_evaluation_records(root, config)
                self.assertEqual(int(restored.validity.sum()), 1)
                self.assertTrue(restored.validity[0, 0, weight_index])
                self.assertTrue(np.isnan(restored.hypervolume).all())
                progress('reload_and_raw_metric_equivalence_passed',
                    evaluation_seeds=plan.eval_seeds, returns_raw=reloaded.returns_raw,
                    episode_metrics=[m.to_dict() for m in reloaded.episode_metrics],
                    completed_training_episodes=len(episodes), tested_weight=weights[weight_index],
                    valid_policy_slots=1, declared_policy_slots=config.n_weights)
            self.assertEqual(torch.get_num_threads(), parent_threads)
            self.assertEqual(numerical_thread_snapshot(), parent_pools)
            self.assertEqual({key: os.environ.get(key) for key in thread_keys}, parent_environment)
            self.assertFalse(mp.active_children())
            self.assertEqual(source_hashes(), before_sources)
            self.assertEqual(production.read_bytes(), original_bytes)
            progress('passed', status='complete', protected_source_count=len(before_sources),
                source_hashes=before_sources, production_config_sha256=hashlib.sha256(original_bytes).hexdigest(),
                parent_threads_preserved=True, worker_cleanup_verified=True)
        except BaseException as error:
            progress('failed', status='failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    unittest.main()

