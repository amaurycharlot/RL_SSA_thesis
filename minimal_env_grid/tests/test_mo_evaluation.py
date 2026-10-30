"""Block 17 frozen preprocessing/guard checks, without training or workers."""
from pathlib import Path
import os
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import ObsNormalizer
from agents.mo.evaluation import (
    make_frozen_obs_normalizer, normalize_policy_observation, evaluation_state_guard,
)


class FrozenPolicyPreprocessing(unittest.TestCase):
    # Params: self, norm_obs; returns: normalizer.
    def normalizer(self, norm_obs=True):
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        class Grid(gym.Env):
            observation_space = gym.spaces.Box(-1e9, 1e9, (2, 3, 3), dtype=np.float32)
            action_space = gym.spaces.MultiDiscrete([2, 3])
            # Params: self, seed, options; returns: (array, dict).
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros((2, 3, 3), np.float32), {}
            # Params: self, action; returns: (array, float, bool, bool, dict).
            def step(self, action):
                return np.zeros((2, 3, 3), np.float32), 2., False, False, {}
        normalizer = VecNormalize(DummyVecEnv([Grid]), norm_obs=norm_obs, norm_reward=True,
                                  gamma=.9, epsilon=1e-6, clip_obs=1.5, clip_reward=2.5)
        self.addCleanup(normalizer.close)
        if norm_obs:
            normalizer.obs_rms.mean = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 10
            normalizer.obs_rms.var = np.full((2, 3, 3), 4., np.float64)
            normalizer.obs_rms.count = 123.
        return normalizer

    # Params: self, normalizer; returns: normalizer state tuple.
    def snapshot_live(self, normalizer):
        return (normalizer.training, normalizer.norm_obs, normalizer.norm_reward,
                normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy(), normalizer.obs_rms.count,
                np.asarray(normalizer.ret_rms.mean).copy(), np.asarray(normalizer.ret_rms.var).copy(),
                normalizer.ret_rms.count, normalizer.returns.copy())

    # Params: self, normalizer, before; returns: None.
    def assert_live_equal(self, normalizer, before):
        after = self.snapshot_live(normalizer)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    # Params: self; returns: None.
    def test_absent_and_observation_disabled_paths_are_identity_without_reward_state(self):
        self.assertIsNone(make_frozen_obs_normalizer(None))
        disabled = self.normalizer(norm_obs=False)
        self.assertIsNone(make_frozen_obs_normalizer(disabled))
        observation = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        actual = normalize_policy_observation(observation, None)
        np.testing.assert_array_equal(actual, observation)
        self.assertEqual(actual.dtype, np.float32)
        self.assertFalse(np.shares_memory(actual, observation))
        actual[0, 0, 0] = 99
        self.assertEqual(observation[0, 0, 0], 0)

    # Params: self; returns: None.
    def test_enabled_snapshot_matches_sb3_exactly_and_never_changes_live_state(self):
        live = self.normalizer()
        before = self.snapshot_live(live)
        frozen = make_frozen_obs_normalizer(live)
        observation = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 2
        raw = observation.copy()
        expected = live.normalize_obs(observation)
        for _ in range(3):
            np.testing.assert_array_equal(normalize_policy_observation(observation, frozen), expected)
        self.assert_live_equal(live, before)
        np.testing.assert_array_equal(observation, raw)
        self.assertEqual(frozen.count, 123)
        self.assertEqual(frozen.epsilon, 1e-6)
        self.assertEqual(frozen.clip_obs, 1.5)
        self.assertNotIn("ret_rms", frozen.__dict__)
        self.assertNotIn("returns", frozen.__dict__)

    # Params: self; returns: None.
    def test_snapshot_is_independent_immutable_and_policy_specific(self):
        first, second = self.normalizer(), self.normalizer()
        second.obs_rms.mean[:] = 100
        a, b = make_frozen_obs_normalizer(first), make_frozen_obs_normalizer(second)
        original = a.mean.copy()
        first.obs_rms.mean[:] = 999
        first.obs_rms.var[:] = 999
        first.obs_rms.count = 999
        np.testing.assert_array_equal(a.mean, original)
        np.testing.assert_array_equal(a.variance, np.full((2, 3, 3), 4))
        self.assertEqual(a.count, 123)
        for array in (a.mean, a.variance):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        obs = np.ones((2, 3, 3), np.float32)
        self.assertFalse(np.array_equal(normalize_policy_observation(obs, a), normalize_policy_observation(obs, b)))

    # Params: self; returns: None.
    def test_snapshot_normalize_is_invoked_exactly_once_and_live_objects_are_rejected(self):
        live = self.normalizer()
        frozen = make_frozen_obs_normalizer(live)
        original = ObsNormalizer.normalize
        calls = []
        # Params: instance, observation; returns: original result.
        def tracked(instance, observation):
            calls.append(instance)
            return original(instance, observation)
        with patch.object(ObsNormalizer, "normalize", tracked):
            normalize_policy_observation(np.ones((2, 3, 3), np.float32), frozen)
        self.assertEqual(calls, [frozen])
        with self.assertRaises(TypeError):
            normalize_policy_observation(np.ones((2, 3, 3), np.float32), live)

    # Params: self; returns: None.
    def test_invalid_raw_observations_and_snapshots_fail_without_changing_live_state(self):
        live = self.normalizer()
        before = self.snapshot_live(live)
        frozen = make_frozen_obs_normalizer(live)
        for observation, error in ((np.zeros((2, 3, 3), np.float64), TypeError),
                                    (np.zeros((2, 3, 2), np.float32), ValueError),
                                    (np.full((2, 3, 3), np.nan, np.float32), ValueError),
                                    (np.zeros((0, 3, 3), np.float32), ValueError),
                                    (np.zeros((3, 3, 3), np.float32), ValueError)):
            with self.subTest(shape=observation.shape), self.assertRaises(error):
                normalize_policy_observation(observation, frozen)
        self.assert_live_equal(live, before)
        with self.assertRaises(TypeError):
            make_frozen_obs_normalizer(SimpleNamespace(norm_obs=True, obs_rms={}))
        with self.assertRaises(TypeError):
            make_frozen_obs_normalizer(SimpleNamespace(norm_obs="yes"))


class EvaluationStatePreservation(unittest.TestCase):
    # Params: self
    # Returns: (random.getstate result, np.random.get_state result, torch.random.get_rng_state().clone result).
    def snapshot(self):
        return (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())

    # Params: self, before; returns: None.
    def assert_rng_equal(self, before):
        self.assertEqual(random.getstate(), before[0])
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before[1][0])
        np.testing.assert_array_equal(after_numpy[1], before[1][1])
        self.assertEqual(after_numpy[2:], before[1][2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before[2]))

    # Params: self; returns: None.
    def consume(self):
        random.random()
        np.random.normal(size=10)
        torch.randn(10)

    # Params: self; returns: SimpleNamespace.
    def model(self):
        return SimpleNamespace(policy=torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.5)))

    # Params: self; returns: None.
    def test_rng_and_mixed_policy_modes_restore_after_success_and_exception(self):
        model = self.model()
        model.policy[0].eval()
        flags = [module.training for module in model.policy.modules()]
        before = self.snapshot()
        for fail in (False, True):
            try:
                with evaluation_state_guard(model):
                    self.assertFalse(any(module.training for module in model.policy.modules()))
                    self.consume()
                    if fail:
                        raise RuntimeError("evaluation failed")
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual([module.training for module in model.policy.modules()], flags)
            self.assert_rng_equal(before)

    # Params: self; returns: None.
    def test_prior_eval_mode_and_model_parameters_remain_unchanged(self):
        model = self.model()
        model.policy.eval()
        weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
        with evaluation_state_guard(model):
            model.policy(torch.ones(3))
        self.assertFalse(any(module.training for module in model.policy.modules()))
        for key, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))

    # Params: self; returns: None.
    def test_nested_guards_restore_their_own_rng_and_mode_boundaries(self):
        model = self.model()
        original = self.snapshot()
        with evaluation_state_guard(model):
            self.consume()
            inner = self.snapshot()
            with evaluation_state_guard(model):
                self.consume()
            self.assert_rng_equal(inner)
            self.assertFalse(model.policy.training)
        self.assert_rng_equal(original)
        self.assertTrue(model.policy.training)

    # Params: self; returns: None.
    def test_failure_during_mode_entry_restores_flags_and_rng(self):
        class Failure(torch.nn.Sequential):
            # Params: instance, mode; returns: instance.
            def train(instance, mode=True):
                super().train(mode)
                if not mode:
                    random.random()
                    np.random.random()
                    torch.rand(1)
                    raise RuntimeError("mode entry failed")
                return instance
        model = SimpleNamespace(policy=Failure(torch.nn.Dropout()))
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "mode entry failed"):
            with evaluation_state_guard(model):
                self.fail("Must not enter evaluation after mode failure")
        self.assertTrue(model.policy.training)
        self.assertTrue(model.policy[0].training)
        self.assert_rng_equal(before)

    # Params: self; returns: None.
    def test_missing_policy_still_protects_rng_and_nonmodule_policy_is_rejected(self):
        before = self.snapshot()
        with evaluation_state_guard():
            self.consume()
        self.assert_rng_equal(before)
        with self.assertRaises(TypeError):
            with evaluation_state_guard(SimpleNamespace(policy=object())):
                pass

    # Params: self; returns: None.
    def test_cuda_snapshot_only_when_already_initialized(self):
        with patch.object(torch.cuda, "is_initialized", return_value=False), \
             patch.object(torch.cuda, "get_rng_state_all") as get_cuda, \
             patch.object(torch.cuda, "set_rng_state_all") as set_cuda:
            with evaluation_state_guard():
                self.consume()
            get_cuda.assert_not_called()
            set_cuda.assert_not_called()
        states = [torch.tensor([1, 2], dtype=torch.uint8)]
        with patch.object(torch.cuda, "is_initialized", return_value=True), \
             patch.object(torch.cuda, "get_rng_state_all", return_value=states), \
             patch.object(torch.cuda, "set_rng_state_all") as restore:
            with self.assertRaises(RuntimeError):
                with evaluation_state_guard():
                    states[0][0] = 99
                    raise RuntimeError("injected CUDA-path failure")
            self.assertTrue(torch.equal(restore.call_args.args[0][0], torch.tensor([1, 2], dtype=torch.uint8)))

    # Params: self; returns: None.
    def test_preprocessing_and_exception_guard_preserve_live_normalizer_and_rng(self):
        normalizer = FrozenPolicyPreprocessing.normalizer(self)
        model = self.model()
        before = self.snapshot()
        flags = (normalizer.training, normalizer.norm_obs, normalizer.norm_reward)
        rms = (normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy(), normalizer.obs_rms.count,
               np.asarray(normalizer.ret_rms.mean).copy(), normalizer.ret_rms.count)
        with self.assertRaises(RuntimeError):
            with evaluation_state_guard(model):
                frozen = make_frozen_obs_normalizer(normalizer)
                normalize_policy_observation(np.ones((2, 3, 3), np.float32), frozen)
                self.consume()
                raise RuntimeError("evaluation failed after preprocessing")
        self.assert_rng_equal(before)
        self.assertEqual((normalizer.training, normalizer.norm_obs, normalizer.norm_reward), flags)
        for actual, expected in zip((normalizer.obs_rms.mean, normalizer.obs_rms.var, normalizer.obs_rms.count,
                                     normalizer.ret_rms.mean, normalizer.ret_rms.count), rms):
            np.testing.assert_array_equal(actual, expected)

    # Params: self; returns: None.
    def test_import_is_passive_and_has_no_training_or_numeric_imports(self):
        script = '''
import sys
import agents.mo.evaluation
assert not any(name in sys.modules for name in
               ("numpy", "torch", "scipy", "gymnasium", "stable_baselines3", "matplotlib"))
print("passive evaluation helpers")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run([sys.executable, "-B", "-c", script], env=environment,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class RawEpisodeEvaluation(unittest.TestCase):
    # Params: self, truncated, endless, durations; returns: Raw.
    def environment(self, truncated=False, endless=False, durations=(1, 1, 1)):
        import gymnasium as gym
        class Raw(gym.Env):
            mode = 'combined'
            reward_order = ('tracking', 'search')
            known_targets, unknown_targets = 1, 1
            grid_shape = (2, 3)
            max_steps, dwell_steps = sum(durations), 1
            observation_space = gym.spaces.Box(-1e9, 1e9, (2, 3, 3), np.float32)
            action_space = gym.spaces.MultiDiscrete([2, 3])
            closed = 0
            calls = 0
            # Params: self, seed, options; returns: (self.buffer, dict).
            def reset(self, seed=None, options=None):
                self.seed_used = seed
                self.calls = 0
                self.total = 0
                self.buffer = np.zeros((2, 3, 3), np.float32)
                return self.buffer, {}
            # Params: self, action; returns: (self.buffer, array, result, result, dict).
            def step(self, action):
                self.calls += 1
                duration = durations[min(self.calls - 1, len(durations) - 1)]
                self.total += duration
                self.buffer.fill(self.calls)
                true = np.zeros((2, 4), np.float32)
                true[:, 0] = self.calls
                est = np.zeros((2, 9), np.float64)
                est[:, 0] = self.calls
                est[:, 4:6] = 1
                done = self.calls == len(durations) and not endless
                return self.buffer, np.array([1, 2], np.float32), done and not truncated, done and truncated, {
                    'true_states': true, 'estimated_states': est,
                    'observed_target_ids': [1] if self.calls == 1 else [0],
                    'num_measurements': 1, 'elapsed_steps': duration, 'total_steps': self.total,
                    'coverage_mean': float(self.calls) / 10}
            # Params: self; returns: None.
            def close(self):
                self.closed += 1
                # Cleanup must not perturb the learner's global generators.
                random.random(); np.random.random(); torch.rand(1)
        return Raw()

    # Params: self, action, consume_rng; returns: Model.
    def model(self, action=(0, 1), consume_rng=False):
        policy = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.5))
        policy[0].eval()
        class Model:
            # Params: self; returns: None.
            def __init__(self):
                self.policy = policy
                self.inputs = []
            # Params: self, observation, deterministic; returns: (array, None).
            def predict(self, observation, deterministic):
                assert deterministic is True
                assert not any(module.training for module in self.policy.modules())
                self.inputs.append(observation.copy())
                if consume_rng:
                    random.random(); np.random.random(); torch.rand(1)
                return np.asarray(action), None
        return Model()

    # Params: self, env, model, **overrides; returns: evaluate_episode result.
    def evaluate(self, env=None, model=None, **overrides):
        from agents.mo.evaluation import evaluate_episode
        kwargs = dict(episode_seed=123, gamma=.5, scales=(2., 4.),
                      frozen_normalizer=None, metric_config={'error_target_set': 'catalogue'},
                      collect_trace=True)
        kwargs.update(overrides)
        return evaluate_episode(model or self.model(), env if env is not None else self.environment(), **kwargs)

    # Params: self; returns: None.
    def test_three_decision_return_and_post_action_alignment(self):
        env, model = self.environment(), self.model()
        result = self.evaluate(env, model)
        np.testing.assert_array_equal(result.return_raw, [1.75, 3.5])
        np.testing.assert_array_equal(result.return_scaled, [.875, .875])
        self.assertEqual(result.return_raw.dtype, np.float64)
        self.assertEqual(result.return_raw.shape, (2,))
        self.assertEqual(env.seed_used, 123)
        self.assertEqual(env.closed, 0)
        trace = result.trace
        np.testing.assert_array_equal(trace.initial_observation, 0)
        np.testing.assert_array_equal(trace.observations[:, 0, 0, 0], [1, 2, 3])
        np.testing.assert_array_equal(trace.true_states[:, 0, 0], [1, 2, 3])
        np.testing.assert_array_equal(trace.estimated_states[:, 0, 0], [1, 2, 3])
        np.testing.assert_array_equal(trace.actions, [[0, 1]] * 3)
        np.testing.assert_array_equal(trace.observed_targets, [[False, True], [True, False], [True, False]])
        np.testing.assert_array_equal([x[0, 0, 0] for x in model.inputs], [0, 1, 2])
        self.assertEqual(result.metrics.unique_new_detections, 1)
        self.assertEqual(result.metrics.mean_final_positional_error, 0.)
        self.assertEqual(result.metrics.search_time_percentage, None)

    # Params: self; returns: None.
    def test_truncation_is_a_real_terminal_transition_without_bootstrap(self):
        result = self.evaluate(self.environment(truncated=True))
        np.testing.assert_array_equal(result.return_raw, [1.75, 3.5])
        np.testing.assert_array_equal(result.trace.terminated, [False] * 3)
        np.testing.assert_array_equal(result.trace.truncated, [False, False, True])

    # Params: self; returns: None.
    def test_slew_discounts_once_per_decision_and_metrics_use_elapsed_time(self):
        result = self.evaluate(self.environment(durations=(2, 4, 3)))
        np.testing.assert_array_equal(result.return_raw, [1.75, 3.5])
        self.assertEqual(result.metrics.elapsed_timesteps, 9)
        self.assertEqual(result.metrics.decisions, 3)
        self.assertEqual(result.metrics.mean_known_observation_frequency, 2 / 9)

    # Params: self; returns: None.
    def test_optional_trace_changes_no_metrics_or_returns(self):
        retained = self.evaluate()
        omitted = self.evaluate(collect_trace=False)
        self.assertIsNone(omitted.trace)
        self.assertEqual(retained.metrics, omitted.metrics)
        np.testing.assert_array_equal(retained.return_raw, omitted.return_raw)

    # Params: self; returns: None.
    def test_frozen_normalization_applied_once_and_trace_stays_raw(self):
        model = self.model()
        normalizer = ObsNormalizer(mean=np.ones((2, 3, 3)), variance=np.ones((2, 3, 3)) * 4,
                                   count=10, epsilon=1e-8, clip_obs=10)
        original = ObsNormalizer.normalize
        with patch.object(ObsNormalizer, 'normalize', autospec=True, side_effect=original) as normalize:
            result = self.evaluate(model=model, frozen_normalizer=normalizer)
        self.assertEqual(normalize.call_count, 3)
        for index, observation in enumerate(model.inputs):
            np.testing.assert_array_equal(observation, original(normalizer, np.full((2, 3, 3), index, np.float32)))
        np.testing.assert_array_equal(result.trace.observations[:, 0, 0, 0], [1, 2, 3])

    # Params: self; returns: None.
    def test_rng_and_mixed_module_modes_survive_success_and_predict_failure(self):
        for failure in (False, True):
            env, model = self.environment(), self.model(consume_rng=True)
            if failure:
                original = model.predict
                # Params: *args, **kwargs; returns: never (raises).
                def fail(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError('prediction failed')
                model.predict = fail
            modes = [module.training for module in model.policy.modules()]
            before = EvaluationStatePreservation.snapshot(self)
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'prediction failed'):
                    self.evaluate(env, model)
                self.assertEqual(env.closed, 1)
            else:
                self.evaluate(env, model)
                self.assertEqual(env.closed, 0)
            EvaluationStatePreservation.assert_rng_equal(self, before)
            self.assertEqual(modes, [module.training for module in model.policy.modules()])

    # Params: self; returns: None.
    def test_invalid_actions_fail_before_step(self):
        for action in ((2, 0), (-1, 1), (0., 1.), ((0, 1),), (True, False)):
            env = self.environment()
            with self.assertRaises(ValueError):
                self.evaluate(env, self.model(action))
            self.assertEqual(env.calls, 0)
            self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_invalid_parameters_fail_before_reset(self):
        for kwargs in ({'gamma': np.nan}, {'gamma': -1}, {'gamma': True},
                {'episode_seed': True}, {'episode_seed': -1}, {'scales': [0, 1]},
                {'scales': [True, 1]}, {'scales': [np.inf, 1]}, {'scales': [1]}, {'collect_trace': 1}):
            env = self.environment()
            with self.assertRaises((ValueError, TypeError)):
                self.evaluate(env, **kwargs)
            self.assertFalse(hasattr(env, 'seed_used'))
            self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_malformed_step_diagnostics_rewards_and_observations_close(self):
        for field, value in (('true_states', np.zeros((1, 4), np.float32)),
                ('estimated_states', np.zeros((2, 8))), ('observed_target_ids', [2]),
                ('observed_target_ids', [1, 1]), ('elapsed_steps', 0), ('elapsed_steps', 1.5),
                ('total_steps', 100), ('coverage_mean', np.nan), ('num_measurements', 0),
                ('reward', np.array([1., 2.])), ('reward', np.float32(1)),
                ('reward', np.array([np.nan, 2], np.float32)), ('obs', np.zeros((1, 1, 3), np.float32)),
                ('terminated', 1)):
            with self.subTest(field=field, value=value):
                env = self.environment()
                original = env.step
                # Params: action; returns: (obs, reward, terminated, truncated, info).
                def broken(action):
                    obs, reward, terminated, truncated, info = original(action)
                    if field == 'reward': reward = value
                    elif field == 'obs': obs = value
                    elif field == 'terminated': terminated = value
                    else: info[field] = value
                    return obs, reward, terminated, truncated, info
                env.step = broken
                with self.assertRaises((ValueError, TypeError)):
                    self.evaluate(env)
                self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_endless_environment_stops_at_horizon_derived_cap(self):
        env = self.environment(endless=True)
        with self.assertRaisesRegex(RuntimeError, 'decision cap'):
            self.evaluate(env)
        self.assertEqual(env.calls, 3)
        self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_reset_and_metric_failures_close_and_restore_rng(self):
        for fail_reset in (True, False):
            env = self.environment()
            if fail_reset:
                # Params: **kwargs; returns: never (raises).
                def reset(**kwargs):
                    random.random(); np.random.random(); torch.rand(1)
                    raise RuntimeError('reset failed')
                env.reset = reset
            model = self.model()
            before = EvaluationStatePreservation.snapshot(self)
            with self.assertRaises((RuntimeError, ValueError)):
                self.evaluate(env, model, metric_config={'error_target_set': 'invalid'})
            EvaluationStatePreservation.assert_rng_equal(self, before)
            self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_real_environment_matches_manual_fixed_actions_and_repeated_seeds(self):
        from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
        # Params: none; returns: env.
        def create():
            env = MinimalSSAEnv(known_targets=1, unknown_targets=1, grid_shape=(2, 3),
                timesteps=3, dwell_steps=1, slew_alpha=0., reward_type='R6', search_reward='R2', mode='combined')
            self.addCleanup(env.close)
            return env
        first = self.evaluate(create())
        second = self.evaluate(create())
        manual = create()
        manual.reset(seed=123)
        rewards, obs_rows, states, estimated, masks = [], [], [], [], []
        while True:
            obs, reward, done, truncated, info = manual.step(np.array([0, 1]))
            rewards.append(reward.copy()); obs_rows.append(obs.copy())
            states.append(info['true_states'].copy()); estimated.append(info['estimated_states'].copy())
            mask = np.zeros(2, bool); mask[info['observed_target_ids']] = True; masks.append(mask)
            if done or truncated: break
        for name, expected in (('observations', obs_rows), ('vector_rewards', rewards),
                ('true_states', states), ('estimated_states', estimated), ('observed_targets', masks)):
            np.testing.assert_array_equal(getattr(first.trace, name), expected)
            np.testing.assert_array_equal(getattr(first.trace, name), getattr(second.trace, name))
        np.testing.assert_array_equal(first.return_raw,
            sum((.5 ** i) * reward.astype(np.float64) for i, reward in enumerate(rewards)))
        self.assertEqual(first.metrics, second.metrics)

    # Params: self; returns: None.
    def test_actual_sb3_policy_predict_works_without_learning(self):
        import gymnasium as gym
        from stable_baselines3 import PPO
        class Scalar(gym.RewardWrapper):
            # Params: self, reward; returns: float.
            def reward(self, reward):
                return float(np.sum(reward))
        training = Scalar(self.environment())
        self.addCleanup(training.close)
        model = PPO('MlpPolicy', training, n_steps=2, batch_size=2, n_epochs=1,
                    device='cpu', seed=456, verbose=0)
        parameters = [parameter.detach().clone() for parameter in model.policy.parameters()]
        before = EvaluationStatePreservation.snapshot(self)
        result = self.evaluate(model=model)
        EvaluationStatePreservation.assert_rng_equal(self, before)
        self.assertEqual(result.metrics.decisions, 3)
        self.assertEqual(model.num_timesteps, 0)
        self.assertTrue(model.policy.training)
        for original, parameter in zip(parameters, model.policy.parameters()):
            self.assertTrue(torch.equal(original, parameter))

    # Params: self; returns: None.
    def test_empty_measurements_missing_tracks_and_horizon_overshoot(self):
        env = self.environment(durations=(2, 2))
        env.dwell_steps, env.max_steps = 2, 3
        original = env.step
        # Params: action; returns: (obs, reward, done, truncated, info).
        def step(action):
            obs, reward, done, truncated, info = original(action)
            info['observed_target_ids'] = []
            info['num_measurements'] = 0
            info['estimated_states'][:] = np.nan
            return obs, reward, done, truncated, info
        env.step = step
        result = self.evaluate(env)
        self.assertEqual(result.metrics.decisions, 2)
        self.assertEqual(result.metrics.elapsed_timesteps, 4)
        self.assertIsNone(result.metrics.mean_final_positional_error)
        self.assertEqual(result.metrics.unique_new_detections, 0)
        np.testing.assert_array_equal(result.trace.observed_targets, False)

    # Params: self; returns: None.
    def test_cleanup_failure_retains_original_error_and_rng(self):
        env = self.environment()
        original_close = env.close
        # Params: none; returns: never (raises).
        def close():
            original_close()
            raise RuntimeError('cleanup failed')
        env.close = close
        model = self.model()
        before = EvaluationStatePreservation.snapshot(self)
        with self.assertRaisesRegex(ValueError, 'gamma') as caught:
            self.evaluate(env, model, gamma=-1)
        self.assertEqual(str(caught.exception.__cause__), 'cleanup failed')
        EvaluationStatePreservation.assert_rng_equal(self, before)


class PolicyEvaluation(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from agents.mo.types import EnvConfig, ExperimentConfig, PPOConfig, PolicyIdentity
        self.fixture = RawEpisodeEvaluation()
        self.config = ExperimentConfig(env=EnvConfig(known_targets=1, unknown_targets=1,
            grid_shape=(2, 3), episode_steps=3), ppo=PPOConfig(gamma=.5),
            objective_scales=(2., 4.), eval_episodes=3, final_episodes=2)
        self.identity = PolicyIdentity('test-fingerprint', 0, 0, 1, (.25, .75), 4096, 'policy-1')

    # Params: self, factory, seeds, model, **overrides; returns: evaluate_policy result.
    def evaluate(self, factory=None, seeds=(19, 3, 11), model=None, **overrides):
        from agents.mo.evaluation import evaluate_policy
        kwargs = dict(config=self.config, identity=self.identity)
        kwargs.update(overrides)
        return evaluate_policy(model or self.fixture.model(), factory or self.fixture.environment,
                               seeds, **kwargs)

    # Params: self; returns: env.
    def tracked_environment(self):
        env = self.fixture.environment()
        env.resets = []
        original = env.reset
        # Params: seed, **kwargs; returns: original result.
        def reset(seed=None, **kwargs):
            env.resets.append(seed)
            return original(seed=seed, **kwargs)
        env.reset = reset
        return env

    # Params: self; returns: None.
    def test_factory_once_seeds_used_once_in_saved_order_and_identity_retained(self):
        env = self.tracked_environment()
        factory = unittest.mock.Mock(return_value=env)
        result = self.evaluate(factory, collect_traces=True)
        factory.assert_called_once_with()
        self.assertEqual(env.resets, [19, 3, 11])
        self.assertEqual(env.closed, 1)
        self.assertEqual(result.episode_seeds, (19, 3, 11))
        self.assertEqual(tuple(trace.seed for trace in result.traces), result.episode_seeds)
        self.assertIs(result.identity, self.identity)
        self.assertEqual(result.actual_step, 4096)
        self.assertEqual(result.identity.weight, (.25, .75))
        self.assertTrue(result.deterministic)
        self.assertEqual(result.returns_raw.shape, (3, 2))
        self.assertEqual(result.returns_scaled.dtype, np.float64)
        self.assertEqual(len(result.episode_metrics), 3)

    # Params: self; returns: None.
    def test_arithmetic_means_sample_sd_and_se_are_episode_vector_summaries(self):
        from dataclasses import replace
        prototype = self.fixture.evaluate()
        vectors = np.array([[1, 10], [5, 2], [3, 6]], np.float64)
        episodes = [replace(prototype, return_raw=row, return_scaled=row / [2, 4], trace=None,
                            metrics=replace(prototype.metrics, decisions=index + 1))
                    for index, row in enumerate(vectors)]
        env = self.tracked_environment()
        with patch('agents.mo.evaluation.evaluate_episode', side_effect=episodes) as episode_call:
            result = self.evaluate(lambda: env)
        self.assertEqual(episode_call.call_count, 3)
        np.testing.assert_array_equal(result.returns_raw, vectors)
        np.testing.assert_array_equal(result.mean_raw, [3, 6])
        np.testing.assert_array_equal(result.sd_raw, [2, 4])
        np.testing.assert_array_equal(result.se_raw, np.array([2, 4]) / np.sqrt(3))
        np.testing.assert_array_equal(result.mean_scaled, result.mean_raw / [2, 4])
        np.testing.assert_array_equal(result.sd_scaled, result.sd_raw / [2, 4])
        np.testing.assert_array_equal(result.se_scaled, result.se_raw / [2, 4])
        self.assertEqual([m.decisions for m in result.episode_metrics], [1, 2, 3])
        self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_different_length_episodes_are_not_pooled_per_decision(self):
        env = self.tracked_environment()
        original_step = env.step
        # Params: action; returns: (obs, array, bool, bool, info) or (obs, array, terminated, truncated, info).
        def step(action):
            obs, _, terminated, truncated, info = original_step(action)
            if env.seed_used == 19:
                return obs, np.array([1, 10], np.float32), True, False, info
            return obs, np.array([2, 1], np.float32), terminated, truncated, info
        env.step = step
        result = self.evaluate(lambda: env, seeds=(19, 3))
        np.testing.assert_array_equal(result.returns_raw, [[1, 10], [3.5, 1.75]])
        np.testing.assert_array_equal(result.mean_raw, [2.25, 5.875])
        self.assertEqual([metric.decisions for metric in result.episode_metrics], [1, 3])

    # Params: self; returns: None.
    def test_one_episode_has_exact_zero_sd_and_standard_error(self):
        result = self.evaluate(seeds=(3,))
        np.testing.assert_array_equal(result.mean_raw, result.returns_raw[0])
        for name in ('sd_raw', 'sd_scaled', 'se_raw', 'se_scaled'):
            value = getattr(result, name)
            self.assertEqual(value.dtype, np.float64)
            np.testing.assert_array_equal(value, [0, 0])

    # Params: self; returns: None.
    def test_trace_retention_does_not_change_returns_or_metric_records(self):
        retained = self.evaluate(collect_traces=True)
        omitted = self.evaluate()
        self.assertEqual(len(retained.traces), 3)
        self.assertEqual(omitted.traces, ())
        self.assertEqual(retained.episode_metrics, omitted.episode_metrics)
        for name in ('returns_raw', 'returns_scaled', 'mean_raw', 'sd_raw', 'se_raw'):
            np.testing.assert_array_equal(getattr(retained, name), getattr(omitted, name))

    # Params: self; returns: None.
    def test_invalid_seed_lists_fail_before_allocating_environment(self):
        for seeds in ((), (1, 1), {1, 2}, (True,), (-1,), (2 ** 32,), (1.,),
                      (np.int64(1),), '12', (s for s in (1, 2))):
            with self.subTest(seeds=seeds):
                factory = unittest.mock.Mock()
                with self.assertRaises((TypeError, ValueError)):
                    self.evaluate(factory, seeds=seeds)
                factory.assert_not_called()

    # Params: self; returns: None.
    def test_invalid_contracts_fail_before_allocating_environment(self):
        for kwargs in ({'config': {}}, {'identity': object()}, {'collect_traces': 1},
                       {'frozen_normalizer': object()}):
            factory = unittest.mock.Mock()
            with self.assertRaises(TypeError):
                self.evaluate(factory, **kwargs)
            factory.assert_not_called()
        from agents.mo.evaluation import evaluate_policy
        with self.assertRaises(TypeError):
            evaluate_policy(self.fixture.model(), None, (1,), self.config, self.identity)

    # Params: self; returns: None.
    def test_seed_list_is_snapshotted_before_factory_can_mutate_it(self):
        seeds = [19, 3, 11]
        env = self.tracked_environment()
        # Params: none; returns: env.
        def factory():
            seeds[:] = [99]
            return env
        result = self.evaluate(factory, seeds=seeds)
        self.assertEqual(env.resets, [19, 3, 11])
        self.assertEqual(result.episode_seeds, (19, 3, 11))

    # Params: self; returns: None.
    def test_factory_episode_and_cleanup_preserve_rng_and_mixed_policy_modes(self):
        model = self.fixture.model(consume_rng=True)
        env = self.tracked_environment()
        # Params: none; returns: env.
        def factory():
            random.random(); np.random.random(); torch.rand(1)
            return env
        before = EvaluationStatePreservation.snapshot(self)
        modes = [m.training for m in model.policy.modules()]
        self.evaluate(factory, model=model)
        EvaluationStatePreservation.assert_rng_equal(self, before)
        self.assertEqual([m.training for m in model.policy.modules()], modes)
        self.assertEqual(env.closed, 1)

    # Params: self; returns: None.
    def test_failed_second_episode_closes_once_and_stops_remaining_seeds(self):
        env = self.tracked_environment()
        original = env.reset
        # Params: seed, **kwargs; returns: (observation, info).
        def reset(seed=None, **kwargs):
            result = original(seed=seed, **kwargs)
            if seed == 3:
                random.random(); np.random.random(); torch.rand(1)
                raise RuntimeError('second reset failed')
            return result
        env.reset = reset
        model = self.fixture.model(consume_rng=True)
        modes = [m.training for m in model.policy.modules()]
        before = EvaluationStatePreservation.snapshot(self)
        with self.assertRaisesRegex(RuntimeError, 'second reset failed'):
            self.evaluate(lambda: env, model=model)
        self.assertEqual(env.resets, [19, 3])
        self.assertEqual(env.closed, 1)
        EvaluationStatePreservation.assert_rng_equal(self, before)
        self.assertEqual([m.training for m in model.policy.modules()], modes)

    # Params: self; returns: None.
    def test_factory_and_final_close_failures_restore_rng_and_modes(self):
        for factory_failure in (False, True):
            env = self.tracked_environment()
            original_close = env.close
            # Params: none; returns: never (raises).
            def close():
                original_close()
                raise RuntimeError('final close failed')
            env.close = close
            # Params: none; returns: env.
            def factory():
                random.random(); np.random.random(); torch.rand(1)
                if factory_failure:
                    raise RuntimeError('factory failed')
                return env
            model = self.fixture.model(consume_rng=True)
            modes = [m.training for m in model.policy.modules()]
            before = EvaluationStatePreservation.snapshot(self)
            with self.assertRaisesRegex(RuntimeError, 'factory failed' if factory_failure else 'final close failed'):
                self.evaluate(factory, model=model)
            EvaluationStatePreservation.assert_rng_equal(self, before)
            self.assertEqual([m.training for m in model.policy.modules()], modes)
            self.assertEqual(env.closed, 0 if factory_failure else 1)

    # Params: self; returns: None.
    def test_aggregation_failure_closes_resource_and_retains_primary_error(self):
        env = self.tracked_environment()
        original_close = env.close
        # Params: none; returns: never (raises).
        def close():
            original_close()
            raise RuntimeError('close also failed')
        env.close = close
        model = self.fixture.model()
        before = EvaluationStatePreservation.snapshot(self)
        with patch('agents.mo.types.EvaluationResult', side_effect=ValueError('assembly failed')):
            with self.assertRaisesRegex(ValueError, 'assembly failed') as caught:
                self.evaluate(lambda: env, model=model)
        self.assertEqual(str(caught.exception.__cause__), 'close also failed')
        self.assertEqual(env.closed, 1)
        EvaluationStatePreservation.assert_rng_equal(self, before)

    # Params: self; returns: None.
    def test_frozen_normalizer_is_forwarded_unchanged_for_every_episode(self):
        normalizer = ObsNormalizer(mean=np.ones((2, 3, 3)), variance=np.ones((2, 3, 3)) * 4,
                                   count=10)
        from agents.mo.evaluation import evaluate_episode
        with patch('agents.mo.evaluation.evaluate_episode', wraps=evaluate_episode) as call:
            result = self.evaluate(frozen_normalizer=normalizer, collect_traces=True)
        for args in call.call_args_list:
            self.assertIs(args.args[5], normalizer)
            self.assertIs(args.args[6], self.config)
        self.assertEqual(result.traces[0].observations.dtype, np.float32)
        self.assertEqual(normalizer.count, 10)

    # Params: self; returns: None.
    def test_repeated_real_policy_evaluations_reproduce_on_cpu(self):
        from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv
        instances = []
        # Params: none; returns: env.
        def factory():
            env = MinimalSSAEnv(known_targets=1, unknown_targets=1, grid_shape=(2, 3),
                timesteps=3, dwell_steps=1, slew_alpha=0., reward_type='R6', search_reward='R2', mode='combined')
            env.close = unittest.mock.Mock(wraps=env.close)
            instances.append(env)
            return env
        model = self.fixture.model(consume_rng=True)
        first = self.evaluate(factory, model=model, collect_traces=True)
        second = self.evaluate(factory, model=model, collect_traces=True)
        for name in ('returns_raw', 'returns_scaled', 'mean_raw', 'mean_scaled', 'sd_raw', 'se_raw'):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertEqual(first.episode_metrics, second.episode_metrics)
        for left, right in zip(first.traces, second.traces):
            np.testing.assert_array_equal(left.observations, right.observations)
            np.testing.assert_array_equal(left.vector_rewards, right.vector_rewards)
            np.testing.assert_array_equal(left.observed_targets, right.observed_targets)
        for env in instances:
            env.close.assert_called_once_with()

    # Params: self; returns: None.
    def test_saved_intermediate_and_final_lists_are_consumed_separately(self):
        from agents.mo.types import SeedPlan
        plan = SeedPlan(run_seed=0, run_index=0, learner_seed=100, base_worker_seed=100,
                        worker_seeds=(100, 101), eval_seeds=(19, 3, 11), final_seeds=(29, 31))
        evaluation_env, final_env = self.tracked_environment(), self.tracked_environment()
        evaluation = self.evaluate(lambda: evaluation_env, seeds=plan.eval_seeds)
        final = self.evaluate(lambda: final_env, seeds=plan.final_seeds)
        self.assertEqual(evaluation_env.resets, list(plan.eval_seeds))
        self.assertEqual(final_env.resets, list(plan.final_seeds))
        self.assertFalse(set(evaluation.episode_seeds) & set(final.episode_seeds))
        self.assertEqual(evaluation_env.closed, 1)
        self.assertEqual(final_env.closed, 1)

    # Params: self; returns: None.
    def test_episode_cleanup_failure_is_not_retried_or_masked_by_policy_owner(self):
        env = self.tracked_environment()
        # Params: **kwargs; returns: never (raises).
        def reset(**kwargs):
            raise ValueError('episode reset failed')
        env.reset = reset
        original_close = env.close
        # Params: none; returns: never (raises).
        def close():
            original_close()
            raise RuntimeError('episode close failed')
        env.close = close
        model = self.fixture.model()
        before = EvaluationStatePreservation.snapshot(self)
        with self.assertRaisesRegex(ValueError, 'episode reset failed') as caught:
            self.evaluate(lambda: env, model=model)
        self.assertEqual(str(caught.exception.__cause__), 'episode close failed')
        self.assertEqual(env.closed, 1)
        EvaluationStatePreservation.assert_rng_equal(self, before)


class FinalPolicySetEvaluation(unittest.TestCase):
    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    # Params: cls; returns: None.
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    # Params: self; returns: None.
    def setUp(self):
        import tempfile
        from agents.mo.types import ExperimentConfig, EnvConfig, PPOConfig, ExecutionConfig
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ExperimentConfig(output_dir=Path(self.temporary.name) / 'experiment',
            run_seeds=(7, 9), n_weights=2, eval_episodes=1, final_episodes=2,
            reference_scaled=(0., 0.), reference_status='frozen',
            requested_timesteps_per_policy=2, eval_interval=1, checkpoint_interval=1,
            env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3), episode_steps=2, slew_alpha=0.),
            ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, verbose=0, device='cpu'),
            execution=ExecutionConfig(subproc=False))

    # Params: self; returns: None.
    def train(self):
        from agents.mo import storage
        from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
        from agents.mo.reporting import LocalLogger
        from agents.mo.trainer import train_one_policy
        self.weights = make_weights(self.config.n_weights)
        self.plans = tuple(make_seed_plan(self.config, i) for i in range(len(self.config.run_seeds)))
        self.budget = make_budget_plan(self.config)
        self.root = storage.create_experiment_dir(self.config, self.weights, self.plans, self.budget)
        results = storage.load_evaluation_records(self.root, self.config)
        with LocalLogger(self.root / 'logs/test-events') as logger:
            for i, plan in enumerate(self.plans):
                for j in range(self.config.n_weights):
                    train_one_policy(self.config, self.root, i, j, self.weights, plan, self.budget, results, logger)

    # Params: self, **kwargs; returns: evaluate_final_policy_set result.
    def evaluate(self, **kwargs):
        from agents.mo.evaluation import evaluate_final_policy_set
        return evaluate_final_policy_set(self.root, kwargs.get('config', self.config), kwargs.get('seed_plan', self.plans))

    # Params: self; returns: None.
    def test_real_final_reload_seed_identity_returns_and_persistence(self):
        import json
        from agents.mo import evaluation, storage
        from agents.mo.envs import make_raw_env
        self.train()
        validation = {p: p.read_bytes() for p in (self.root / 'results/evaluations').rglob('step_*.*')}
        calls, evaluate = [], evaluation.evaluate_policy
        # Params: *args, **kwargs; returns: EvaluationResult.
        def track(*args, **kwargs):
            self.assertEqual(args[2], self.plans[0].final_seeds)
            self.assertFalse(set(args[2]) & set(self.plans[0].eval_seeds))
            self.assertEqual(args[0].num_timesteps, self.budget.effective_total)
            result = evaluate(*args, **kwargs); calls.append(result); return result
        with patch.object(evaluation, 'evaluate_policy', side_effect=track), \
             patch.object(storage, 'discover_latest_checkpoint', side_effect=AssertionError('checkpoint selection')):
            result = self.evaluate()
        self.assertEqual(result.means_raw.shape, (2, 2, 2))
        self.assertEqual(result.episode_returns_raw.shape, (2, 2, 2, 2))
        self.assertTrue(result.validity.all())
        for call in calls:
            i, j = call.identity.run_index, call.identity.weight_index
            self.assertEqual(call.identity.weight, tuple(self.weights[j].tolist()))
            np.testing.assert_array_equal(result.episode_returns_raw[i, j], call.returns_raw)
            self.assertEqual(result.operational_metrics[i][j], call.episode_metrics)
        with storage.load_policy_bundle(self.root / 'models/seed_0/w_0/final', self.config) as bundle:
            expected = evaluate(bundle.model, lambda: make_raw_env(self.config.env), result.test_seeds,
                                self.config, bundle.identity)
        np.testing.assert_array_equal(result.episode_returns_raw[0, 0], expected.returns_raw)
        for path, content in validation.items(): self.assertEqual(path.read_bytes(), content)
        meta = json.loads((self.root / 'results/final_results.json').read_text())
        self.assertEqual(meta['actual_step'], self.budget.effective_total)
        self.assertEqual(len(meta['policy_records']), 4)
        with np.load(self.root / 'results/final_results.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['episode_returns_raw'], result.episode_returns_raw)
            np.testing.assert_array_equal(saved['test_seeds'], result.test_seeds)
        self.assertEqual(storage.load_evaluation_records(self.root, self.config).means_raw.shape, (2, 1, 2, 2))

    # Params: self; returns: None.
    def test_missing_last_final_bundle_blocks_set_before_any_model_load(self):
        from agents.mo import storage
        self.train()
        (self.root / 'models/seed_1/w_1/final').rename(Path(self.temporary.name) / 'missing-final')
        with patch.object(storage, 'load_policy_bundle') as load:
            with self.assertRaises(storage.StorageError): self.evaluate()
        load.assert_not_called()
        self.assertFalse((self.root / 'results/final_results.npz').exists())
        self.assertFalse(list((self.root / 'results/final').rglob('*.json')))

    # Params: self; returns: None.
    def test_corrupt_final_bundle_never_falls_back_to_valid_checkpoint(self):
        from agents.mo import storage
        self.train()
        (self.root / 'models/seed_1/w_1/final/final_model.zip').write_bytes(b'corrupt')
        with patch.object(storage, 'load_policy_bundle') as load:
            with self.assertRaises(storage.StorageError): self.evaluate()
        load.assert_not_called()
        self.assertTrue((self.root / 'models/seed_1/w_1/checkpoints/step_2/model.zip').is_file())

    # Params: self; returns: None.
    def test_only_exact_saved_ordered_seed_plans_and_science_are_accepted(self):
        from dataclasses import replace
        from agents.mo import storage
        self.train()
        wrong = replace(self.plans[0], final_seeds=self.plans[0].final_seeds[::-1])
        with patch.object(storage, 'load_policy_bundle') as load:
            for plans in (self.plans[::-1], self.plans[:1], (wrong, self.plans[1])):
                with self.assertRaises(storage.StorageError): self.evaluate(seed_plan=plans)
            with self.assertRaises(storage.StorageError):
                self.evaluate(config=replace(self.config, final_seed_root=self.config.final_seed_root + 1))
        load.assert_not_called()

    # Params: self; returns: None.
    def test_scaled_fronts_and_hv_per_seed_with_stable_duplicate_indices(self):
        from dataclasses import replace
        from agents.mo import evaluation
        self.config = replace(self.config, n_weights=3, objective_scales=(2., 4.), scale_status='frozen')
        self.train()
        geometry = np.array([[[1, 5], [4, 2], [1, 5]], [[2, 1], [1, 1], [0, 0]]], dtype=np.float64)
        evaluate = evaluation.evaluate_policy
        # Params: *args, **kwargs; returns: replace result.
        def synthetic(*args, **kwargs):
            result = evaluate(*args, **kwargs)
            point = geometry[result.identity.run_index, result.identity.weight_index]
            scaled = np.tile(point, (self.config.final_episodes, 1)); raw = scaled * self.config.objective_scales
            return replace(result, returns_raw=raw, returns_scaled=scaled, mean_raw=raw[0],
                mean_scaled=scaled[0], sd_raw=np.zeros(2), sd_scaled=np.zeros(2),
                se_raw=np.zeros(2), se_scaled=np.zeros(2))
        with patch.object(evaluation, 'evaluate_policy', side_effect=synthetic): result = self.evaluate()
        np.testing.assert_array_equal(result.means_scaled, geometry)
        np.testing.assert_array_equal(result.means_raw, geometry * [2., 4.])
        self.assertEqual(result.front_indices, ((0, 1), (0,)))
        np.testing.assert_array_equal(result.hypervolume, [11., 2.])

    # Params: self; returns: None.
    def test_committed_records_reused_and_disposable_aggregate_rebuilt(self):
        from agents.mo import evaluation, storage
        self.train(); first = self.evaluate()
        records = {p: p.read_bytes() for p in (self.root / 'results/final').rglob('step_*.*')}
        (self.root / 'results/final_results.npz').write_bytes(b'disposable corruption')
        with patch.object(storage, 'load_policy_bundle') as load, patch.object(evaluation, 'evaluate_policy') as evaluate:
            second = self.evaluate()
        load.assert_not_called(); evaluate.assert_not_called()
        for p, content in records.items(): self.assertEqual(p.read_bytes(), content)
        np.testing.assert_array_equal(second.episode_returns_raw, first.episode_returns_raw)
        with np.load(self.root / 'results/final_results.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['hypervolume'], first.hypervolume)

    # Params: self; returns: None.
    def test_corrupt_cached_test_record_fails_preflight_without_overwrite(self):
        from agents.mo import storage
        self.train(); self.evaluate()
        path = self.root / 'results/final/seed_1/w_1/step_2.npz'
        path.write_bytes(b'corrupt committed test payload')
        with patch.object(storage, 'load_policy_bundle') as load:
            with self.assertRaises(storage.StorageError): self.evaluate()
        load.assert_not_called()
        self.assertEqual(path.read_bytes(), b'corrupt committed test payload')

    # Params: self; returns: None.
    def test_interrupted_testing_reuses_committed_slots_and_closes_bundles(self):
        from agents.mo import evaluation, storage
        self.train()
        loaded, load, evaluate = [], storage.load_policy_bundle, evaluation.evaluate_policy
        # Params: *args, **kwargs; returns: bundle.
        def track(*args, **kwargs):
            self.assertTrue(all(b.closed for b in loaded))
            bundle = load(*args, **kwargs); loaded.append(bundle); return bundle
        # Params: *args, **kwargs; returns: evaluate result.
        def fail_second(*args, **kwargs):
            if args[4].weight_index == 1: raise RuntimeError('injected final test failure')
            return evaluate(*args, **kwargs)
        with patch.object(storage, 'load_policy_bundle', side_effect=track), \
             patch.object(evaluation, 'evaluate_policy', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'final test failure'): self.evaluate()
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(b.closed for b in loaded))
        self.assertFalse((self.root / 'results/final_results.npz').exists())
        first = self.root / 'results/final/seed_0/w_0/step_2.json'; original = first.read_bytes()
        with patch.object(storage, 'load_policy_bundle', wraps=load) as reload: result = self.evaluate()
        self.assertEqual(reload.call_count, 3)
        self.assertTrue(result.validity.all()); self.assertEqual(first.read_bytes(), original)

    # Params: self; returns: None.
    def test_normalized_models_use_frozen_obs_and_close_carriers(self):
        from dataclasses import replace
        from agents.mo import evaluation, storage
        from agents.mo.types import NormalizationConfig
        self.config = replace(self.config, run_seeds=(7,), normalization=NormalizationConfig(norm_obs=True, norm_reward=True))
        self.train()
        loaded, load, evaluate = [], storage.load_policy_bundle, evaluation.evaluate_policy
        # Params: *args, **kwargs; returns: bundle.
        def track(*args, **kwargs):
            self.assertTrue(all(b.closed for b in loaded))
            bundle = load(*args, **kwargs); loaded.append(bundle); return bundle
        # Params: *args, **kwargs; returns: EvaluationResult.
        def check(*args, **kwargs):
            normalizer = loaded[-1].vec_normalize; frozen = kwargs['frozen_normalizer']
            self.assertFalse(normalizer.training); self.assertFalse(normalizer.norm_reward)
            np.testing.assert_array_equal(frozen.mean, normalizer.obs_rms.mean)
            before = (normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy(), normalizer.obs_rms.count,
                      normalizer.ret_rms.mean.copy(), normalizer.ret_rms.var.copy(), normalizer.ret_rms.count)
            result = evaluate(*args, **kwargs)
            after = (normalizer.obs_rms.mean, normalizer.obs_rms.var, normalizer.obs_rms.count,
                     normalizer.ret_rms.mean, normalizer.ret_rms.var, normalizer.ret_rms.count)
            for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
            return result
        files = {p: p.read_bytes() for p in (self.root / 'models').rglob('vecnormalize.pkl')}
        with patch.object(storage, 'load_policy_bundle', side_effect=track), \
             patch.object(evaluation, 'evaluate_policy', side_effect=check): result = self.evaluate()
        self.assertTrue(result.validity.all())
        self.assertEqual(len(loaded), 2)
        self.assertTrue(all(b.closed and b.normalization_env is not None for b in loaded))
        for p, content in files.items(): self.assertEqual(p.read_bytes(), content)

    # Params: self; returns: None.
    def test_normalization_carriers_close_on_evaluation_failure(self):
        from dataclasses import replace
        from agents.mo import evaluation, storage
        from agents.mo.types import NormalizationConfig
        self.config = replace(self.config, run_seeds=(7,), normalization=NormalizationConfig(norm_obs=True))
        self.train()
        loaded, load = [], storage.load_policy_bundle
        # Params: *args, **kwargs; returns: bundle.
        def track(*args, **kwargs):
            bundle = load(*args, **kwargs); loaded.append(bundle); return bundle
        with patch.object(storage, 'load_policy_bundle', side_effect=track), \
             patch.object(evaluation, 'evaluate_policy', side_effect=RuntimeError('failed normalized testing')):
            with self.assertRaisesRegex(RuntimeError, 'normalized testing'): self.evaluate()
        self.assertEqual(len(loaded), 1); self.assertTrue(loaded[0].closed)
        self.assertIsNotNone(loaded[0].normalization_env)

    # Params: self; returns: None.
    def test_whole_final_phase_preserves_python_numpy_and_torch_rng(self):
        self.train()
        python = random.getstate(); numpy = np.random.get_state(); tensor = torch.random.get_rng_state().clone()
        self.evaluate()
        self.assertEqual(random.getstate(), python)
        current = np.random.get_state()
        self.assertEqual(current[0], numpy[0]); np.testing.assert_array_equal(current[1], numpy[1])
        self.assertEqual(current[2:], numpy[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), tensor))


if __name__ == "__main__":
    unittest.main()
