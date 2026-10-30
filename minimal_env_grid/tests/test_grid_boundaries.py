"""Boundary regressions and bounded spawned-worker failure/learning probes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from agents.catalogue import CatalogueState
from ssa_env.core.grid_geometry import point_to_grid_cell, wrap_position_float32
from ssa_env.core.sensor import Sensor
from ssa_env.core.target import Target
from ssa_env.envs.minimal_ssa_env_RL import MinimalSSAEnv


class ToroidalBoundaries(unittest.TestCase):
    # Params: self; returns: None.
    def test_tiny_negative_rounding_reproducer_stays_in_last_cell(self):
        point = np.array([-1e-7, -1e-7], np.float32)
        self.assertTrue(np.all(point % 100. == 100.))  # Original crash mechanism.
        catalogue = CatalogueState(1)
        wrapped = catalogue._wrap_pos(point)
        self.assertEqual(wrapped.dtype, np.float32)
        self.assertTrue(np.all(wrapped >= 0)); self.assertTrue(np.all(wrapped < 100.))
        catalogue.create_track(0, wrapped, [0, 0], 0)
        obs = catalogue.build_obs_track()
        self.assertGreater(obs[9, 9], 0)
        self.assertEqual(np.count_nonzero(obs), 1)
        np.testing.assert_array_equal(point, np.array([-1e-7, -1e-7], np.float32))

    # Params: self; returns: None.
    def test_exact_seams_and_periodic_offsets(self):
        for point, expected in (([0, 0], (0, 0)), ([100, 100], (0, 0)),
                ([-100, 200], (0, 0)), ([-1, -1], (9, 9)), ([101, 199], (0, 9)),
                ([np.nextafter(100., 0.), np.nextafter(0., -1.)], (9, 9)),
                ([np.nextafter(np.float32(10), np.float32(0)), 10], (0, 1))):
            with self.subTest(point=point):
                self.assertEqual(point_to_grid_cell(point, 100., (10, 10)), expected)

    # Params: self; returns: None.
    def test_subnormal_and_float64_remainders_that_round_up(self):
        for value in (-1e-14, -1e-100, float(np.nextafter(np.float32(0), np.float32(-1)))):
            wrapped = wrap_position_float32(np.array([value, value], np.float64), 100.)
            self.assertTrue(np.all(wrapped < 100.))
            self.assertEqual(point_to_grid_cell(wrapped, 100., (10, 10)), (9, 9))

    # Params: self; returns: None.
    def test_rectangular_grids_and_nonrepresentable_world_size(self):
        for size in (100., 100.1):
            for point in ([size, size], [-1e-7, -1e-7], [0, size / 2], [3 * size + 1, -2 * size - 1]):
                wrapped = wrap_position_float32(np.array(point), size)
                self.assertTrue(np.all(wrapped >= 0)); self.assertTrue(np.all(wrapped.astype(np.float64) < size))
                gx, gy = point_to_grid_cell(wrapped, size, (3, 7))
                self.assertTrue(0 <= gx < 3 and 0 <= gy < 7)
        self.assertEqual(point_to_grid_cell([-1e-7, -1e-7], 100., (3, 7)), (2, 6))

    # Params: self; returns: None.
    def test_ordinary_wrapping_preserves_legacy_arithmetic_and_dtype(self):
        rng = np.random.default_rng(512)
        for dtype in (np.float32, np.float64):
            for point in rng.uniform(-300, 300, (200, 2)).astype(dtype):
                expected = np.asarray(point % 100., dtype=np.float32)
                np.testing.assert_array_equal(wrap_position_float32(point, 100.), expected)
        point = np.array([101.234567890123, -101.234567890123], np.float64)
        np.testing.assert_array_equal(wrap_position_float32(point, 100.), (point % 100.).astype(np.float32))

    # Params: self; returns: None.
    def test_invalid_values_fail_explicitly(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    point_to_grid_cell([bad, 1], 100., (10, 10))
                with self.assertRaisesRegex(ValueError, 'finite'):
                    wrap_position_float32([bad, 1], 100.)

    # Params: self; returns: None.
    def test_catalogue_all_creation_paths_and_stale_unwrapped_tracks(self):
        catalogue = CatalogueState(3)
        catalogue.maybe_create_track_from_measurement(0, np.array([-1e-7, -1e-7], np.float32))
        catalogue.create_track(1, [-1e-7, -1e-7], [2, 3], 0)
        catalogue.create_track(2, [20, 30], [4, 5], 0)
        catalogue.tracks[2].x[:2] = [100, 100]  # Old/checkpoint state also maps safely.
        obs = catalogue.build_obs_track()
        self.assertGreater(obs[0, 0], 0); self.assertGreater(obs[9, 9], 0)
        self.assertEqual(np.count_nonzero(obs), 2)
        np.testing.assert_array_equal(catalogue.tracks[1].x[2:], [2, 3])

    # Params: self; returns: None.
    def test_target_and_measurement_wrap_do_not_store_the_upper_endpoint(self):
        target = Target(0, 0, 0, -1e-7, -1e-7)
        target.propagate(1., 100., 0.)
        self.assertTrue(np.all(target.position < 100.))
        self.assertEqual(point_to_grid_cell(target.position, 100., (10, 10)), (9, 9))
        sensor = Sensor(0, 100.)
        point = np.asarray(sensor._wrap_pos(np.float32(-1e-7), np.float32(-1e-7)), np.float32)
        self.assertTrue(np.all(point < 100.))
        sensor.set_pointing(100, 100)
        self.assertTrue(sensor.is_in_fov(Target(1, 5, 5, 0, 0)))
        self.assertFalse(sensor.is_in_fov(Target(2, 55, 55, 0, 0)))

    # Params: self; returns: None.
    def test_raw_combined_observation_and_reward_mapping_at_seams(self):
        env = MinimalSSAEnv(mode='combined', known_targets=1, unknown_targets=1,
                            grid_shape=(10, 10), reward_type='R6', search_reward='R2')
        self.addCleanup(env.close)
        env.reset(seed=9)
        env.catalogue.tracks[0].x[:2] = [-1e-7, 100]
        obs = env._build_observation([])
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(np.isfinite(obs).all())
        self.assertGreater(obs[9, 0, 0], 0)
        self.assertEqual(env._point_to_grid_cell([-1e-7, 100]), (9, 0))


# Params: kind; returns: None.
def probe(kind):
    import gymnasium as gym
    import torch
    from agents.mo.envs import TrainingSubprocVecEnv, worker_thread_environment, set_worker_threads, make_scalar_env
    from agents.mo.types import EnvConfig
    torch.set_num_threads(1)

    class Crash(gym.Env):
        observation_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        # Params: self, rank; returns: None.
        def __init__(self, rank):
            self.rank = rank
        # Params: self, seed, options; returns: (array, dict).
        def reset(self, *, seed=None, options=None):
            return np.zeros(1, np.float32), {}
        # Params: self, action; returns: (array, float, bool, bool, dict).
        def step(self, action):
            if self.rank == 1:
                time.sleep(.1)  # Healthy rank 0's response is consumed first.
                raise ValueError('injected simulation failure')
            return np.zeros(1, np.float32), 0., False, False, {}

    class Boundary(gym.Wrapper):
        # Params: self, env; returns: None.
        def __init__(self, env):
            super().__init__(env)
            self.boundary_checks = 0
        # Params: self, action; returns: self.env.step result.
        def step(self, action):
            raw = self.unwrapped
            raw.catalogue.tracks[0].x[:2] = [-1e-7, -1e-7]
            raw.catalogue._wrap_track_position(raw.catalogue.tracks[0])
            assert raw.observation_space.contains(raw._build_observation([]))
            self.boundary_checks += 1
            return self.env.step(action)

    # Params: rank; returns: construct.
    def factory(rank):
        # Params: none; returns: Crash or Boundary.
        def construct():
            set_worker_threads(1)
            if kind == 'crash':
                return Crash(rank)
            return Boundary(make_scalar_env(EnvConfig(known_targets=1, unknown_targets=1,
                episode_steps=3, slew_alpha=0.), [0.5, 0.5], 9 + rank, [1., 1.]))
        return construct

    with worker_thread_environment(1):
        vec = TrainingSubprocVecEnv([factory(0), factory(1)], start_method='spawn')
    try:
        vec.reset()
        if kind == 'crash':
            started = time.monotonic()
            try:
                vec.step(np.zeros(2, dtype=np.int64))
            except (EOFError, OSError):
                pass
            else:
                raise AssertionError('Worker failure was not propagated')
            assert vec.closed and not vec.waiting
            assert all(not item.is_alive() for item in vec.processes)
            assert all(remote.closed for remote in vec.remotes)
            print(json.dumps({'probe': kind, 'closed': True, 'failure_cleanup_seconds': time.monotonic() - started}), flush=True)
        else:
            from stable_baselines3 import PPO
            from agents.RL.grid_cnn_extractor import GridCNNExtractor
            model = PPO('MlpPolicy', vec, n_steps=2, batch_size=4, n_epochs=1, device='cpu', verbose=0,
                policy_kwargs={'features_extractor_class': GridCNNExtractor,
                               'features_extractor_kwargs': {'features_dim': 128}})
            model.learn(total_timesteps=8)
            assert model.num_timesteps == 8
            assert isinstance(model.policy.features_extractor, GridCNNExtractor)
            assert vec.get_attr('boundary_checks') == [4, 4]
            print(json.dumps({'probe': kind, 'transitions': 8, 'extractor': 'GridCNNExtractor',
                              'boundary_checks': [4, 4]}), flush=True)
    finally:
        vec.close()
        vec.close()
    assert all(not item.is_alive() for item in vec.processes)


class SpawnBoundaries(unittest.TestCase):
    # Params: self, kind; returns: probe-result dict.
    def run_probe(self, kind):
        import psutil
        env = dict(os.environ, PYTHONPATH=str(PROJECT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--probe', kind],
            cwd=str(PROJECT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = process.communicate(timeout=45)
            self.assertEqual(process.returncode, 0, out + err)
            result = json.loads(out.strip().splitlines()[-1])
            self.assertEqual(result['probe'], kind)
            return result
        finally:
            if process.poll() is None:
                tree = psutil.Process(process.pid).children(recursive=True) + [psutil.Process(process.pid)]
                for member in tree:
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(tree, timeout=5)
                process.communicate(timeout=5)

    # Params: self; returns: None.
    def test_worker_exception_closes_after_partially_consumed_step_without_hanging(self):
        self.assertTrue(self.run_probe('crash')['closed'])

    # Params: self; returns: None.
    def test_two_spawn_workers_train_cnn_with_boundary_injection_every_step(self):
        result = self.run_probe('boundary')
        self.assertEqual(result['extractor'], 'GridCNNExtractor')
        self.assertEqual(result['transitions'], 8)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--probe':
        probe(sys.argv[2])
    else:
        unittest.main()
