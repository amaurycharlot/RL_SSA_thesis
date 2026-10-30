"""Parallel plan block 1: shared setup, immutable saved plans and passive loading."""
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import config as planning, storage
from agents.mo.experiment import initialize_experiment, load_experiment
from agents.mo.types import ExperimentConfig, LoggingConfig, PPOConfig


class SharedExperimentSetup(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mo_parallel_setup_')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = ExperimentConfig(output_dir=self.directory / 'experiment',
            run_seeds=(9, 2), n_weights=3, eval_episodes=2, final_episodes=3,
            reference_scaled=(-10., -10.), reference_status='frozen',
            requested_timesteps_per_policy=9, eval_interval=3, checkpoint_interval=5,
            ppo=PPOConfig(n_envs=2, n_steps=2, batch_size=4, n_epochs=1, device='cpu', verbose=0))

    # Params: self, root; returns: dict.
    def snapshot(self, root):
        return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(),
                    p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}

    # Params: self; returns: None.
    def test_initialization_matches_established_storage_and_identity_format(self):
        weights = planning.make_weights(self.config.n_weights)
        plans = tuple(planning.make_seed_plan(self.config, i) for i in range(2))
        budget = planning.make_budget_plan(self.config)
        expected = storage.experiment_fingerprint(self.config, weights, plans, budget)
        legacy = storage.create_experiment_dir(replace(self.config, output_dir=self.directory / 'direct'),
                                               weights, plans, budget)
        initialized = initialize_experiment(self.config)
        self.assertEqual(initialized.root, self.config.output_dir.resolve())
        self.assertEqual(initialized.config, self.config)
        self.assertEqual(initialized.fingerprint, expected)
        self.assertEqual(initialized.seed_plans, plans)
        self.assertEqual(initialized.budget_plan, budget)
        np.testing.assert_array_equal(initialized.weights, weights)
        for name in ('weights.npy', 'seed_plan.json', 'budget_plan.json'):
            self.assertEqual((legacy / name).read_bytes(), (initialized.root / name).read_bytes())
        direct = storage._read_json(legacy / 'manifest.json')
        saved = storage._read_json(initialized.root / 'manifest.json')
        self.assertEqual(direct['identity'], saved['identity'])
        self.assertEqual(direct['fingerprint'], saved['fingerprint'])
        self.assertEqual(set(saved['initial_file_hashes']),
                         {'config.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json'})

    def test_load_uses_complete_saved_grid_without_regeneration_or_writes(self):
        # Save a valid non-generated preference order and a distinct learner seed.
        weights = planning.make_weights(3)[::-1].copy()
        plans = tuple(replace(planning.make_seed_plan(self.config, i), learner_seed=111 + 10 * i,
                          base_worker_seed=111 + 10 * i, worker_seeds=(111 + 10 * i, 112 + 10 * i))
                      for i in range(2))
        budget = planning.make_budget_plan(self.config)
        root = storage.create_experiment_dir(self.config, weights, plans, budget)
        before = self.snapshot(root)
        with patch.object(planning, 'make_weights', side_effect=AssertionError('regenerated weights')), \
             patch.object(planning, 'make_seed_plan', side_effect=AssertionError('regenerated seeds')), \
             patch.object(planning, 'make_budget_plan', side_effect=AssertionError('regenerated budgets')), \
             patch.object(storage, 'resolve_resume_state', side_effect=AssertionError('reconciled policies')), \
             patch.object(storage, 'load_evaluation_records', side_effect=AssertionError('loaded global results')):
            loaded = load_experiment(root)
        self.assertEqual(loaded.config, self.config)
        self.assertEqual(loaded.seed_plans, plans)
        self.assertEqual(loaded.budget_plan, budget)
        self.assertEqual(loaded.weights.tobytes(), weights.tobytes())
        self.assertEqual(loaded.weights.dtype, np.float32)
        self.assertEqual(loaded.fingerprint, storage.experiment_fingerprint(self.config, weights, plans, budget))
        self.assertEqual(self.snapshot(root), before)

    # Params: self; returns: None.
    def test_preferences_and_context_are_immutable_for_both_helpers(self):
        initialized = initialize_experiment(self.config)
        for context in (initialized, load_experiment(initialized.root)):
            with self.assertRaises(ValueError):
                context.weights.setflags(write=True)
            with self.assertRaises(ValueError):
                context.weights[0, 0] = 1.
            with self.assertRaises(FrozenInstanceError):
                context.fingerprint = 'different'

    # Params: self; returns: None.
    def test_explicit_destination_does_not_mutate_caller_config(self):
        destination = self.directory / 'explicit'
        initialized = initialize_experiment(self.config, destination)
        self.assertEqual(initialized.root, destination.resolve())
        self.assertEqual(initialized.config, replace(self.config, output_dir=destination))
        self.assertFalse(self.config.output_dir.exists())

    # Params: self; returns: None.
    def test_default_destination_and_saved_logging_survive_loading(self):
        previous = Path.cwd()
        self.addCleanup(__import__('os').chdir, previous)
        __import__('os').chdir(self.directory)
        config = replace(self.config, output_dir=None,
            logging=LoggingConfig(mode='offline', project='owner-project', entity='owner-entity'))
        initialized = initialize_experiment(config)
        self.assertEqual(initialized.root, self.directory / 'runs_mo' / config.experiment_id)
        loaded = load_experiment(initialized.root)
        self.assertEqual(loaded.config, replace(config, output_dir=initialized.root))

    # Params: self; returns: None.
    def test_relocated_experiment_keeps_fingerprint_and_saved_plans(self):
        initialized = initialize_experiment(self.config)
        moved = self.directory / 'copy'
        shutil.copytree(initialized.root, moved)
        before = self.snapshot(moved)
        loaded = load_experiment(moved)
        self.assertEqual(loaded.root, moved.resolve())
        self.assertEqual(loaded.config.output_dir, moved.resolve())
        self.assertEqual(loaded.fingerprint, initialized.fingerprint)
        self.assertEqual(loaded.seed_plans, initialized.seed_plans)
        self.assertEqual(loaded.budget_plan, initialized.budget_plan)
        self.assertEqual(self.snapshot(moved), before)

    # Params: self; returns: None.
    def test_nonempty_or_unfrozen_initialization_does_not_overwrite(self):
        initialized = initialize_experiment(self.config)
        before = self.snapshot(initialized.root)
        with self.assertRaises(storage.StorageError):
            initialize_experiment(self.config)
        self.assertEqual(self.snapshot(initialized.root), before)
        destination = self.directory / 'unfrozen'
        with self.assertRaises(planning.ConfigError):
            initialize_experiment(replace(self.config, reference_scaled=None,
                reference_status='unselected'), destination)
        self.assertFalse(destination.exists())

    # Params: self; returns: None.
    def test_load_rejects_changed_science_and_provenance_without_writes(self):
        initialized = initialize_experiment(self.config)
        before = self.snapshot(initialized.root)
        changed = replace(self.config, objective_scales=(2., 1.), scale_status='frozen')
        with self.assertRaises(storage.StorageError):
            load_experiment(initialized.root, config=changed)
        for function, value in (('_source_hashes', {'changed.py': 'different'}),
                                ('_versions', {'python': 'different'})):
            with self.subTest(provenance=function), patch.object(storage, function, return_value=value):
                with self.assertRaises(storage.StorageError):
                    load_experiment(initialized.root)
        self.assertEqual(self.snapshot(initialized.root), before)

    # Params: self; returns: None.
    def test_supplied_output_and_logging_overrides_keep_saved_identity(self):
        initialized = initialize_experiment(self.config)
        changed = replace(self.config, output_dir=self.directory / 'unused',
            logging=LoggingConfig(mode='offline', project='different', entity='different'))
        loaded = load_experiment(initialized.root, config=changed)
        self.assertEqual(loaded.config, changed)
        self.assertEqual(loaded.root, initialized.root)
        self.assertEqual(loaded.fingerprint, initialized.fingerprint)
        self.assertFalse(changed.output_dir.exists())

    # Params: self; returns: None.
    def test_load_rejects_corrupt_initial_files_and_manifest(self):
        initialized = initialize_experiment(self.config)
        for name in ('config.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json', 'manifest.json'):
            with self.subTest(file=name):
                copied = self.directory / name.replace('.', '_')
                shutil.copytree(initialized.root, copied)
                path = copied / name
                path.write_bytes(b'{}' if name.endswith('.json') else b'injected corruption')
                before = self.snapshot(copied)
                with self.assertRaises(storage.StorageError):
                    load_experiment(copied)
                self.assertEqual(self.snapshot(copied), before)

    # Params: self; returns: None.
    def test_load_never_creates_missing_experiment_or_accepts_initializing_marker(self):
        absent = self.directory / 'absent'
        with self.assertRaises(storage.StorageError):
            load_experiment(absent)
        self.assertFalse(absent.exists())
        initialized = initialize_experiment(self.config)
        (initialized.root / '.INITIALIZING').touch()
        before = self.snapshot(initialized.root)
        with self.assertRaises(storage.StorageError):
            load_experiment(initialized.root)
        self.assertEqual(self.snapshot(initialized.root), before)

    # Params: self; returns: None.
    def test_import_and_helpers_do_not_start_training_logging_or_final_evaluation(self):
        script = '''import sys
from unittest.mock import patch
with patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir during import')):
    from agents.mo.experiment import initialize_experiment, load_experiment
assert 'torch' not in sys.modules
assert 'stable_baselines3' not in sys.modules
assert 'agents.mo.trainer' not in sys.modules
'''
        completed = subprocess.run([sys.executable, '-B', '-c', script], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with patch('agents.mo.trainer.train_one_policy', side_effect=AssertionError('training')), \
             patch('agents.mo.reporting.make_logger', side_effect=AssertionError('logger')), \
             patch('agents.mo.evaluation.evaluate_final_policy_set', side_effect=AssertionError('final evaluation')):
            initialized = initialize_experiment(self.config)
            load_experiment(initialized.root)
        self.assertEqual(len(list(initialized.root.rglob('COMPLETE.json'))), 0)
        self.assertFalse(any((initialized.root / 'logs').iterdir()))


class PolicyTaskMapping(unittest.TestCase):
    """Parallel plan block 2: stable full-grid task identities, without training."""

    # Params: self; returns: None.
    def test_seed_major_order_covers_every_declared_slot_exactly_once(self):
        from agents.mo.experiment import task_to_policy
        config = ExperimentConfig(run_seeds=(9, 2), n_weights=3)
        slots = [task_to_policy(task, config) for task in range(6)]
        self.assertEqual(slots, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        self.assertEqual(len(set(slots)), 6)
        self.assertEqual([config.run_seeds[seed] for seed, _ in slots], [9, 9, 9, 2, 2, 2])

    # Params: self; returns: None.
    def test_standard_grid_has_110_unique_slots_and_stable_row_boundaries(self):
        from agents.mo.experiment import task_to_policy
        config = ExperimentConfig()
        self.assertEqual((len(config.run_seeds), config.n_weights), (10, 11))
        slots = [task_to_policy(task, config) for task in range(110)]
        self.assertEqual(set(slots), {(seed, weight) for seed in range(10) for weight in range(11)})
        self.assertEqual(len(set(slots)), len(slots))
        for task, expected in ((0, (0, 0)), (10, (0, 10)), (11, (1, 0)), (109, (9, 10))):
            with self.subTest(task=task):
                self.assertEqual(slots[task], expected)

    # Params: self; returns: None.
    def test_one_replicate_and_smallest_supported_preference_grid(self):
        from agents.mo.experiment import task_to_policy
        config = ExperimentConfig(run_seeds=(81,), n_weights=2)
        self.assertEqual([task_to_policy(task, config) for task in range(2)], [(0, 0), (0, 1)])

    # Params: self; returns: None.
    def test_task_bounds_reject_negative_end_and_large_ids(self):
        from agents.mo.experiment import task_to_policy
        config = ExperimentConfig(run_seeds=(9, 2), n_weights=3)
        for task in (-1, -100, 6, 7, 2**100, np.int64(-1), np.int64(6)):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, r'\[0, 6\)'):
                task_to_policy(task, config)

    # Params: self; returns: None.
    def test_task_ids_reject_booleans_nonintegral_and_container_values(self):
        from agents.mo.experiment import task_to_policy
        config = ExperimentConfig(run_seeds=(9, 2), n_weights=3)
        for task in (True, False, np.bool_(True), '1', 1., np.float64(1.), None, [1], np.array([1])):
            with self.subTest(task=repr(task)), self.assertRaises(TypeError):
                task_to_policy(task, config)
        for task in (np.int32(3), np.int64(3)):
            self.assertEqual(task_to_policy(task, config), (1, 0))
            self.assertTrue(all(type(index) is int for index in task_to_policy(task, config)))

    # Params: self; returns: None.
    def test_mapping_requires_a_valid_full_experiment_config(self):
        from agents.mo.experiment import task_to_policy
        for config in (None, {}, {'n_weights': 3, 'run_seeds': (9, 2)}, 3):
            with self.subTest(config=config), self.assertRaises(TypeError):
                task_to_policy(0, config)

    # Params: self; returns: None.
    def test_saved_reordered_preferences_and_noncontiguous_seeds_keep_their_identity(self):
        from agents.mo.experiment import task_to_policy
        with tempfile.TemporaryDirectory(prefix='mo_parallel_mapping_') as temporary:
            config = ExperimentConfig(output_dir=Path(temporary) / 'experiment',
                run_seeds=(9, 2), n_weights=3, reference_scaled=(-10., -10.), reference_status='frozen')
            weights = planning.make_weights(3)[::-1].copy()
            plans = tuple(planning.make_seed_plan(config, index) for index in range(2))
            budget = planning.make_budget_plan(config)
            root = storage.create_experiment_dir(config, weights, plans, budget)
            # Params: none; returns: dict.
            def snapshot():
                return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                        for p in root.rglob('*') if p.is_file()}
            before = snapshot()
            loaded = load_experiment(root)
            full_weights, full_plans = loaded.weights, loaded.seed_plans
            identities = []
            for task in range(6):
                seed_index, weight_index = task_to_policy(task, loaded.config)
                plan = loaded.seed_plans[seed_index]
                identities.append((plan.run_index, plan.run_seed, weight_index,
                                   tuple(loaded.weights[weight_index].tolist())))
                self.assertIs(loaded.weights, full_weights)
                self.assertIs(loaded.seed_plans, full_plans)
                self.assertEqual(plan, plans[seed_index])
                self.assertEqual(loaded.budget_plan, budget)
            self.assertEqual(identities, [
                (0, 9, 0, (1., 0.)), (0, 9, 1, (.5, .5)), (0, 9, 2, (0., 1.)),
                (1, 2, 0, (1., 0.)), (1, 2, 1, (.5, .5)), (1, 2, 2, (0., 1.))])
            self.assertEqual(loaded.config, config)
            self.assertEqual(loaded.weights.shape, (3, 2))
            self.assertEqual(len(loaded.seed_plans), 2)
            self.assertFalse(loaded.weights.flags.writeable)
            self.assertEqual(snapshot(), before)

    # Params: self; returns: None.
    def test_mapping_does_not_import_training_or_touch_files(self):
        script = '''import sys
from unittest.mock import patch
from agents.mo.experiment import task_to_policy
from agents.mo.types import ExperimentConfig
with patch('pathlib.Path.open', side_effect=AssertionError('opened file')), \\
     patch('pathlib.Path.mkdir', side_effect=AssertionError('created directory')):
    assert task_to_policy(109, ExperimentConfig()) == (9, 10)
assert 'torch' not in sys.modules
assert 'stable_baselines3' not in sys.modules
assert 'agents.mo.trainer' not in sys.modules
'''
        completed = subprocess.run([sys.executable, '-B', '-c', script], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
