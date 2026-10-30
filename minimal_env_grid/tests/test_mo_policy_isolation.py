"""Parallel block 4: scoped recovery/reads, private logging and concurrent writers."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import storage, trainer, reporting
from agents.mo.experiment import initialize_experiment
from agents.mo.types import ExperimentConfig, EnvConfig, PPOConfig, ExecutionConfig, ExperimentResults


# Params: root; returns: dict.
def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


# Params: root; returns: dict.
def shared_snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file() and (
                p.parent == root or p.parent == root / 'results'
                or root / 'logs' in p.parents or root / 'reports' in p.parents)}


class ScopedPolicyRecords(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from test_mo_reporting import SummaryTableExports
        self.fixture = SummaryTableExports(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.config = self.fixture.root, self.fixture.config

    # Params: self; returns: None.
    def test_scoped_loader_keeps_full_dimensions_but_reads_only_selected_records(self):
        own = self.fixture.record(0, 0, 0)
        self.fixture.record(1, 1, 0)
        before = snapshot(self.root)
        with patch.object(storage, '_atomic_write', side_effect=AssertionError('reader wrote files')):
            selected = storage.load_policy_evaluation_records(self.root, self.config, 0, 0)
        self.assertEqual(selected.validity.shape, (2, 2, 2))
        self.assertEqual(set(selected.identities), {(0, 0, 0)})
        np.testing.assert_array_equal(selected.means_raw[0, 0, 0], own.mean_raw)
        self.assertTrue(np.isnan(selected.means_raw[1]).all())
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_foreign_pending_reconciliation_does_not_block_selected_policy(self):
        storage.resolve_resume_state(self.root, self.config, 1, 1)
        journal = self.root / 'results/evaluations/seed_1/w_1/RESUME.json'
        pending = storage._read_json(journal); pending['status'] = 'pending'
        storage.atomic_write_json(journal, pending)
        foreign = snapshot(journal.parent)
        globals_before = shared_snapshot(self.root)
        with self.assertRaisesRegex(storage.StorageError, 'pending'):
            storage.load_evaluation_records(self.root, self.config)
        with patch.object(storage, 'load_evaluation_records', side_effect=AssertionError('global rebuild')):
            state = storage.resolve_resume_state(self.root, self.config, 0, 0)
            selected = storage.load_policy_evaluation_records(self.root, self.config, 0, 0)
        self.assertEqual(state.current_step, 0)
        self.assertFalse(selected.validity.any())
        self.assertEqual(snapshot(journal.parent), foreign)
        self.assertEqual(shared_snapshot(self.root), globals_before)

    # Params: self; returns: None.
    def test_foreign_corrupt_commit_is_ignored_but_selected_corruption_fails(self):
        self.fixture.record(0, 0, 0); self.fixture.record(1, 1, 0)
        foreign = self.root / 'results/evaluations/seed_1/w_1/step_2.json'
        foreign.write_text('{}')
        before = snapshot(self.root)
        selected = storage.load_policy_evaluation_records(self.root, self.config, 0, 0)
        self.assertEqual(set(selected.identities), {(0, 0, 0)})
        with self.assertRaises((storage.StorageError, ValueError)):
            storage.load_policy_evaluation_records(self.root, self.config, 1, 1)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_invalid_scoped_indices_never_write(self):
        before = snapshot(self.root)
        for seed, weight in ((-1, 0), (2, 0), (0, 2), (True, 0), (0, False)):
            with self.subTest(slot=(seed, weight)), self.assertRaises(storage.StorageError):
                storage.load_policy_evaluation_records(self.root, self.config, seed, weight)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_refresh_clears_only_selected_stale_slots_and_preserves_other_policies(self):
        self.fixture.record(0, 0, 0); self.fixture.record(0, 1, 0); self.fixture.record(1, 1, 1)
        destination = storage.load_evaluation_records(self.root, self.config)
        before_raw, before_scaled = destination.means_raw.copy(), destination.means_scaled.copy()
        before_returns = destination.episode_returns.copy()
        before_ids, before_hashes = dict(destination.identities), dict(destination._evaluation_slot_hashes)
        selected = storage.load_policy_evaluation_records(self.root, self.config, 0, 0)
        selected.validity[0, :, 0] = False
        selected.means_raw[0, :, 0] = np.nan; selected.means_scaled[0, :, 0] = np.nan
        selected.episode_returns[0, :, 0] = np.nan
        selected.identities.clear(); selected.episode_metric_tables.clear(); selected._evaluation_slot_hashes.clear()
        trainer._sync_policy_results(destination, selected, 0, 0)
        self.assertFalse(destination.validity[0, :, 0].any())
        for key in ((0, 0, 1), (1, 1, 1)):
            np.testing.assert_array_equal(destination.means_raw[key], before_raw[key])
            np.testing.assert_array_equal(destination.means_scaled[key], before_scaled[key])
            np.testing.assert_array_equal(destination.episode_returns[key], before_returns[key])
            self.assertEqual(destination.identities[key], before_ids[key])
            self.assertEqual(destination._evaluation_slot_hashes[key], before_hashes[key])
        destination.validate()


class IsolatedPolicyTraining(unittest.TestCase):
    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(1)

    # Params: cls; returns: None.
    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    # Params: self; returns: None.
    def setUp(self):
        from test_mo_training import PolicyOrchestrationTests
        self.fixture = PolicyOrchestrationTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.config = self.fixture.root, self.fixture.config

    # Params: self, **kwargs; returns: self.fixture.run_policy result.
    def run_policy(self, **kwargs):
        return self.fixture.run_policy(result_store=None, logger=None, **kwargs)

    # Params: self; returns: None.
    def test_default_private_logger_and_table_cannot_touch_shared_caches_or_logs(self):
        before = shared_snapshot(self.root)
        loggers = []
        original = reporting.make_logger
        # Params: config, directory; returns: logger.
        def local(config, directory):
            self.assertEqual(directory, self.fixture.policy)
            self.assertTrue((directory / 'OWNER.LOCK').is_dir())
            logger = original(config, directory); loggers.append(logger); return logger
        with patch.object(reporting, 'make_logger', side_effect=local), \
             patch.object(storage, 'load_evaluation_records', side_effect=AssertionError('global read')), \
             patch.object(trainer, '_cache_policy_results', side_effect=AssertionError('shared cache')):
            summary = self.run_policy()
        self.assertTrue(summary.complete); self.assertTrue(loggers[0]._closed)
        self.assertEqual(shared_snapshot(self.root), before)
        events = [storage._read_json(p) for p in sorted((self.fixture.policy / 'logs/events').glob('*.json'))]
        self.assertTrue(events)
        self.assertEqual([e['event_index'] for e in events], list(range(len(events))))
        self.assertTrue(all(e['identity']['weight_index'] == 0 for e in events))
        before_skip = snapshot(self.root)
        with patch.object(trainer, 'build_model', side_effect=AssertionError('completed policy retrained')), \
             patch.object(trainer, '_cache_policy_results', side_effect=AssertionError('shared cache')):
            skipped = self.run_policy()
        self.assertTrue(skipped.resume_metadata['skipped_complete'])
        self.assertEqual(snapshot(self.root), before_skip)

    # Params: self; returns: None.
    def test_owned_logger_closes_on_learning_failure_and_claim_is_released(self):
        loggers, original = [], reporting.make_logger
        # Params: *args; returns: logger.
        def local(*args):
            logger = original(*args); loggers.append(logger); return logger
        with patch.object(reporting, 'make_logger', side_effect=local), \
             patch.object(trainer, 'learn_one_update', side_effect=OSError('learning interrupted')):
            with self.assertRaisesRegex(OSError, 'learning interrupted'):
                self.run_policy()
        self.assertTrue(loggers[0]._closed)
        self.assertFalse((self.fixture.policy / 'OWNER.LOCK').exists())
        self.assertFalse((self.fixture.policy / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_owned_logger_close_failure_cannot_publish_completion(self):
        original = reporting.make_logger
        # Params: *args; returns: logger.
        def local(*args):
            logger = original(*args)
            logger.close = lambda: (_ for _ in ()).throw(OSError('logger close failed'))
            return logger
        with patch.object(reporting, 'make_logger', side_effect=local):
            with self.assertRaisesRegex(OSError, 'logger close failed'):
                self.run_policy()
        self.assertTrue((self.fixture.policy / 'final/COMMITTED').exists())
        self.assertFalse((self.fixture.policy / 'COMPLETE.json').exists())
        self.assertFalse((self.fixture.policy / 'OWNER.LOCK').exists())

    # Params: self; returns: None.
    def test_borrowed_logger_can_disable_caches_and_remains_caller_owned(self):
        before = shared_snapshot(self.root)
        with patch.object(trainer, '_cache_policy_results', side_effect=AssertionError('shared cache')):
            summary = self.fixture.run_policy(cache_results=False)
        self.assertTrue(summary.complete)
        self.assertFalse(self.fixture.logger._closed)
        # Borrowed logger is explicitly supplied; only its caller-owned event path changes.
        before_caches = {k: v for k, v in before.items() if k.startswith('results/')}
        self.assertEqual({k: v for k, v in shared_snapshot(self.root).items() if k.startswith('results/')}, before_caches)

    # Params: self; returns: None.
    def test_invalid_cache_mode_fails_before_allocating_workers_or_claim(self):
        for mode, error in ((True, ValueError), (0, TypeError), ('false', TypeError)):
            with self.subTest(mode=mode), patch('agents.mo.envs.make_train_vec_env') as workers:
                with self.assertRaises(error):
                    self.run_policy(cache_results=mode)
                workers.assert_not_called()
                self.assertFalse((self.fixture.policy / 'OWNER.LOCK').exists())


WORKER = '''import json, sys, time
from pathlib import Path
from unittest.mock import patch
import torch
torch.set_num_threads(1)
from agents.mo import storage, trainer, reporting
from agents.mo.experiment import load_experiment
saved = load_experiment(sys.argv[1])
weight = int(sys.argv[2]); ready, go = Path(sys.argv[3]), Path(sys.argv[4])
policy = saved.root / 'models/seed_0' / ('w_' + str(weight))
allowed = [policy] + [saved.root / 'results' / ns / 'seed_0' / ('w_' + str(weight))
    for ns in ('evaluations', 'resume_history', 'superseded')]
writes, loggers = [], []
atomic, factory = storage._atomic_write, reporting.make_logger
def check_write(path, writer):
    path = Path(path).resolve()
    assert any(parent == path or parent in path.parents for parent in allowed), str(path)
    writes.append(path.relative_to(saved.root).as_posix())
    return atomic(path, writer)
def make_logger(config, destination):
    assert destination == policy
    assert (policy / 'OWNER.LOCK').is_dir()
    logger = factory(config, destination); loggers.append(logger)
    # Both claims must be held together before either policy starts recovery.
    (ready.parent / ('claimed_' + str(weight))).write_text('claimed')
    deadline = time.monotonic() + 30
    while not all((ready.parent / ('claimed_' + str(index))).exists() for index in range(2)):
        if time.monotonic() > deadline: raise TimeoutError('claims did not overlap')
        time.sleep(.02)
    return logger
ready.write_text('ready')
deadline = time.monotonic() + 30
while not go.exists():
    if time.monotonic() > deadline: raise TimeoutError('test barrier expired')
    time.sleep(.02)
with patch.object(storage, '_atomic_write', side_effect=check_write), \\
     patch.object(storage, 'load_evaluation_records', side_effect=AssertionError('global results rebuild')), \\
     patch.object(trainer, '_cache_policy_results', side_effect=AssertionError('shared cache')), \\
     patch.object(reporting, 'make_logger', side_effect=make_logger):
    summary = trainer.train_one_policy(saved.config, saved.root, 0, weight,
        saved.weights, saved.seed_plans[0], saved.budget_plan)
assert summary.complete and summary.actual_transitions == 4
assert all(logger._closed for logger in loggers)
print(json.dumps({'weight_index': weight, 'actual_step': summary.actual_transitions,
    'fingerprint': summary.identity.experiment_fingerprint, 'writes': writes}), flush=True)
'''


class ConcurrentPolicyWrites(unittest.TestCase):
    # Params: self; returns: None.
    def test_two_processes_train_distinct_slots_without_overwriting_shared_artifacts(self):
        with tempfile.TemporaryDirectory(prefix='mo_concurrent_slots_') as temporary:
            harness = Path(temporary)
            saved = initialize_experiment(ExperimentConfig(output_dir=harness / 'experiment',
                run_seeds=(7,), n_weights=2, eval_episodes=1, final_episodes=1,
                reference_scaled=(-10., -10.), reference_status='frozen',
                requested_timesteps_per_policy=4, eval_interval=2, checkpoint_interval=2,
                env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3), episode_steps=2, slew_alpha=0.),
                ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0),
                execution=ExecutionConfig(subproc=False)))
            storage.load_evaluation_records(saved.root, saved.config)
            with reporting.make_logger(saved.config, saved.root) as logger:
                logger.record('global_sentinel', None, {'unchanged': True})
            (saved.root / 'reports/sentinel.txt').write_text('unchanged')
            before = shared_snapshot(saved.root)
            go, processes = harness / 'go', []
            try:
                for weight in range(2):
                    processes.append(subprocess.Popen([sys.executable, '-B', '-c', WORKER, str(saved.root),
                        str(weight), str(harness / ('ready_' + str(weight))), str(go)], cwd=str(PROJECT_ROOT),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                deadline = time.monotonic() + 30
                while not all((harness / ('ready_' + str(weight))).exists() for weight in range(2)):
                    for process in processes:
                        if process.poll() is not None:
                            out, err = process.communicate()
                            self.fail('Worker exited before barrier: ' + out + err)
                    if time.monotonic() > deadline:
                        self.fail('Workers did not reach test barrier')
                    time.sleep(.05)
                go.write_text('start')
                evidence = []
                for process in processes:
                    out, err = process.communicate(timeout=60)
                    self.assertEqual(process.returncode, 0, out + err)
                    evidence.append(json.loads(out.strip().splitlines()[-1]))
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.communicate(timeout=15)
            self.assertEqual(shared_snapshot(saved.root), before)
            self.assertTrue(all((harness / ('claimed_' + str(weight))).exists() for weight in range(2)))
            self.assertEqual({item['weight_index'] for item in evidence}, {0, 1})
            for item in evidence:
                self.assertEqual(item['fingerprint'], saved.fingerprint)
                self.assertTrue(item['writes'])
                weight = item['weight_index']
                policy = saved.root / 'models/seed_0' / ('w_' + str(weight))
                self.assertTrue((policy / 'COMPLETE.json').is_file())
                events = [storage._read_json(path) for path in sorted((policy / 'logs/events').glob('*.json'))]
                self.assertTrue(events)
                self.assertEqual([e['event_index'] for e in events], list(range(len(events))))
                self.assertTrue(all(e['identity']['weight_index'] == weight for e in events))
            self.assertFalse(list(saved.root.rglob('OWNER.LOCK')))
            # Only the separate sequential reader now refreshes the shared caches.
            rebuilt = storage.load_evaluation_records(saved.root, saved.config)
            self.assertTrue(rebuilt.validity.all())
            self.assertTrue(np.isfinite(rebuilt.hypervolume).all())


if __name__ == '__main__':
    unittest.main()
