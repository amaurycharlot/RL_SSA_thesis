"""Parallel block 3: process-level policy ownership and training lifetime checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import ownership, storage
from agents.mo.experiment import initialize_experiment
from agents.mo.types import ExperimentConfig


CHILD = '''import json, os, sys
from agents.mo.experiment import load_experiment
from agents.mo.ownership import claim_policy_slot
from agents.mo.storage import StorageError
saved = load_experiment(sys.argv[1])
try:
    with claim_policy_slot(saved.root, saved.config, 0, 0) as claim:
        print(json.dumps(dict(claim.metadata)), flush=True)
        if sys.argv[2] == 'crash':
            os._exit(23)
        sys.stdin.readline()
except StorageError as error:
    print(str(error), file=sys.stderr)
    sys.exit(19)
'''


class PolicyOwnership(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mo_policy_claim_')
        self.addCleanup(temporary.cleanup)
        self.saved = initialize_experiment(ExperimentConfig(output_dir=Path(temporary.name) / 'experiment',
            run_seeds=(9, 2), n_weights=2, reference_scaled=(-10., -10.), reference_status='frozen'))
        self.root, self.config = self.saved.root, self.saved.config
        self.policy = self.root / 'models/seed_0/w_0'
        self.directory = self.policy / 'OWNER.LOCK'

    # Params: self, seed, weight; returns: ownership.claim_policy_slot result.
    def claim(self, seed=0, weight=0):
        return ownership.claim_policy_slot(self.root, self.config, seed, weight)

    # Params: self; returns: json.loads result.
    def crashed_owner(self):
        result = subprocess.run([sys.executable, '-B', '-c', CHILD, str(self.root), 'crash'],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 23, result.stderr)
        return json.loads(result.stdout)

    # Params: self, token, **kwargs; returns: ownership.recover_policy_claim result.
    def recover(self, token, **kwargs):
        return ownership.recover_policy_claim(self.root, self.config, 0, 0,
            expected_token=token, owner_stopped=True, reason='Verified test child has exited', **kwargs)

    # Params: self; returns: None.
    def test_owner_records_slot_host_process_and_scheduler_identity_and_releases(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123_2', 'SLURM_ARRAY_JOB_ID': '123',
                 'SLURM_ARRAY_TASK_ID': '2', 'SLURM_RESTART_COUNT': '1'}):
            with self.claim() as claim:
                record = storage._read_json(claim.directory / (claim.token + '.json'))
                self.assertEqual(record, dict(claim.metadata))
                self.assertEqual(record['host'], socket.gethostname())
                self.assertEqual(record['pid'], os.getpid())
                self.assertEqual(record['experiment_fingerprint'], self.saved.fingerprint)
                self.assertEqual((record['seed_index'], record['run_seed'], record['weight_index']), (0, 9, 0))
                self.assertEqual((record['slurm_job_id'], record['slurm_array_job_id'],
                    record['slurm_array_task_id'], record['slurm_restart_count']), ('123_2', '123', '2', '1'))
                self.assertTrue(record['created_at'])
                with self.assertRaises(TypeError):
                    claim.metadata['pid'] = -1
        self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_duplicate_claim_is_refused_and_other_slots_are_independent(self):
        with self.claim() as first:
            before = (first.directory / (first.token + '.json')).read_bytes()
            with self.assertRaisesRegex(storage.StorageError, 'already claimed'):
                with self.claim():
                    self.fail('Duplicate acquired')
            with self.claim(0, 1), self.claim(1, 0):
                self.assertEqual((first.directory / (first.token + '.json')).read_bytes(), before)
        self.assertFalse(list((self.root / 'models').rglob('OWNER.LOCK')))

    # Params: self; returns: None.
    def test_exceptions_and_keyboard_interrupt_release_and_allow_retry(self):
        for error in (OSError('normal failure'), KeyboardInterrupt('cancelled')):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                with self.claim():
                    raise error
            self.assertFalse(self.directory.exists())
            with self.claim():
                pass

    # Params: self; returns: None.
    def test_failed_metadata_flush_releases_partial_claim(self):
        with patch.object(ownership.os, 'fsync', side_effect=OSError('metadata flush failed')):
            with self.assertRaisesRegex(OSError, 'metadata flush failed'):
                with self.claim():
                    self.fail('Uncommitted claim yielded')
        self.assertFalse(self.directory.exists())
        with self.claim():
            pass

    # Params: self; returns: None.
    def test_invalid_indices_and_provenance_fail_before_claim_creation(self):
        for seed, weight in ((True, 0), (0, False), (-1, 0), (2, 0), (0, 2)):
            with self.subTest(slot=(seed, weight)), self.assertRaises(storage.StorageError):
                with self.claim(seed, weight):
                    self.fail('Invalid slot claimed')
        with patch.object(storage, '_source_hashes', return_value={'changed.py': 'different'}):
            with self.assertRaises(storage.StorageError):
                with self.claim():
                    self.fail('Changed provenance claimed')
        self.assertFalse(any((self.root / 'models').iterdir()))

    # Params: self; returns: None.
    def test_separate_process_rejects_duplicate_while_owner_is_live(self):
        with self.claim() as claim:
            result = subprocess.run([sys.executable, '-B', '-c', CHILD, str(self.root), 'hold'],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
            self.assertIn('already claimed', result.stderr)
            self.assertTrue((self.directory / (claim.token + '.json')).is_file())

    # Params: self; returns: None.
    def test_hard_exit_leaves_claim_until_exact_explicit_recovery(self):
        old = self.crashed_owner()
        token = old['token']
        before = (self.directory / (token + '.json')).read_bytes()
        with self.assertRaises(storage.StorageError):
            with self.claim():
                self.fail('Automatically stole stale claim')
        with self.assertRaisesRegex(storage.StorageError, 'owner_stopped'):
            ownership.recover_policy_claim(self.root, self.config, 0, 0,
                expected_token=token, reason='Missing stop verification')
        with self.assertRaisesRegex(storage.StorageError, 'Expected owner token'):
            self.recover('0' * 32)
        self.assertEqual((self.directory / (token + '.json')).read_bytes(), before)
        audit = storage._read_json(self.recover(token))
        self.assertEqual(audit['owner_metadata'], old)
        self.assertEqual(audit['expected_token'], token)
        self.assertTrue(audit['owner_stopped'])
        self.assertEqual(audit['reason'], 'Verified test child has exited')
        self.assertFalse(self.directory.exists())
        with self.claim() as replacement:
            self.assertNotEqual(replacement.token, token)

    # Params: self; returns: None.
    def test_delayed_former_owner_and_recovery_cannot_remove_successor(self):
        token = self.crashed_owner()['token']
        self.recover(token)
        with self.claim() as successor:
            before = (self.directory / (successor.token + '.json')).read_bytes()
            with self.assertRaises(storage.StorageError):
                ownership._release(self.directory, token)
            with self.assertRaises(storage.StorageError):
                self.recover(token)
            self.assertEqual((self.directory / (successor.token + '.json')).read_bytes(), before)

    # Params: self; returns: None.
    def test_current_process_claim_cannot_be_recovered(self):
        with self.claim() as claim:
            with self.assertRaisesRegex(storage.StorageError, 'current process'):
                self.recover(claim.token)
            self.assertTrue((self.directory / (claim.token + '.json')).exists())

    # Params: self; returns: None.
    def test_recovery_requires_reason_safe_token_and_stop_confirmation(self):
        for reason, token in (('', '0' * 32), (' ', '0' * 32), ('test', '../outside'), ('test', 1)):
            with self.subTest(reason=reason, token=token), self.assertRaises(ValueError):
                ownership.recover_policy_claim(self.root, self.config, 0, 0,
                    expected_token=token, owner_stopped=True, reason=reason)

    # Params: self; returns: None.
    def test_interrupted_empty_or_partial_metadata_requires_explicit_recovery(self):
        self.directory.mkdir(parents=True)
        with self.assertRaises(storage.StorageError):
            with self.claim():
                self.fail('Stole empty interrupted claim')
        with self.assertRaisesRegex(ValueError, 'tokenless recovery'):
            self.recover(None)
        self.assertTrue(self.directory.is_dir())
        # Simulate an operator removing a verified empty directory offline.
        self.directory.rmdir()
        self.directory.mkdir()
        token = 'a' * 32
        (self.directory / (token + '.json')).write_bytes(b'{"pid":')
        with self.assertRaisesRegex(ValueError, 'tokenless recovery'):
            self.recover(None)
        audit = self.recover(token)
        self.assertIsNone(storage._read_json(audit)['owner_metadata'])
        self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_failed_recovery_audit_keeps_stale_claim_intact(self):
        old = self.crashed_owner()
        owner = self.directory / (old['token'] + '.json')
        before = owner.read_bytes()
        with patch.object(storage, 'atomic_write_json', side_effect=OSError('audit unavailable')):
            with self.assertRaisesRegex(OSError, 'audit unavailable'):
                self.recover(old['token'])
        self.assertEqual(owner.read_bytes(), before)
        self.recover(old['token'])

    # Params: self; returns: None.
    def test_import_is_passive(self):
        code = '''import sys
from unittest.mock import patch
with patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir on import')):
    from agents.mo.ownership import claim_policy_slot, recover_policy_claim
assert 'torch' not in sys.modules
assert 'stable_baselines3' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class TrainingClaimLifetime(unittest.TestCase):
    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    # Params: cls; returns: None.
    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.previous_threads)

    # Params: self; returns: None.
    def setUp(self):
        from test_mo_training import PolicyOrchestrationTests
        self.fixture = PolicyOrchestrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.policy / 'OWNER.LOCK'

    # Params: self; returns: record['token'].
    def assert_owned(self):
        records = list(self.directory.glob('*.json'))
        self.assertEqual(len(records), 1)
        record = storage._read_json(records[0])
        self.assertEqual(record['pid'], os.getpid())
        return record['token']

    # Params: self; returns: None.
    def test_one_claim_spans_reconciliation_training_cleanup_and_completion(self):
        from agents.mo import envs
        events, tokens = [], []
        resolve, complete, factory = storage.resolve_resume_state, storage.mark_policy_complete, envs.make_train_vec_env
        # Params: phase; returns: None.
        def check(phase):
            events.append(phase); tokens.append(self.assert_owned())
        # Params: *args, **kwargs; returns: resolve result.
        def reconciliation(*args, **kwargs):
            check('resume'); return resolve(*args, **kwargs)
        # Params: *args, **kwargs; returns: handle.
        def environment(*args, **kwargs):
            check('environment')
            handle = factory(*args, **kwargs)
            close = handle.close
            # Params: none; returns: close result.
            def cleanup():
                check('cleanup'); return close()
            handle.close = cleanup
            return handle
        # Params: *args, **kwargs; returns: complete result.
        def completion(*args, **kwargs):
            check('completion'); return complete(*args, **kwargs)
        with patch.object(storage, 'resolve_resume_state', side_effect=reconciliation), \
             patch.object(envs, 'make_train_vec_env', side_effect=environment), \
             patch.object(storage, 'mark_policy_complete', side_effect=completion):
            summary = self.fixture.run_policy()
        self.assertTrue(summary.complete)
        self.assertEqual(events, ['resume', 'environment', 'cleanup', 'completion'])
        self.assertEqual(len(set(tokens)), 1)
        self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_duplicate_training_is_rejected_before_reconciliation_or_workers(self):
        from agents.mo import envs, trainer
        with ownership.claim_policy_slot(self.fixture.root, self.fixture.config, 0, 0):
            with patch.object(storage, 'resolve_resume_state') as resume, \
                 patch.object(envs, 'make_train_vec_env') as workers, patch.object(trainer, 'build_model') as model:
                with self.assertRaisesRegex(storage.StorageError, 'already claimed'):
                    self.fixture.run_policy()
                resume.assert_not_called(); workers.assert_not_called(); model.assert_not_called()
        self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_completed_skip_is_claimed_and_releases_without_model_allocation(self):
        from agents.mo import envs, trainer
        self.fixture.run_policy()
        resume = storage.resolve_resume_state
        # Params: *args, **kwargs; returns: resume result.
        def check(*args, **kwargs):
            self.assert_owned(); return resume(*args, **kwargs)
        with patch.object(storage, 'resolve_resume_state', side_effect=check), \
             patch.object(envs, 'make_train_vec_env') as workers, patch.object(trainer, 'build_model') as model:
            summary = self.fixture.run_policy()
        self.assertTrue(summary.resume_metadata['skipped_complete'])
        workers.assert_not_called(); model.assert_not_called()
        self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_failures_at_each_policy_phase_release_the_claim(self):
        from test_mo_training import PolicyOrchestrationTests
        for target in ('agents.mo.storage.resolve_resume_state', 'agents.mo.envs.make_train_vec_env',
                       'agents.mo.trainer.build_model', 'agents.mo.trainer.learn_one_update',
                       'agents.mo.evaluation.evaluate_policy', 'agents.mo.storage.mark_policy_complete'):
            with self.subTest(phase=target):
                fixture = PolicyOrchestrationTests(); fixture.setUp()
                try:
                    with patch(target, side_effect=OSError('injected policy phase failure')):
                        with self.assertRaisesRegex(OSError, 'injected policy phase failure'):
                            fixture.run_policy()
                    self.assertFalse((fixture.policy / 'OWNER.LOCK').exists())
                    with ownership.claim_policy_slot(fixture.root, fixture.config, 0, 0):
                        pass
                finally:
                    fixture.doCleanups()

    # Params: self; returns: None.
    def test_sequential_prepass_reconciliation_is_also_claimed(self):
        from agents.mo import trainer
        destination = Path(self.fixture.temporary.name) / 'sequential'
        # Params: root, config, seed, weight; returns: never (raises).
        def stop(root, config, seed, weight):
            path = root / 'models' / f'seed_{seed}' / f'w_{weight}' / 'OWNER.LOCK'
            self.assertEqual(len(list(path.glob('*.json'))), 1)
            raise OSError('stop after owned prepass')
        with patch.object(storage, 'resolve_resume_state', side_effect=stop):
            with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(OSError, 'owned prepass'):
                trainer.run_experiment(self.fixture.config, experiment_dir=destination)
        self.assertFalse(list(destination.rglob('OWNER.LOCK')))


if __name__ == '__main__':
    unittest.main()
