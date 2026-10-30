"""Parallel block 6: exclusive finalizer, complete inputs and interruption retry."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import storage, ownership, finalization, trainer, reporting, evaluation
from agents.mo.experiment import initialize_experiment
from agents.mo.types import ExperimentConfig, EnvConfig, PPOConfig, ExecutionConfig, ReportManifest
from agents.train import train_ppo_mo as cli


# Params: root; returns: dict.
def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


# Params: root; returns: ExperimentConfig.
def tiny_config(root):
    return ExperimentConfig(output_dir=root, run_seeds=(7,), n_weights=2,
        reference_scaled=(-10., -10.), reference_status='frozen', eval_episodes=1, final_episodes=2,
        requested_timesteps_per_policy=4, eval_interval=2, checkpoint_interval=2,
        env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3), episode_steps=2, slew_alpha=0.),
        ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0),
        execution=ExecutionConfig(subproc=False))


def fixture_report(root, config):
    """Small durable report substitute for failure tests; CLI test uses real plots."""
    path = root / 'reports/report.md'; path.write_text('Finalization test report')
    budgets = tuple(storage._read_json(root / 'budget_plan.json')['eval_steps'])
    manifest = ReportManifest(('reports/report.md',),
        {'reports/report.md': hashlib.sha256(path.read_bytes()).hexdigest()},
        storage._read_json(root / 'manifest.json')['fingerprint'], budgets,
        tuple(1 for _ in budgets), datetime.now(timezone.utc).isoformat())
    storage.atomic_write_json(root / 'reports/report_manifest.json', manifest)
    return manifest


class FinalizerCoordination(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mo_finalize_claim_')
        self.addCleanup(temporary.cleanup)
        self.saved = initialize_experiment(tiny_config(Path(temporary.name) / 'experiment'))
        self.root, self.config = self.saved.root, self.saved.config

    # Params: self; returns: None.
    def test_exclusive_finalizer_blocks_second_finalizer_and_new_workers(self):
        with ownership.claim_finalizer(self.root, self.config) as claim:
            record = storage._read_json(claim.directory / (claim.token + '.json'))
            self.assertEqual(record['kind'], 'finalizer')
            self.assertEqual(record['pid'], os.getpid())
            self.assertEqual(record['experiment_fingerprint'], self.saved.fingerprint)
            before = snapshot(self.root)
            with self.assertRaisesRegex(storage.StorageError, 'already claimed'):
                with ownership.claim_finalizer(self.root, self.config):
                    self.fail('Duplicate finalizer')
            with self.assertRaisesRegex(storage.StorageError, 'finalizer'):
                with ownership.claim_policy_slot(self.root, self.config, 0, 0):
                    self.fail('Worker started during finalization')
            self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.root / 'FINALIZER.LOCK').exists())

    # Params: self; returns: None.
    def test_existing_worker_is_refused_before_any_shared_artifact_changes(self):
        with ownership.claim_policy_slot(self.root, self.config, 0, 0):
            before = snapshot(self.root)
            with self.assertRaisesRegex(storage.StorageError, 'active/interrupted'):
                finalization.finalize_experiment(self.root)
            self.assertEqual(snapshot(self.root), before)
        self.assertFalse((self.root / 'FINALIZER.LOCK').exists())

    # Params: self; returns: None.
    def test_finalizer_starting_during_worker_publication_prevents_worker_yield(self):
        original = ownership._claim_directory
        # Params: directory, metadata, label; returns: context manager.
        @contextmanager
        def interleave(directory, metadata, label):
            with original(directory, metadata, label) as claim:
                if directory.name == 'OWNER.LOCK':
                    with ownership.claim_finalizer(self.root, self.config):
                        yield claim
                else:
                    yield claim
        with patch.object(ownership, '_claim_directory', side_effect=interleave):
            with self.assertRaisesRegex(storage.StorageError, 'during policy claim publication'):
                with ownership.claim_policy_slot(self.root, self.config, 0, 0):
                    self.fail('Worker yielded across finalizer handshake')
        self.assertFalse(list(self.root.rglob('*.LOCK')))

    # Params: self; returns: None.
    def test_incomplete_grid_and_stale_storage_lock_are_refused_read_only(self):
        before = snapshot(self.root)
        with self.assertRaisesRegex(storage.StorageError, 'every policy complete'):
            finalization.finalize_experiment(self.root)
        self.assertEqual(snapshot(self.root), before)
        directory = self.root / 'results/evaluations/seed_0/w_0'
        directory.mkdir(parents=True); (directory / 'RESUME.LOCK').touch()
        before = snapshot(self.root)
        with self.assertRaisesRegex(storage.StorageError, 'active/interrupted'):
            finalization.finalize_experiment(self.root)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_hard_exit_global_claim_requires_explicit_token_recovery(self):
        script = '''import os, sys
from agents.mo.experiment import load_experiment
from agents.mo.ownership import claim_finalizer
saved = load_experiment(sys.argv[1])
with claim_finalizer(saved.root, saved.config) as claim:
    print(claim.token, flush=True)
    os._exit(23)
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, str(self.root)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 23, result.stderr)
        token = result.stdout.strip()
        with self.assertRaises(storage.StorageError):
            with ownership.claim_policy_slot(self.root, self.config, 0, 0):
                self.fail('Stole finalizer ownership')
        with self.assertRaisesRegex(storage.StorageError, 'owner_stopped'):
            ownership.recover_finalizer_claim(self.root, self.config, expected_token=token, reason='Stopped test child')
        with self.assertRaises(storage.StorageError):
            ownership.recover_finalizer_claim(self.root, self.config, expected_token='0' * 32,
                owner_stopped=True, reason='Stopped test child')
        audit = ownership.recover_finalizer_claim(self.root, self.config, expected_token=token,
            owner_stopped=True, reason='Verified child exited')
        self.assertEqual(storage._read_json(audit)['expected_token'], token)
        with ownership.claim_finalizer(self.root, self.config) as successor:
            with self.assertRaises(storage.StorageError):
                ownership._release(successor.directory, token)
            self.assertTrue((successor.directory / (successor.token + '.json')).exists())

    # Params: self; returns: None.
    def test_current_finalizer_cannot_be_recovered(self):
        with ownership.claim_finalizer(self.root, self.config) as claim:
            with self.assertRaisesRegex(storage.StorageError, 'current process'):
                ownership.recover_finalizer_claim(self.root, self.config, expected_token=claim.token,
                    owner_stopped=True, reason='Invalid active recovery')

    # Params: self; returns: None.
    def test_finalize_parser_and_cli_dispatch_without_whole_grid_training(self):
        parsed = cli.parse_args(['--finalize', '--experiment', str(self.root)])
        self.assertTrue(parsed.finalize)
        for args in (['--finalize'], ['--finalize', '--experiment', 'x', '--config', 'x'],
                     ['--finalize', '--experiment', 'x', '--task-id', '0'],
                     ['--finalize', '--experiment', 'x', '--initialize'],
                     ['--finalize', '--experiment', 'x', '--preflight'],
                     ['--finalize', '--experiment', 'x', '--output', 'y']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(args)
        manifest = ReportManifest(('reports/fixture',), {'reports/fixture': '0' * 64},
            self.saved.fingerprint, (4,), (1,), 'fixture')
        out, err = io.StringIO(), io.StringIO()
        with patch.object(cli, '_check_device', return_value='cpu'), \
             patch.object(finalization, 'finalize_experiment', return_value=manifest) as finish, \
             patch.object(trainer, 'run_experiment', side_effect=AssertionError('training')), \
             patch.object(trainer, 'train_one_policy', side_effect=AssertionError('policy training')), \
             redirect_stdout(out), redirect_stderr(err):
            code = cli.main(['--finalize', '--experiment', str(self.root)])
        self.assertEqual(code, 0, err.getvalue())
        finish.assert_called_once_with(self.root)
        result = json.loads(out.getvalue())
        self.assertEqual((result['status'], result['completed_policies']), ('finalized', 2))

    # Params: self; returns: None.
    def test_finalize_failure_and_interrupt_return_nonzero(self):
        for error, expected in ((OSError('finalization failed'), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(error=type(error).__name__), patch.object(cli, '_check_device', return_value='cpu'), \
                 patch.object(finalization, 'finalize_experiment', side_effect=error), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['--finalize', '--experiment', str(self.root)]), expected)

    # Params: self; returns: None.
    def test_finalization_import_is_passive(self):
        script = '''import sys
from unittest.mock import patch
with patch('pathlib.Path.mkdir', side_effect=AssertionError('mkdir at import')):
    from agents.mo.finalization import finalize_experiment
assert 'torch' not in sys.modules
assert 'stable_baselines3' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class FinalizerLifecycle(unittest.TestCase):
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
        temporary = tempfile.TemporaryDirectory(prefix='mo_finalize_complete_')
        self.addCleanup(temporary.cleanup)
        self.saved = initialize_experiment(tiny_config(Path(temporary.name) / 'experiment'))
        self.root, self.config = self.saved.root, self.saved.config
        for weight in range(2):
            trainer.train_one_policy(self.config, self.root, 0, weight, self.saved.weights,
                self.saved.seed_plans[0], self.saved.budget_plan, cache_results=False)
        storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: finalization.finalize_experiment result.
    def finalize(self):
        with patch.object(reporting, 'generate_reports', side_effect=fixture_report):
            return finalization.finalize_experiment(self.root)

    # Params: self; returns: None.
    def assert_unlocked(self):
        self.assertFalse((self.root / 'FINALIZER.LOCK').exists())
        self.assertFalse(list((self.root / 'models').rglob('OWNER.LOCK')))

    # Params: self; returns: None.
    def test_success_publishes_verified_hashes_without_changing_training_inputs(self):
        before_models = snapshot(self.root / 'models')
        before_evaluations = snapshot(self.root / 'results/evaluations')
        with patch.object(trainer, 'train_one_policy', side_effect=AssertionError('retraining')):
            manifest = self.finalize()
        record = storage._read_json(self.root / 'FINALIZED.json')
        self.assertEqual(record['experiment_fingerprint'], self.saved.fingerprint)
        self.assertEqual((record['n_policies'], record['actual_step']), (2, 4))
        self.assertEqual(record['test_seeds'], list(self.saved.seed_plans[0].final_seeds))
        for section in ('canonical_evaluation_hashes', 'policy_commitments', 'derived_file_hashes'):
            self.assertTrue(record[section])
            for relative, digest in record[section].items():
                self.assertEqual(hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest)
        self.assertEqual(manifest.experiment_fingerprint, self.saved.fingerprint)
        self.assertEqual(snapshot(self.root / 'models'), before_models)
        self.assertEqual(snapshot(self.root / 'results/evaluations'), before_evaluations)
        self.assert_unlocked()

    # Params: self; returns: None.
    def test_readonly_full_rebuild_does_not_refresh_caches(self):
        before = snapshot(self.root)
        results = storage.load_evaluation_records(self.root, self.config, write_cache=False)
        self.assertTrue(results.validity.all()); self.assertTrue(np.isfinite(results.hypervolume).all())
        self.assertEqual(snapshot(self.root), before)
        with self.assertRaises(TypeError):
            storage.load_evaluation_records(self.root, self.config, write_cache=0)

    # Params: self; returns: None.
    def test_missing_or_corrupt_policy_marker_refuses_before_cache_writes(self):
        marker = self.root / 'models/seed_0/w_1/COMPLETE.json'
        original = marker.read_bytes()
        for content in (None, b'{}'):
            if content is None: marker.unlink()
            else: marker.write_bytes(content)
            before = snapshot(self.root)
            with self.subTest(content=content), patch.object(trainer, '_cache_policy_results') as cache:
                with self.assertRaises(storage.StorageError): self.finalize()
                cache.assert_not_called()
                self.assertEqual(snapshot(self.root), before)
            marker.write_bytes(original)
        self.assert_unlocked()

    # Params: self; returns: None.
    def test_missing_intermediate_budget_refuses_even_with_valid_final_completion(self):
        (self.root / 'results/evaluations/seed_0/w_1/step_2.json').unlink()
        before = snapshot(self.root)
        with patch.object(trainer, '_cache_policy_results') as cache:
            with self.assertRaisesRegex(storage.StorageError, 'every declared intermediate'):
                self.finalize()
            cache.assert_not_called()
        self.assertEqual(snapshot(self.root), before)
        self.assert_unlocked()

    # Params: self; returns: None.
    def test_interrupted_final_evaluation_reuses_first_committed_policy_on_retry(self):
        evaluate, count = evaluation.evaluate_policy, []
        # Params: *args, **kwargs; returns: evaluate result.
        def interrupt(*args, **kwargs):
            count.append(1)
            if len(count) == 2: raise KeyboardInterrupt('interrupted final testing')
            return evaluate(*args, **kwargs)
        with patch.object(evaluation, 'evaluate_policy', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt): self.finalize()
        first = self.root / 'results/final/seed_0/w_0'
        before = snapshot(first)
        self.assertTrue(before)
        self.assertFalse((self.root / 'FINALIZED.json').exists()); self.assert_unlocked()
        with patch.object(evaluation, 'evaluate_policy', wraps=evaluate) as remaining:
            self.finalize()
        self.assertEqual(remaining.call_count, 1)
        self.assertEqual(snapshot(first), before)
        self.assertTrue((self.root / 'FINALIZED.json').exists()); self.assert_unlocked()

    # Params: self; returns: None.
    def test_report_failure_retries_without_re_evaluating_committed_policies(self):
        # Params: root, config; returns: never (raises).
        def broken(root, config):
            (root / 'reports/partial.txt').write_text('interrupted report')
            raise OSError('report interrupted')
        with patch.object(reporting, 'generate_reports', side_effect=broken):
            with self.assertRaisesRegex(OSError, 'report interrupted'):
                finalization.finalize_experiment(self.root)
        before = snapshot(self.root / 'results/final')
        self.assertFalse((self.root / 'FINALIZED.json').exists()); self.assert_unlocked()
        with patch.object(evaluation, 'evaluate_policy', side_effect=AssertionError('repeat final episode')):
            self.finalize()
        self.assertEqual(snapshot(self.root / 'results/final'), before)
        self.assertTrue((self.root / 'FINALIZED.json').exists())

    # Params: self; returns: None.
    def test_commit_write_failure_leaves_no_success_and_retry_preserves_final_records(self):
        original = storage.atomic_write_json
        # Params: path, payload; returns: original result.
        def fail_marker(path, payload):
            if Path(path) == self.root / 'FINALIZED.json': raise OSError('final marker interrupted')
            return original(path, payload)
        with patch.object(storage, 'atomic_write_json', side_effect=fail_marker):
            with self.assertRaisesRegex(OSError, 'final marker interrupted'): self.finalize()
        self.assertFalse((self.root / 'FINALIZED.json').exists()); self.assert_unlocked()
        before = snapshot(self.root / 'results/final')
        self.finalize()
        self.assertEqual(snapshot(self.root / 'results/final'), before)

    # Params: self; returns: None.
    def test_old_success_marker_is_archived_before_interrupted_rebuild(self):
        self.finalize()
        old = (self.root / 'FINALIZED.json').read_bytes()
        with patch.object(reporting, 'generate_reports', side_effect=OSError('rebuild interrupted')):
            with self.assertRaises(OSError): finalization.finalize_experiment(self.root)
        self.assertFalse((self.root / 'FINALIZED.json').exists())
        history = list((self.root / 'results/finalization_history').glob('*.json'))
        self.assertEqual(len(history), 1); self.assertEqual(history[0].read_bytes(), old)
        self.assert_unlocked()
        self.finalize()
        self.assertTrue((self.root / 'FINALIZED.json').exists())

    # Params: self; returns: None.
    def test_logger_close_failure_precedes_finalization_commit(self):
        original = reporting.make_logger
        # Params: *args; returns: logger.
        def failed_logger(*args):
            logger = original(*args)
            # Params: none; returns: never (raises).
            def fail(): raise OSError('finalizer logger close failed')
            logger.close = fail
            return logger
        with patch.object(reporting, 'make_logger', side_effect=failed_logger):
            with self.assertRaisesRegex(OSError, 'logger close failed'): self.finalize()
        self.assertFalse((self.root / 'FINALIZED.json').exists()); self.assert_unlocked()

    # Params: self; returns: None.
    def test_actual_finalize_cli_generates_reports_and_reuses_canonical_inputs(self):
        command = [sys.executable, '-B', '-m', 'agents.train.train_ppo_mo', '--finalize', '--experiment', str(self.root)]
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        before = snapshot(self.root / 'models')
        result = subprocess.run(command, cwd=str(self.root.parent), env=env,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual((status['mode'], status['status'], status['completed_policies']), ('finalize', 'finalized', 2))
        marker = storage._read_json(self.root / 'FINALIZED.json')
        self.assertTrue((self.root / 'reports/report_manifest.json').is_file())
        for relative, digest in marker['derived_file_hashes'].items():
            path = self.root / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            if path.suffix == '.png': self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(snapshot(self.root / 'models'), before)
        final_before = snapshot(self.root / 'results/final')
        again = subprocess.run(command, cwd=str(self.root.parent), env=env,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertEqual(snapshot(self.root / 'results/final'), final_before)
        self.assertEqual(snapshot(self.root / 'models'), before)
        self.assertEqual(len(list((self.root / 'results/finalization_history').glob('*.json'))), 1)
        self.assert_unlocked()


if __name__ == '__main__':
    unittest.main()
