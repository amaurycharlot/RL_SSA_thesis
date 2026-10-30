"""Block 34 CLI integration, immutable datasets and training-free reporting."""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Params: module, *args; returns: (code, output.getvalue result, errors.getvalue result).
def invoke(module, *args):
    output, errors = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        code = module.main(list(args))
    return code, output.getvalue(), errors.getvalue()


# Params: root; returns: dict.
def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class EvaluationReportCLI(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from test_mo_reporting import SummaryTableExports
        self.fixture = SummaryTableExports(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    # Params: self; returns: None.
    def test_parser_rejects_scientific_overrides_abbreviations_and_unlabelled_seeds(self):
        from scripts import evaluate_ppo_mo as evaluate, plot_ppo_mo_results as plot
        for module in (evaluate, plot):
            for args in ([], ['--experiment', ''], ['--experiment', 'run', '--gamma', '.8'],
                         ['--experiment', 'run', '--weights', 'a.npy'], ['--exp', 'run']):
                self.assertEqual(invoke(module, *args)[0], 2)
            self.assertEqual(invoke(module, '--help')[0], 0)
        for args in (['--test-seeds', '10'], ['--dataset-label', '../final'],
                     ['--dataset-label', 'a', '--test-seeds', '-1'],
                     ['--dataset-label', 'a', '--test-seeds', '4294967296'],
                     ['--dataset-label', 'a', '--test-seeds', '5', '5']):
            self.assertEqual(invoke(evaluate, '--experiment', str(self.root), *args)[0], 2)

    # Params: self; returns: None.
    def test_import_help_direct_execution_and_windows_child_are_passive(self):
        code = '''
import sys, pathlib, runpy, io
from contextlib import redirect_stdout
before=set(pathlib.Path('.').iterdir())
from scripts import evaluate_ppo_mo, plot_ppo_mo_results
for cli in (evaluate_ppo_mo,plot_ppo_mo_results):
 runpy.run_path(cli.__file__,run_name='__mp_main__')
 with redirect_stdout(io.StringIO()): assert cli.main(['--help'])==0
 assert not any(n.split('.')[0] in ('numpy','torch','gymnasium','stable_baselines3','matplotlib') for n in sys.modules)
assert set(pathlib.Path('.').iterdir())==before
'''
        environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=self.root.parent,
            env=environment, capture_output=True, text=True, check=True)
        for name in ('evaluate_ppo_mo', 'plot_ppo_mo_results'):
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / (name + '.py')), '--help'],
                cwd=self.root.parent, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    # Params: self; returns: None.
    def test_saved_config_uses_exact_weights_scales_and_plans_without_generation(self):
        from scripts._ppo_mo_cli import saved_experiment
        from agents.mo import config
        with patch.object(config, 'make_weights', side_effect=AssertionError('regenerated weights')), \
             patch.object(config, 'make_seed_plan', side_effect=AssertionError('regenerated seeds')):
            root, saved, plans, payload, fingerprint = saved_experiment(self.root)
        self.assertEqual(saved, self.fixture.config)
        self.assertEqual(plans, self.fixture.plans)
        self.assertEqual(fingerprint, self.fixture.fingerprint)
        self.assertEqual(payload['weights']['values'], self.fixture.weights.tolist())
        self.assertEqual(root, self.root)

    # Params: self; returns: None.
    def test_missing_initialization_no_records_and_missing_final_models_fail_without_writes(self):
        from scripts import evaluate_ppo_mo as evaluate, plot_ppo_mo_results as plot
        before = snapshot(self.root)
        for module in (evaluate, plot):
            code, _, error = invoke(module, '--experiment', str(self.root.parent / 'absent'))
            self.assertEqual(code, 2); self.assertIn('Cannot', error)
            code, _, error = invoke(module, '--experiment', str(self.root))
            self.assertEqual(code, 2); self.assertIn('Cannot', error)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_plotting_reads_records_only_exports_verifiable_manifest_and_preserves_inputs(self):
        from scripts import plot_ppo_mo_results as plot
        from agents.mo import storage, evaluation, trainer, envs
        self.fixture.populate()
        before = snapshot(self.root)
        export = self.root.parent / 'report-export'
        with patch.object(trainer, 'run_experiment', side_effect=AssertionError('training')), \
             patch.object(trainer, 'build_model', side_effect=AssertionError('model')), \
             patch.object(storage, 'load_policy_bundle', side_effect=AssertionError('load')), \
             patch.object(evaluation, 'evaluate_policy', side_effect=AssertionError('evaluate')), \
             patch.object(envs, 'make_raw_env', side_effect=AssertionError('environment')):
            code, output, error = invoke(plot, '--experiment', str(self.root), '--report-dir', str(export))
        self.assertEqual(code, 0, error)
        self.assertFalse(json.loads(output)['training_started'])
        manifest = json.loads((export / 'reports/report_manifest.json').read_text())
        for name, digest in manifest['file_hashes'].items():
            self.assertEqual(hashlib.sha256((export / name).read_bytes()).hexdigest(), digest)
        after = snapshot(self.root)
        self.assertTrue(all(after[name] == value for name, value in before.items()))
        self.assertTrue(all(name in before or name.startswith('reports/') for name in after))

    # Params: self; returns: None.
    def test_bad_report_destination_and_corrupt_record_are_actionable_nonzero(self):
        from scripts import plot_ppo_mo_results as plot
        self.fixture.populate()
        for target in (self.root, self.root / 'results', self.root.parent):
            self.assertEqual(invoke(plot, '--experiment', str(self.root), '--report-dir', str(target))[0], 2)
        path = next((self.root / 'results/final').rglob('*.npz'))
        path.write_bytes(b'corrupt')
        before = snapshot(self.root)
        code, _, error = invoke(plot, '--experiment', str(self.root))
        self.assertEqual(code, 2); self.assertIn('checksum', error)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_partial_report_is_explicit_without_filling_missing_slots(self):
        from scripts import plot_ppo_mo_results as plot
        self.fixture.record()
        code, _, error = invoke(plot, '--experiment', str(self.root))
        self.assertEqual(code, 0, error)
        text = (self.root / 'reports/report.md').read_text(encoding='utf-8')
        self.assertIn('0/2 final evaluations', text)

    # Params: self; returns: None.
    def test_interrupts_return_130(self):
        from scripts import evaluate_ppo_mo as evaluate, plot_ppo_mo_results as plot
        with patch.object(evaluate, '_evaluate', side_effect=KeyboardInterrupt):
            self.assertEqual(invoke(evaluate, '--experiment', str(self.root))[0], 130)
        with patch('scripts._ppo_mo_cli.saved_experiment', side_effect=KeyboardInterrupt):
            self.assertEqual(invoke(plot, '--experiment', str(self.root))[0], 130)


class RealFinalCLI(unittest.TestCase):
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
        from test_mo_evaluation import FinalPolicySetEvaluation
        from dataclasses import replace
        from agents.mo.types import NormalizationConfig
        self.fixture = FinalPolicySetEvaluation(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.config = replace(self.fixture.config,
            normalization=NormalizationConfig(norm_obs=True, norm_reward=True))
        self.fixture.train()
        self.root = self.fixture.root

    # Params: self; returns: None.
    def test_real_canonical_cli_reuses_records_and_matches_direct_saved_policy(self):
        import numpy as np
        from scripts import evaluate_ppo_mo as cli
        from agents.mo import storage, evaluation, trainer
        from agents.mo.envs import make_raw_env
        before = snapshot(self.root)
        with patch.object(trainer, 'run_experiment', side_effect=AssertionError('training')):
            code, output, error = invoke(cli, '--experiment', str(self.root), '--device', 'cpu')
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(output)['test_seeds'], list(self.fixture.plans[0].final_seeds))
        self.assertEqual(json.loads(output)['policies'], 4)
        path = next((self.root / 'results/final/seed_0/w_0').glob('*.json'))
        result, _ = storage._load_evaluation_record(path)
        with storage.load_policy_bundle(self.root / 'models/seed_0/w_0/final', self.fixture.config) as bundle:
            expected = evaluation.evaluate_policy(bundle.model, lambda: make_raw_env(self.fixture.config.env),
                self.fixture.plans[0].final_seeds, self.fixture.config, bundle.identity,
                frozen_normalizer=evaluation.make_frozen_obs_normalizer(bundle.vec_normalize))
        np.testing.assert_array_equal(result.returns_raw, expected.returns_raw)
        np.testing.assert_array_equal(result.returns_scaled, expected.returns_scaled)
        committed = {p: p.read_bytes() for p in (self.root / 'results/final').rglob('*') if p.is_file()}
        with patch.object(storage, 'load_policy_bundle', side_effect=AssertionError('cached model load')):
            self.assertEqual(invoke(cli, '--experiment', str(self.root))[0], 0)
        self.assertTrue(all(p.read_bytes() == data for p, data in committed.items()))
        after = snapshot(self.root)
        self.assertTrue(all(after[name] == data for name, data in before.items()))

    # Params: self; returns: None.
    def test_labelled_alternate_seeds_and_traces_leave_canonical_results_unchanged(self):
        from scripts import evaluate_ppo_mo as cli
        from agents.mo import storage
        self.assertEqual(invoke(cli, '--experiment', str(self.root))[0], 0)
        before = snapshot(self.root)
        args = ['--experiment', str(self.root), '--dataset-label', 'test_123', '--test-seeds', '123', '124', '--retain-traces']
        code, output, error = invoke(cli, *args)
        self.assertEqual(code, 0, error)
        metadata = json.loads(Path(json.loads(output)['dataset']).read_text())
        self.assertEqual(metadata['test_seeds'], [123, 124])
        self.assertEqual(metadata['canonical_test_seeds'], list(self.fixture.plans[0].final_seeds))
        self.assertEqual(len(metadata['policy_records']), 4)
        self.assertFalse(metadata['canonical_results_replaced'])
        folder = self.root / 'results/alternate_tests/test_123'
        for item in metadata['policy_records']:
            result, saved = storage._load_evaluation_record(folder / item['path'])
            self.assertEqual(result.episode_seeds, (123, 124))
            self.assertEqual(len(result.traces), 2)
            self.assertEqual(saved['kind'], 'alternate_test')
        self.assertEqual(invoke(cli, *args)[0], 2)
        after = snapshot(self.root)
        self.assertTrue(all(after[name] == data for name, data in before.items()))
        self.assertTrue(all(name in before or name.startswith('results/alternate_tests/test_123/') for name in after))

    # Params: self; returns: None.
    def test_canonical_traces_retained_and_cannot_replace_previous_no_trace_records(self):
        from scripts import evaluate_ppo_mo as cli
        from agents.mo import storage
        code, _, error = invoke(cli, '--experiment', str(self.root), '--retain-traces')
        self.assertEqual(code, 0, error)
        for path in (self.root / 'results/final').rglob('*.json'):
            self.assertEqual(len(storage._load_evaluation_record(path)[0].traces), 2)
        # Independently exercise the existing-no-trace preflight using a real fixture.
        self.fixture.doCleanups()
        self.setUp()
        self.assertEqual(invoke(cli, '--experiment', str(self.root))[0], 0)
        before = snapshot(self.root)
        code, _, error = invoke(cli, '--experiment', str(self.root), '--retain-traces')
        self.assertEqual(code, 2); self.assertIn('--dataset-label', error)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_missing_last_bundle_blocks_all_model_loading_and_alternate_output(self):
        from scripts import evaluate_ppo_mo as cli
        from agents.mo import storage
        (self.root / 'models/seed_1/w_1/final').rename(self.root.parent / 'held-final')
        before = snapshot(self.root)
        with patch.object(storage, 'load_policy_bundle', side_effect=AssertionError('premature load')):
            code, _, error = invoke(cli, '--experiment', str(self.root), '--dataset-label', 'missing')
        self.assertEqual(code, 2); self.assertIn('absent', error)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_evaluation_failure_closes_loaded_bundle_and_preserves_incomplete_marker(self):
        from scripts import evaluate_ppo_mo as cli
        from agents.mo import storage, evaluation
        loaded = []
        original = storage.load_policy_bundle
        # Params: *args, **kwargs; returns: bundle.
        def load(*args, **kwargs):
            bundle = original(*args, **kwargs); loaded.append(bundle); return bundle
        with patch.object(storage, 'load_policy_bundle', side_effect=load), \
             patch.object(evaluation, 'evaluate_policy', side_effect=RuntimeError('injected evaluation failure')):
            code, _, error = invoke(cli, '--experiment', str(self.root), '--dataset-label', 'failed')
        self.assertEqual(code, 2); self.assertIn('injected evaluation failure', error)
        self.assertEqual(len(loaded), 1); self.assertTrue(loaded[0].closed)
        self.assertTrue((self.root / 'results/alternate_tests/failed').is_dir())
        self.assertFalse((self.root / 'results/alternate_tests/failed/dataset.json').exists())

    # Params: self; returns: None.
    def test_alternate_overlap_and_unavailable_device_fail_before_loading(self):
        from scripts import evaluate_ppo_mo as cli
        from agents.mo import storage
        import torch
        before = snapshot(self.root)
        with patch.object(storage, 'load_policy_bundle', side_effect=AssertionError('load')), \
             patch.object(torch.cuda, 'is_available', return_value=False):
            self.assertEqual(invoke(cli, '--experiment', str(self.root), '--device', 'cuda')[0], 2)
            for seed in (self.fixture.plans[0].worker_seeds[0], self.fixture.plans[0].eval_seeds[0]):
                self.assertEqual(invoke(cli, '--experiment', str(self.root), '--dataset-label', 'overlap', '--test-seeds', str(seed))[0], 2)
        self.assertEqual(snapshot(self.root), before)

    # Params: self; returns: None.
    def test_module_cli_evaluation_and_report_from_another_cwd(self):
        environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for module in ('scripts.evaluate_ppo_mo', 'scripts.plot_ppo_mo_results'):
            result = subprocess.run([sys.executable, '-B', '-m', module, '--experiment', str(self.root)],
                cwd=self.root.parent, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'reports/report_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
