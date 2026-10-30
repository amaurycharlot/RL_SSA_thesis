"""Block 33 isolated CLI integration checks; preserves concurrent report/logger tests."""
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import reporting, storage


class TrainingCLI(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from test_mo_reporting import SummaryTableExports
        import json
        self.fixture = SummaryTableExports(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.temporary.name)
        self.output = self.root / 'cli-output'
        self.config_path = self.root / 'cli.json'
        values = storage._json_value(self.fixture.config); values.pop('output_dir')
        self.config_path.write_text(json.dumps(values), encoding='utf-8')

    # Params: self, extra, default_config; returns: (code, documents, errors.getvalue result).
    def main(self, extra=(), *, default_config=True):
        import io, json
        from contextlib import redirect_stdout, redirect_stderr
        from agents.train.train_ppo_mo import main
        args = ['--config', str(self.config_path)] if default_config else []
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors): code = main(args + list(extra))
        documents = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
        return code, documents, errors.getvalue()

    # Params: self; returns: None.
    def test_parser_explicit_paths_unknown_flags_conflicts_and_help(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from agents.train.train_ppo_mo import parse_args
        args = parse_args(['--config', 'a.json', '--resume', 'run', '--preflight'])
        self.assertEqual(args.config, Path('a.json')); self.assertEqual(args.resume, Path('run'))
        self.assertTrue(args.preflight)
        for arguments in (['--output', 'a', '--resume', 'b'], ['--resume'], ['--output', ''],
                          ['--n-envs', '1'], ['--pref']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught: parse_args(arguments)
            self.assertEqual(caught.exception.code, 2)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught: parse_args(['--help'])
        self.assertEqual(caught.exception.code, 0)

    # Params: self; returns: None.
    def test_import_help_and_windows_child_guard_are_passive(self):
        import os
        code = '''
import sys, pathlib, runpy, multiprocessing as mp, io
from contextlib import redirect_stdout
from unittest.mock import patch
before=set(pathlib.Path('.').iterdir())
import agents.train.train_ppo_mo as cli
runpy.run_path(cli.__file__,run_name='__mp_main__')
with patch.object(mp,'freeze_support') as freeze, patch.object(sys,'argv',['train_ppo_mo','--help']), redirect_stdout(io.StringIO()):
 try: runpy.run_path(cli.__file__,run_name='__main__')
 except SystemExit as e: assert e.code==0
 freeze.assert_called_once_with()
assert not any(n.split('.')[0] in ('numpy','torch','gymnasium','stable_baselines3','matplotlib') for n in sys.modules)
assert set(pathlib.Path('.').iterdir())==before
'''
        subprocess.run([sys.executable, '-B', '-c', code], cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)), capture_output=True, text=True, check=True)

    def unselected_example(self):
        """Own the unfrozen template contract without assuming production choices."""
        from agents.mo.types import ExperimentConfig
        path = self.root / 'unselected-example/configs/ppo_mo_r6_r2.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        values = storage._json_value(ExperimentConfig()); values.pop('output_dir')
        path.write_text(json.dumps(values), encoding='utf-8')
        return path

    # Params: self; returns: None.
    def test_unselected_example_preflight_checks_runtime_without_training_resources(self):
        import torch, multiprocessing as mp
        from agents.mo import envs, trainer
        example = self.unselected_example()
        threads, children = torch.get_num_threads(), mp.active_children()
        with patch.object(envs, 'make_train_vec_env', side_effect=AssertionError('training worker group')), \
             patch.object(envs, 'DummyVecEnv', side_effect=AssertionError('sequential training group')), \
             patch.object(envs, 'TrainingSubprocVecEnv', side_effect=AssertionError('spawn group')), \
             patch.object(trainer, 'build_model', side_effect=AssertionError('model')), \
             patch.object(trainer, 'run_experiment', side_effect=AssertionError('training')):
            code, docs, errors = self.main(['--config', str(example), '--output', str(self.output), '--preflight'], default_config=False)
        self.assertEqual(code, 0, errors); plan = docs[0]
        self.assertFalse(plan['training_ready']); self.assertIsNone(plan['resolved_config']['reference_scaled'])
        self.assertEqual(plan['resolved_config']['ppo']['n_envs'], 8)
        self.assertEqual(plan['budget_plan']['effective_total'], 503808)
        self.assertEqual(plan['cost']['effective_training_transitions'], 55_418_880)
        self.assertTrue(plan['runtime']['single_environment_transition_checked'])
        self.assertEqual(torch.get_num_threads(), threads); self.assertEqual(mp.active_children(), children)
        self.assertFalse(self.output.exists())

    # Params: self; returns: None.
    def test_default_example_works_for_preflight_and_unset_training_is_rejected(self):
        from agents.mo import trainer
        from agents.train import train_ppo_mo as cli
        example = self.unselected_example()
        fake_cli = example.parent.parent / 'agents/train/train_ppo_mo.py'
        with patch.object(cli, '__file__', str(fake_cli)):
            with patch('agents.train.train_ppo_mo._check_runtime', return_value={'checked': True}):
                code, docs, errors = self.main(['--preflight', '--output', str(self.output)], default_config=False)
            self.assertEqual(code, 0, errors); self.assertFalse(docs[0]['training_ready'])
            with patch.object(trainer, 'run_experiment') as run:
                code, _, errors = self.main(['--output', str(self.output)], default_config=False)
            self.assertEqual(code, 2); self.assertIn('requires a selected reference', errors)
            run.assert_not_called(); self.assertFalse(self.output.exists())

    # Params: self; returns: None.
    def test_nonempty_destination_is_preserved_and_rejected_before_runtime(self):
        self.output.mkdir(); sentinel = self.output / 'existing.txt'; sentinel.write_text('keep')
        with patch('agents.train.train_ppo_mo._check_runtime') as check:
            code, _, errors = self.main(['--output', str(self.output), '--preflight'])
        self.assertEqual(code, 2); self.assertIn('--resume', errors); check.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'keep')

    # Params: self; returns: None.
    def test_resume_preflight_uses_saved_configuration_weights_and_seed_plans(self):
        from agents.mo import config as planning
        root = self.fixture.root
        before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
        with patch.object(planning, 'make_weights', side_effect=AssertionError('regenerated weights')), \
             patch.object(planning, 'make_seed_plan', side_effect=AssertionError('regenerated seed plans')):
            code, docs, errors = self.main(['--resume', str(root), '--preflight'], default_config=False)
        self.assertEqual(code, 0, errors)
        self.assertTrue(docs[0]['resume'])
        self.assertEqual(docs[0]['weights']['values'], self.fixture.weights.tolist())
        self.assertEqual(docs[0]['seed_plans'], storage._json_value(self.fixture.plans))
        for path, content in before.items(): self.assertEqual(path.read_bytes(), content)
        self.assertEqual(set(before), {p for p in root.rglob('*') if p.is_file()})

    # Params: self; returns: None.
    def test_resume_rejects_changed_science_before_runtime_or_training(self):
        import json
        values = json.loads(self.config_path.read_text()); values['error_target_set'] = 'initial'
        self.config_path.write_text(json.dumps(values))
        with patch('agents.train.train_ppo_mo._check_runtime') as check:
            code, _, errors = self.main(['--resume', str(self.fixture.root), '--preflight'])
        self.assertEqual(code, 2); self.assertIn('mismatch', errors); check.assert_not_called()

    # Params: self; returns: None.
    def test_unavailable_explicit_device_has_no_silent_cpu_fallback(self):
        import json, torch
        values = json.loads(self.config_path.read_text()); values['ppo']['device'] = 'cuda'
        self.config_path.write_text(json.dumps(values))
        with patch.object(torch.cuda, 'is_available', return_value=False):
            code, _, errors = self.main(['--output', str(self.output), '--preflight'])
        self.assertEqual(code, 2); self.assertIn('unavailable', errors); self.assertFalse(self.output.exists())

    # Params: self; returns: None.
    def test_runtime_probe_closes_owned_environment_on_reset_failure(self):
        from agents.mo import envs
        env = envs.make_scalar_env(self.fixture.config.env, self.fixture.weights[0], None, self.fixture.config.objective_scales)
        with patch.object(env, 'reset', side_effect=RuntimeError('injected reset failure')), \
             patch.object(env, 'close', wraps=env.close) as close, \
             patch.object(envs, 'make_scalar_env', return_value=env):
            code, _, errors = self.main(['--output', str(self.output), '--preflight'])
        self.assertEqual(code, 2); self.assertIn('reset failure', errors); close.assert_called_once_with()
        self.assertFalse(self.output.exists())

    # Params: self; returns: None.
    def test_training_failure_and_interrupt_return_nonzero_without_reporting(self):
        from agents.mo import trainer
        for error, expected in ((RuntimeError('injected training failure'), 1), (KeyboardInterrupt(), 130)):
            with patch('agents.train.train_ppo_mo._check_runtime', return_value={}), \
                 patch.object(trainer, 'run_experiment', side_effect=error), patch.object(reporting, 'generate_reports') as report:
                code, _, errors = self.main(['--output', str(self.output)])
            self.assertEqual(code, expected); self.assertIn('training/final evaluation', errors)
            report.assert_not_called()

    # Params: self; returns: None.
    def test_reporting_failure_returns_nonzero_with_clear_phase(self):
        from types import SimpleNamespace
        from agents.mo import trainer
        with patch('agents.train.train_ppo_mo._check_runtime', return_value={}), \
             patch.object(trainer, 'run_experiment', return_value=SimpleNamespace(completed_policies=4, total_policies=4)), \
             patch.object(reporting, 'generate_reports', side_effect=OSError('injected report failure')):
            code, _, errors = self.main(['--output', str(self.output)])
        self.assertEqual(code, 1); self.assertIn('report generation failed', errors)

    # Params: self; returns: None.
    def test_enabled_logging_preflight_requires_owner_settings_without_starting_logger(self):
        values = json.loads(self.config_path.read_text())
        values['logging'] = {'mode': 'online', 'project': 'owner-project', 'entity': 'owner-account'}
        self.config_path.write_text(json.dumps(values))
        with patch('agents.train.train_ppo_mo._check_runtime', return_value={}), \
             patch.object(reporting, 'make_logger', side_effect=AssertionError('preflight must not start logging')):
            code, docs, errors = self.main(['--output', str(self.output), '--preflight'])
        self.assertEqual(code, 0, errors)
        self.assertEqual(docs[0]['resolved_config']['logging']['entity'], 'owner-account')
        values['logging'].pop('entity'); self.config_path.write_text(json.dumps(values))
        code, _, errors = self.main(['--output', str(self.output), '--preflight'])
        self.assertEqual(code, 2); self.assertIn('project and entity', errors)
        self.assertFalse(self.output.exists())

    # Params: self; returns: None.
    def test_invalid_json_and_unknown_flags_return_nonzero(self):
        self.config_path.write_text('{invalid')
        code, _, errors = self.main(['--preflight'])
        self.assertEqual(code, 2); self.assertIn('Cannot read configuration', errors)
        code, _, _ = self.main(['--unknown'])
        self.assertEqual(code, 2)

    # Params: self; returns: None.
    def test_actual_small_cli_training_reports_and_completed_resume(self):
        import os, json
        environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, '-B', '-m', 'agents.train.train_ppo_mo']
        first = subprocess.run(command + ['--config', str(self.config_path), '--output', str(self.output)],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=180)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        summary = [json.loads(line) for line in first.stdout.splitlines() if line.startswith('{')][-1]
        self.assertEqual((summary['completed_policies'], summary['total_policies']), (4, 4))
        self.assertEqual(summary['additional_training_transitions'], 16)
        self.assertTrue(Path(summary['report_manifest']).is_file())
        before = {p: p.read_bytes() for folder in ('models', 'results') for p in (self.output / folder).rglob('*') if p.is_file()}
        second = subprocess.run(command + ['--resume', str(self.output)], cwd=self.root, env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        resumed = [json.loads(line) for line in second.stdout.splitlines() if line.startswith('{')][-1]
        self.assertEqual(resumed['additional_training_transitions'], 0)
        for path, content in before.items(): self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len(list(self.output.glob('models/seed_*/w_*/COMPLETE.json'))), 4)

    # Params: self; returns: None.
    def test_actual_cli_windows_spawn_workers_complete_a_tiny_run(self):
        import os, json
        values = json.loads(self.config_path.read_text())
        values.update(run_seeds=[7], requested_timesteps_per_policy=2, eval_episodes=1, final_episodes=1)
        values['ppo'].update(n_envs=2, n_steps=1, batch_size=2)
        values['env']['episode_steps'] = 2
        values['execution']['subproc'] = True
        values['normalization'].update(norm_obs=False, norm_reward=False)
        self.config_path.write_text(json.dumps(values))
        result = subprocess.run([sys.executable, '-B', '-m', 'agents.train.train_ppo_mo', '--config',
            str(self.config_path), '--output', str(self.output)], cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
            capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')][-1]
        self.assertEqual(summary['completed_policies'], 2)
        self.assertEqual(summary['additional_training_transitions'], 4)
        self.assertTrue(Path(summary['report_manifest']).is_file())


if __name__ == '__main__':
    unittest.main()
