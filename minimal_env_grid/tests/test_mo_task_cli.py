"""Parallel block 5: explicit initialize and isolated saved-policy task CLI."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import config as planning, storage, trainer, reporting, ownership
from agents.mo.experiment import initialize_experiment
from agents.mo.types import (ExperimentConfig, EnvConfig, PPOConfig, ExecutionConfig,
                             PolicyIdentity, PolicyTrainingSummary)
from agents.train import train_ppo_mo as cli


# Params: root; returns: dict.
def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


class PolicyTaskCLI(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mo_task_cli_')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / 'experiment'
        self.config = ExperimentConfig(output_dir=self.output, run_seeds=(9, 2), n_weights=3,
            reference_scaled=(-10., -10.), reference_status='frozen',
            requested_timesteps_per_policy=3, eval_interval=2, checkpoint_interval=2,
            eval_episodes=1, final_episodes=1,
            env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3), episode_steps=2, slew_alpha=0.),
            ppo=PPOConfig(n_envs=2, n_steps=1, batch_size=2, n_epochs=1, device='cpu', verbose=0),
            execution=ExecutionConfig(subproc=False))
        self.source = self.directory / 'config.json'
        self.write_config(self.config)

    # Params: self, config; returns: None.
    def write_config(self, config):
        values = storage._json_value(config); values.pop('output_dir')
        self.source.write_text(json.dumps(values), encoding='utf-8')

    # Params: self, arguments; returns: (code, docs, err.getvalue result).
    def main(self, arguments):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(arguments)
        docs = [json.loads(line) for line in out.getvalue().splitlines() if line.startswith('{')]
        return code, docs, err.getvalue()

    # Params: self, task; returns: self.main result.
    def task(self, task=5):
        return self.main(['--experiment', str(self.output), '--task-id', str(task)])

    # Params: self; returns: initialize_experiment result.
    def saved(self):
        return initialize_experiment(self.config)

    # Params: self, saved, weight_index; returns: PolicyTrainingSummary.
    def summary(self, saved, *, weight_index=2):
        identity = PolicyIdentity(saved.fingerprint, 2, 1, weight_index,
            tuple(saved.weights[weight_index].tolist()), saved.budget_plan.effective_total, 'test-summary')
        return PolicyTrainingSummary(identity, saved.budget_plan.requested_total,
            saved.budget_plan.effective_total, saved.budget_plan.effective_total, True,
            saved.root / 'models/seed_1' / f'w_{weight_index}' / 'final',
            resume_metadata={'additional_training_transitions': 4, 'restored_step': 0, 'skipped_complete': False})

    # Params: self; returns: None.
    def test_parser_explicit_modes_and_incompatible_combinations(self):
        initialized = cli.parse_args(['--initialize', '--config', 'config.json', '--output', 'new'])
        self.assertTrue(initialized.initialize)
        task = cli.parse_args(['--experiment', 'saved', '--task-id', '0'])
        self.assertEqual(task.task_id, 0); self.assertEqual(task.experiment, Path('saved'))
        invalid = [
            ['--initialize', '--resume', 'saved'], ['--initialize', '--preflight'],
            ['--initialize', '--task-id', '0', '--experiment', 'saved'],
            ['--task-id', '0'], ['--experiment', 'saved'],
            ['--task-id', '0', '--experiment', 'saved', '--config', 'config.json'],
            ['--task-id', '0', '--experiment', 'saved', '--output', 'new'],
            ['--task-id', '0', '--experiment', 'saved', '--resume', 'saved'],
            ['--task-id', '0', '--experiment', 'saved', '--preflight'],
            ['--task-id', '-1', '--experiment', 'saved'],
            ['--task-id', '1.5', '--experiment', 'saved'],
            ['--task-id', 'true', '--experiment', 'saved'], ['--experiment', ''],
        ]
        for args in invalid:
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                cli.parse_args(args)
            self.assertEqual(caught.exception.code, 2)

    # Params: self; returns: None.
    def test_initialize_saves_full_grid_without_training_runtime_or_reports(self):
        with patch.object(cli, '_check_runtime', side_effect=AssertionError('runtime probe')), \
             patch.object(cli, '_check_device', side_effect=AssertionError('device allocation')), \
             patch.object(trainer, 'run_experiment', side_effect=AssertionError('experiment training')), \
             patch.object(reporting, 'make_logger', side_effect=AssertionError('logger')):
            code, docs, err = self.main(['--initialize', '--config', str(self.source), '--output', str(self.output)])
        self.assertEqual(code, 0, err)
        result = docs[-1]
        self.assertEqual((result['mode'], result['status'], result['n_tasks']), ('initialize', 'initialized', 6))
        self.assertEqual(result['task_id_range'], [0, 5])
        self.assertEqual(storage._read_json(self.output / 'config.json')['run_seeds'], [9, 2])
        self.assertEqual(len(storage._read_json(self.output / 'seed_plan.json')), 2)
        self.assertEqual(storage._read_json(self.output / 'manifest.json')['fingerprint'], result['experiment_fingerprint'])
        self.assertFalse(any((self.output / 'models').iterdir()))
        self.assertFalse(any((self.output / 'logs').iterdir()))
        self.assertFalse(list((self.output / 'results').rglob('*.npy')))
        self.assertFalse(any((self.output / 'reports').iterdir()))

    # Params: self; returns: None.
    def test_initialization_refuses_nonempty_unfrozen_or_invalid_logging_without_overwrite(self):
        self.saved(); before = snapshot(self.output)
        code, _, err = self.main(['--initialize', '--config', str(self.source), '--output', str(self.output)])
        self.assertNotEqual(code, 0); self.assertIn('nonempty', err)
        self.assertEqual(snapshot(self.output), before)
        destination = self.directory / 'unfrozen'
        self.write_config(replace(self.config, reference_scaled=None, reference_status='unselected'))
        code, _, _ = self.main(['--initialize', '--config', str(self.source), '--output', str(destination)])
        self.assertEqual(code, 2); self.assertFalse(destination.exists())

    # Params: self; returns: None.
    def test_initialization_does_not_import_torch_or_probe_worker_device(self):
        self.write_config(replace(self.config, ppo=replace(self.config.ppo, device='cuda:999')))
        script = '''import sys
from agents.train.train_ppo_mo import main
assert main(sys.argv[1:]) == 0
assert 'torch' not in sys.modules
assert 'stable_baselines3' not in sys.modules
assert 'agents.mo.envs' not in sys.modules
'''
        process = subprocess.run([sys.executable, '-B', '-c', script, '--initialize', '--config', str(self.source),
            '--output', str(self.output)], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    # Params: self; returns: None.
    def test_worker_uses_saved_full_matrix_original_seed_and_only_isolated_policy_call(self):
        reversed_weights = planning.make_weights(3)[::-1].copy()
        with patch.object(planning, 'make_weights', return_value=reversed_weights):
            saved = self.saved()
        before = snapshot(self.output)
        with patch.object(planning, 'make_weights', side_effect=AssertionError('regenerated weights')), \
             patch.object(planning, 'make_seed_plan', side_effect=AssertionError('regenerated seeds')), \
             patch.object(cli, '_prepare', side_effect=AssertionError('sequential prepare')), \
             patch.object(cli, '_check_runtime', side_effect=AssertionError('environment probe')), \
             patch.object(cli, '_check_device', return_value='cpu'), \
             patch.object(trainer, 'run_experiment', side_effect=AssertionError('whole grid')), \
             patch.object(reporting, 'generate_reports', side_effect=AssertionError('reports')), \
             patch.object(trainer, 'train_one_policy', return_value=self.summary(saved)) as train:
            code, docs, err = self.task()
        self.assertEqual(code, 0, err)
        args, kwargs = train.call_args
        self.assertEqual(args[0], saved.config)
        self.assertEqual(args[2:4], (1, 2))
        self.assertEqual(args[4].shape, (3, 2))
        self.assertEqual(args[4].tobytes(), reversed_weights.tobytes())
        self.assertFalse(args[4].flags.writeable)
        self.assertEqual(args[5], saved.seed_plans[1])
        self.assertEqual(args[6], saved.budget_plan)
        self.assertEqual(kwargs, {'cache_results': False})
        self.assertEqual(docs[-1]['status'], 'policy_complete')
        self.assertEqual((docs[-1]['seed_index'], docs[-1]['run_seed'], docs[-1]['weight_index']), (1, 2, 2))
        self.assertEqual(snapshot(self.output), before)

    # Params: self; returns: None.
    def test_missing_experiment_or_out_of_range_task_fails_before_runtime(self):
        with patch.object(cli, '_check_device') as device, patch.object(trainer, 'train_one_policy') as train:
            code, _, _ = self.task(0)
            self.assertEqual(code, 2); self.assertFalse(self.output.exists())
            self.saved()
            before = snapshot(self.output)
            code, _, err = self.task(6)
            self.assertEqual(code, 2); self.assertIn('[0, 6)', err)
            self.assertEqual(snapshot(self.output), before)
            device.assert_not_called(); train.assert_not_called()

    # Params: self; returns: None.
    def test_worker_rejects_corrupt_initialization_and_changed_provenance_without_writes(self):
        self.saved()
        for function, value in (('_source_hashes', {'different.py': 'changed'}), ('_versions', {'python': 'different'})):
            before = snapshot(self.output)
            with self.subTest(function=function), patch.object(storage, function, return_value=value), \
                 patch.object(cli, '_check_device') as device, patch.object(trainer, 'train_one_policy') as train:
                code, _, _ = self.task()
                self.assertEqual(code, 2); device.assert_not_called(); train.assert_not_called()
                self.assertEqual(snapshot(self.output), before)
        (self.output / 'weights.npy').write_bytes(b'corrupt saved matrix')
        before = snapshot(self.output)
        with patch.object(cli, '_check_device') as device:
            code, _, _ = self.task()
        self.assertEqual(code, 2); device.assert_not_called()
        self.assertEqual(snapshot(self.output), before)

    # Params: self; returns: None.
    def test_duplicate_claim_is_rejected_before_worker_model_or_global_writes(self):
        from agents.mo import envs
        saved = self.saved()
        with ownership.claim_policy_slot(saved.root, saved.config, 1, 2):
            before = snapshot(self.output)
            with patch.object(cli, '_check_device', return_value='cpu'), \
                 patch.object(envs, 'make_train_vec_env') as workers, patch.object(trainer, 'build_model') as model:
                code, _, err = self.task()
            self.assertEqual(code, 1); self.assertIn('already claimed', err)
            workers.assert_not_called(); model.assert_not_called()
            self.assertEqual(snapshot(self.output), before)

    # Params: self; returns: None.
    def test_device_failure_prevents_policy_execution(self):
        self.saved(); before = snapshot(self.output)
        with patch.object(cli, '_check_device', side_effect=ValueError('device unavailable')), \
             patch.object(trainer, 'train_one_policy') as train:
            code, _, err = self.task()
        self.assertEqual(code, 2); self.assertIn('runtime preflight', err)
        train.assert_not_called(); self.assertEqual(snapshot(self.output), before)

    # Params: self; returns: None.
    def test_policy_failure_and_interrupt_are_nonzero_without_finalization(self):
        self.saved()
        for error, expected in ((OSError('policy failure'), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(error=type(error).__name__), patch.object(cli, '_check_device', return_value='cpu'), \
                 patch.object(trainer, 'train_one_policy', side_effect=error), \
                 patch.object(reporting, 'generate_reports') as reports:
                code, docs, err = self.task()
                self.assertEqual(code, expected); self.assertFalse(docs)
                self.assertIn('policy training/resume', err); reports.assert_not_called()

    # Params: self; returns: None.
    def test_incompatible_policy_summary_cannot_report_success(self):
        saved = self.saved()
        for result in (object(), self.summary(saved, weight_index=1)):
            with self.subTest(result=type(result).__name__), patch.object(cli, '_check_device', return_value='cpu'), \
                 patch.object(trainer, 'train_one_policy', return_value=result):
                code, docs, err = self.task()
                self.assertEqual(code, 1); self.assertFalse(docs)
                self.assertIn('Task summary does not match', err)

    # Params: self; returns: None.
    def test_actual_spawn_task_interruption_resume_and_completed_direct_script_skip(self):
        config = replace(self.config, execution=ExecutionConfig(subproc=True))
        self.write_config(config)
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [sys.executable, '-B', '-m', 'agents.train.train_ppo_mo']
        # Params: args, expected, prefix; returns: docs.
        def execute(args, expected=0, prefix=None):
            result = subprocess.run((prefix or command) + args, cwd=str(self.directory), env=env,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            docs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            return docs
        initialized = execute(['--initialize', '--config', str(self.source), '--output', str(self.output)])[-1]
        self.assertEqual(initialized['n_tasks'], 6)
        original_plans = storage._read_json(self.output / 'seed_plan.json')
        other = self.output / 'results/evaluations/seed_0/w_0'
        other.mkdir(parents=True)
        (other / 'RESUME.json').write_text('{"status":"pending"}')
        # Shared file sentinels need not be valid caches: a worker must never read/write them.
        for name in ('eval_returns_raw', 'eval_returns_scaled', 'eval_valid', 'actual_steps', 'hypervolume'):
            (self.output / 'results' / (name + '.npy')).write_bytes(b'shared cache sentinel')
        (self.output / 'logs/sentinel.txt').write_text('global log sentinel')
        (self.output / 'reports/sentinel.txt').write_text('report sentinel')
        before = snapshot(self.output)
        stop = '''import sys
from unittest.mock import patch
from agents.mo import trainer
from agents.train.train_ppo_mo import main
learn = trainer.learn_one_update
def interrupted(model, *args, **kwargs):
    if model.num_timesteps == 2: raise KeyboardInterrupt('stop after committed update')
    return learn(model, *args, **kwargs)
if __name__ == '__main__':
    with patch.object(trainer, 'learn_one_update', side_effect=interrupted):
        raise SystemExit(main(sys.argv[1:]))
'''
        task_args = ['--experiment', str(self.output), '--task-id', '4']
        execute(task_args, expected=130, prefix=[sys.executable, '-B', '-c', stop])
        policy = self.output / 'models/seed_1/w_1'
        self.assertTrue((policy / 'checkpoints/step_2/COMMITTED').is_file())
        self.assertFalse((policy / 'COMPLETE.json').exists())
        self.assertFalse((policy / 'OWNER.LOCK').exists())
        result = execute(task_args)[-1]
        self.assertEqual(result['status'], 'policy_complete')
        self.assertEqual((result['seed_index'], result['run_seed'], result['weight_index']), (1, 2, 1))
        self.assertEqual(result['actual_transitions'], 4)
        self.assertEqual(result['resume']['restored_step'], 2)
        self.assertEqual(result['resume']['additional_training_transitions'], 2)
        self.assertEqual(len(list(self.output.rglob('COMPLETE.json'))), 1)
        for relative, unchanged in before.items():
            path = self.output / relative
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), unchanged)
        evaluation = self.output / 'results/evaluations/seed_1/w_1/step_4.json'
        record, _ = storage._load_evaluation_record(evaluation)
        self.assertEqual(record.episode_seeds, tuple(original_plans[1]['eval_seeds']))
        before_skip = snapshot(self.output)
        skipped = execute(task_args, prefix=[sys.executable, '-B', str(Path(cli.__file__).resolve())])[-1]
        self.assertTrue(skipped['resume']['skipped_complete'])
        self.assertEqual(skipped['resume']['additional_training_transitions'], 0)
        self.assertEqual(snapshot(self.output), before_skip)
        self.assertFalse(any((self.output / 'results/final').iterdir()))
        self.assertFalse((self.output / 'reports/report_manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
