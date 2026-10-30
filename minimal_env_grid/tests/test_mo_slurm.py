"""Block 8: submission graph, fixed execution inputs and batch launch contracts."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from agents.mo import storage
from agents.mo.types import ExperimentConfig, PPOConfig, ExecutionConfig

spec = importlib.util.spec_from_file_location('mo_slurm', PROJECT / 'scripts/slurm/mo_slurm.py')
slurm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slurm)


class SlurmWorkflow(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='mo slurm ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = ExperimentConfig(run_seeds=(9, 2), n_weights=3,
            reference_scaled=(-10., -10.), reference_status='frozen',
            requested_timesteps_per_policy=8, eval_interval=8, checkpoint_interval=8,
            eval_episodes=1, final_episodes=1,
            ppo=PPOConfig(n_envs=8, n_steps=1, batch_size=8, n_epochs=1, device='cpu', verbose=0))
        self.config_path = self.root / 'science config.json'
        self.write_config()
        self.experiment = self.root / 'experiment'
        self.directory = self.root / 'submission'

    # Params: self; returns: None.
    def write_config(self):
        values = storage._json_value(self.config); values.pop('output_dir')
        self.config_path.write_text(json.dumps(values), encoding='utf-8')

    # Params: self, *extra; returns: slurm.parser().parse_args result.
    def args(self, *extra):
        return slurm.parser().parse_args(['submit', '--config', str(self.config_path),
            '--experiment', str(self.experiment), '--submission-dir', str(self.directory),
            '--account', 'project-account', '--partition', 'cpu-partition', *extra])

    # Params: self, *extra; returns: slurm.prepare result.
    def manifest(self, *extra):
        return slurm.prepare(self.args(*extra))

    # Params: self, manifest, stage; returns: dict.
    def environment(self, manifest, stage):
        return dict(os.environ, SLURM_JOB_ID='101', SLURM_NTASKS='1', SLURM_JOB_NUM_NODES='1',
            SLURM_CPUS_PER_TASK=str(manifest['resources'][stage]['cpus']), SLURM_ARRAY_TASK_ID='5')

    # Params: self; returns: None.
    def test_dry_run_creates_full_grid_plan_without_submitting_or_initializing(self):
        before = self.config_path.read_bytes()
        with patch.object(slurm.subprocess, 'run', side_effect=AssertionError('dry run submitted jobs')):
            result = slurm.submit_workflow(self.args('--dry-run', '--max-concurrent', '2'))
        self.assertEqual(result['n_tasks'], 6)
        self.assertEqual(result['status'], 'dry_run')
        self.assertFalse(self.experiment.exists())
        self.assertEqual(self.config_path.read_bytes(), before)
        state = json.loads((self.directory / 'submission.json').read_text())
        self.assertIn('--array=0-5%2', state['commands']['policy'])
        self.assertIn('--dependency=afterok:INITIALIZE_JOB_ID', state['commands']['policy'])
        self.assertIn('--dependency=afterok:POLICY_JOB_ID', state['commands']['finalize'])
        self.assertTrue((self.directory / 'logs').is_dir())

    # Params: self; returns: None.
    def test_three_submissions_chain_whole_array_success_and_preserve_arguments(self):
        calls = []
        # Params: command, **kwargs; returns: subprocess.CompletedProcess result.
        def sbatch(command, **kwargs):
            calls.append(command)
            self.assertEqual(kwargs, {'check': True, 'capture_output': True, 'text': True})
            return subprocess.CompletedProcess(command, 0, str(100 + len(calls)) + ';cluster\n', '')
        result = slurm.submit_workflow(self.args('--max-concurrent', '3', '--policy-cpus', '10',
            '--learner-threads', '2', '--policy-memory', '12G', '--policy-time', '12:30:00',
            '--finalizer-cpus', '2', '--finalizer-threads', '2', '--gres', 'gpu:1'), runner=sbatch)
        self.assertEqual(result['jobs'], {'initialize': '101', 'policy': '102', 'finalize': '103'})
        self.assertEqual(len(calls), 3)
        self.assertFalse(any(x.startswith('--dependency') for x in calls[0]))
        self.assertIn('--dependency=afterok:101', calls[1])
        self.assertIn('--dependency=afterok:102', calls[2])
        self.assertIn('--kill-on-invalid-dep=yes', calls[2])
        self.assertIn('--array=0-5%3', calls[1])
        self.assertIn('--cpus-per-task=10', calls[1])
        self.assertIn('--mem=12G', calls[1]); self.assertIn('--time=12:30:00', calls[1])
        self.assertIn('--gres=gpu:1', calls[1]); self.assertIn('--gres=gpu:1', calls[2])
        self.assertFalse(any(x.startswith('--gres') for x in calls[0]))
        for stage, command in zip(slurm.STAGES, calls):
            self.assertIn('--nodes=1', command); self.assertIn('--ntasks=1', command)
            self.assertIn('--account=project-account', command)
            self.assertIn('--partition=cpu-partition', command)
            self.assertEqual(command[-3], str(Path(sys.executable).absolute()))
            self.assertEqual(command[-1], str(self.directory / 'manifest.json'))
        self.assertIn('%A_%a.out', next(x for x in calls[1] if x.startswith('--output=')))
        self.assertEqual(json.loads((self.directory / 'submission.json').read_text())['status'], 'submitted')

    # Params: self; returns: None.
    def test_failed_array_submission_records_init_and_does_not_submit_finalizer(self):
        calls = []
        # Params: command, **kwargs; returns: subprocess.CompletedProcess result.
        def sbatch(command, **kwargs):
            calls.append(command)
            if len(calls) == 2:
                raise subprocess.CalledProcessError(1, command, stderr='rejected resources')
            return subprocess.CompletedProcess(command, 0, '101\n', '')
        with self.assertRaises(subprocess.CalledProcessError):
            slurm.submit_workflow(self.args(), runner=sbatch)
        state = json.loads((self.directory / 'submission.json').read_text())
        self.assertEqual(len(calls), 2)
        self.assertEqual(state['status'], 'submission_failed')
        self.assertEqual(state['scheduler_stderr'], 'rejected resources')
        self.assertEqual(state['jobs'], {'initialize': '101'})
        self.assertNotIn('finalize', state['commands'])

    # Params: self; returns: None.
    def test_unrecognized_sbatch_response_stops_chain(self):
        with self.assertRaisesRegex(RuntimeError, 'Unrecognized sbatch'):
            slurm.submit_workflow(self.args(), runner=lambda *a, **k:
                subprocess.CompletedProcess(a[0], 0, 'Submitted batch job 101', ''))
        self.assertEqual(json.loads((self.directory / 'submission.json').read_text())['jobs'], {})

    # Params: self; returns: None.
    def test_undersized_policy_allocation_fails_before_writes(self):
        for options in (('--policy-cpus', '8'), ('--learner-threads', '2')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'CPU demand'):
                self.manifest(*options)
            self.assertFalse(self.directory.exists()); self.assertFalse(self.experiment.exists())

    # Params: self; returns: None.
    def test_saved_worker_threads_are_counted_without_override(self):
        self.config = replace(self.config, execution=ExecutionConfig(worker_threads=2))
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'CPU demand 17'):
            self.manifest()
        manifest = self.manifest('--policy-cpus', '17')
        self.assertEqual(manifest['worker_threads'], 2)
        self.assertEqual(manifest['cpu_demand'], 17)

    # Params: self; returns: None.
    def test_wrong_worker_topology_is_refused_without_science_changes(self):
        for config in (replace(self.config, execution=ExecutionConfig(subproc=False)),
                       replace(self.config, ppo=replace(self.config.ppo, n_envs=4, batch_size=4))):
            self.config = config; self.write_config()
            before = self.config_path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'eight subprocess'):
                self.manifest()
            self.assertEqual(self.config_path.read_bytes(), before)
            self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_invalid_config_and_finalizer_allocation_fail_before_writes(self):
        with self.assertRaisesRegex(ValueError, 'finalizer threads'):
            self.manifest('--finalizer-threads', '2')
        self.config = replace(self.config, reference_scaled=None, reference_status='unselected')
        self.write_config()
        with self.assertRaises(ValueError):
            self.manifest()
        self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_nonempty_experiment_existing_submission_and_nested_logs_refused(self):
        self.experiment.mkdir(); (self.experiment / 'keep').write_text('sentinel')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            self.manifest()
        self.assertEqual((self.experiment / 'keep').read_text(), 'sentinel')
        nested = self.args(); nested.submission_dir = self.experiment / 'slurm'
        with self.assertRaisesRegex(ValueError, 'outside the experiment'):
            slurm.prepare(nested)
        self.assertFalse(nested.submission_dir.exists())
        self.directory.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.manifest()

    # Params: self; returns: None.
    def test_manifest_detects_environment_and_configuration_drift(self):
        manifest = self.manifest()
        slurm.verify(manifest)
        with patch.object(slurm, 'environment_state', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Python environment changed'):
                slurm.verify(manifest)
        self.config_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            slurm.verify(manifest)

    # Params: self; returns: None.
    def test_manifest_detects_added_removed_and_modified_sources(self):
        manifest = self.manifest()
        project = self.root / 'source snapshot'
        (project / 'agents').mkdir(parents=True)
        source = project / 'agents/model.py'; source.write_text('original')
        manifest['project'] = str(project)
        manifest['source_hashes'] = slurm.source_state(project)
        with patch.object(slurm, 'PROJECT', project):
            slurm.verify(manifest)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                slurm.verify(manifest)
            source.write_text('original')
            extra = project / 'agents/extra.py'; extra.write_text('added')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                slurm.verify(manifest)
            extra.unlink(); source.unlink()
            with self.assertRaisesRegex(ValueError, 'source changed'):
                slurm.verify(manifest)

    # Params: self; returns: None.
    def test_stage_commands_share_interpreter_source_and_full_saved_experiment(self):
        manifest = self.manifest('--learner-threads', '2', '--policy-cpus', '10')
        for stage in slurm.STAGES:
            command, environ = slurm.stage_command(manifest, stage, self.environment(manifest, stage))
            self.assertEqual(command[:5], [manifest['environment']['python'], '-s', '-B', '-m',
                                          'agents.train.train_ppo_mo'])
            self.assertEqual(environ['PYTHONPATH'], str(PROJECT))
            self.assertTrue(all(environ[name] == str(manifest['resources'][stage]['threads'])
                                for name in slurm.THREAD_ENV))
            if stage == 'initialize':
                self.assertEqual(command[5:], ['--initialize', '--config', str(self.config_path),
                                               '--output', str(self.experiment)])
            elif stage == 'policy':
                self.assertEqual(command[5:], ['--experiment', str(self.experiment), '--task-id', '5',
                                               '--learner-threads', '2', '--cpus-per-task', '10'])
                self.assertNotIn('--config', command)
            else:
                self.assertEqual(command[5:], ['--finalize', '--experiment', str(self.experiment)])

    # Params: self; returns: None.
    def test_bad_scheduler_identity_allocation_topology_and_array_bounds_fail(self):
        manifest = self.manifest()
        for change in ({'SLURM_JOB_ID': ''}, {'SLURM_CPUS_PER_TASK': '8'},
                       {'SLURM_NTASKS': '2'}, {'SLURM_JOB_NUM_NODES': '2'},
                       {'SLURM_ARRAY_TASK_ID': '-1'}, {'SLURM_ARRAY_TASK_ID': '6'},
                       {'SLURM_ARRAY_TASK_ID': ''}):
            environ = dict(self.environment(manifest, 'policy'), **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                slurm.stage_command(manifest, 'policy', environ)

    # Params: self; returns: None.
    def test_run_entry_checks_manifest_before_exec_and_preserves_exit_contract(self):
        manifest = self.manifest()
        environ = self.environment(manifest, 'policy')
        expected, expected_env = slurm.stage_command(manifest, 'policy', environ)
        with patch.dict(os.environ, environ, clear=True), patch.object(slurm.os, 'chdir') as chdir, \
             patch.object(slurm.os, 'execve') as execute:
            code = slurm.main(['run', '--manifest', str(self.directory / 'manifest.json'), '--stage', 'policy'])
        self.assertEqual(code, 0)
        chdir.assert_called_once_with(str(PROJECT))
        execute.assert_called_once_with(expected[0], expected, expected_env)
        self.config_path.write_text('{}')
        with patch.object(slurm.os, 'execve') as execute:
            self.assertEqual(slurm.main(['run', '--manifest', str(self.directory / 'manifest.json'),
                                         '--stage', 'policy']), 1)
        execute.assert_not_called()

    # Params: self; returns: None.
    def test_real_initialization_command_retains_full_grid_and_no_training(self):
        manifest = self.manifest()
        command, environ = slurm.stage_command(manifest, 'initialize', self.environment(manifest, 'initialize'))
        result = subprocess.run(command, cwd=str(PROJECT), env=environ, text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual((summary['mode'], summary['n_tasks']), ('initialize', 6))
        saved = json.loads((self.experiment / 'config.json').read_text())
        self.assertEqual(saved['run_seeds'], [9, 2]); self.assertEqual(saved['ppo']['n_envs'], 8)
        self.assertEqual(saved['execution']['worker_threads'], 1)
        self.assertEqual(len(json.loads((self.experiment / 'seed_plan.json').read_text())), 2)
        self.assertFalse(list(self.experiment.rglob('COMPLETE.json')))
        self.assertFalse(list(self.experiment.rglob('model.zip')))

    # Params: self; returns: None.
    def test_batch_templates_are_lf_and_exec_exactly_one_guarded_stage(self):
        for stage in slurm.STAGES:
            body = (PROJECT / 'scripts/slurm' / (stage + '.sh')).read_bytes()
            self.assertTrue(body.startswith(b'#!/bin/bash\n'))
            self.assertNotIn(b'\r', body)
            text = body.decode()
            self.assertIn('set -euo pipefail', text)
            self.assertIn('#SBATCH --nodes=1', text); self.assertIn('#SBATCH --ntasks=1', text)
            self.assertEqual(text.count('exec '), 1)
            self.assertIn('exec "$1" -s -B "$2" run --manifest "$3" --stage ' + stage, text)
            self.assertNotIn('pip install', text)

    # Params: self; returns: None.
    def test_help_and_import_are_passive(self):
        result = subprocess.run([sys.executable, '-s', '-B', str(PROJECT / 'scripts/slurm/mo_slurm.py'), '--help'],
                                cwd=str(self.root), capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('submit', result.stdout)
        self.assertFalse(self.experiment.exists()); self.assertFalse(self.directory.exists())

    # Params: self; returns: None.
    def test_bash_syntax_and_argument_forwarding_with_spaces(self):
        bash = (Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt'
                else Path(shutil.which('bash') or '/missing/bash'))
        if not bash.is_file():
            self.skipTest('Bash is not available')
        stub = self.root / 'fake python.sh'
        stub.write_bytes(b'#!/bin/bash\nprintf \'%s\\n\' "$@"\n')
        stub.chmod(0o755)
        launcher = str(self.root / 'shared source/mo_slurm.py')
        manifest = str(self.root / 'submission plan/manifest.json')
        for stage in slurm.STAGES:
            script = PROJECT / 'scripts/slurm' / (stage + '.sh')
            syntax = subprocess.run([str(bash), '-n', str(script)], capture_output=True, text=True, timeout=10)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            result = subprocess.run([str(bash), str(script), str(stub), launcher, manifest],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ['-s', '-B', launcher, 'run', '--manifest',
                                                         manifest, '--stage', stage])
            missing = subprocess.run([str(bash), str(script)], capture_output=True, text=True, timeout=10)
            self.assertEqual(missing.returncode, 2)
        stub.write_bytes(b'#!/bin/bash\nexit 7\n')
        result = subprocess.run([str(bash), str(PROJECT / 'scripts/slurm/policy.sh'), str(stub), launcher, manifest],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)


if __name__ == '__main__':
    unittest.main()
