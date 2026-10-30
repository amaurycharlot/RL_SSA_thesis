"""Execution limits must preserve science and bound independent worker teams."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from threadpoolctl import threadpool_info
from agents.mo import trainer, envs, storage
from agents.mo.runtime import execution_resources, cpu_allocation, THREAD_ENV
from agents.mo.types import ExperimentConfig, ExecutionConfig
from agents.train import train_ppo_mo as cli


# Params: none; returns: sorted result.
def pools():
    return sorted((p['filepath'], p['num_threads']) for p in threadpool_info())


class ResourceLimits(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start(); self.addCleanup(self.env.stop)
        os.environ.pop('SLURM_CPUS_PER_TASK', None)
        self.config = ExperimentConfig()

    # Params: self; returns: None.
    def test_eight_workers_require_nine_cpus_for_one_thread_learner(self):
        with patch('agents.mo.runtime.cpu_allocation', return_value={
                'allocated_cpus': 8, 'allocation_bounds': {'cpus_per_task': 8}, 'visible_cpus': 8}):
            with self.assertRaisesRegex(ValueError, 'CPU demand 9'):
                with execution_resources(self.config, learner_threads=1, cpus_per_task=8):
                    self.fail('oversubscribed execution entered')

    # Params: self; returns: None.
    def test_scheduler_and_explicit_bounds_use_minimum(self):
        os.environ['SLURM_CPUS_PER_TASK'] = '10'
        with patch('agents.mo.runtime.os.cpu_count', return_value=24), \
             patch('agents.mo.runtime.os.sched_getaffinity', return_value=set(range(24)), create=True):
            self.assertEqual(cpu_allocation(20)['allocated_cpus'], 10)
            self.assertEqual(cpu_allocation(9)['allocated_cpus'], 9)
        with patch('agents.mo.runtime.os.sched_getaffinity', return_value=set(range(4)), create=True):
            self.assertEqual(cpu_allocation(20)['allocated_cpus'], 4)

    # Params: self; returns: None.
    def test_bad_scheduler_allocations_fail_closed(self):
        for value in ('0', '-1', 'two', '4(x2)', ''):
            with self.subTest(value=value):
                os.environ['SLURM_CPUS_PER_TASK'] = value
                with self.assertRaises(ValueError):
                    cpu_allocation()

    # Params: self; returns: None.
    def test_managed_allocation_requires_explicit_learner_limit(self):
        os.environ['SLURM_CPUS_PER_TASK'] = '12'
        with self.assertRaisesRegex(ValueError, 'explicit learner_threads'):
            with execution_resources(self.config):
                self.fail('unbounded learner entered')

    # Params: self; returns: None.
    def test_unknown_allocation_needs_explicit_cpu_count(self):
        with patch('agents.mo.runtime.cpu_allocation', return_value={
                'allocated_cpus': None, 'allocation_bounds': {}, 'visible_cpus': None}):
            with self.assertRaisesRegex(ValueError, 'allocation is unknown'):
                with execution_resources(self.config, learner_threads=1):
                    self.fail('unknown allocation entered')

    # Params: self; returns: None.
    def test_missing_thread_control_dependency_restores_environment(self):
        config = replace(self.config, execution=ExecutionConfig(subproc=False))
        before = (torch.get_num_threads(), pools(), dict(os.environ))
        with patch.dict(sys.modules, {'threadpoolctl': None}):
            with self.assertRaisesRegex(RuntimeError, 'requirements-mo-runtime.txt'):
                with execution_resources(config, learner_threads=1):
                    self.fail('missing numerical controls entered')
        self.assertEqual((torch.get_num_threads(), pools(), dict(os.environ)), before)

    # Params: self; returns: None.
    def test_sequential_runner_keeps_limits_through_final_testing(self):
        config = replace(self.config, execution=ExecutionConfig(subproc=False))
        before = (torch.get_num_threads(), pools(), dict(os.environ))
        # Params: *args; returns: str.
        def run(*args):
            self.assertIs(args[0], config)
            self.assertEqual(args[3], {'learner_threads': 1})
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertTrue(all(count == 1 for _, count in pools()))
            return 'summary'
        with patch.object(trainer, '_run_experiment', side_effect=run):
            self.assertEqual(trainer.run_experiment(config, learner_threads=1), 'summary')
        self.assertEqual((torch.get_num_threads(), pools(), dict(os.environ)), before)

    # Params: self; returns: None.
    def test_invalid_python_options(self):
        for key in ('learner_threads', 'cpus_per_task'):
            for value in (0, -1, True, 1.5, '2'):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    with execution_resources(self.config, **{key: value}):
                        self.fail('invalid limit entered')

    # Params: self; returns: None.
    def test_default_preserves_parent_counts_and_environment(self):
        before = (torch.get_num_threads(), pools(), dict(os.environ))
        with execution_resources(self.config) as record:
            self.assertIsNone(record['allocated_cpus'])
            self.assertFalse(record['allocation_checked'])
            self.assertEqual(record['worker_processes'], 8)
        self.assertEqual((torch.get_num_threads(), pools(), dict(os.environ)), before)

    # Params: self; returns: None.
    def test_control_and_exception_restore_all_parent_limits(self):
        config = replace(self.config, execution=ExecutionConfig(subproc=False))
        for failure in (False, True):
            before = (torch.get_num_threads(), pools(), dict(os.environ))
            try:
                with execution_resources(config, learner_threads=1) as record:
                    self.assertEqual(torch.get_num_threads(), 1)
                    self.assertTrue(all(count == 1 for _, count in pools()))
                    self.assertEqual(record['cpu_demand'], 1)
                    self.assertEqual(record['worker_processes'], 0)
                    self.assertTrue(all(os.environ[k] == '1' for k in THREAD_ENV))
                    if failure:
                        raise OSError('interrupted')
            except OSError:
                self.assertTrue(failure)
            self.assertEqual((torch.get_num_threads(), pools(), dict(os.environ)), before)

    # Params: self; returns: None.
    def test_parser_strict_limits_and_modes(self):
        args = cli.parse_args(['--preflight', '--learner-threads', '2', '--cpus-per-task', '10'])
        self.assertEqual(cli._execution_options(args), {'learner_threads': 2, 'cpus_per_task': 10})
        for args in (['--learner-threads', '0'], ['--learner-threads', '1.5'],
                     ['--cpus-per-task', '10'], ['--initialize', '--learner-threads', '1'],
                     ['--finalize', '--experiment', 'saved', '--learner-threads', '1']):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                cli.parse_args(args)


class RealPolicyResources(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from test_mo_task_cli import PolicyTaskCLI
        self.fixture = PolicyTaskCLI(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.config = replace(self.fixture.config, execution=ExecutionConfig(subproc=True))
        self.saved = self.fixture.saved()
        self.env = patch.dict(os.environ)
        self.env.start(); self.addCleanup(self.env.stop)
        os.environ.pop('SLURM_CPUS_PER_TASK', None)

    # Params: self, threads; returns: trainer.train_one_policy result.
    def run_policy(self, threads):
        saved = self.saved
        return trainer.train_one_policy(saved.config, saved.root, 0, 0, saved.weights,
            saved.seed_plans[0], saved.budget_plan, learner_threads=threads,
            cpus_per_task=threads + 2, cache_results=False)

    # Params: self; returns: None.
    def test_spawn_resume_limits_before_fresh_and_reloaded_models_and_science_unchanged(self):
        from test_mo_task_cli import snapshot
        before = (torch.get_num_threads(), pools(), dict(os.environ))
        science = {name: (self.saved.root / name).read_bytes() for name in
                   ('config.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json', 'fingerprint.json')
                   if (self.saved.root / name).exists()}
        original_factory, original_build = envs.make_env_factory, trainer.build_model
        original_load, original_learn = storage.load_policy_bundle, trainer.learn_one_update
        seen = {'build': 0, 'reload': 0, 'workers': 0}

        # Params: *args, **kwargs; returns: construct.
        def factory(*args, **kwargs):
            base = original_factory(*args, **kwargs)
            # Params: none; returns: env.
            def construct():
                import torch
                from threadpoolctl import threadpool_info
                env = base()
                env._block7_threads = (torch.get_num_threads(),
                    [p['num_threads'] for p in threadpool_info()])
                return env
            return construct

        expected_threads = [2]
        # Params: none; returns: None.
        def check_parent():
            self.assertEqual(torch.get_num_threads(), expected_threads[0])
            self.assertTrue(all(count <= expected_threads[0] for _, count in pools()))

        # Params: config, handle, plan; returns: original_build result.
        def build(config, handle, plan):
            check_parent(); seen['build'] += 1
            rows = handle.vec_env.get_attr('_block7_threads')
            self.assertEqual(len(rows), 2)
            for count, numerical in rows:
                self.assertEqual(count, 1)
                self.assertTrue(numerical); self.assertTrue(all(n == 1 for n in numerical))
                seen['workers'] += 1
            return original_build(config, handle, plan)

        # Params: *args, **kwargs; returns: original_load result.
        def load(*args, **kwargs):
            check_parent(); seen['reload'] += 1
            return original_load(*args, **kwargs)

        # Params: model, *args, **kwargs; returns: original_learn result.
        def interrupted(model, *args, **kwargs):
            if model.num_timesteps == 2:
                raise KeyboardInterrupt()
            return original_learn(model, *args, **kwargs)

        with patch.object(envs, 'make_env_factory', side_effect=factory), \
             patch.object(trainer, 'build_model', side_effect=build), \
             patch.object(trainer, 'learn_one_update', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_policy(2)
        self.assertEqual((torch.get_num_threads(), pools(), dict(os.environ)), before)
        expected_threads[0] = 1
        with patch.object(storage, 'load_policy_bundle', side_effect=load):
            summary = self.run_policy(1)
        self.assertTrue(summary.complete)
        self.assertEqual(summary.resume_metadata['restored_step'], 2)
        self.assertEqual(summary.resume_metadata['additional_training_transitions'], 2)
        self.assertEqual(seen, {'build': 1, 'reload': 1, 'workers': 2})
        self.assertEqual((torch.get_num_threads(), pools(), dict(os.environ)), before)
        for name, contents in science.items():
            self.assertEqual((self.saved.root / name).read_bytes(), contents)
        policy = self.saved.root / 'models/seed_0/w_0'
        events = [storage._read_json(p) for p in (policy / 'logs/events').glob('*.json')]
        resource_events = [e for e in events if e['event'] == 'execution_resources']
        self.assertEqual(len(resource_events), 2)
        self.assertEqual(sorted(e['payload']['learner_torch_threads'] for e in resource_events), [1, 2])
        self.assertTrue(all(e['payload']['resolved_device'] == 'cpu' for e in resource_events))
        completed = snapshot(self.saved.root)
        with patch.object(trainer, 'build_model', side_effect=AssertionError('completed retry allocated model')):
            self.assertTrue(self.run_policy(1).resume_metadata['skipped_complete'])
        self.assertEqual(snapshot(self.saved.root), completed)

    # Params: self; returns: None.
    def test_cli_allocation_failure_is_read_only_before_training(self):
        from test_mo_task_cli import snapshot
        before = snapshot(self.saved.root)
        with patch.object(trainer, 'train_one_policy') as train:
            code, docs, error = self.fixture.main(['--experiment', str(self.saved.root),
                '--task-id', '0', '--learner-threads', '2', '--cpus-per-task', '3'])
        self.assertEqual(code, 2); self.assertFalse(docs)
        self.assertIn('CPU demand 4', error)
        train.assert_not_called()
        self.assertEqual(snapshot(self.saved.root), before)

    # Params: self; returns: None.
    def test_cli_forwards_limits_and_reports_scheduler_bound(self):
        os.environ['SLURM_CPUS_PER_TASK'] = '4'
        with patch.object(trainer, 'train_one_policy', return_value=self.fixture.summary(self.saved)) as train:
            code, docs, error = self.fixture.main(['--experiment', str(self.saved.root),
                '--task-id', '5', '--learner-threads', '1', '--cpus-per-task', '3'])
        self.assertEqual(code, 0, error)
        self.assertEqual(train.call_args.kwargs, {
            'cache_results': False, 'learner_threads': 1, 'cpus_per_task': 3})
        record = docs[-1]['execution_resources']
        self.assertEqual(record['allocated_cpus'], 3)
        self.assertEqual(record['allocation_bounds']['SLURM_CPUS_PER_TASK'], 4)
        self.assertEqual(record['cpu_demand'], 3)
        self.assertEqual(record['learner_torch_threads'], 1)


if __name__ == '__main__':
    unittest.main()
