"""Real sequential versus concurrent MO smoke check, independent of production.

check --output NEW_DIRECTORY retains evidence, complete test runs and logs.
worker is a test-only subprocess entry point; barriers hold real policy claims,
and interruption happens only before the update after a committed checkpoint.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT / 'tests/fixtures/mo_parallel_smoke.json'
sys.path.insert(0, str(PROJECT))


# Params: condition, message; returns: None.
def require(condition, message):
    if not condition:
        raise AssertionError(message)


# Params: none; returns: CPU count or None.
def available_cpus():
    from agents.mo.runtime import cpu_allocation
    allocation = cpu_allocation()
    return allocation['allocated_cpus'] or allocation['visible_cpus']


# Params: root, shared; returns: dict.
def snapshot(root, shared=False):
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file() and (not shared or p.parent in (root, root / 'results')
            or root / 'logs' in p.parents or root / 'reports' in p.parents)}


# Params: args; returns: code.
def worker(args):
    from unittest.mock import patch
    from agents.mo import reporting, trainer
    from agents.train.train_ppo_mo import main as cli
    factory, learn = reporting.make_logger, trainer.learn_one_update
    claims = []

    # Params: config, destination; returns: owned.
    def logger(config, destination):
        owned = factory(config, destination)
        claims.append(owned)
        if args.gate is not None:
            require((destination / 'OWNER.LOCK').is_dir(), 'Barrier must hold the real policy claim')
            (args.gate / ('claimed_' + str(args.task_id))).write_text(str(os.getpid()))
            deadline = time.monotonic() + 90
            while not (args.gate / 'release').exists():
                if time.monotonic() > deadline:
                    owned.close()
                    raise TimeoutError('Integration claim barrier expired')
                time.sleep(.025)
        return owned

    # Params: model, config, *rest, **kwargs; returns: learn result.
    def update(model, config, *rest, **kwargs):
        if args.interrupt and model.num_timesteps == config.ppo.n_envs * config.ppo.n_steps:
            raise KeyboardInterrupt('Integration interruption after the first committed checkpoint')
        return learn(model, config, *rest, **kwargs)

    with patch.object(reporting, 'make_logger', side_effect=logger), \
         patch.object(trainer, 'learn_one_update', side_effect=update):
        code = cli(['--experiment', str(args.experiment), '--task-id', str(args.task_id),
                    '--learner-threads', '1', '--cpus-per-task', '3'])
    require(all(item._closed for item in claims), 'Worker left an owned logger open')
    require(not mp.active_children(), 'Worker leaked a simulation process')
    print(json.dumps({'smoke_worker': args.task_id, 'exit_code': code,
                      'simulation_children_after_exit': 0}), flush=True)
    return code


class Processes:
    """Keep bounded child invocations and kill their full trees after a timeout."""
    # Params: self, directory; returns: None.
    def __init__(self, directory):
        import psutil
        self.psutil = psutil
        self.directory = directory
        self.live = []
        self.records = []
        self.environment = dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONNOUSERSITE='1')
        self.environment.pop('SLURM_CPUS_PER_TASK', None)
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            self.environment[name] = '1'

    # Params: self, name, command; returns: process.
    def start(self, name, command):
        out = (self.directory / (name + '.out')).open('w', encoding='utf-8')
        err = (self.directory / (name + '.err')).open('w', encoding='utf-8')
        try:
            process = subprocess.Popen(command, cwd=str(PROJECT), env=self.environment, stdout=out, stderr=err)
        except BaseException:
            out.close(); err.close()
            raise
        self.live.append((name, process, out, err))
        return process

    # Params: self, process, expected; returns: (record, stderr).
    def finish(self, process, expected=0):
        name, _, out, err = next(item for item in self.live if item[1] is process)
        process.wait(timeout=150)
        out.close(); err.close()
        stdout = (self.directory / (name + '.out')).read_text(encoding='utf-8')
        stderr = (self.directory / (name + '.err')).read_text(encoding='utf-8')
        require(process.returncode == expected,
                f'{name}: expected exit {expected}, received {process.returncode}\n{stdout}\n{stderr}')
        documents = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
        record = {'name': name, 'exit_code': process.returncode, 'documents': documents}
        self.records.append(record)
        return record, stderr

    # Params: self, name, arguments, expected; returns: self.finish result.
    def cli(self, name, arguments, expected=0):
        return self.finish(self.start(name, [sys.executable, '-s', '-B', '-m',
                          'agents.train.train_ppo_mo', *arguments]), expected)

    # Params: self; returns: None.
    def close(self):
        for _, process, out, err in self.live:
            if process.poll() is None:
                try:
                    owner = self.psutil.Process(process.pid)
                    tree = owner.children(recursive=True) + [owner]
                    for member in tree:
                        try:
                            member.terminate()
                        except self.psutil.NoSuchProcess:
                            pass
                    _, alive = self.psutil.wait_procs(tree, timeout=5)
                    for member in alive:
                        try:
                            member.kill()
                        except self.psutil.NoSuchProcess:
                            pass
                except self.psutil.NoSuchProcess:
                    pass
                process.wait(timeout=15)
            out.close(); err.close()


# Params: gate, tasks, processes; returns: dict.
def wait_claims(gate, tasks, processes):
    deadline = time.monotonic() + 60
    while not all((gate / ('claimed_' + str(task))).is_file() for task in tasks):
        require(all(p.poll() is None for p in processes), 'A learner exited before overlapping claims')
        if time.monotonic() > deadline:
            raise TimeoutError('Learners did not overlap their policy claims')
        time.sleep(.025)
    return {str(task): int((gate / ('claimed_' + str(task))).read_text()) for task in tasks}


# Params: record; returns: next result.
def policy_summary(record):
    return next(doc for doc in record['documents'] if doc.get('status') == 'policy_complete')


# Params: saved; returns: dict.
def validate_records(saved):
    import numpy as np
    from agents.mo import storage
    from agents.mo.pareto import hypervolume_2d_max
    config, root, budget = saved.config, saved.root, saved.budget_plan
    intermediate = storage.load_evaluation_records(root, config, write_cache=False)
    require(intermediate.validity.shape == (2, 2, 2) and intermediate.validity.all(),
            'Intermediate grid is incomplete or has renumbered axes')
    require(np.isfinite(intermediate.means_raw).all(), 'Intermediate returns are not finite')
    identities, evaluations = [], []
    for i, plan in enumerate(saved.seed_plans):
        for j, weight in enumerate(saved.weights):
            bundle, _, _ = storage._committed_bundle(root / 'models' / f'seed_{i}' / f'w_{j}' / 'final',
                                                    saved.fingerprint)
            identity = bundle.identity
            expected = (i, plan.run_seed, j, tuple(weight.tolist()), budget.effective_total)
            require((identity.run_index, identity.run_seed, identity.weight_index, identity.weight,
                     identity.actual_step) == expected, 'Final bundle changed policy identity or actual budget')
            require((bundle.directory.parent / 'COMPLETE.json').is_file(), 'Missing policy completion marker')
            identities.append(list(expected[:3]) + [list(expected[3]), expected[4]])
            for namespace, steps, seeds in (
                    ('evaluations', budget.eval_steps, plan.eval_seeds),
                    ('final', (budget.effective_total,), plan.final_seeds)):
                for step in steps:
                    path = root / 'results' / namespace / f'seed_{i}' / f'w_{j}' / f'step_{step}.json'
                    result, metadata = storage._load_evaluation_record(path)
                    require(result.episode_seeds == seeds, 'Evaluation changed saved ordered episode seeds')
                    require(result.actual_step == step and result.deterministic, 'Evaluation budget/mode changed')
                    require((result.identity.run_index, result.identity.run_seed, result.identity.weight_index,
                             result.identity.weight, result.identity.experiment_fingerprint)
                            == (i, plan.run_seed, j, tuple(weight.tolist()), saved.fingerprint),
                            'Evaluation identity does not match its saved slot')
                    require(np.isfinite(result.returns_raw).all() and np.isfinite(result.returns_scaled).all(),
                            'Evaluation contains nonfinite returns')
                    np.testing.assert_allclose(result.returns_scaled,
                        result.returns_raw / np.asarray(config.objective_scales), rtol=1e-12, atol=1e-12)
                    require(metadata['kind'] == namespace,
                            'Evaluation used the wrong namespace')
                    evaluations.append({'namespace': namespace, 'run_index': i, 'weight_index': j,
                                        'actual_step': step, 'episode_seeds': list(seeds)})
    final = storage._read_json(root / 'results/final_results.json')
    require(final['run_seeds'] == [9, 2], 'Final aggregation renumbered run seeds')
    with np.load(root / 'results/final_results.npz', allow_pickle=False) as numeric:
        require(numeric['validity'].shape == (2, 2) and numeric['validity'].all(), 'Final grid incomplete')
        require(numeric['hypervolume'].shape == (2,), 'Hypervolume lost its per-training-seed axis')
        for i in range(2):
            expected_hv = hypervolume_2d_max(numeric['means_scaled'][i], config.reference_scaled)
            np.testing.assert_allclose(numeric['hypervolume'][i], expected_hv, rtol=1e-12, atol=1e-12)
    report = storage._read_json(root / 'reports/report_manifest.json')
    require(report['experiment_fingerprint'] == saved.fingerprint, 'Report fingerprint mismatch')
    require(report['represented_budgets'] == list(budget.eval_steps) and report['seed_counts'] == [2, 2],
            'Reports do not represent both complete training seeds at both evaluation budgets')
    for relative, expected_hash in report['file_hashes'].items():
        require(hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected_hash,
                'Report artifact hash does not match its manifest')
    return {'policy_identities': identities, 'evaluation_contracts': evaluations,
            'effective_total': budget.effective_total, 'requested_total': budget.requested_total,
            'eval_steps': list(budget.eval_steps), 'complete_policies': len(identities)}


# Params: output; returns: evidence.
def run_check(output):
    import numpy as np
    from agents.mo import storage
    from agents.mo.experiment import load_experiment
    capacity = available_cpus()
    require(capacity is None or capacity >= 6,
            'Concurrent smoke check needs six allocated/visible CPUs (two learners with two workers each)')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True)  # Refuse to mix evidence with an existing directory.
    logs = output / 'logs'; logs.mkdir()
    started = time.monotonic()
    progress_path = output / 'progress.jsonl'

    # Params: event, **fields; returns: None.
    def progress(event, **fields):
        line = json.dumps({'event': event, **fields}, allow_nan=False)
        print(line, flush=True)
        with progress_path.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    production = PROJECT / 'configs/ppo_mo_r6_r2.json'
    production_hash = hashlib.sha256(production.read_bytes()).hexdigest() if production.is_file() else None
    fixture_hash = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    seq, concurrent = output / 'sequential', output / 'concurrent'
    suite = Processes(logs)
    try:
        progress('sequential_grid_started')
        seq_record, _ = suite.cli('sequential', ['--config', str(FIXTURE), '--output', str(seq),
            '--learner-threads', '1', '--cpus-per-task', '3'])
        seq_saved = load_experiment(seq)
        require(seq_saved.budget_plan.requested_total == 5 and seq_saved.budget_plan.effective_total == 8,
                'Fixture must exercise rollout budget rounding')
        progress('sequential_grid_complete', policies=4, effective_transitions=32)
        suite.cli('initialize', ['--initialize', '--config', str(FIXTURE), '--output', str(concurrent)])
        saved = load_experiment(concurrent)
        require(saved.fingerprint == seq_saved.fingerprint, 'Execution mode changed scientific fingerprint')
        np.testing.assert_array_equal(saved.weights, seq_saved.weights)
        require(saved.seed_plans == seq_saved.seed_plans and saved.budget_plan == seq_saved.budget_plan,
                'Execution mode changed saved plans')
        storage.load_evaluation_records(concurrent, saved.config)
        shared = snapshot(concurrent, shared=True)
        gate = output / 'claims'; gate.mkdir()
        processes = []
        for task in (0, 3):
            command = [sys.executable, '-s', '-B', str(Path(__file__).resolve()), 'worker',
                '--experiment', str(concurrent), '--task-id', str(task), '--gate', str(gate)]
            if task == 0:
                command.append('--interrupt')
            processes.append(suite.start('concurrent_' + str(task), command))
        overlapping = wait_claims(gate, (0, 3), processes)
        require(len(set(overlapping.values())) == 2, 'Learners must be separate overlapping processes')
        progress('distinct_policy_claims_overlap', learner_pids=overlapping)
        paused = snapshot(concurrent)
        _, duplicate_error = suite.cli('duplicate', ['--experiment', str(concurrent), '--task-id', '0',
            '--learner-threads', '1', '--cpus-per-task', '3'], expected=1)
        require('already claimed' in duplicate_error, 'Duplicate writer was not refused by ownership')
        require(snapshot(concurrent) == paused, 'Duplicate writer changed artifacts')
        _, active_error = suite.cli('active_finalizer', ['--finalize', '--experiment', str(concurrent)], expected=1)
        require('active/interrupted' in active_error, 'Finalizer did not refuse live writers')
        require(snapshot(concurrent) == paused, 'Active-grid finalization changed artifacts')
        (gate / 'release').write_text('continue')
        interrupted, _ = suite.finish(processes[0], expected=130)
        completed, _ = suite.finish(processes[1])
        require(policy_summary(completed)['actual_transitions'] == 8, 'Concurrent policy exceeded its budget')
        require(snapshot(concurrent, shared=True) == shared, 'Policy workers changed shared caches/logs/reports')
        checkpoint = concurrent / 'models/seed_0/w_0/checkpoints/step_4'
        require((checkpoint / 'COMMITTED').is_file(), 'Interruption did not retain a committed checkpoint')
        require(not (checkpoint.parent.parent / 'COMPLETE.json').exists(), 'Interrupted policy marked complete')
        require(not list(concurrent.rglob('OWNER.LOCK')), 'Worker ownership leaked after interruption')
        progress('duplicate_and_active_finalization_refused', interrupted_at=4)
        remaining = [suite.start('concurrent_' + str(task), [sys.executable, '-s', '-B', '-m',
            'agents.train.train_ppo_mo', '--experiment', str(concurrent), '--task-id', str(task),
            '--learner-threads', '1', '--cpus-per-task', '3']) for task in (1, 2)]
        for process in remaining:
            suite.finish(process)
        require(snapshot(concurrent, shared=True) == shared, 'Remaining workers changed shared outputs')
        partial = snapshot(concurrent)
        _, incomplete_error = suite.cli('incomplete_finalizer', ['--finalize', '--experiment', str(concurrent)], expected=1)
        require('requires every policy complete' in incomplete_error, 'Incomplete grid was not refused')
        require(snapshot(concurrent) == partial, 'Incomplete finalizer changed committed/shared artifacts')
        require(not (concurrent / 'FINALIZED.json').exists(), 'Incomplete grid has a finalization marker')
        resumed, _ = suite.cli('resume', ['--experiment', str(concurrent), '--task-id', '0',
            '--learner-threads', '1', '--cpus-per-task', '3'])
        resume = policy_summary(resumed)['resume']
        require(resume['restored_step'] == 4 and resume['additional_training_transitions'] == 4,
                'Resume repeated or skipped training transitions')
        require(snapshot(concurrent, shared=True) == shared, 'Resumed worker changed shared outputs')
        before_skip = snapshot(concurrent)
        skipped, _ = suite.cli('completed_retry', ['--experiment', str(concurrent), '--task-id', '3',
            '--learner-threads', '1', '--cpus-per-task', '3'])
        skip = policy_summary(skipped)['resume']
        require(skip['skipped_complete'] and skip['additional_training_transitions'] == 0,
                'Completed task retrained')
        require(snapshot(concurrent) == before_skip, 'Completed task retry changed artifacts')
        progress('incomplete_finalization_refused_and_checkpoint_resumed', restored_step=4, additional_transitions=4)
        suite.cli('finalize', ['--finalize', '--experiment', str(concurrent)])
        require((concurrent / 'FINALIZED.json').is_file(), 'Finalizer did not publish success')
        require(not list(concurrent.rglob('OWNER.LOCK')) and not (concurrent / 'FINALIZER.LOCK').exists(),
                'Completed grid leaked ownership')
        sequential_contract = validate_records(seq_saved)
        concurrent_contract = validate_records(saved)
        require(sequential_contract == concurrent_contract,
                'Sequential/concurrent policy identities, budgets or evaluation seeds differ')
        require(hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == fixture_hash, 'Fixture was modified')
        require(production_hash is None or hashlib.sha256(production.read_bytes()).hexdigest() == production_hash,
                'Production configuration was modified')
        sequential_summary = next(d for d in seq_record['documents'] if d.get('status') == 'complete')
        require(sequential_summary['effective_training_transitions'] == 32, 'Sequential grid cost changed')
        concurrent_cost = 4 + sum(doc['resume']['additional_training_transitions']
            for record in suite.records for doc in record['documents'] if doc.get('status') == 'policy_complete')
        require(concurrent_cost == sequential_summary['effective_training_transitions'],
                'Interrupted/concurrent execution repeated or omitted training transitions')
        evidence = {'status': 'passed', 'experiment_fingerprint': saved.fingerprint,
            'fixture': str(FIXTURE), 'fixture_sha256': fixture_hash, 'production_config_sha256': production_hash,
            'contract': concurrent_contract, 'overlapping_claim_pids': overlapping,
            'duplicate_claim_refused': True, 'active_finalizer_refused': True,
            'incomplete_finalizer_refused': True, 'shared_worker_outputs_unchanged': True,
            'sequential_effective_training_transitions': 32,
            'concurrent_effective_training_transitions': concurrent_cost,
            'available_cpus': capacity,
            'resume': resume, 'completed_retry': skip, 'processes': suite.records,
            'comparison': 'Policy identity, saved plans, budgets, episode seeds and completeness; no reward/model bitwise equality requirement',
            'elapsed_seconds': time.monotonic() - started}
        (output / 'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        progress('passed', evidence=str(output / 'evidence.json'), elapsed_seconds=evidence['elapsed_seconds'])
        return evidence
    except BaseException as error:
        progress('failed', error=repr(error))
        raise
    finally:
        suite.close()


# Params: argv; returns: worker result or int.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check'); check.add_argument('--output', type=Path, required=True)
    child = commands.add_parser('worker')
    child.add_argument('--experiment', type=Path, required=True)
    child.add_argument('--task-id', type=int, required=True)
    child.add_argument('--gate', type=Path)
    child.add_argument('--interrupt', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'worker':
        return worker(args)
    run_check(args.output)
    return 0


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
