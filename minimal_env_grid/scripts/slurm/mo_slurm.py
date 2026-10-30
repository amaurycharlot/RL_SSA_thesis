"""Submit one immutable MO grid as initialization, policy array and finalizer.

Run this with the shared training environment's Python. Jobs use that same
absolute interpreter and project directory; they never install or update code.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
PACKAGES = ('numpy', 'scipy', 'torch', 'gymnasium', 'stable_baselines3', 'threadpoolctl')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
STAGES = ('initialize', 'policy', 'finalize')


# Params: value; returns: positive int.
def positive(value):
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


# Params: path; returns: SHA-256 hex str.
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Params: project; returns: dict.
def source_state(project):
    files = []
    for directory in ('agents', 'ssa_env', 'scripts/slurm'):
        files.extend(p for p in (project / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.sh'))
    return {p.relative_to(project).as_posix(): digest(p) for p in sorted(files)}


def environment_state():
    # Do not resolve the interpreter symlink: doing so bypasses Linux venvs.
    return {'python': str(Path(sys.executable).absolute()),
            'python_version': sys.version,
            'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


# Params: path, value; returns: None.
def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare(args):
    """Validate the frozen full grid and resources before creating run files."""
    sys.path.insert(0, str(PROJECT))
    from agents.mo.config import load_config, validate_config
    config_path = args.config.expanduser().resolve(strict=True)
    experiment = args.experiment.expanduser().resolve()
    directory = args.submission_dir.expanduser().resolve()
    if directory == experiment or experiment in directory.parents:
        raise ValueError('submission-dir must be outside the experiment (initialization needs an empty destination)')
    if directory == PROJECT or directory in PROJECT.parents:
        raise ValueError('submission-dir must not replace the source project or an ancestor')
    if directory.exists():
        raise ValueError('submission-dir already exists; choose a new directory')
    if experiment.exists() and (not experiment.is_dir() or any(experiment.iterdir())):
        raise ValueError('experiment destination must be new or empty')
    config = load_config(config_path, {'output_dir': experiment})
    validate_config(config)
    if config.logging.mode != 'off' and (not config.logging.project or not config.logging.entity):
        raise ValueError('External logging requires owner-supplied project and entity')
    if not config.execution.subproc or config.ppo.n_envs != 8:
        raise ValueError('Slurm policy template requires the saved eight subprocess simulation workers')
    demand = args.learner_threads + 8 * config.execution.worker_threads
    if args.policy_cpus < demand:
        raise ValueError(f'policy CPU demand {demand} exceeds --policy-cpus {args.policy_cpus}')
    if args.finalizer_threads > args.finalizer_cpus:
        raise ValueError('finalizer threads exceed finalizer CPUs')
    n_tasks = len(config.run_seeds) * config.n_weights
    resources = {
        'initialize': {'cpus': args.init_cpus, 'memory': args.init_memory, 'time': args.init_time, 'threads': 1},
        'policy': {'cpus': args.policy_cpus, 'memory': args.policy_memory, 'time': args.policy_time,
                   'threads': args.learner_threads},
        'finalize': {'cpus': args.finalizer_cpus, 'memory': args.finalizer_memory, 'time': args.finalizer_time,
                     'threads': args.finalizer_threads}}
    for value in (args.account, args.partition, args.init_memory, args.policy_memory, args.finalizer_memory,
                  args.init_time, args.policy_time, args.finalizer_time) + (() if args.gres is None else (args.gres,)):
        if not value.strip() or any(c in value for c in '\r\n\x00'):
            raise ValueError('Slurm account, partition, memory and time must be nonempty single-line values')
    manifest = {'schema_version': 1, 'project': str(PROJECT), 'config': str(config_path),
        'config_sha256': digest(config_path), 'experiment': str(experiment),
        'submission_dir': str(directory), 'environment': environment_state(),
        'source_hashes': source_state(PROJECT), 'n_tasks': n_tasks,
        'n_seeds': len(config.run_seeds), 'n_weights': config.n_weights,
        'worker_processes': 8, 'worker_threads': config.execution.worker_threads,
        'cpu_demand': demand, 'resources': resources, 'account': args.account,
        'partition': args.partition, 'gres': args.gres, 'max_concurrent': args.max_concurrent}
    directory.mkdir(parents=True)
    (directory / 'logs').mkdir()
    write_json(directory / 'manifest.json', manifest)
    return manifest


# Params: manifest, stage, dependency; returns: sbatch argument list.
def sbatch_command(manifest, stage, dependency=None):
    directory = Path(manifest['submission_dir'])
    resource = manifest['resources'][stage]
    log_name = stage + ('-%A_%a' if stage == 'policy' else '-%j')
    command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1',
        '--account=' + manifest['account'], '--partition=' + manifest['partition'],
        '--cpus-per-task=' + str(resource['cpus']), '--mem=' + resource['memory'],
        '--time=' + resource['time'], '--chdir=' + manifest['project'], '--export=ALL',
        '--job-name=mo-' + stage, '--output=' + str(directory / 'logs' / (log_name + '.out')),
        '--error=' + str(directory / 'logs' / (log_name + '.err'))]
    if stage == 'policy':
        command.append(f"--array=0-{manifest['n_tasks'] - 1}%{manifest['max_concurrent']}")
    if dependency is not None:
        command.extend(['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes'])
    if stage != 'initialize' and manifest['gres']:
        command.append('--gres=' + manifest['gres'])
    command.extend([str(PROJECT / 'scripts/slurm' / (stage + '.sh')),
        manifest['environment']['python'], str(PROJECT / 'scripts/slurm/mo_slurm.py'),
        str(directory / 'manifest.json')])
    return command


# Params: args, runner; returns: dict.
def submit_workflow(args, runner=None):
    manifest = prepare(args)
    state = {'status': 'dry_run' if args.dry_run else 'submitting', 'jobs': {}, 'commands': {}}
    destination = Path(manifest['submission_dir']) / 'submission.json'
    runner = subprocess.run if runner is None else runner
    dependency = None
    try:
        for stage in STAGES:
            command = sbatch_command(manifest, stage, dependency)
            state['commands'][stage] = command
            write_json(destination, state)
            if args.dry_run:
                job_id = stage.upper() + '_JOB_ID'
            else:
                result = runner(command, check=True, capture_output=True, text=True)
                # --parsable may return jobid;cluster. Dependencies use the jobid.
                response = result.stdout.strip()
                if not re.fullmatch(r'[1-9][0-9]*(?:;[^\s;]+)?', response):
                    raise RuntimeError('Unrecognized sbatch job ID: ' + response)
                job_id = response.split(';', 1)[0]
            state['jobs'][stage] = job_id
            dependency = job_id
            write_json(destination, state)
        state['status'] = 'dry_run' if args.dry_run else 'submitted'
        write_json(destination, state)
    except BaseException as error:
        state['status'] = 'submission_failed'
        state['error'] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            state['scheduler_stderr'] = error.stderr
        write_json(destination, state)
        raise
    return {'status': state['status'], 'manifest': str(Path(manifest['submission_dir']) / 'manifest.json'),
            'submission': str(destination), 'n_tasks': manifest['n_tasks'], 'jobs': state['jobs']}


# Params: manifest; returns: None.
def verify(manifest):
    if manifest['schema_version'] != 1 or Path(manifest['project']) != PROJECT:
        raise ValueError('Manifest does not belong to this shared source project')
    if manifest['environment'] != environment_state():
        raise ValueError('Shared Python environment changed since submission')
    if manifest['source_hashes'] != source_state(PROJECT):
        raise ValueError('Shared source changed since submission')
    if manifest['config_sha256'] != digest(Path(manifest['config'])):
        raise ValueError('Frozen input configuration changed since submission')


# Params: manifest, stage, environ; returns: (command, environment).
def stage_command(manifest, stage, environ):
    resource = manifest['resources'][stage]
    if not environ.get('SLURM_JOB_ID'):
        raise ValueError('Run stages inside their submitted Slurm job')
    if positive(environ.get('SLURM_CPUS_PER_TASK')) != resource['cpus']:
        raise ValueError('Slurm CPU allocation differs from the submission manifest')
    for name in ('SLURM_NTASKS', 'SLURM_JOB_NUM_NODES'):
        if positive(environ.get(name, '1')) != 1:
            raise ValueError('Each job must use one node and one Slurm task')
    command = [manifest['environment']['python'], '-s', '-B', '-m', 'agents.train.train_ppo_mo']
    if stage == 'initialize':
        command.extend(['--initialize', '--config', manifest['config'], '--output', manifest['experiment']])
    elif stage == 'policy':
        value = environ.get('SLURM_ARRAY_TASK_ID', '')
        if not re.fullmatch(r'[0-9]+', value) or not 0 <= int(value) < manifest['n_tasks']:
            raise ValueError('SLURM_ARRAY_TASK_ID must identify a declared saved policy')
        command.extend(['--experiment', manifest['experiment'], '--task-id', str(int(value)),
                        '--learner-threads', str(resource['threads']),
                        '--cpus-per-task', str(resource['cpus'])])
    else:
        command.extend(['--finalize', '--experiment', manifest['experiment']])
    environment = dict(environ, PYTHONPATH=manifest['project'], PYTHONNOUSERSITE='1')
    for name in THREAD_ENV:
        environment[name] = str(resource['threads'])
    return command, environment


# Params: none; returns: ArgumentParser.
def parser():
    root = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = root.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('submit', allow_abbrev=False, help='Submit all three stages, or write a dry-run plan')
    for name in ('config', 'experiment', 'submission-dir'):
        submit.add_argument('--' + name, type=Path, required=True)
    for name in ('account', 'partition'):
        submit.add_argument('--' + name, required=True)
    submit.add_argument('--learner-threads', type=positive, default=1)
    submit.add_argument('--max-concurrent', type=positive, default=4)
    submit.add_argument('--finalizer-threads', type=positive, default=1)
    submit.add_argument('--gres', help='Optional shared policy/finalizer GRES request, e.g. gpu:1')
    for prefix, cpus, memory, time in (('init', 1, '2G', '00:10:00'),
                                      ('policy', 9, '8G', '24:00:00'),
                                      ('finalizer', 1, '8G', '04:00:00')):
        submit.add_argument('--' + prefix + '-cpus', type=positive, default=cpus)
        submit.add_argument('--' + prefix + '-memory', default=memory)
        submit.add_argument('--' + prefix + '-time', default=time)
    submit.add_argument('--dry-run', action='store_true', help='Write manifest and command plan without calling sbatch')
    run = commands.add_parser('run', allow_abbrev=False, help='Internal entry point used by batch templates')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--stage', choices=STAGES, required=True)
    return root


# Params: argv; returns: int.
def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'submit':
            print(json.dumps(submit_workflow(args), allow_nan=False), flush=True)
        else:
            manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
            verify(manifest)
            command, environment = stage_command(manifest, args.stage, os.environ)
            os.chdir(manifest['project'])
            os.execve(command[0], command, environment)
        return 0
    except KeyboardInterrupt:
        print('Slurm workflow interrupted; inspect submission.json for accepted jobs', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'Slurm workflow failed: {type(error).__name__}: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
