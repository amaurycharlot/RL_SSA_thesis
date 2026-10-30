"""Explicit MO training, read-only preflight and validated checkpoint resume.

Importing or requesting help allocates no environments, workers, models or run
files. Scientific settings belong in the JSON configuration, not CLI overrides.
"""
import argparse
from pathlib import Path


# Params: value; returns: Path.
def _path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError('path must be nonempty')
    return Path(value)


# Params: value; returns: number.
def _positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


# Params: args; returns: dict.
def _execution_options(args):
    return {key: getattr(args, key) for key in ('learner_threads', 'cpus_per_task')
            if getattr(args, key) is not None}


def parse_args(argv=None):
    """Select sequential, read-only preflight, initialize-only or one saved task."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=_path, help='Raw experiment JSON; defaults to the checked-in example, or saved configuration on resume')
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument('--output', type=_path, help='Exact new experiment destination; refuses nonempty directories')
    destination.add_argument('--resume', type=_path, metavar='EXPERIMENT', help='Resume the complete saved experiment sequentially')
    destination.add_argument('--experiment', type=_path, metavar='EXPERIMENT', help='Initialized experiment for --task-id; reads its full saved configuration')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='Check resolved plans and one local environment transition; create no run artifacts or training resources')
    mode.add_argument('--finalize', action='store_true', help='Finalize a complete saved grid in --experiment; no policy training')
    mode.add_argument('--initialize', action='store_true', help='Commit the full experiment grid only; no training, evaluation or reports')
    mode.add_argument('--task-id', type=int, metavar='N', help='Run/resume exactly one zero-based policy task in --experiment')
    parser.add_argument('--learner-threads', type=_positive_int, metavar='N', help='Execution-only learner Torch/BLAS thread limit; separate from saved worker threads')
    parser.add_argument('--cpus-per-task', type=_positive_int, metavar='N', help='Execution-only CPU allocation; also bounded by SLURM_CPUS_PER_TASK and CPU affinity')
    args = parser.parse_args(argv)
    if (args.initialize or args.finalize) and _execution_options(args):
        parser.error('learner resource options apply to training/preflight, not initialize/finalize')
    if args.cpus_per_task is not None and args.learner_threads is None:
        parser.error('--cpus-per-task requires --learner-threads')
    if args.task_id is not None:
        if args.task_id < 0:
            parser.error('--task-id must be nonnegative')
        if args.experiment is None:
            parser.error('--task-id requires --experiment')
        if args.config is not None:
            parser.error('--task-id loads saved configuration; --config is not permitted')
    elif args.finalize:
        if args.experiment is None:
            parser.error('--finalize requires --experiment')
        if args.config is not None:
            parser.error('--finalize loads saved configuration; --config is not permitted')
    elif args.experiment is not None:
        parser.error('--experiment requires --task-id or --finalize')
    if args.initialize and args.resume is not None:
        parser.error('--initialize cannot use --resume')
    return args



def _prepare(args):
    """Resolve configuration/plans and validate destinations without writing."""
    import numpy as np
    from agents.mo import config as planning, storage
    from agents.mo.types import ExperimentConfig, SeedPlan, BudgetPlan
    destination = args.resume if args.resume is not None else args.output
    if args.config is None and args.resume is not None:
        root, payload, _ = storage._evaluation_context(args.resume)
        values = dict(payload['config'])
        values['logging'] = storage._read_json(root / 'config.json')['logging']
        for name, record_type in planning._NESTED.items():
            values[name] = record_type(**values[name])
        config = ExperimentConfig(**values, output_dir=root)
    else:
        source = args.config if args.config is not None else Path(__file__).resolve().parents[2] / 'configs/ppo_mo_r6_r2.json'
        overrides = {} if destination is None else {'output_dir': destination}
        config = planning.load_config(source, overrides)
    planning.validate_config(config, require_frozen_objectives=not args.preflight)
    if config.logging.mode != 'off' and (not config.logging.project or not config.logging.entity):
        raise ValueError('External logging requires owner-supplied project and entity')
    if config.output_dir is None:
        if config.experiment_id in ('.', '..') or any(c in config.experiment_id for c in '/\\:'):
            raise ValueError('experiment_id must be a single directory name')
        root = Path('runs_mo') / config.experiment_id
    else:
        root = config.output_dir
    root = root.expanduser().absolute().resolve()
    if args.resume is not None:
        weights = planning.validate_weights(np.load(root / 'weights.npy', allow_pickle=False))
        plans = tuple(SeedPlan(**p) for p in storage._read_json(root / 'seed_plan.json'))
        budget = BudgetPlan(**storage._read_json(root / 'budget_plan.json'))
        storage.create_experiment_dir(config, weights, plans, budget, resume=True)  # Validation only.
    else:
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise storage.StorageError('Experiment destination is nonempty or not a directory; use --resume EXPERIMENT')
        weights = planning.make_weights(config.n_weights)
        plans = tuple(planning.make_seed_plan(config, i) for i in range(len(config.run_seeds)))
        budget = planning.make_budget_plan(config)
    return config, root, weights, plans, budget


def _check_device(config):
    """Validate device availability without creating any environment or learner."""
    import torch
    from stable_baselines3.common.utils import get_device
    configured = config.ppo.device
    device = get_device(configured)
    if configured != 'auto':
        requested = torch.device(configured)
        if device.type != requested.type:
            raise ValueError(f'Requested device {configured} is unavailable; implicit CPU fallback is refused')
        if requested.type in ('cuda', 'xpu'):
            backend = getattr(torch, requested.type, None)
            if backend is None or not backend.is_available():
                raise ValueError(f'Requested device {configured} is unavailable')
            if requested.index is not None and requested.index >= backend.device_count():
                raise ValueError(f'Requested device index {configured} is unavailable')
        if requested.type == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('Requested MPS device is unavailable')
    return device


def _check_runtime(config, weights, seed_plan):
    """Own one scalarized raw environment; check the actual reward/obs contract."""
    from contextlib import closing
    import numpy as np
    from agents.mo.envs import make_scalar_env
    device = _check_device(config)
    configured = config.ppo.device
    with closing(make_scalar_env(config.env, weights[0], None, config.objective_scales)) as env:
        raw = env.unwrapped
        if (raw.mode != 'combined' or raw.reward_order != ('tracking', 'search')
                or raw.reward_dim != 2 or raw.reward_space.shape != (2,)
                or raw.reward_space.dtype != np.float32
                or not np.array_equal(raw.action_space.nvec, config.env.action_nvec)):
            raise ValueError('Environment reward/action metadata does not match the combined MO contract')
        observation, reset_info = env.reset(seed=seed_plan.base_worker_seed)
        # Params: value; returns: None.
        def check_observation(value):
            if (not isinstance(value, np.ndarray) or value.shape != config.env.observation_shape
                    or value.dtype != np.float32 or not np.isfinite(value).all()
                    or not env.observation_space.contains(value)):
                raise ValueError('Environment observation does not match the configured finite float32 grid')
        check_observation(observation)
        if not isinstance(reset_info, dict):
            raise ValueError('Environment reset diagnostics must be a dictionary')
        observation, reward, terminated, truncated, info = env.step(np.zeros(2, np.int64))
        check_observation(observation)
        if not np.isscalar(reward) or not np.isfinite(reward):
            raise ValueError('Scalarized training reward must be finite')
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
            raise ValueError('Environment termination flags must be boolean')
        raw_reward = np.asarray(info['vector_reward_raw'])
        scaled = np.asarray(info['vector_reward_scaled'])
        if (raw_reward.shape != (2,) or raw_reward.dtype != np.float32 or not np.isfinite(raw_reward).all()
                or not np.array_equal(scaled, raw_reward / np.asarray(config.objective_scales, np.float32))
                or float(reward) != float(np.dot(weights[0], scaled))):
            raise ValueError('Raw/scaled vector rewards and scalarization disagree')
        for field in ('true_states', 'estimated_states', 'observed_target_ids', 'elapsed_steps', 'total_steps', 'coverage_mean'):
            if field not in info:
                raise ValueError('Environment is missing required diagnostic: ' + field)
        return {'configured_device': configured, 'resolved_device': str(device),
            'observation_shape': list(observation.shape), 'observation_dtype': str(observation.dtype),
            'action_nvec': raw.action_space.nvec.tolist(), 'raw_reward_order': list(raw.reward_order),
            'raw_reward_dtype': str(raw_reward.dtype), 'single_environment_transition_checked': True}


def main(argv=None):
    """Return zero on successful preflight/training/reporting, nonzero on failure."""
    import sys
    import json
    try:
        args = parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    if not __package__:  # Also support direct script execution from another cwd.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    phase = 'configuration/resume validation'
    try:
        from agents.mo import config as planning, storage
        if args.initialize:
            from agents.mo.experiment import initialize_experiment
            source = args.config if args.config is not None else Path(__file__).resolve().parents[2] / 'configs/ppo_mo_r6_r2.json'
            overrides = {} if args.output is None else {'output_dir': args.output}
            config = planning.load_config(source, overrides)
            planning.validate_config(config)
            if config.logging.mode != 'off' and (not config.logging.project or not config.logging.entity):
                raise ValueError('External logging requires owner-supplied project and entity')
            phase = 'experiment initialization'
            saved = initialize_experiment(config)
            task_count = len(saved.seed_plans) * len(saved.weights)
            print(json.dumps({'mode': 'initialize', 'status': 'initialized',
                'experiment_dir': str(saved.root), 'experiment_fingerprint': saved.fingerprint,
                'n_seeds': len(saved.seed_plans), 'n_weights': len(saved.weights),
                'n_tasks': task_count, 'task_id_range': [0, task_count - 1],
                'budget_plan': storage._json_value(saved.budget_plan),
                'cost': planning.training_transition_cost(saved.config)}, allow_nan=False), flush=True)
            return 0
        if args.finalize:
            from agents.mo.experiment import load_experiment
            from agents.mo.finalization import finalize_experiment
            saved = load_experiment(args.experiment)
            phase = 'runtime preflight'
            _check_device(saved.config)
            phase = 'experiment finalization'
            manifest = finalize_experiment(saved.root)
            print(json.dumps({'mode': 'finalize', 'status': 'finalized',
                'experiment_dir': str(saved.root), 'experiment_fingerprint': manifest.experiment_fingerprint,
                'completed_policies': len(saved.seed_plans) * len(saved.weights),
                'actual_step': saved.budget_plan.effective_total,
                'report_manifest': str(saved.root / 'reports/report_manifest.json'),
                'finalization_record': str(saved.root / 'FINALIZED.json')}, allow_nan=False), flush=True)
            return 0
        if args.task_id is not None:
            from agents.mo.experiment import load_experiment, task_to_policy
            from agents.mo.types import PolicyTrainingSummary
            saved = load_experiment(args.experiment)
            seed_index, weight_index = task_to_policy(args.task_id, saved.config)
            phase = 'runtime preflight'
            device = _check_device(saved.config)
            # Validate managed allocation before touching a policy claim/resume journal.
            from agents.mo.runtime import execution_resources
            with execution_resources(saved.config, **_execution_options(args)) as resources:
                pass
            phase = 'policy training/resume'
            from agents.mo.trainer import train_one_policy
            summary = train_one_policy(saved.config, saved.root, seed_index, weight_index,
                saved.weights, saved.seed_plans[seed_index], saved.budget_plan, cache_results=False, **_execution_options(args))
            expected = (seed_index, saved.seed_plans[seed_index].run_seed, weight_index,
                        tuple(saved.weights[weight_index].tolist()))
            if (not isinstance(summary, PolicyTrainingSummary) or not summary.complete
                    or summary.identity.experiment_fingerprint != saved.fingerprint
                    or (summary.identity.run_index, summary.identity.run_seed,
                        summary.identity.weight_index, summary.identity.weight) != expected
                    or summary.requested_transitions != saved.budget_plan.requested_total
                    or summary.effective_transitions != saved.budget_plan.effective_total
                    or summary.actual_transitions != saved.budget_plan.effective_total
                    or summary.final_bundle_path != saved.root / 'models' / f'seed_{seed_index}' / f'w_{weight_index}' / 'final'):
                raise storage.StorageError('Task summary does not match its saved policy slot/budget')
            print(json.dumps({'mode': 'task', 'status': 'policy_complete',
                'experiment_dir': str(saved.root), 'experiment_fingerprint': saved.fingerprint,
                'task_id': args.task_id, 'seed_index': seed_index, 'run_seed': summary.identity.run_seed,
                'weight_index': weight_index, 'weight': list(summary.identity.weight),
                'requested_transitions': summary.requested_transitions,
                'effective_transitions': summary.effective_transitions,
                'actual_transitions': summary.actual_transitions,
                'resolved_device': str(device), 'execution_resources': resources, 'resume': storage._json_value(summary.resume_metadata),
                'final_bundle': str(summary.final_bundle_path)}, allow_nan=False), flush=True)
            return 0
        config, root, weights, plans, budget = _prepare(args)
        phase = 'runtime preflight'
        from agents.mo.runtime import execution_resources
        with execution_resources(config, **_execution_options(args)) as resources:
            runtime = _check_runtime(config, weights, plans[0])
        runtime['execution_resources'] = resources
        cost = planning.training_transition_cost(config)
        cost['intermediate_evaluation_episodes'] = cost['n_policies'] * len(budget.eval_steps) * config.eval_episodes
        cost['final_test_episodes'] = cost['n_policies'] * config.final_episodes
        cost['evaluation_transitions'] = 'Additional to training; depend on episode lengths'
        ready = config.reference_status == 'frozen' and config.scale_status != 'unselected'
        print(json.dumps({'mode': 'preflight' if args.preflight else 'training', 'resume': args.resume is not None,
            'experiment_dir': str(root), 'resolved_config': storage._json_value(config),
            'weights': {'dtype': str(weights.dtype), 'values': weights.tolist()},
            'seed_plans': storage._json_value(plans), 'budget_plan': storage._json_value(budget),
            'cost': cost, 'runtime': runtime, 'training_ready': ready,
            'reference_notice': None if ready else 'Select and freeze reference_scaled and objective scales before training.'},
            allow_nan=False), flush=True)
        if args.preflight:
            return 0
        phase = 'training/final evaluation'
        from agents.mo.trainer import run_experiment
        summary = run_experiment(config, experiment_dir=root, resume=args.resume is not None, **_execution_options(args))
        if summary.completed_policies != summary.total_policies:
            raise RuntimeError('Experiment returned an incomplete training policy set')
        phase = 'report generation'
        from agents.mo.reporting import generate_reports
        manifest = generate_reports(root, config)
        from dataclasses import replace
        summary = replace(summary, report_manifest=manifest)
        print(json.dumps({'status': 'complete', 'experiment_dir': str(root),
            'experiment_fingerprint': summary.experiment_fingerprint,
            'completed_policies': summary.completed_policies, 'total_policies': summary.total_policies,
            'effective_training_transitions': summary.effective_total_transitions,
            'additional_training_transitions': sum(p.resume_metadata['additional_training_transitions'] for p in summary.policies),
            'report_manifest': str(root / 'reports/report_manifest.json')}, allow_nan=False), flush=True)
        return 0
    except KeyboardInterrupt:
        print(f'{phase} interrupted', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'{phase} failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 2 if phase in ('configuration/resume validation', 'runtime preflight') else 1


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
