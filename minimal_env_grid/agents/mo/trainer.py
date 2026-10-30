"""Independent PPO construction and bounded completed-update learning.

Primitives borrow live resources; train_one_policy owns its training handle
and loaded bundle, and borrows the caller's result table and event logger.
Importing this module allocates no model, workers, files or remote clients.
"""
from numbers import Integral

from .config import validate_config
from .types import SeedPlan, TrainingEnvHandle


def run_experiment(config, experiment_dir=None, resume=False, *, learner_threads=None, cpus_per_task=None):
    """Keep execution limits active through training and final model reloads."""
    from .runtime import execution_resources
    options = {}
    if learner_threads is not None:
        options['learner_threads'] = learner_threads
    if cpus_per_task is not None:
        options['cpus_per_task'] = cpus_per_task
    with execution_resources(config, **options):
        return _run_experiment(config, experiment_dir, resume, options)


def _run_experiment(config, experiment_dir=None, resume=False, execution_options=None):
    """Train the saved seed/preference grid, then test the final policy sets.

    Own the local logger; each train_one_policy call owns and closes its worker
    group before returning. Resume reads persisted plans, validates provenance,
    and delegates completion checks to the policy trainer. Final testing reloads
    committed final bundles on disjoint seeds; reporting belongs to later blocks.
    """
    from dataclasses import replace
    from pathlib import Path
    import warnings
    from . import config as planning, storage
    from .experiment import initialize_experiment, load_experiment
    from .ownership import claim_policy_slot
    from .reporting import make_logger
    from .types import ExperimentSummary, PolicyTrainingSummary

    if type(resume) is not bool:
        raise TypeError('resume must be bool')
    validate_config(config)
    if experiment_dir is not None:
        config = replace(config, output_dir=Path(experiment_dir))
    if resume:
        destination = (config.output_dir if config.output_dir is not None
                       else Path('runs_mo') / config.experiment_id)
        initialized = load_experiment(destination, config=config)
    else:
        initialized = initialize_experiment(config)
    root, weights = initialized.root, initialized.weights
    plans, budget = initialized.seed_plans, initialized.budget_plan
    fingerprint = initialized.fingerprint
    cost = planning.training_transition_cost(config)
    cost['intermediate_evaluation_episodes'] = cost['n_policies'] * len(budget.eval_steps) * config.eval_episodes
    cost['final_test_episodes'] = cost['n_policies'] * config.final_episodes
    cost['evaluation_transitions'] = 'additional; depend on episode lengths; excluded from training budget'
    warnings.warn(
        f"Experiment budget: {len(plans)} seeds * {len(weights)} weights * "
        f"{budget.effective_total} = {cost['effective_training_transitions']} training transitions "
        f"plus {cost['intermediate_evaluation_episodes']} intermediate evaluation episodes "
        f"and {cost['final_test_episodes']} final-test episodes. Resume may reuse committed work.",
        RuntimeWarning, stacklevel=2)
    logger = make_logger(config, root)
    summaries = []
    try:
        logger.record('experiment_start', None, dict(cost, resume=resume, scope='training_and_final_testing'))
        # Finish interrupted reconciliation before any global record rebuild.
        # No model or worker is allocated during this storage-only pass.
        for i in range(len(plans)):
            for j in range(len(weights)):
                with claim_policy_slot(root, config, i, j):
                    storage.resolve_resume_state(root, config, i, j)
        results = storage.load_evaluation_records(root, config)
        for i, plan in enumerate(plans):
            for j in range(len(weights)):
                summary = train_one_policy(config, root, i, j, weights, plan, budget, results, logger, **(execution_options or {}))
                if (not isinstance(summary, PolicyTrainingSummary)
                        or summary.identity.experiment_fingerprint != fingerprint
                        or (summary.identity.run_index, summary.identity.run_seed, summary.identity.weight_index,
                            summary.identity.weight) != (i, plan.run_seed, j, tuple(weights[j].tolist()))
                        or summary.requested_transitions != budget.requested_total
                        or summary.effective_transitions != budget.effective_total):
                    raise storage.StorageError('Policy summary does not match its saved experiment slot/budget')
                summaries.append(summary)
        _cache_policy_results(root, results, config)
        summary = ExperimentSummary(fingerprint, root, tuple(summaries),
            sum(p.complete for p in summaries), cost['n_policies'], cost['effective_training_transitions'])
        logger.record('experiment_training_complete', None, {
            'completed_policies': summary.completed_policies, **cost,
            'additional_training_transitions': sum(
                p.resume_metadata['additional_training_transitions'] for p in summaries)})
        from .evaluation import evaluate_final_policy_set
        final = evaluate_final_policy_set(root, config, plans)
        logger.record('experiment_final_testing_complete', None, {
            'hypervolume_per_training_seed': final.hypervolume,
            'front_indices_per_training_seed': final.front_indices,
            'test_seeds': final.test_seeds, 'final_test_episodes': cost['final_test_episodes']})
        logger.flush()
        return summary
    except BaseException as error:
        try:
            logger.record('experiment_failure', None, {'error': str(error), 'returned_policies': len(summaries)})
        except Exception:
            pass
        raise
    finally:
        logger.close()


def build_model(config, training_handle, seed_plan):
    """Construct a fresh network and optimizer on a borrowed scalar VecEnv.

    Validate frozen objectives before constructing PPO. Each call creates new
    parameters/optimizer state; no previous preference policy is consulted.
    PPO initializes its own learner RNG from the explicitly saved learner seed.
    The caller owns the handle on both success and construction failure.
    """
    validate_config(config)
    if not isinstance(training_handle, TrainingEnvHandle):
        raise TypeError('training_handle must be TrainingEnvHandle')
    if training_handle._closed or training_handle.vec_env is None:
        raise ValueError('training_handle must own a live vector environment')
    if not isinstance(seed_plan, SeedPlan):
        raise TypeError('seed_plan must be SeedPlan')
    if (not 0 <= seed_plan.run_index < len(config.run_seeds)
            or config.run_seeds[seed_plan.run_index] != seed_plan.run_seed
            or len(seed_plan.worker_seeds) != config.ppo.n_envs
            or len(seed_plan.eval_seeds) != config.eval_episodes
            or len(seed_plan.final_seeds) != config.final_episodes):
        raise ValueError('seed_plan must match the experiment replicate and dimensions')
    if training_handle.vec_env.num_envs != config.ppo.n_envs:
        raise ValueError('training environment count must match PPO configuration')

    from stable_baselines3 import PPO
    from agents.RL.grid_cnn_extractor import GridCNNExtractor

    ppo = config.ppo
    return PPO(
        policy=ppo.policy, env=training_handle.vec_env,
        policy_kwargs={'features_extractor_class': GridCNNExtractor,
                       'features_extractor_kwargs': {'features_dim': ppo.features_dim}},
        learning_rate=ppo.learning_rate, n_steps=ppo.n_steps,
        batch_size=ppo.batch_size, n_epochs=ppo.n_epochs,
        gamma=ppo.gamma, gae_lambda=ppo.gae_lambda, clip_range=ppo.clip_range,
        clip_range_vf=ppo.clip_range_vf, normalize_advantage=ppo.normalize_advantage,
        ent_coef=ppo.ent_coef, vf_coef=ppo.vf_coef, max_grad_norm=ppo.max_grad_norm,
        use_sde=ppo.use_sde, sde_sample_freq=ppo.sde_sample_freq,
        target_kl=ppo.target_kl, stats_window_size=ppo.stats_window_size,
        tensorboard_log=None, verbose=ppo.verbose, seed=seed_plan.learner_seed,
        device=ppo.device,
    )


def learn_one_update(model, config, diagnostics=None):
    """Borrow a PPO and advance exactly one full rollout plus its PPO update.

    Returns the actual post-update transition count. Reuse the same model,
    optimizer, environment and diagnostics instance across calls. No resets or
    resource cleanup occur here. Only read-only episode diagnostics are accepted
    during collection; evaluation/checkpoint scheduling runs after this returns.

    SB3 counts epochs in _n_updates: a full PPO update normally adds n_epochs,
    with fewer epochs permitted only when configured target_kl stops optimization.
    Exceptions leave ownership with the caller and must not imply completion.
    """
    validate_config(config)
    from .callbacks import EpisodeDiagnosticsCallback
    if diagnostics is not None and not isinstance(diagnostics, EpisodeDiagnosticsCallback):
        raise TypeError('only EpisodeDiagnosticsCallback is supported during collection')
    ppo = config.ppo
    if (model.n_envs != ppo.n_envs or model.n_steps != ppo.n_steps
            or model.n_epochs != ppo.n_epochs or model.target_kl != ppo.target_kl):
        raise ValueError('model rollout/update settings must match configuration')
    if callable(model.learning_rate):
        raise ValueError('chunked baseline requires a constant numeric learning rate')
    current = model.num_timesteps
    if (isinstance(current, bool) or not isinstance(current, Integral)
            or current < 0 or current % ppo.rollout_size):
        raise ValueError('training step must be a nonnegative completed-update boundary')
    env = model.get_env()
    if env is None:
        raise ValueError('model must have a live training environment')
    optimizer = model.policy.optimizer
    updates = model._n_updates
    model.learn(total_timesteps=ppo.rollout_size, reset_num_timesteps=False,
                callback=diagnostics, log_interval=None, progress_bar=False)
    if model.num_timesteps != current + ppo.rollout_size or not model.rollout_buffer.full:
        raise RuntimeError('learning did not complete exactly one full rollout')
    epochs = model._n_updates - updates
    if epochs != ppo.n_epochs and not (ppo.target_kl is not None and 1 <= epochs <= ppo.n_epochs):
        raise RuntimeError('learning did not complete one PPO update')
    if model.get_env() is not env or model.policy.optimizer is not optimizer:
        raise RuntimeError('learning replaced the live environment or optimizer')
    return int(model.num_timesteps)


def _sync_policy_results(destination, source, seed_index, weight_index):
    """Refresh only the selected policy, preserving all other borrowed slots."""
    import numpy as np
    for name in ('means_raw', 'means_scaled', 'validity'):
        getattr(destination, name)[seed_index, :, weight_index] = getattr(source, name)[seed_index, :, weight_index]
    destination.hypervolume[seed_index] = np.nan
    if destination.episode_returns is not None:
        destination.episode_returns[seed_index, :, weight_index] = source.episode_returns[seed_index, :, weight_index]
    for name in ('identities', 'episode_metric_tables', '_evaluation_slot_hashes'):
        target = getattr(destination, name, {})
        for key in list(target):
            if key[0] == seed_index and key[2] == weight_index:
                del target[key]
        target.update({key: value for key, value in getattr(source, name, {}).items()
                       if key[0] == seed_index and key[2] == weight_index})
        setattr(destination, name, target)
    destination._resume_context = source._resume_context



def _cache_policy_results(root, results, config):
    """Refresh disposable caches without rereading all earlier policy records."""
    import numpy as np
    from .pareto import compute_hv_curve
    from .storage import _atomic_write
    results.hypervolume[:] = compute_hv_curve(results, config.reference_scaled)
    for name, array in (('eval_returns_raw', results.means_raw), ('eval_returns_scaled', results.means_scaled),
            ('eval_valid', results.validity), ('actual_steps', results.actual_budgets),
            ('hypervolume', results.hypervolume)):
        _atomic_write(root / 'results' / (name + '.npy'), lambda stream, a=array: np.save(stream, a, allow_pickle=False))


def _final_bundle_from_checkpoint(checkpoint, destination, identity):
    """Publish the exact committed final-budget snapshot without another learn().

    A reloaded PPO has no completed live rollout buffer. Copying the validated
    checkpoint also supports final-budget recovery without forging buffer flags
    or serializing a different learner/normalizer snapshot.
    """
    import hashlib
    import os
    from pathlib import Path
    import shutil
    import tempfile
    from . import storage
    destination = Path(destination)
    record, meta, payload = storage._committed_bundle(checkpoint, identity.experiment_fingerprint)
    if (record.identity != identity or identity.actual_step != payload['budget_plan']['effective_total']
            or destination != record.directory.parents[1] / 'final' and destination != record.directory):
        raise storage.StorageError('Final publication requires this policy final-budget checkpoint')
    if destination.exists():
        existing, _, _ = storage._committed_bundle(destination, identity.experiment_fingerprint)
        if existing.identity != identity:
            raise storage.StorageError('Existing final bundle conflicts with the final evaluation identity')
        return destination
    stage = Path(tempfile.mkdtemp(prefix='.final.', suffix='.tmp', dir=destination.parent))
    try:
        names = {}
        for name, digest in meta['member_hashes'].items():
            output = 'final_model.zip' if name == record.model_filename else name
            shutil.copyfile(record.directory / name, stage / output)
            if hashlib.sha256((stage / output).read_bytes()).hexdigest() != digest:
                raise storage.StorageError('Final snapshot copy checksum mismatch')
            with (stage / output).open('r+b') as stream:
                stream.flush(); os.fsync(stream.fileno())
            names[output] = digest
        meta = dict(meta, kind='final', member_hashes=names)
        storage.atomic_write_json(stage / 'bundle.json', meta)
        storage.atomic_write_json(stage / 'COMMITTED', {'bundle_schema_version': 1,
            'bundle_id': identity.bundle_id, 'actual_step': identity.actual_step,
            'bundle_sha256': hashlib.sha256((stage / 'bundle.json').read_bytes()).hexdigest()})
        # Both absolute directory targets must remain inside the owned policy.
        stage.resolve().relative_to(destination.parent.resolve())
        destination.resolve().relative_to(destination.parent.resolve())
        if destination.exists():
            raise storage.StorageError('Final bundle was created by another writer')
        os.rename(stage, destination)
        storage._committed_bundle(destination, identity.experiment_fingerprint)
        return destination
    finally:
        if stage.exists():
            stage.resolve().relative_to(destination.parent.resolve())
            if not stage.name.startswith('.final.') or not stage.name.endswith('.tmp'):
                raise storage.StorageError('Refusing cleanup outside owned final staging directory')
            shutil.rmtree(stage)


def train_one_policy(config, experiment_dir, seed_index, weight_index,
                     weights, seed_plan, budget_plan, result_store=None, logger=None, *, cache_results=None, learner_threads=None, cpus_per_task=None):
    """Own one learner through completed updates, evaluation and final completion.

    weights is the full saved float32 (W,2) matrix; result_store is a borrowed
    ExperimentResults table. If omitted, allocate a private full-grid table.
    With logger=None, own a policy-local logger inside the claim and disable
    shared caches by default. Borrowed loggers retain sequential compatibility
    (caches enabled by default); concurrent callers supplying a policy-local
    logger must explicitly pass cache_results=False. The caller owns borrowed
    loggers. Only validated success/skip returns a summary.
    """
    import numpy as np
    from contextlib import ExitStack
    from dataclasses import replace
    from . import storage
    from .callbacks import EpisodeDiagnosticsCallback
    from .config import validate_weights
    from .envs import make_train_vec_env, make_raw_env
    from .evaluation import evaluate_policy, make_frozen_obs_normalizer
    from .types import BudgetPlan, ExperimentResults, PolicyIdentity, PolicyTrainingSummary, ResumeState

    validate_config(config)
    root, payload, fingerprint = storage._evaluation_context(experiment_dir, config)
    weights = validate_weights(weights)
    if (weights.shape != tuple(payload['weights']['shape'])
            or weights.tobytes(order='C').hex() != payload['weights']['bytes_hex']):
        raise storage.StorageError('Training weights must match the exact saved matrix/order')
    if (type(seed_index) is not int or not 0 <= seed_index < len(config.run_seeds)
            or type(weight_index) is not int or not 0 <= weight_index < config.n_weights):
        raise storage.StorageError('Training indices must identify a declared policy')
    if not isinstance(seed_plan, SeedPlan) or storage._json_value(seed_plan) != payload['seed_plans'][seed_index]:
        raise storage.StorageError('Training must use the saved replicate seed plan')
    if not isinstance(budget_plan, BudgetPlan) or storage._json_value(budget_plan) != payload['budget_plan']:
        raise storage.StorageError('Training must use the saved actual budget plan')
    if cache_results is None:
        cache_results = logger is not None
    if type(cache_results) is not bool:
        raise TypeError('cache_results must be bool or None')
    if logger is None and cache_results:
        raise ValueError('Shared cache publication requires an explicit sequential logger/owner')
    if result_store is None:
        result_store = ExperimentResults.empty(len(config.run_seeds), budget_plan.eval_steps,
                                                config.n_weights, config.eval_episodes)
    if not isinstance(result_store, ExperimentResults):
        raise TypeError('result_store must be ExperimentResults')
    result_store.validate()
    if (result_store.validity.shape != (len(config.run_seeds), len(budget_plan.eval_steps), config.n_weights)
            or not np.array_equal(result_store.actual_budgets, budget_plan.eval_steps)
            or result_store.episode_returns is not None and result_store.episode_returns.shape[-2] != config.eval_episodes):
        raise storage.StorageError('Result table axes/episodes must match saved budgets and policies')
    for array in (result_store.means_raw, result_store.means_scaled, result_store.validity,
                  result_store.hypervolume, result_store.episode_returns):
        if array is not None and not array.flags.writeable:
            raise storage.StorageError('The borrowed result table must be writable')
    if logger is not None and (not callable(getattr(logger, 'record', None)) or not callable(getattr(logger, 'flush', None))):
        raise TypeError('logger must support record and flush; ownership stays with the caller')

    policy = root / 'models' / f'seed_{seed_index}' / f'w_{weight_index}'
    identity = PolicyIdentity(fingerprint, seed_plan.run_seed, seed_index, weight_index,
                              tuple(weights[weight_index].tolist()), 0, 'initial')
    from .ownership import claim_policy_slot
    with claim_policy_slot(root, config, seed_index, weight_index):
        with ExitStack() as log_ownership:
            if logger is None:
                from .reporting import make_logger
                logger = log_ownership.enter_context(make_logger(config, policy))
            phase = 'resume'
            try:
                state = storage.resolve_resume_state(root, config, seed_index, weight_index)
                _sync_policy_results(result_store,
                    storage.load_policy_evaluation_records(root, config, seed_index, weight_index), seed_index, weight_index)
                if cache_results:
                    _cache_policy_results(root, result_store, config)
                restored = state.current_step
                if state.complete:
                    record, _, _ = storage._committed_bundle(policy / 'final', fingerprint)
                    return PolicyTrainingSummary(record.identity, budget_plan.requested_total,
                        budget_plan.effective_total, state.current_step, True, policy / 'final',
                        state.completed_evaluation_ids, resume_metadata={'skipped_complete': True,
                            'restored_step': restored, 'additional_training_transitions': 0,
                            'resume_attempt_id': state.resume_attempt_id})
                logger.record('policy_start', identity, {'restored_step': restored,
                    'effective_target': budget_plan.effective_total, 'resume_attempt_id': state.resume_attempt_id,
                    'continuation': 'checkpoint continuation; live environment/RNG replay is not exact'})
                with ExitStack() as owned:
                    phase = 'execution resources'
                    from .runtime import execution_resources
                    resources = owned.enter_context(execution_resources(config,
                        learner_threads=learner_threads, cpus_per_task=cpus_per_task))
                    from stable_baselines3.common.utils import get_device
                    logger.record('execution_resources', identity, dict(resources,
                        resolved_device=str(get_device(config.ppo.device))))
                    phase = 'environment'
                    handle = owned.enter_context(make_train_vec_env(config, weights[weight_index], seed_plan))
                    phase = 'model'
                    if state.selected_bundle_path is None:
                        model = build_model(config, handle, seed_plan)
                    else:
                        bundle = owned.enter_context(storage.load_policy_bundle(state.selected_bundle_path, config, handle))
                        model, identity = bundle.model, bundle.identity
                    if model.num_timesteps != state.current_step:
                        raise storage.StorageError('Loaded learner step does not match selected resume state')
                    diagnostics = EpisodeDiagnosticsCallback(logger, seed_index, weight_index, config.ppo.n_envs)
                    checkpoint = state.selected_bundle_path

                    # Params: none; returns: None.
                    def evaluate_current():
                        nonlocal phase
                        phase = 'evaluation'
                        frozen = make_frozen_obs_normalizer(handle.vec_normalize)
                        result = evaluate_policy(model, lambda: make_raw_env(config.env), seed_plan.eval_seeds,
                                                 config, identity, frozen_normalizer=frozen, collect_traces=False)
                        phase = 'evaluation_write'
                        storage.save_evaluation_record(root, identity, result)
                        index = budget_plan.eval_steps.index(identity.actual_step)
                        storage.store_eval_slot(result_store, seed_index, index, weight_index, result)
                        if cache_results:
                            _cache_policy_results(root, result_store, config)
                        logger.record('policy_evaluation', identity, {'mean_raw': result.mean_raw,
                            'mean_scaled': result.mean_scaled, 'resume_attempt_id': state.resume_attempt_id})

                    current = int(model.num_timesteps)
                    if current in budget_plan.eval_steps:
                        index = budget_plan.eval_steps.index(current)
                        if not result_store.validity[seed_index, index, weight_index]:
                            evaluate_current()  # Repair before advancing or finalizing.
                    while current < budget_plan.effective_total:
                        phase = 'training'
                        current = learn_one_update(model, config, diagnostics)
                        if current > budget_plan.effective_total:
                            raise RuntimeError('Learner advanced beyond the effective policy budget')
                        identity = replace(identity, actual_step=current,
                            bundle_id=f'{state.resume_attempt_id}-s{seed_index}-w{weight_index}-step{current}')
                        if current in budget_plan.eval_steps or current in budget_plan.checkpoint_steps:
                            phase = 'checkpoint'
                            checkpoint = policy / 'checkpoints' / f'step_{current}'
                            snapshot = ResumeState(False, current, budget_plan.effective_total,
                                selected_bundle_path=checkpoint, last_committed_checkpoint=checkpoint,
                                completed_evaluation_ids=tuple(storage._evaluation_id(v) for key, v in
                                    sorted(result_store.identities.items()) if key[0] == seed_index and key[2] == weight_index),
                                next_evaluation_budgets=tuple(v for v in budget_plan.eval_steps if v > current),
                                next_checkpoint_budgets=tuple(v for v in budget_plan.checkpoint_steps if v > current),
                                resume_attempt_id=state.resume_attempt_id, canonical_attempt_ids=state.canonical_attempt_ids,
                                superseded_attempt_ids=state.superseded_attempt_ids)
                            storage.save_policy_bundle(model, handle, checkpoint, identity, snapshot)
                        if current in budget_plan.eval_steps:
                            evaluate_current()
                    phase = 'final_bundle'
                    final = _final_bundle_from_checkpoint(checkpoint, policy / 'final', identity)
                    phase = 'cleanup'
                phase = 'completion'
                logger.record('policy_artifacts_committed', identity, {'final_bundle': str(final),
                    'resume_attempt_id': state.resume_attempt_id})
                logger.flush()
                log_ownership.close()  # Owned logger close failures must precede completion.
                storage.mark_policy_complete(policy, final, storage._resume_directory(root, seed_index, weight_index) / f'step_{current}.json')
                completed = tuple(storage._evaluation_id(v) for key, v in sorted(result_store.identities.items())
                                  if key[0] == seed_index and key[2] == weight_index)
                return PolicyTrainingSummary(identity, budget_plan.requested_total, budget_plan.effective_total,
                    current, True, final, completed, resume_metadata={'skipped_complete': False,
                        'restored_step': restored, 'additional_training_transitions': current - restored,
                        'resume_attempt_id': state.resume_attempt_id, 'canonical_attempt_ids': state.canonical_attempt_ids,
                        'superseded_attempt_ids': state.superseded_attempt_ids,
                        'continuation': 'checkpoint continuation; live environment/RNG replay is not exact'})
            except BaseException as error:
                try:
                    logger.record('policy_failure', identity, {'phase': phase, 'error_type': type(error).__name__,
                                                             'error': str(error)})
                    logger.flush()
                except Exception:
                    pass  # Keep the original training/persistence/cancellation failure primary.
                raise
