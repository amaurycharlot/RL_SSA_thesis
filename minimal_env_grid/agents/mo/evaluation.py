"""Observation-only frozen preprocessing and learner-state preservation.

Imports are passive. Episode evaluation uses a borrowed raw environment;
these helpers never step a training environment or normalize evaluation rewards.
"""
from contextlib import contextmanager
from collections.abc import Mapping
import random

from .types import ObsNormalizer


def make_frozen_obs_normalizer(train_normalizer):
    """Copy observation RMS only; never modify the live VecNormalize object.

    Absent/disabled observation normalization returns None even when scalar
    reward normalization is enabled. Dict RMS is unsupported for this fixed
    Box grid. The snapshot owns immutable copies of mean/variance/count.
    """
    if train_normalizer is None:
        return None
    enabled = getattr(train_normalizer, "norm_obs", None)
    if type(enabled) is not bool:
        raise TypeError("train_normalizer must expose a boolean norm_obs flag")
    if not enabled:
        return None
    rms = getattr(train_normalizer, "obs_rms", None)
    if rms is None or isinstance(rms, Mapping):
        raise TypeError("Expected observation RMS for the fixed Box grid")
    try:
        return ObsNormalizer(mean=rms.mean, variance=rms.var, count=float(rms.count), enabled=True,
                             epsilon=float(train_normalizer.epsilon), clip_obs=float(train_normalizer.clip_obs))
    except AttributeError as error:
        raise TypeError("Missing observation RMS or preprocessing settings") from error


def normalize_policy_observation(raw_obs, frozen_normalizer):
    """Return an independent float32 (H,W,3) policy input, normalized once.

    Only ObsNormalizer snapshots are accepted, never a live training normalizer.
    The identity path copies without casting; malformed/nonfinite raw grid
    observations fail rather than silently changing dtype or shape.
    """
    import numpy as np
    observation = np.asarray(raw_obs)
    if observation.dtype != np.float32:
        raise TypeError("Raw observations must have dtype float32")
    if observation.ndim != 3 or observation.shape[-1] != 3 or min(observation.shape[:2]) < 1:
        raise ValueError("Raw observations must have positive shape (H,W,3)")
    if not np.all(np.isfinite(observation)):
        raise ValueError("Raw observations must be finite")
    if frozen_normalizer is None:
        return observation.copy()
    if not isinstance(frozen_normalizer, ObsNormalizer):
        raise TypeError("Use an immutable ObsNormalizer snapshot, not a live normalizer")
    return frozen_normalizer.normalize(observation)


@contextmanager
def evaluation_state_guard(model=None):
    """Restore learner Python/NumPy/Torch RNG and policy modes in finally.

    Temporarily select evaluation mode for a Torch policy. Preserve every
    module's prior flag, including mixed train/eval submodules. A model without
    a policy (or None) still receives RNG protection for synthetic evaluators.
    Snapshot CUDA generators only if CUDA was already initialized; this guard
    does not initialize CUDA just to inspect state. This is not a snapshot of
    live environments, optimizer/model parameters or normalization statistics.
    """
    import numpy as np
    import torch
    policy = getattr(model, "policy", None) if model is not None else None
    if policy is not None and not isinstance(policy, torch.nn.Module):
        raise TypeError("model.policy must be a Torch module")
    modes = tuple((module, module.training) for module in policy.modules()) if policy is not None else ()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    numpy_state = (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:])
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                   if torch.cuda.is_initialized() else None)
    try:
        if policy is not None:
            setter = getattr(policy, "set_training_mode", None)
            if callable(setter):
                setter(False)
            else:
                policy.train(False)
        yield
    finally:
        # Direct restoration avoids recursively flattening originally mixed
        # modes, or invoking a failing/custom train() hook on the way out.
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def evaluate_episode(model, raw_env, episode_seed, gamma, scales, frozen_normalizer,
                     metric_config, collect_trace):
    """Evaluate one raw combined episode; discount once per decision.

    The caller owns raw_env on success (for reuse across seeds). On any failure
    it is closed before propagating the original exception. A policy-level
    owner should also close in finally, including successful evaluation.
    Traces contain raw post-action observations/rewards, not policy inputs.
    """
    import numpy as np
    from numbers import Real
    from .types import EnvConfig, EpisodeTrace, EpisodeEvaluation, MAX_SEED
    from .metrics import summarize_episode

    try:
        with evaluation_state_guard(model):
            if type(episode_seed) is not int or not 0 <= episode_seed <= MAX_SEED:
                raise ValueError("episode_seed must be an explicit uint32 Python integer")
            if isinstance(gamma, bool) or not isinstance(gamma, Real) or not np.isfinite(gamma) or not 0 <= gamma <= 1:
                raise ValueError("gamma must be finite and in [0,1]")
            if isinstance(scales, (tuple, list)) and any(isinstance(s, (bool, np.bool_)) for s in scales):
                raise ValueError("scales must contain real numbers, not bools")
            scale_input = np.asarray(scales)
            if scale_input.shape != (2,) or scale_input.dtype.kind not in 'fiu':
                raise ValueError("scales must be a numeric pair")
            scales = scale_input.astype(np.float64, copy=True)
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("scales must be finite and positive")
            if type(collect_trace) is not bool:
                raise TypeError("collect_trace must be bool")
            if (getattr(raw_env, 'unwrapped', raw_env) is not raw_env
                    or getattr(raw_env, 'mode', None) != 'combined'
                    or getattr(raw_env, 'reward_order', None) != ('tracking', 'search')):
                raise TypeError("Expected an unwrapped raw combined vector-reward environment")
            config = EnvConfig(known_targets=raw_env.known_targets,
                unknown_targets=raw_env.unknown_targets, grid_shape=tuple(raw_env.grid_shape),
                episode_steps=raw_env.max_steps, dwell_steps=raw_env.dwell_steps)
            if (tuple(raw_env.observation_space.shape) != config.observation_shape
                    or np.dtype(raw_env.observation_space.dtype) != np.dtype('float32')
                    or not np.array_equal(getattr(raw_env.action_space, 'nvec', None), config.grid_shape)):
                raise ValueError("Raw environment spaces must match the original grid")
            targets = config.known_targets + config.unknown_targets
            # Slew can only increase elapsed duration; ceil(horizon/dwell) is
            # a conservative upper bound even when the last action overshoots.
            decision_cap = (config.episode_steps + config.dwell_steps - 1) // config.dwell_steps
            obs, _ = raw_env.reset(seed=episode_seed)
            obs = normalize_policy_observation(obs, None)
            if obs.shape != config.observation_shape:
                raise ValueError("Reset observation shape disagrees with environment grid")
            initial = obs.copy()
            rows = {name: [] for name in ('observations', 'vector_rewards', 'true_states',
                'estimated_states', 'actions', 'observed_targets', 'elapsed_steps',
                'coverage_values', 'terminated', 'truncated')}
            total = np.zeros(2, dtype=np.float64)
            discount = 1.0
            elapsed_total = 0
            for _ in range(decision_cap):
                policy_obs = normalize_policy_observation(obs, frozen_normalizer)
                action, _ = model.predict(policy_obs, deterministic=True)
                action = np.asarray(action)
                if (action.shape != (2,) or action.dtype.kind not in 'iu'
                        or np.any(action < 0) or np.any(action >= config.grid_shape)):
                    raise ValueError("Policy action must be an integer (row,column) within the grid")
                action = action.copy()
                obs, reward, terminated, truncated, info = raw_env.step(action.copy())
                obs = normalize_policy_observation(obs, None)
                reward = np.asarray(reward)
                if obs.shape != initial.shape:
                    raise ValueError("Post-action observation changed shape")
                if reward.shape != (2,) or reward.dtype != np.float32 or not np.isfinite(reward).all():
                    raise ValueError("Raw reward must be a finite float32 vector of shape (2,)")
                if not isinstance(terminated, (bool, np.bool_)) or not isinstance(truncated, (bool, np.bool_)):
                    raise TypeError("Termination flags must be boolean")
                if not isinstance(info, Mapping):
                    raise TypeError("Step diagnostics must be a mapping")
                true = np.asarray(info['true_states'])
                estimated = np.asarray(info['estimated_states'])
                if (true.shape != (targets, 4) or true.dtype.kind != 'f' or not np.isfinite(true).all()
                        or estimated.shape != (targets, 9) or estimated.dtype.kind != 'f'):
                    raise ValueError("Post-action states must preserve physical target ID rows")
                # Unknown tracks may be absent/nonfinite; shared metrics own
                # the established valid-catalogue mask and missing semantics.
                ids = np.asarray(info['observed_target_ids'])
                if ids.ndim != 1 or (ids.size and (ids.dtype.kind not in 'iu'
                        or np.any(ids < 0) or np.any(ids >= targets)
                        or np.unique(ids).size != ids.size)):
                    raise ValueError("Observed IDs must be distinct physical target indices")
                mask = np.zeros(targets, dtype=bool)
                mask[ids.astype(np.int64)] = True
                if 'num_measurements' in info and info['num_measurements'] != int(mask.sum()):
                    raise ValueError("Observed IDs and measurement count disagree")
                elapsed = info['elapsed_steps']
                if isinstance(elapsed, (bool, np.bool_)) or not isinstance(elapsed, (int, np.integer)) or elapsed < config.dwell_steps:
                    raise ValueError("Elapsed steps must be an integer at least dwell_steps")
                elapsed_total += int(elapsed)
                if info['total_steps'] != elapsed_total:
                    raise ValueError("Post-action total_steps is not aligned with elapsed steps")
                coverage = info['coverage_mean']
                if isinstance(coverage, (bool, np.bool_)) or not isinstance(coverage, Real) or not np.isfinite(coverage) or coverage < 0:
                    raise ValueError("coverage_mean must be finite nonnegative raw-grid coverage")
                total += discount * reward.astype(np.float64)
                discount *= float(gamma)
                if not np.isfinite(total).all():
                    raise ValueError("Nonfinite accumulated episode return")
                for name, value in (('observations', obs), ('vector_rewards', reward),
                        ('true_states', true), ('estimated_states', estimated), ('actions', action),
                        ('observed_targets', mask), ('elapsed_steps', int(elapsed)),
                        ('coverage_values', float(coverage)), ('terminated', bool(terminated)),
                        ('truncated', bool(truncated))):
                    rows[name].append(value.copy() if isinstance(value, np.ndarray) else value)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Episode exceeded the horizon/dwell decision cap without termination")
            trace = EpisodeTrace(initial_observation=initial, seed=episode_seed,
                initial_known_count=config.known_targets,
                **{name: np.asarray(values) for name, values in rows.items()})
            metrics = summarize_episode(trace, config, metric_config)
            with np.errstate(over='ignore', invalid='ignore'):
                scaled = total / scales
            return EpisodeEvaluation(return_raw=total, return_scaled=scaled,
                                     metrics=metrics, trace=trace if collect_trace else None)
    except BaseException as error:
        try:
            with evaluation_state_guard():
                raw_env.close()
        except BaseException as close_error:
            # Keep the evaluation error primary while retaining cleanup context.
            raise error from close_error
        raise


def evaluate_final_policy_set(experiment_dir, config, seed_plan):
    """Reload matched-budget final policies and test each saved preference.

    Require every declared seed/weight bundle before any evaluation. Results
    use only the saved ordered final-test episodes; intermediate records never
    select a checkpoint or enter these fronts. Validated committed final records
    are reused after interruption. Derived aggregate files are disposable.
    """
    import hashlib
    import numpy as np
    from . import storage
    from .envs import make_raw_env
    from .types import FinalResults, SeedPlan
    from .pareto import unique_front_indices, hypervolume_2d_max

    root, payload, fingerprint = storage._evaluation_context(experiment_dir, config)
    plans = (seed_plan,) if isinstance(seed_plan, SeedPlan) else tuple(seed_plan)
    if (any(not isinstance(p, SeedPlan) for p in plans)
            or storage._json_value(plans) != payload['seed_plans']):
        raise storage.StorageError('Final testing requires every saved ordered replicate seed plan')
    seeds = plans[0].final_seeds
    n_seeds, n_weights = len(plans), config.n_weights
    target = payload['budget_plan']['effective_total']
    slots = []
    # Preflight the entire declared set, not just the first available policy.
    # This never falls back to checkpoints or discovers a validation-best model.
    for i, plan in enumerate(plans):
        for j in range(n_weights):
            directory = root / 'models' / f'seed_{i}' / f'w_{j}' / 'final'
            record, meta, _ = storage._committed_bundle(directory, fingerprint)
            identity = record.identity
            if (meta['kind'] != 'final' or identity.actual_step != target
                    or (identity.run_index, identity.run_seed, identity.weight_index, list(identity.weight))
                    != (i, plan.run_seed, j, payload['weights']['values'][j])):
                raise storage.StorageError('Final bundle does not match the saved policy slot/final budget')
            journal = root / 'results/evaluations' / f'seed_{i}' / f'w_{j}' / 'RESUME.json'
            if journal.exists():
                saved = storage._read_json(journal)
                storage._validate_resume_plan(root, saved, payload, fingerprint, i, j)
                if saved['status'] != 'ready':
                    raise storage.StorageError('Final testing cannot use pending resume reconciliation')
            path = root / 'results/final' / f'seed_{i}' / f'w_{j}' / f'step_{target}.json'
            previous = None
            if path.exists():
                previous, saved = storage._load_evaluation_record(path)
                expected, kind = storage._evaluation_location(root, payload, fingerprint, identity, previous)
                if (path != expected or kind != 'final' or saved['kind'] != 'final'
                        or saved['objective_scales'] != payload['config']['objective_scales']):
                    raise storage.StorageError('Committed final evaluation has incompatible provenance')
            slots.append((i, j, record, path, previous))

    raw = np.full((n_seeds, n_weights, len(seeds), 2), np.nan, dtype=np.float64)
    scaled = np.full_like(raw, np.nan)
    valid = np.zeros((n_seeds, n_weights), dtype=bool)
    metrics = [[() for _ in range(n_weights)] for _ in range(n_seeds)]
    provenance = []
    # Guard factory/loading/cleanup randomness as well as policy inference.
    with evaluation_state_guard():
        for i, j, record, path, result in slots:
            if result is None:
                with storage.load_policy_bundle(record.directory, config) as bundle:
                    if bundle.identity != record.identity:
                        raise storage.StorageError('Final bundle identity changed after preflight')
                    frozen = make_frozen_obs_normalizer(bundle.vec_normalize)
                    result = evaluate_policy(bundle.model, lambda: make_raw_env(config.env),
                        seeds, config, bundle.identity, frozen_normalizer=frozen, collect_traces=False)
                    # Commit each independent policy before moving to the next.
                    storage.save_evaluation_record(root, bundle.identity, result)
            raw[i, j] = result.returns_raw
            scaled[i, j] = result.returns_scaled
            valid[i, j] = True
            metrics[i][j] = result.episode_metrics
            provenance.append({'identity': storage._json_value(record.identity),
                'bundle_sha256': record.bundle_sha256,
                'evaluation_record': path.relative_to(root).as_posix(),
                'evaluation_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    means_raw, means_scaled = raw.mean(axis=2), scaled.mean(axis=2)
    fronts = tuple(tuple(int(j) for j in unique_front_indices(means_scaled[i], range(n_weights)))
                   for i in range(n_seeds))
    hv = np.asarray([hypervolume_2d_max(means_scaled[i], config.reference_scaled)
                     for i in range(n_seeds)], dtype=np.float64)
    final = FinalResults(means_raw, means_scaled, raw, scaled, valid, seeds,
                         fronts, hv, tuple(tuple(row) for row in metrics))
    front_mask = np.zeros_like(valid)
    for i, indices in enumerate(fronts):
        front_mask[i, list(indices)] = True
    numeric = root / 'results/final_results.npz'
    storage.atomic_write_npz(numeric, {
        'means_raw': final.means_raw, 'means_scaled': final.means_scaled,
        'episode_returns_raw': final.episode_returns_raw,
        'episode_returns_scaled': final.episode_returns_scaled,
        'validity': final.validity, 'test_seeds': np.asarray(seeds, dtype=np.uint32),
        'front_mask': front_mask, 'hypervolume': final.hypervolume,
    })
    storage.atomic_write_json(root / 'results/final_results.json', {
        'schema_version': 1, 'kind': 'derived_final_policy_set',
        'experiment_fingerprint': fingerprint, 'actual_step': target,
        'run_seeds': config.run_seeds, 'weights': payload['weights']['values'],
        'test_seeds': seeds, 'objective_scales': config.objective_scales,
        'reference_scaled': config.reference_scaled, 'front_indices': fronts,
        'policy_records': provenance, 'numeric_file': numeric.name,
        'numeric_sha256': hashlib.sha256(numeric.read_bytes()).hexdigest(),
    })
    return final


def evaluate_policy(model, raw_env_factory, episode_seeds, config, identity,
                    frozen_normalizer=None, collect_traces=False):
    """Evaluate a policy on the supplied saved, ordered episode-seed list.

    The caller selects the intermediate or disjoint final list; never generate,
    sort or resample seeds here. Return both raw/scaled episode vectors, their
    arithmetic means, sample SD and SE, plus separate per-episode diagnostics.
    One raw environment is reused across resets and owned until cleanup.
    """
    import numpy as np
    from collections.abc import Sequence
    from .types import ExperimentConfig, PolicyIdentity, EvaluationResult, ObsNormalizer, MAX_SEED

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    if not isinstance(identity, PolicyIdentity):
        raise TypeError("identity must be PolicyIdentity")
    if not callable(raw_env_factory):
        raise TypeError("raw_env_factory must be a zero-argument callable")
    if type(collect_traces) is not bool:
        raise TypeError("collect_traces must be bool")
    if frozen_normalizer is not None and not isinstance(frozen_normalizer, ObsNormalizer):
        raise TypeError("Use a matching frozen ObsNormalizer snapshot")
    if isinstance(episode_seeds, (str, bytes)) or not isinstance(episode_seeds, Sequence):
        raise TypeError("episode_seeds must be an ordered sequence of Python integers")
    seeds = tuple(episode_seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed <= MAX_SEED for seed in seeds):
        raise ValueError("episode_seeds must be nonempty explicit uint32 Python integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("episode_seeds must be distinct")

    # Guard the whole ownership interval, including random draws made by a
    # factory or close hook, not only inference inside evaluate_episode.
    with evaluation_state_guard(model):
        raw_env = raw_env_factory()
        episode_failed = False
        failure = None
        try:
            episodes = []
            for seed in seeds:
                try:
                    episode = evaluate_episode(model, raw_env, seed, config.ppo.gamma,
                        config.objective_scales, frozen_normalizer, config, collect_traces)
                except BaseException:
                    # Block 18 already attempts close on every episode error,
                    # even reset/metric errors. Do not close that resource twice
                    # or replace its original error if its cleanup also failed.
                    episode_failed = True
                    raise
                episodes.append(episode)
            raw = np.stack([episode.return_raw for episode in episodes])
            scaled = np.stack([episode.return_scaled for episode in episodes])
            summaries = {}
            for suffix, values in (('raw', raw), ('scaled', scaled)):
                sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(2, np.float64)
                summaries['mean_' + suffix] = values.mean(axis=0)
                summaries['sd_' + suffix] = sd
                summaries['se_' + suffix] = sd / np.sqrt(len(seeds))
            return EvaluationResult(episode_seeds=seeds, returns_raw=raw, returns_scaled=scaled,
                **summaries, episode_metrics=tuple(episode.metrics for episode in episodes),
                deterministic=True, actual_step=identity.actual_step, identity=identity,
                traces=tuple(episode.trace for episode in episodes) if collect_traces else ())
        except BaseException as error:
            failure = error
            raise
        finally:
            if not episode_failed:
                try:
                    raw_env.close()
                except BaseException as close_error:
                    if failure is not None:
                        raise failure from close_error
                    raise
