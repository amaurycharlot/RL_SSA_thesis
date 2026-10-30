"""Experiment identity and atomic local persistence; no training imports/side effects.

The initialization manifest is written last. It commits only initialization,
never policy completion. Resume validates identity and immutable initial files;
policy/evaluation recovery belongs to the later storage blocks.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from contextlib import contextmanager

from .config import validate_config, validate_weights, make_budget_plan
from .types import SeedPlan, BudgetPlan

STORAGE_SCHEMA_VERSION = 1


class StorageError(ValueError):
    """Invalid experiment identity, initialization or resume request."""


def _json_value(value):
    """Own primitive JSON values; map all nonfinite numbers to JSON null."""
    import numpy as np
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError('JSON mapping keys must be strings')
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


# Params: payload; returns: UTF-8 JSON bytes.
def _json_bytes(payload):
    return (json.dumps(_json_value(payload), sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def _atomic_write(path, writer):
    """Flush and fsync a temporary sibling before replace; remove it on failure."""
    path = Path(path)
    # Do not create unrelated parent trees as a side effect of a record write.
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_json(path, payload):
    """Replace one JSON record atomically; never write nonstandard NaN tokens."""
    data = _json_bytes(payload)  # Validate before opening a temporary file.
    _atomic_write(path, lambda stream: stream.write(data))


def atomic_write_npz(path, arrays):
    """Write named numeric arrays without pickle, retaining NaNs and exact dtypes."""
    import numpy as np
    if not isinstance(arrays, Mapping) or not arrays:
        raise TypeError('arrays must be a nonempty mapping of named numeric arrays')
    copied = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or name == 'file' or '/' in name or '\\' in name:
            raise StorageError('NPZ array names must be nonempty safe names other than file')
        array = np.asarray(value)
        if array.dtype.kind not in 'biufc':
            raise TypeError('NPZ payloads must be numeric/bool arrays without objects')
        copied[name] = array.copy()
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **copied))


# Params: none; returns: package-version dict.
def _versions():
    result = {'python': platform.python_version()}
    for package in ('numpy', 'gymnasium', 'stable-baselines3', 'torch', 'scipy', 'cloudpickle'):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


# Params: none; returns: dict.
def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [root / 'agents/RL/grid_cnn_extractor.py', root / 'agents/catalogue.py']
    for pattern in ('agents/mo/*.py', 'agents/ckf/*.py', 'ssa_env/core/*.py', 'ssa_env/envs/*.py'):
        paths.extend(root.glob(pattern))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


# Params: config; returns: dict.
def _architecture(config):
    return {'policy': 'MlpPolicy', 'features_extractor': 'agents.RL.grid_cnn_extractor.GridCNNExtractor',
            'features_dim': 128, 'observation_shape': list(config.env.observation_shape),
            'observation_dtype': 'float32', 'action_type': 'MultiDiscrete',
            'action_nvec': list(config.env.action_nvec), 'mode': 'combined', 'simple_obs': False,
            'world_size': 100.0, 'reward_order': list(config.reward_order),
            'num_objectives': 2, 'objectives': 'maximize', 'discount_clock': 'decision'}


# Params: config, weights, seed_plan, budget_plan; returns: dict.
def _identity_payload(config, weights, seed_plan, budget_plan):
    validate_config(config, require_frozen_objectives=False)
    weights = validate_weights(weights)
    if weights.shape[0] != config.n_weights:
        raise StorageError('Weight count must match config.n_weights')
    plans = (seed_plan,) if isinstance(seed_plan, SeedPlan) else tuple(seed_plan)
    if len(plans) != len(config.run_seeds) or not all(isinstance(p, SeedPlan) for p in plans):
        raise StorageError('Provide one ordered SeedPlan for every declared training replicate')
    training_seeds = set()
    for index, plan in enumerate(plans):
        if (plan.run_index != index or plan.run_seed != config.run_seeds[index]
                or len(plan.worker_seeds) != config.ppo.n_envs
                or len(plan.eval_seeds) != config.eval_episodes
                or len(plan.final_seeds) != config.final_episodes):
            raise StorageError('Seed plan identity/dimensions do not match the experiment')
        if plan.eval_seeds != plans[0].eval_seeds or plan.final_seeds != plans[0].final_seeds:
            raise StorageError('Holdout seeds must be shared across training replicates')
        if training_seeds.intersection(plan.worker_seeds):
            raise StorageError('Training workers must have disjoint seeds across replicates')
        training_seeds.update(plan.worker_seeds)
    if not isinstance(budget_plan, BudgetPlan) or budget_plan != make_budget_plan(config):
        raise StorageError('Budget plan does not match resolved configuration')
    science = _json_value(config)
    science.pop('output_dir')
    # Logging transport/account identifiers are presentation destinations only.
    science.pop('logging')
    return {'storage_schema_version': STORAGE_SCHEMA_VERSION, 'config': science,
            'weights': {'dtype': 'float32', 'shape': list(weights.shape), 'values': weights.tolist(),
                        'bytes_hex': weights.tobytes(order='C').hex()},
            'seed_plans': _json_value(plans), 'budget_plan': _json_value(budget_plan),
            'architecture': _architecture(config), 'versions': _versions(), 'source_hashes': _source_hashes()}


def experiment_fingerprint(config, weights, seed_plan, budget_plan):
    """SHA256 of canonical resolved science, exact weights, plans and provenance.

    seed_plan is an ordered sequence covering all declared replicates (or a
    single SeedPlan for a one-replicate experiment). Only output_dir/logging
    destinations are excluded. Device/execution settings remain part of identity.
    """
    return hashlib.sha256(_json_bytes(_identity_payload(config, weights, seed_plan, budget_plan))).hexdigest()


# Params: path; returns: decoded JSON.
def _read_json(path):
    # Params: items; returns: dict.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise StorageError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    # Params: value; returns: never (raises).
    def constant(value):
        raise StorageError(f'Nonstandard JSON number: {value}')
    with path.open('r', encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=constant)


# Params: path, payload, fingerprint; returns: None.
def _validate_initialization(path, payload, fingerprint):
    import numpy as np
    try:
        manifest = _read_json(path / 'manifest.json')
        if (manifest['storage_schema_version'] != STORAGE_SCHEMA_VERSION
                or manifest['fingerprint'] != fingerprint or manifest['identity'] != payload):
            raise StorageError('Resume science/config/source/version identity mismatch')
        required = {'config.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json'}
        if set(manifest['initial_file_hashes']) != required:
            raise StorageError('Resume manifest has an invalid initial file inventory')
        for name, digest in manifest['initial_file_hashes'].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
                raise StorageError(f'Resume initialization checksum mismatch: {name}')
        saved_config = _read_json(path / 'config.json')
        science = dict(saved_config)
        resolved = science.pop('resolved')
        science.pop('output_dir')
        science.pop('logging')
        if science != payload['config'] or resolved != {k: v for k, v in payload.items() if k != 'config'}:
            raise StorageError('Saved resolved configuration does not match requested identity')
        if _read_json(path / 'seed_plan.json') != payload['seed_plans']:
            raise StorageError('Saved seed plans do not match identity')
        if _read_json(path / 'budget_plan.json') != payload['budget_plan']:
            raise StorageError('Saved budget plan does not match identity')
        weights = np.load(path / 'weights.npy', allow_pickle=False)
        if (weights.dtype != np.float32 or list(weights.shape) != payload['weights']['shape']
                or weights.tobytes(order='C').hex() != payload['weights']['bytes_hex']):
            raise StorageError('Saved exact weights do not match identity')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise StorageError(f'Cannot resume experiment {path}: {error}') from error


def create_experiment_dir(config, weights, seed_plan, budget_plan, *, resume=False):
    """Initialize output_dir (exact destination) or runs_mo/experiment_id.

    A nonempty directory is never overwritten. Explicit resume validates the
    manifest/initial-file checksums and returns without writing anything.
    Failed initialization remains visibly incomplete and protected from reuse.
    Strict frozen-objective validation applies before any filesystem mutation.
    """
    import numpy as np
    if type(resume) is not bool:
        raise TypeError('resume must be bool')
    validate_config(config)
    payload = _identity_payload(config, weights, seed_plan, budget_plan)
    fingerprint = hashlib.sha256(_json_bytes(payload)).hexdigest()
    if config.output_dir is None:
        name = config.experiment_id
        if name in ('.', '..') or any(c in name for c in '/\\:'):
            raise StorageError('experiment_id must be a single directory name')
        path = Path('runs_mo') / name
    else:
        path = config.output_dir
    path = path.expanduser().absolute().resolve()
    if resume:
        if not path.is_dir() or (path / '.INITIALIZING').exists():
            raise StorageError('Explicit resume requires a committed experiment initialization')
        _validate_initialization(path, payload, fingerprint)
        return path
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise StorageError(f'Experiment destination is nonempty or not a directory: {path}; use explicit resume')
    path.mkdir(parents=True, exist_ok=True)
    lock = path / '.INITIALIZING'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise StorageError('Experiment initialization is already in progress') from error
    os.close(descriptor)
    try:
        # Recheck after taking ownership: a competing initializer may have committed.
        if any(p.name != lock.name for p in path.iterdir()):
            raise StorageError('Experiment destination became nonempty during initialization')
        resolved = _json_value(config)
        resolved['output_dir'] = str(path)
        resolved['resolved'] = {k: v for k, v in payload.items() if k != 'config'}
        atomic_write_json(path / 'config.json', resolved)
        _atomic_write(path / 'weights.npy', lambda stream: np.save(stream, weights, allow_pickle=False))
        atomic_write_json(path / 'seed_plan.json', payload['seed_plans'])
        atomic_write_json(path / 'budget_plan.json', payload['budget_plan'])
        for directory in ('results/evaluations', 'results/final', 'models', 'logs', 'reports'):
            (path / directory).mkdir(parents=True, exist_ok=True)
        hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                  for name in ('config.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json')}
        atomic_write_json(path / 'manifest.json', {
            'storage_schema_version': STORAGE_SCHEMA_VERSION, 'fingerprint': fingerprint,
            'status': 'initialized', 'created_at': datetime.now(timezone.utc).isoformat(),
            'identity': payload, 'initial_file_hashes': hashes})
    finally:
        lock.unlink()
    return path


def _bundle_context(directory, identity, resume_state):
    """Resolve frozen initialization and canonical policy slot before allocation."""
    from .types import PolicyIdentity, ResumeState
    if not isinstance(identity, PolicyIdentity) or not isinstance(resume_state, ResumeState):
        raise TypeError('identity/resume_state must be PolicyIdentity/ResumeState')
    directory = Path(directory).absolute().resolve()
    root = next((p for p in directory.parents if (p / 'manifest.json').is_file()), None)
    if root is None:
        raise StorageError('Policy bundles require an initialized enclosing experiment')
    manifest = _read_json(root / 'manifest.json')
    payload = manifest['identity']
    fingerprint = hashlib.sha256(_json_bytes(payload)).hexdigest()
    if fingerprint != identity.experiment_fingerprint or manifest['fingerprint'] != fingerprint:
        raise StorageError('Bundle identity does not match experiment fingerprint')
    _validate_initialization(root, payload, fingerprint)
    if payload['source_hashes'] != _source_hashes() or payload['versions'] != _versions():
        raise StorageError('Cannot save a policy with changed source/package provenance')
    config = payload['config']
    run, weight, step = identity.run_index, identity.weight_index, identity.actual_step
    if (not 0 <= run < len(config['run_seeds']) or config['run_seeds'][run] != identity.run_seed
            or not 0 <= weight < len(payload['weights']['values'])
            or list(identity.weight) != payload['weights']['values'][weight]):
        raise StorageError('Policy run/weight identity does not match the saved slot')
    budget = payload['budget_plan']
    if step <= 0 or step % budget['rollout_size'] or step > budget['effective_total']:
        raise StorageError('Bundle step must be a positive completed PPO update within the budget')
    if (resume_state.current_step != step or resume_state.effective_target != budget['effective_total']
            or resume_state.complete):
        raise StorageError('Resume counters must describe this update without declaring completion')
    for field, schedule in (('next_evaluation_budgets', 'eval_steps'),
                            ('next_checkpoint_budgets', 'checkpoint_steps')):
        if tuple(getattr(resume_state, field)) != tuple(s for s in budget[schedule] if s > step):
            raise StorageError('Resume pending budgets must follow the saved actual step')
    policy = root / 'models' / f'seed_{run}' / f'w_{weight}'
    checkpoint = policy / 'checkpoints' / f'step_{step}'
    final = policy / 'final'
    if directory not in (checkpoint, final) or (directory == final and step != budget['effective_total']):
        raise StorageError('Bundle directory must identify this checkpoint step or the matched final budget')
    return directory, payload, directory == final


# Params: model, handle, identity, payload; returns: (dict, normalizer).
def _validate_live_bundle(model, handle, identity, payload):
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    from .types import TrainingEnvHandle
    if not isinstance(model, PPO) or type(model.policy) is not MlpPolicy:
        raise StorageError('The bundle must contain the configured PPO MlpPolicy')
    if not isinstance(handle, TrainingEnvHandle) or handle.closed or handle.vec_env is None:
        raise StorageError('Saving requires an open matching TrainingEnvHandle')
    if model.get_env() is not handle.vec_env:
        raise StorageError('Learner and training handle must own the same VecEnv')
    config, architecture = payload['config'], payload['architecture']
    ppo = config['ppo']
    if model.seed != payload['seed_plans'][identity.run_index]['learner_seed']:
        raise StorageError('Learner seed does not match the saved run seed plan')
    updates = identity.actual_step // payload['budget_plan']['rollout_size']
    if (model.num_timesteps != identity.actual_step or model.n_envs != ppo['n_envs']
            or model.n_steps != ppo['n_steps'] or model.n_epochs != ppo['n_epochs']
            or not model.rollout_buffer.full or not model.rollout_buffer.generator_ready
            or model._n_updates < updates):
        raise StorageError('Learner counters/rollout do not describe a completed update')
    if ppo['target_kl'] is None and model._n_updates != updates * ppo['n_epochs']:
        raise StorageError('Save after PPO optimization, not from a rollout-collection callback')
    extractor = model.policy.features_extractor
    if (type(extractor).__module__ + '.' + type(extractor).__name__ != architecture['features_extractor']
            or extractor.features_dim != architecture['features_dim']
            or list(model.observation_space.shape) != architecture['observation_shape']
            or not np.array_equal(model.action_space.nvec, architecture['action_nvec'])):
        raise StorageError('Live policy architecture/action/observation contract mismatch')
    for name in ('batch_size', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm',
                 'normalize_advantage', 'use_sde', 'sde_sample_freq', 'target_kl', 'learning_rate'):
        if getattr(model, name) != ppo[name]:
            raise StorageError(f'Live PPO setting mismatch: {name}')
    if model.clip_range(1.) != ppo['clip_range'] or model.clip_range(0.) != ppo['clip_range']:
        raise StorageError('Live clipping schedule does not match constant configured clipping')
    expected_vf = ppo['clip_range_vf']
    if ((model.clip_range_vf is None) != (expected_vf is None)
            or (expected_vf is not None and
                (model.clip_range_vf(1.) != expected_vf or model.clip_range_vf(0.) != expected_vf))):
        raise StorageError('Live value clipping does not match the configured constant')
    for name, expected in (('weights', identity.weight), ('scales', config['objective_scales'])):
        values = handle.vec_env.get_attr(name)
        if len(values) != ppo['n_envs'] or any(not np.array_equal(v, np.asarray(expected, np.float32)) for v in values):
            raise StorageError(f'Live scalarization {name} do not match the policy identity')
    settings = config['normalization']
    enabled = settings['norm_obs'] or settings['norm_reward']
    normalizer = handle.vec_normalize
    if (normalizer is not None) != enabled or model.get_vec_normalize_env() is not normalizer:
        raise StorageError('Model/handle normalization state does not match the policy configuration')
    if enabled:
        if handle.vec_env is not normalizer:
            raise StorageError('Normalizer must be the matching outer training environment')
        for name, value in settings.items():
            if getattr(normalizer, name) != value:
                raise StorageError(f'Normalizer setting mismatch: {name}')
    return {'num_timesteps': int(model.num_timesteps), 'n_updates': int(model._n_updates),
            'completed_rollouts': updates}, normalizer


def _validate_saved_normalizer(path, live):
    """Validate only the freshly serialized, trusted live policy normalizer."""
    import pickle
    import numpy as np
    with path.open('rb') as stream:
        saved = pickle.load(stream)
        if stream.read(1):
            raise StorageError('Normalizer pickle contains trailing/incomplete serialization')
    if type(saved) is not type(live):
        raise StorageError('Saved normalizer type does not match the live normalizer')
    for name in ('norm_obs', 'norm_reward', 'gamma', 'epsilon', 'clip_obs', 'clip_reward', 'training'):
        if getattr(saved, name) != getattr(live, name):
            raise StorageError(f'Saved normalizer setting mismatch: {name}')
    for name in ('ret_rms',) + (('obs_rms',) if live.norm_obs else ()):
        for field in ('mean', 'var', 'count'):
            expected, actual = getattr(getattr(live, name), field), getattr(getattr(saved, name), field)
            if not np.all(np.isfinite(actual)) or not np.array_equal(expected, actual):
                raise StorageError(f'Saved normalizer statistics mismatch: {name}.{field}')


def save_policy_bundle(model, training_handle, directory, identity, resume_state):
    """Commit a new bundle from one synchronous post-update learner snapshot.

    directory is models/seed_<run_index>/w_<weight_index>/checkpoints/step_<actual>
    or that policy's final directory at its effective final budget. Existing
    directories are never overwritten. The caller must not train concurrently;
    this API supports checkpoint continuation, not exact live-state crash replay.
    """
    import shutil
    import zipfile
    directory, payload, final = _bundle_context(directory, identity, resume_state)
    if directory.exists():
        raise StorageError(f'Bundle destination already exists: {directory}')
    counters, normalizer = _validate_live_bundle(model, training_handle, identity, payload)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + directory.name + '.', suffix='.tmp', dir=directory.parent))
    try:
        model_name = 'final_model.zip' if final else 'model.zip'
        model_path = stage / model_name
        with model_path.open('wb') as stream:
            model.save(stream)
            stream.flush()
            os.fsync(stream.fileno())
        with zipfile.ZipFile(model_path) as archive:
            if (not {'data', 'policy.pth', 'policy.optimizer.pth'}.issubset(archive.namelist())
                    or archive.testzip() is not None):
                raise StorageError('Model ZIP is incomplete or corrupt')
            data = json.loads(archive.read('data'))
            if data['num_timesteps'] != counters['num_timesteps'] or data['_n_updates'] != counters['n_updates']:
                raise StorageError('Serialized learner counters do not match the bundle update')
        names = [model_name]
        if normalizer is not None:
            path = stage / 'vecnormalize.pkl'
            normalizer.save(path)
            _validate_saved_normalizer(path, normalizer)
            names.append(path.name)
        # Verify the borrowed learner and scalarization still describe the same update.
        after, same_normalizer = _validate_live_bundle(model, training_handle, identity, payload)
        if after != counters or same_normalizer is not normalizer:
            raise StorageError('Learner changed while saving the bundle')
        for path in stage.iterdir():
            if path.name not in names or path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise StorageError('Unexpected or empty member in staged bundle')
            with path.open('r+b') as stream:
                stream.flush()
                os.fsync(stream.fileno())
        hashes = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in names}
        metadata_payload = {
            'storage_schema_version': STORAGE_SCHEMA_VERSION, 'bundle_schema_version': 1,
            'kind': 'final' if final else 'checkpoint', 'identity': _json_value(identity),
            'config_fingerprint': identity.experiment_fingerprint,
            'actual_step': identity.actual_step, 'objective_scales': payload['config']['objective_scales'],
            'reference_scaled': payload['config']['reference_scaled'], 'architecture': payload['architecture'],
            'normalization': payload['config']['normalization'], 'counters': counters,
            'resume_state': _json_value(resume_state), 'member_hashes': hashes,
            'versions': payload['versions'], 'source_hashes': payload['source_hashes']}
        atomic_write_json(stage / 'bundle.json', metadata_payload)
        # The marker binds metadata and all members; readers must validate both.
        atomic_write_json(stage / 'COMMITTED', {
            'bundle_schema_version': 1, 'bundle_id': identity.bundle_id, 'actual_step': identity.actual_step,
            'bundle_sha256': hashlib.sha256((stage / 'bundle.json').read_bytes()).hexdigest()})
        if directory.exists():
            raise StorageError('Bundle destination was created by another writer')
        os.rename(stage, directory)
    finally:
        # stage is an owned mkdtemp sibling, never a user-supplied cleanup path.
        if stage.exists():
            if stage.resolve().parent != directory.parent.resolve() or not stage.name.endswith('.tmp'):
                raise StorageError('Refusing cleanup outside the owned staging directory')
            shutil.rmtree(stage)


from dataclasses import dataclass


@dataclass(frozen=True)
class BundleMetadata:
    """Validated local bundle reference; contains no loaded learner/resources."""
    directory: Path
    identity: object
    model_filename: str
    normalization_enabled: bool
    bundle_sha256: str

    # Params: self; returns: self.identity.actual_step.
    @property
    def actual_step(self):
        return self.identity.actual_step

    # Params: self; returns: self.identity.experiment_fingerprint.
    @property
    def config_fingerprint(self):
        return self.identity.experiment_fingerprint


def _committed_bundle(directory, expected_fingerprint=None):
    """Validate the canonical directory, marker, metadata and all member hashes."""
    import zipfile
    from .types import PolicyIdentity, ResumeState
    try:
        directory = Path(directory)
        if directory.is_symlink():
            raise StorageError('Bundle directories cannot be symlinks')
        directory = directory.absolute().resolve()
        if not directory.is_dir():
            raise StorageError('Bundle directory is absent')
        for name in ('bundle.json', 'COMMITTED'):
            if (directory / name).is_symlink():
                raise StorageError('Bundle records cannot be symlinks')
        meta = _read_json(directory / 'bundle.json')
        marker = _read_json(directory / 'COMMITTED')
        identity = PolicyIdentity(**meta['identity'])
        digest = hashlib.sha256((directory / 'bundle.json').read_bytes()).hexdigest()
        if (type(meta['bundle_schema_version']) is not int or meta['bundle_schema_version'] != 1
                or type(meta['storage_schema_version']) is not int or meta['storage_schema_version'] != STORAGE_SCHEMA_VERSION
                or _json_bytes(marker) != _json_bytes({'bundle_schema_version': 1, 'bundle_id': identity.bundle_id,
                              'actual_step': identity.actual_step, 'bundle_sha256': digest})
                or meta['config_fingerprint'] != identity.experiment_fingerprint
                or meta['actual_step'] != identity.actual_step
                or (expected_fingerprint is not None and expected_fingerprint != identity.experiment_fingerprint)):
            raise StorageError('Bundle commit/step/fingerprint mismatch')
        _, payload, final = _bundle_context(directory, identity, ResumeState(**meta['resume_state']))
        for field, expected in (
                ('architecture', payload['architecture']), ('objective_scales', payload['config']['objective_scales']),
                ('reference_scaled', payload['config']['reference_scaled']),
                ('normalization', payload['config']['normalization']), ('versions', payload['versions']),
                ('source_hashes', payload['source_hashes']), ('kind', 'final' if final else 'checkpoint')):
            if _json_bytes(meta[field]) != _json_bytes(expected):
                raise StorageError(f'Bundle compatibility mismatch: {field}')
        settings = meta['normalization']
        enabled = settings['norm_obs'] or settings['norm_reward']
        model_name = 'final_model.zip' if final else 'model.zip'
        required = {model_name} | ({'vecnormalize.pkl'} if enabled else set())
        if set(meta['member_hashes']) != required or {p.name for p in directory.iterdir()} != required | {'bundle.json', 'COMMITTED'}:
            raise StorageError('Bundle member inventory is incomplete or unexpected')
        for name, expected in meta['member_hashes'].items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or not path.stat().st_size:
                raise StorageError('Missing/invalid bundle member')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise StorageError(f'Bundle member checksum mismatch: {name}')
        counters = meta['counters']
        updates = identity.actual_step // payload['budget_plan']['rollout_size']
        if (counters['num_timesteps'] != identity.actual_step or counters['completed_rollouts'] != updates
                or type(counters['n_updates']) is not int or counters['n_updates'] < updates):
            raise StorageError('Bundle update counters are inconsistent')
        if (payload['config']['ppo']['target_kl'] is None
                and counters['n_updates'] != updates * payload['config']['ppo']['n_epochs']):
            raise StorageError('Bundle optimization counters do not match completed updates')
        with zipfile.ZipFile(directory / model_name) as archive:
            if not {'data', 'policy.pth', 'policy.optimizer.pth'}.issubset(archive.namelist()) or archive.testzip() is not None:
                raise StorageError('Incomplete/corrupt model archive')
            data = json.loads(archive.read('data'))
            if data['num_timesteps'] != identity.actual_step or data['_n_updates'] != counters['n_updates']:
                raise StorageError('Serialized model step/counters mismatch')
        return BundleMetadata(directory, identity, model_name, bool(enabled), digest), meta, payload
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise StorageError(f'Invalid committed bundle {directory}: {error}') from error


def discover_latest_checkpoint(policy_dir, expected_fingerprint):
    """Return the largest numeric compatible actual step; ignore incomplete bundles."""
    import re
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise TypeError('expected_fingerprint must be a nonempty string')
    parent = Path(policy_dir) / 'checkpoints'
    if not parent.is_dir():
        return None
    valid = []
    for path in parent.iterdir():
        if not re.fullmatch(r'step_[0-9]+', path.name) or not path.is_dir():
            continue
        try:
            record, _, _ = _committed_bundle(path, expected_fingerprint)
        except StorageError:
            continue
        valid.append(record)
    return max(valid, key=lambda record: record.actual_step) if valid else None


# Params: config, payload, evaluation; returns: None.
def _load_config_compatibility(config, payload, evaluation):
    validate_config(config)
    requested = _json_value(config)
    requested.pop('output_dir')
    requested.pop('logging')
    if evaluation:
        # Evaluation may choose a loading device; training continuation stays strict.
        requested['ppo']['device'] = payload['config']['ppo']['device']
    if _json_bytes(requested) != _json_bytes(payload['config']):
        raise StorageError('Requested config is incompatible with the saved experiment')


# Params: handle, payload, identity; returns: None.
def _loading_handle(handle, payload, identity):
    import numpy as np
    from .types import TrainingEnvHandle
    if not isinstance(handle, TrainingEnvHandle) or handle.closed or handle.vec_env is None:
        raise StorageError('Loading for training requires an open TrainingEnvHandle')
    if handle.vec_env.num_envs != payload['config']['ppo']['n_envs']:
        raise StorageError('Training worker count mismatch')
    for name, expected in (('weights', identity.weight), ('scales', payload['config']['objective_scales'])):
        values = handle.vec_env.get_attr(name)
        if len(values) != handle.vec_env.num_envs or any(not np.array_equal(v, np.asarray(expected, np.float32)) for v in values):
            raise StorageError(f'Training scalarization mismatch: {name}')
    settings = payload['config']['normalization']
    enabled = settings['norm_obs'] or settings['norm_reward']
    if (handle.vec_normalize is not None) != enabled:
        raise StorageError('Training normalization presence mismatch')
    if enabled:
        if handle.vec_env is not handle.vec_normalize:
            raise StorageError('Training normalizer must be the outer environment')
        for name, expected in settings.items():
            if getattr(handle.vec_normalize, name) != expected:
                raise StorageError(f'Training normalizer setting mismatch: {name}')


# Params: normalizer, settings, observation_shape; returns: None.
def _loaded_normalizer(normalizer, settings, observation_shape):
    import numpy as np
    for name, expected in settings.items():
        if _json_bytes(getattr(normalizer, name)) != _json_bytes(expected):
            raise StorageError(f'Loaded normalizer setting mismatch: {name}')
    for name in ('ret_rms',) + (('obs_rms',) if settings['norm_obs'] else ()):
        rms = getattr(normalizer, name)
        for field in ('mean', 'var', 'count'):
            if not np.all(np.isfinite(getattr(rms, field))):
                raise StorageError('Nonfinite normalization statistics')
        if np.ndim(rms.count) != 0 or np.any(rms.var < 0) or rms.count < 0:
            raise StorageError('Invalid normalization statistics')
        expected_shape = tuple(observation_shape) if name == 'obs_rms' else ()
        if np.shape(rms.mean) != expected_shape or np.shape(rms.var) != expected_shape:
            raise StorageError('Normalization statistics shape mismatch')


# Params: model, record, meta, payload; returns: None.
def _loaded_model(model, record, meta, payload):
    from stable_baselines3.ppo import MlpPolicy
    import numpy as np
    ppo, architecture = payload['config']['ppo'], payload['architecture']
    extractor = model.policy.features_extractor
    if (type(model.policy) is not MlpPolicy or model.num_timesteps != record.actual_step
            or model.n_envs != ppo['n_envs']
            or model._n_updates != meta['counters']['n_updates']
            or model.seed != payload['seed_plans'][record.identity.run_index]['learner_seed']
            or type(extractor).__module__ + '.' + type(extractor).__name__ != architecture['features_extractor']
            or extractor.features_dim != architecture['features_dim']
            or list(model.observation_space.shape) != architecture['observation_shape']
            or not np.array_equal(model.action_space.nvec, architecture['action_nvec'])):
        raise StorageError('Loaded PPO identity/architecture/counters mismatch')
    for name in ('n_steps', 'n_epochs', 'batch_size', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
                 'max_grad_norm', 'normalize_advantage', 'use_sde', 'sde_sample_freq', 'target_kl', 'learning_rate'):
        if getattr(model, name) != ppo[name]:
            raise StorageError(f'Loaded PPO setting mismatch: {name}')
    if any(model.clip_range(progress) != ppo['clip_range'] for progress in (0., 1.)):
        raise StorageError('Loaded clipping schedule mismatch')
    expected_vf = ppo['clip_range_vf']
    if ((model.clip_range_vf is None) != (expected_vf is None)
            or (expected_vf is not None and any(model.clip_range_vf(p) != expected_vf for p in (0., 1.)))):
        raise StorageError('Loaded value clipping mismatch')
    if not model.policy.optimizer.state_dict()['state']:
        raise StorageError('Loaded trained policy is missing optimizer state')


def load_policy_bundle(directory, config, training_handle=None):
    """Load validated local SB3 files, owning only evaluation normalization carriers.

    Evaluation models have no training environment. Normalize raw observations
    once using make_frozen_obs_normalizer(bundle.vec_normalize); rewards still
    come from the separate raw evaluator. Training handles are borrowed and are
    updated only after model/normalizer restoration succeeds completely.
    """
    from .types import PolicyBundle
    record, meta, payload = _committed_bundle(directory)
    evaluation = training_handle is None
    _load_config_compatibility(config, payload, evaluation)
    if not evaluation:
        _loading_handle(training_handle, payload, record.identity)
    bundle = PolicyBundle(identity=record.identity, config_fingerprint=record.config_fingerprint, metadata=meta)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        normalizer = None
        env = None if evaluation else training_handle.vec_env
        if record.normalization_enabled:
            if evaluation:
                from .envs import _make_dummy_vec_env, make_scalar_env
                carrier = _make_dummy_vec_env([lambda: make_scalar_env(config.env, record.identity.weight,
                                                     None, config.objective_scales)])
                bundle.normalization_env = carrier  # register before unpickling can fail
                normalizer = VecNormalize.load(record.directory / 'vecnormalize.pkl', carrier)
                bundle.normalization_env = normalizer
                bundle.owned_resources = (carrier,)
            else:
                normalizer = VecNormalize.load(record.directory / 'vecnormalize.pkl', training_handle.vec_normalize.venv)
            _loaded_normalizer(normalizer, meta['normalization'], payload['architecture']['observation_shape'])
            if evaluation:
                normalizer.training = False
                normalizer.norm_reward = False
            else:
                normalizer.training = True
                env = normalizer
        from .evaluation import evaluation_state_guard
        with evaluation_state_guard():
            model = PPO.load(record.directory / record.model_filename, env=env, device=config.ppo.device)
        _loaded_model(model, record, meta, payload)
        if evaluation:
            model.policy.set_training_mode(False)
        bundle.model = model
        bundle.vec_normalize = normalizer
        if not evaluation and normalizer is not None:
            training_handle.vec_normalize = normalizer
            training_handle.vec_env = normalizer
        return bundle
    except BaseException:
        bundle.close()
        raise


def _completion_evaluation(path, record, payload):
    """Validate an immutable JSON commit record and its numeric NPZ payload.

    Evaluation-record schema v1: committed=true, data_sha256, identity,
    actual_step, deterministic, episode_seeds, objective_scales and per-episode
    episode_metrics. The NPZ contains the eight EvaluationResult return/summary
    arrays. The JSON is published last by the evaluation persistence layer.
    """
    import numpy as np
    from .types import EvaluationResult, EpisodeMetrics, TargetObservationMetrics
    path = Path(path).absolute().resolve()
    policy = record.directory.parent
    root = policy.parents[2]
    expected = root / 'results/evaluations' / f'seed_{record.identity.run_index}' / f'w_{record.identity.weight_index}' / f'step_{record.actual_step}.json'
    if path != expected:
        raise StorageError('Completion requires the canonical final-budget intermediate evaluation')
    meta = _read_json(path)
    numeric = path.with_suffix('.npz')
    if 'record_schema_version' in meta:
        # Use Block 21's codec for new records, including optional traces.
        # The earlier flat completion-record schema remains readable below.
        result, meta = _load_evaluation_record(path)
        expected, kind = _evaluation_location(root, payload,
            record.identity.experiment_fingerprint, record.identity, result)
        if (path != expected or kind != 'evaluations' or meta['kind'] != kind
                or meta['objective_scales'] != payload['config']['objective_scales']):
            raise StorageError('Completion evaluation path/namespace/scales mismatch')
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), meta['npz_sha256']
    if (type(meta['storage_schema_version']) is not int or meta['storage_schema_version'] != STORAGE_SCHEMA_VERSION
            or meta['committed'] is not True
            or _json_bytes(meta['identity']) != _json_bytes(record.identity.to_dict()) or meta['actual_step'] != record.actual_step
            or meta['deterministic'] is not True
            or meta['episode_seeds'] != payload['seed_plans'][record.identity.run_index]['eval_seeds']
            or meta['objective_scales'] != payload['config']['objective_scales']
            or hashlib.sha256(numeric.read_bytes()).hexdigest() != meta['data_sha256']):
        raise StorageError('Completion evaluation identity/seeds/scales/commit/checksum mismatch')
    metrics = []
    for item in meta['episode_metrics']:
        values = dict(item)
        values['target_observation_metrics'] = tuple(TargetObservationMetrics(**t) for t in values.get('target_observation_metrics', ()))
        metrics.append(EpisodeMetrics(**values))
    names = {prefix + '_' + suffix for prefix in ('returns', 'mean', 'sd', 'se') for suffix in ('raw', 'scaled')}
    with np.load(numeric, allow_pickle=False) as arrays:
        if set(arrays.files) != names:
            raise StorageError('Completion evaluation return-array inventory mismatch')
        values = {name: arrays[name].copy() for name in names}
    result = EvaluationResult(episode_seeds=meta['episode_seeds'], **values,
        episode_metrics=tuple(metrics), deterministic=True, actual_step=record.actual_step, identity=record.identity)
    if not np.allclose(result.returns_scaled, result.returns_raw / np.asarray(meta['objective_scales']), rtol=1e-12, atol=1e-12):
        raise StorageError('Completion evaluation raw/scaled returns disagree')
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), meta['data_sha256']


def mark_policy_complete(policy_dir, final_bundle, final_eval_record):
    """Commit completion only with a valid matched-final bundle and saved validation.

    final_bundle is a directory or BundleMetadata. EvaluationResult objects alone
    cannot establish persistence. Identical repeat marks are idempotent; an
    existing conflicting completion record is never overwritten.
    """
    try:
        directory = final_bundle.directory if isinstance(final_bundle, BundleMetadata) else final_bundle
        record, meta, payload = _committed_bundle(directory)
        policy = Path(policy_dir).absolute().resolve()
        if record.directory != policy / 'final' or meta['kind'] != 'final' or record.actual_step != payload['budget_plan']['effective_total']:
            raise StorageError('Completion requires the valid final bundle at the effective target')
        path, json_hash, numeric_hash = _completion_evaluation(final_eval_record, record, payload)
        marker = {'storage_schema_version': STORAGE_SCHEMA_VERSION, 'identity': record.identity.to_dict(),
                  'actual_step': record.actual_step, 'config_fingerprint': record.config_fingerprint,
                  'final_bundle': 'final', 'bundle_sha256': record.bundle_sha256,
                  'evaluation_json': str(path.relative_to(policy.parents[2]).as_posix()),
                  'evaluation_json_sha256': json_hash, 'evaluation_npz_sha256': numeric_hash}
        destination = policy / 'COMPLETE.json'
        if destination.exists():
            if _json_bytes(_read_json(destination)) != _json_bytes(marker):
                raise StorageError('Conflicting policy completion record already exists')
            return
        atomic_write_json(destination, marker)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise StorageError(f'Cannot mark policy complete: {error}') from error


_EVAL_ARRAYS = ('returns_raw', 'returns_scaled', 'mean_raw', 'mean_scaled',
                'sd_raw', 'sd_scaled', 'se_raw', 'se_scaled')
_TRACE_ARRAYS = ('initial_observation', 'observations', 'vector_rewards', 'true_states',
                 'estimated_states', 'actions', 'observed_targets', 'elapsed_steps',
                 'coverage_values', 'terminated', 'truncated')


# Params: experiment_dir, config; returns: (root, payload, fingerprint).
def _evaluation_context(experiment_dir, config=None):
    root = Path(experiment_dir).expanduser().resolve()
    if (root / '.INITIALIZING').exists():
        raise StorageError('Evaluation storage requires a committed experiment initialization')
    manifest = _read_json(root / 'manifest.json')
    payload = manifest['identity']
    fingerprint = hashlib.sha256(_json_bytes(payload)).hexdigest()
    _validate_initialization(root, payload, fingerprint)
    if config is not None:
        validate_config(config)
        science = _json_value(config)
        science.pop('output_dir')
        science.pop('logging')
        if science != payload['config']:
            raise StorageError('Evaluation config does not match saved experiment science')
    return root, payload, fingerprint


# Params: result; returns: (metadata, arrays, SHA-256).
def _encode_evaluation(result):
    from .types import EvaluationResult
    if not isinstance(result, EvaluationResult):
        raise TypeError('result must be EvaluationResult')
    arrays = {name: getattr(result, name) for name in _EVAL_ARRAYS}
    for index, trace in enumerate(result.traces):
        arrays.update({f'trace_{index}__{name}': getattr(trace, name) for name in _TRACE_ARRAYS})
    body = {name: _json_value(getattr(result, name)) for name in
            ('episode_seeds', 'episode_metrics', 'deterministic', 'actual_step', 'identity')}
    body['trace_count'] = len(result.traces)
    body['trace_initial_known_counts'] = [trace.initial_known_count for trace in result.traces]
    digest = hashlib.sha256(_json_bytes(body))
    for name, array in sorted(arrays.items()):
        digest.update(_json_bytes([name, array.dtype.str, array.shape]))
        digest.update(array.tobytes(order='C'))
    return body, arrays, digest.hexdigest()


# Params: root, payload, fingerprint, identity, result; returns: (record path, namespace).
def _evaluation_location(root, payload, fingerprint, identity, result):
    import numpy as np
    from .types import PolicyIdentity
    if not isinstance(identity, PolicyIdentity) or result.identity != identity:
        raise StorageError('Evaluation identity argument must match the complete result identity')
    run, weight, step = identity.run_index, identity.weight_index, identity.actual_step
    config = payload['config']
    if (identity.experiment_fingerprint != fingerprint
            or not 0 <= run < len(config['run_seeds']) or config['run_seeds'][run] != identity.run_seed
            or not 0 <= weight < len(payload['weights']['values'])
            or list(identity.weight) != payload['weights']['values'][weight]
            or not result.deterministic or result.actual_step != step):
        raise StorageError('Evaluation fingerprint/run/weight/step or deterministic identity mismatch')
    seeds = payload['seed_plans'][run]
    if list(result.episode_seeds) == seeds['eval_seeds']:
        kind = 'evaluations'
        if step not in payload['budget_plan']['eval_steps']:
            raise StorageError('Intermediate evaluation must use a saved actual evaluation budget')
    elif list(result.episode_seeds) == seeds['final_seeds']:
        kind = 'final'
        if step != payload['budget_plan']['effective_total']:
            raise StorageError('Final evaluation requires the matched final actual budget')
    else:
        raise StorageError('Episode seeds must exactly match a saved ordered evaluation/final list')
    with np.errstate(over='ignore', invalid='ignore'):
        expected = result.returns_raw / np.asarray(config['objective_scales'], np.float64)
    if not np.array_equal(result.returns_scaled, expected):
        raise StorageError('Scaled episode returns do not match the saved fixed objective scales')
    for trace in result.traces:
        if (trace.initial_observation.shape != tuple(payload['architecture']['observation_shape'])
                or trace.true_states.shape[1] != config['env']['known_targets'] + config['env']['unknown_targets']
                or trace.initial_known_count != config['env']['known_targets']):
            raise StorageError('Evaluation trace dimensions/catalogue do not match saved science')
    return root / 'results' / kind / f'seed_{run}' / f'w_{weight}' / f'step_{step}.json', kind


def _load_evaluation_record(path):
    """Read a committed JSON/NPZ pair, validating checksums and record contracts."""
    import numpy as np
    import zipfile
    from .types import (EvaluationResult, PolicyIdentity, EpisodeMetrics,
                        TargetObservationMetrics, EpisodeTrace)
    path = Path(path)
    try:
        meta = _read_json(path)
        if (set(meta) - {'resume_attempt_id'} != {'record_schema_version', 'kind', 'objective_scales', 'result',
                         'content_sha256', 'npz_sha256'} or meta['record_schema_version'] != 1
                or ('resume_attempt_id' in meta and (not isinstance(meta['resume_attempt_id'], str) or not meta['resume_attempt_id']))):
            raise StorageError('Invalid committed evaluation metadata schema')
        numeric = path.with_suffix('.npz')
        if hashlib.sha256(numeric.read_bytes()).hexdigest() != meta['npz_sha256']:
            raise StorageError('Evaluation numeric checksum mismatch')
        body = dict(meta['result'])
        count = body.pop('trace_count')
        known_counts = body.pop('trace_initial_known_counts')
        seeds = body['episode_seeds']
        if type(count) is not int or count not in (0, len(seeds)) or len(known_counts) != count:
            raise StorageError('Trace count must be zero or the full ordered episode count')
        expected = set(_EVAL_ARRAYS) | {f'trace_{i}__{name}' for i in range(count) for name in _TRACE_ARRAYS}
        with np.load(numeric, allow_pickle=False) as archive:
            if len(archive.files) != len(expected) or set(archive.files) != expected:
                raise StorageError('Evaluation numeric inventory mismatch')
            arrays = {name: archive[name] for name in expected}
        body['identity'] = PolicyIdentity(**body['identity'])
        metrics = []
        for item in body['episode_metrics']:
            item = dict(item)
            item['target_observation_metrics'] = tuple(TargetObservationMetrics(**t)
                                                       for t in item['target_observation_metrics'])
            metrics.append(EpisodeMetrics(**item))
        body['episode_metrics'] = tuple(metrics)
        traces = tuple(EpisodeTrace(seed=seeds[i],
            initial_known_count=known_counts[i],
            **{name: arrays[f'trace_{i}__{name}'] for name in _TRACE_ARRAYS}) for i in range(count))
        result = EvaluationResult(**body, **{name: arrays[name] for name in _EVAL_ARRAYS}, traces=traces)
        if _encode_evaluation(result)[2] != meta['content_sha256']:
            raise StorageError('Evaluation semantic checksum mismatch')
        return result, meta
    except (OSError, KeyError, TypeError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise StorageError(f'Cannot load committed evaluation {path}: {error}') from error


def _save_evaluation_record_unlocked(experiment_dir, identity, result):
    """Commit numeric data first, then its checksummed JSON marker; never replace a record.

    Identical committed writes are no-ops. A per-slot exclusive lock prevents
    competing writers from interleaving the two files. Orphan NPZ data from an
    interrupted first write is not a record and can be retried under the lock.
    """
    root, payload, fingerprint = _evaluation_context(experiment_dir)
    body, arrays, digest = _encode_evaluation(result)
    path, kind = _evaluation_location(root, payload, fingerprint, identity, result)
    if payload['source_hashes'] != _source_hashes() or payload['versions'] != _versions():
        raise StorageError('Cannot commit an evaluation with changed source/package provenance')
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.LOCK')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise StorageError('Evaluation slot has an active/interrupted writer; resolve its lock explicitly') from error
    os.close(descriptor)
    try:
        if path.exists():
            previous, meta = _load_evaluation_record(path)
            _evaluation_location(root, payload, fingerprint, previous.identity, previous)
            if meta['kind'] != kind or meta['objective_scales'] != payload['config']['objective_scales'] or _encode_evaluation(previous)[2] != digest:
                raise StorageError('Conflicting committed evaluation; use a separate explicit attempt')
            return
        numeric = path.with_suffix('.npz')
        atomic_write_npz(numeric, arrays)
        journal = _resume_directory(root, identity.run_index, identity.weight_index) / 'RESUME.json'
        attempt = _read_json(journal)['state']['resume_attempt_id'] if journal.exists() else 'initial'
        atomic_write_json(path, {'record_schema_version': 1, 'kind': kind,
            'objective_scales': payload['config']['objective_scales'], 'result': body,
            'content_sha256': digest, 'npz_sha256': hashlib.sha256(numeric.read_bytes()).hexdigest(),
            'resume_attempt_id': attempt})
    finally:
        lock.unlink()


def store_eval_slot(results, seed_index, budget_index, weight_index, result):
    """Populate one actual-budget slot, preserving missing NaNs and conflict identity."""
    import numpy as np
    from .types import ExperimentResults
    if not isinstance(results, ExperimentResults):
        raise TypeError('results must be ExperimentResults')
    results.validate()
    key = (seed_index, budget_index, weight_index)
    if any(type(index) is not int or not 0 <= index < results.validity.shape[axis]
           for axis, index in enumerate(key)):
        raise StorageError('Result slot indices must be in-range Python integers')
    _, _, digest = _encode_evaluation(result)
    if (not result.deterministic or result.identity.run_index != seed_index or result.identity.weight_index != weight_index
            or result.actual_step != int(results.actual_budgets[budget_index])):
        raise StorageError('Evaluation must match its seed/weight/actual-budget slot')
    if results.episode_returns is not None and results.episode_returns.shape[-2] != len(result.episode_seeds):
        raise StorageError('Evaluation episode count does not match result tensor')
    hashes = getattr(results, '_evaluation_slot_hashes', {})
    if results.validity[key]:
        if (hashes.get(key) != digest or not np.array_equal(results.means_raw[key], result.mean_raw)
                or not np.array_equal(results.means_scaled[key], result.mean_scaled)
                or results.identities.get(key) != result.identity
                or results.episode_metric_tables.get(key) != result.episode_metrics
                or (results.episode_returns is not None
                    and not np.array_equal(results.episode_returns[key], result.returns_raw))):
            raise StorageError('Conflicting populated evaluation slot')
        return
    results.means_raw[key] = result.mean_raw
    results.means_scaled[key] = result.mean_scaled
    if results.episode_returns is not None:
        results.episode_returns[key] = result.returns_raw
    results.identities[key] = result.identity
    results.episode_metric_tables[key] = result.episode_metrics
    results.hypervolume[seed_index, budget_index] = np.nan
    results.validity[key] = True
    results._evaluation_slot_hashes = dict(hashes)
    results._evaluation_slot_hashes[key] = digest


def load_evaluation_records(experiment_dir, config, *, write_cache=True):
    """Rebuild the entire intermediate grid and refresh disposable shared caches.

    This is a sequential/finalizer operation, never a concurrent policy worker
    operation. Use load_policy_evaluation_records for read-only slot loading.
    """
    if type(write_cache) is not bool:
        raise TypeError('write_cache must be bool')
    return _load_evaluation_records(experiment_dir, config, write_cache=write_cache)


def load_policy_evaluation_records(experiment_dir, config, seed_index, weight_index):
    """Read only one policy's committed intermediate records without any writes.

    Retain the full saved grid dimensions/indices; other slots stay missing.
    Other policies' pending/corrupt records cannot block this policy's resume.
    """
    validate_config(config)
    if (type(seed_index) is not int or not 0 <= seed_index < len(config.run_seeds)
            or type(weight_index) is not int or not 0 <= weight_index < config.n_weights):
        raise StorageError('Policy record indices must identify a declared slot')
    return _load_evaluation_records(experiment_dir, config,
        policy_slot=(seed_index, weight_index), write_cache=False)


def _load_evaluation_records(experiment_dir, config, *, policy_slot=None, write_cache=True):
    """Rebuild intermediate tensors/HV from authoritative records, then refresh caches.

    Final-test records live in their separate namespace and never enter an
    intermediate curve. Cache files are disposable and are never read here.
    Numeric orphans/temporary files are not committed records; malformed JSON
    commits, corrupt checksums or a misplaced identity fail explicitly.
    """
    import numpy as np
    from .types import ExperimentResults
    from .pareto import compute_hv_curve
    root, payload, fingerprint = _evaluation_context(experiment_dir, config)
    if policy_slot is None:
        journals = (root / 'results/evaluations').glob('seed_*/w_*/RESUME.json')
        records = (root / 'results/evaluations').glob('seed_*/w_*/step_*.json')
    else:
        directory = _resume_directory(root, *policy_slot)
        journals = directory.glob('RESUME.json')
        records = directory.glob('step_*.json')
    resume_plans = {}
    for journal in journals:
        plan = _read_json(journal)
        _validate_resume_plan(root, plan, payload, fingerprint, plan.get('run_index'), plan.get('weight_index'))
        if journal != _resume_directory(root, plan['run_index'], plan['weight_index']) / 'RESUME.json':
            raise StorageError('Resume journal is in the wrong policy directory')
        if plan['status'] != 'ready':
            raise StorageError('Resume reconciliation is pending; resolve it before reading curves')
        resume_plans[(plan['run_index'], plan['weight_index'])] = plan
    results = ExperimentResults.empty(len(config.run_seeds),
        payload['budget_plan']['eval_steps'], config.n_weights, config.eval_episodes)
    for path in sorted(records):
        result, meta = _load_evaluation_record(path)
        expected, kind = _evaluation_location(root, payload, fingerprint, result.identity, result)
        if path != expected or kind != 'evaluations' or meta['kind'] != kind or meta['objective_scales'] != list(config.objective_scales):
            raise StorageError('Evaluation commit path/namespace/scales do not match its saved slot')
        plan = resume_plans.get((result.identity.run_index, result.identity.weight_index))
        if (plan is not None and result.actual_step > plan['state']['current_step']
                and plan['state']['resume_attempt_id'] != 'initial'
                and (meta.get('resume_attempt_id') != plan['state']['resume_attempt_id']
                     or not _resumed_bundle_matches(root, result.identity, plan['state']['resume_attempt_id'], fingerprint))):
            raise StorageError('Evaluation is not part of the active resumed trajectory')
        budget_index = payload['budget_plan']['eval_steps'].index(result.actual_step)
        store_eval_slot(results, result.identity.run_index, budget_index, result.identity.weight_index, result)
    results.hypervolume[:] = compute_hv_curve(results, config.reference_scaled)
    results.validate()
    if write_cache:
        cache = root / 'results'
        for name, array in (('eval_returns_raw', results.means_raw), ('eval_returns_scaled', results.means_scaled),
                            ('eval_valid', results.validity), ('actual_steps', results.actual_budgets),
                            ('hypervolume', results.hypervolume)):
            _atomic_write(cache / (name + '.npy'), lambda stream, a=array: np.save(stream, a, allow_pickle=False))
    results._resume_context = (root, payload, fingerprint)
    return results


# Params: root, run, weight; returns: root / 'results/evaluations' / f'seed_{run}' / f'w_{weight}'.
def _resume_directory(root, run, weight):
    return root / 'results/evaluations' / f'seed_{run}' / f'w_{weight}'


# Params: root, identity, attempt, fingerprint; returns: bool.
def _resumed_bundle_matches(root, identity, attempt, fingerprint):
    policy = root / 'models' / f'seed_{identity.run_index}' / f'w_{identity.weight_index}'
    for candidate in (policy / 'checkpoints' / f'step_{identity.actual_step}', policy / 'final'):
        try:
            bundle, meta, _ = _committed_bundle(candidate, fingerprint)
        except StorageError:
            continue
        if bundle.identity == identity and meta['resume_state']['resume_attempt_id'] == attempt:
            return True
    return False


@contextmanager
def _policy_storage_lock(directory):
    """Serialize canonical evaluation publication and resume reconciliation."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'RESUME.LOCK'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise StorageError('Policy storage is busy; an interrupted lock requires explicit recovery') from error
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink()


# Params: experiment_dir, identity, result; returns: None.
def save_evaluation_record(experiment_dir, identity, result):
    root, payload, fingerprint = _evaluation_context(experiment_dir)
    _encode_evaluation(result)
    _evaluation_location(root, payload, fingerprint, identity, result)
    directory = _resume_directory(root, identity.run_index, identity.weight_index)
    with _policy_storage_lock(directory):
        journal = directory / 'RESUME.json'
        if journal.exists():
            plan = _read_json(journal)
            state = _validate_resume_plan(root, plan, payload, fingerprint, identity.run_index, identity.weight_index)
            if plan['status'] != 'ready':
                raise StorageError('Resume reconciliation is pending; resolve it before writing evaluations')
            path, _ = _evaluation_location(root, payload, fingerprint, identity, result)
            if result.actual_step < state.current_step and not path.exists():
                raise StorageError('Cannot reconstruct an earlier missing evaluation from a later restored model')
            if result.actual_step == state.current_step and state.selected_bundle_path is not None:
                selected, _, _ = _committed_bundle(state.selected_bundle_path, fingerprint)
                if result.identity != selected.identity:
                    raise StorageError('Recovery evaluation must identify the selected restored bundle')
            if result.actual_step > state.current_step and state.resume_attempt_id != 'initial':
                if not _resumed_bundle_matches(root, identity, state.resume_attempt_id, fingerprint):
                    raise StorageError('Later evaluation requires a committed bundle from the active resume attempt')
        _save_evaluation_record_unlocked(root, identity, result)


# Params: identity; returns: evaluation ID.
def _evaluation_id(identity):
    return f'seed_{identity.run_index}/w_{identity.weight_index}/step_{identity.actual_step}'


def reconcile_evaluations_after_checkpoint(results, checkpoint_step, policy_identity):
    """Reconcile one loaded in-memory policy row; perform no filesystem writes.

    Use results returned by load_evaluation_records so the saved distinct eval
    and checkpoint schedules are available. Superseded slots are retained in an
    owned audit table before clearing them. Missing restored-budget evaluation
    is absent from completed_evaluation_ids; pending schedules are strictly later.
    """
    import numpy as np
    import uuid
    from .types import ExperimentResults, PolicyIdentity, ResumeState
    if not isinstance(results, ExperimentResults) or not isinstance(policy_identity, PolicyIdentity):
        raise TypeError('Expected ExperimentResults and PolicyIdentity')
    results.validate()
    if not hasattr(results, '_resume_context'):
        raise StorageError('Resume reconciliation requires results loaded with their saved budget context')
    _, payload, fingerprint = results._resume_context
    run, weight = policy_identity.run_index, policy_identity.weight_index
    budget = payload['budget_plan']
    if (type(checkpoint_step) is not int or checkpoint_step < 0
            or checkpoint_step > budget['effective_total'] or checkpoint_step % budget['rollout_size']
            or checkpoint_step != policy_identity.actual_step
            or policy_identity.experiment_fingerprint != fingerprint
            or not 0 <= run < len(payload['config']['run_seeds'])
            or policy_identity.run_seed != payload['config']['run_seeds'][run]
            or not 0 <= weight < len(payload['weights']['values'])
            or list(policy_identity.weight) != payload['weights']['values'][weight]):
        raise StorageError('Restored identity/step must match a saved completed-update policy slot')
    for key, identity in results.identities.items():
        if key[0] == run and key[2] == weight and (identity.experiment_fingerprint != fingerprint
                or identity.run_seed != policy_identity.run_seed or identity.weight != policy_identity.weight):
            raise StorageError('Cannot reconcile incompatible policy records')
    completed, removed = [], []
    for index, step in enumerate(results.actual_budgets):
        key = run, index, weight
        if not results.validity[key]:
            continue
        if int(step) <= checkpoint_step:
            completed.append(_evaluation_id(results.identities[key]))
            continue
        removed.append({'slot': key, 'identity': results.identities[key],
            'mean_raw': results.means_raw[key].copy(), 'mean_scaled': results.means_scaled[key].copy(),
            'episode_returns': None if results.episode_returns is None else results.episode_returns[key].copy(),
            'metrics': results.episode_metric_tables[key]})
    # All validation precedes mutation. Other policies and earlier slots retain
    # their arrays, metrics, identities and content-hash entries unchanged.
    for item in removed:
        key = item['slot']
        results.validity[key] = False
        results.means_raw[key] = np.nan
        results.means_scaled[key] = np.nan
        if results.episode_returns is not None:
            results.episode_returns[key] = np.nan
        results.identities.pop(key)
        results.episode_metric_tables.pop(key)
        getattr(results, '_evaluation_slot_hashes', {}).pop(key, None)
        results.hypervolume[key[:2]] = np.nan
    results._superseded_evaluations = tuple(getattr(results, '_superseded_evaluations', ())) + tuple(removed)
    results.validate()
    attempt = getattr(results, '_resume_attempt_id', 'resume-' + uuid.uuid4().hex)
    canonical = tuple(dict.fromkeys(tuple(getattr(results, '_canonical_attempt_ids', ())) + (attempt,)))
    superseded = tuple(getattr(results, '_superseded_attempt_ids', ()))
    if removed:
        superseded += ('superseded-' + attempt,)
    return ResumeState(False, checkpoint_step, budget['effective_total'],
        completed_evaluation_ids=tuple(completed),
        next_evaluation_budgets=tuple(s for s in budget['eval_steps'] if s > checkpoint_step),
        next_checkpoint_budgets=tuple(s for s in budget['checkpoint_steps'] if s > checkpoint_step),
        resume_attempt_id=attempt, canonical_attempt_ids=canonical,
        superseded_attempt_ids=tuple(dict.fromkeys(superseded)))


# Params: path; returns: SHA-256 hex str.
def _resume_member_hash(path):
    if path.is_symlink():
        raise StorageError('Resume members cannot be symlinks')
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise StorageError('Resume member is absent')
    entries = {}
    for member in path.rglob('*'):
        if member.is_symlink():
            raise StorageError('Resume directory members cannot be symlinks')
        if member.is_file():
            entries[member.relative_to(path).as_posix()] = hashlib.sha256(member.read_bytes()).hexdigest()
    return hashlib.sha256(_json_bytes(entries)).hexdigest()


def _finish_resume_moves(root, directory, plan):
    """Complete or retry same-volume moves from an already-published journal."""
    from .types import ResumeState
    root = root.resolve()
    state = ResumeState(**plan['state'])
    for item in plan['moves']:
        source, target = root / item['source'], root / item['destination']
        # Validate resolved absolute paths before any move, including directory
        # moves on Windows. No recursive deletion or overwrite is performed.
        source.resolve().relative_to(root)
        target.resolve().relative_to(root / 'results/superseded')
        if target.exists():
            if source.exists() or _resume_member_hash(target) != item['sha256']:
                raise StorageError('Resume archive conflicts with its journal')
            continue
        if _resume_member_hash(source) != item['sha256']:
            raise StorageError('Resume source changed after reconciliation was planned')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
    ready = dict(plan, status='ready')
    history = root / 'results/resume_history' / f'seed_{plan["run_index"]}' / f'w_{plan["weight_index"]}'
    history.mkdir(parents=True, exist_ok=True)
    record = history / (state.resume_attempt_id + '.json')
    if record.exists():
        if _json_bytes(_read_json(record)) != _json_bytes(ready):
            raise StorageError('Resume attempt history is immutable')
    else:
        atomic_write_json(record, ready)
    atomic_write_json(directory / 'RESUME.json', ready)
    return state


def _validate_resume_plan(root, plan, payload, fingerprint, run, weight):
    """Validate a journal's identity and exact move boundaries before replay."""
    import re
    from .types import ResumeState
    if (plan.get('resume_schema_version') != 1 or plan.get('fingerprint') != fingerprint
            or plan.get('run_index') != run or plan.get('weight_index') != weight
            or type(run) is not int or not 0 <= run < len(payload['config']['run_seeds'])
            or type(weight) is not int or not 0 <= weight < len(payload['weights']['values'])
            or plan.get('status') not in ('pending', 'ready')):
        raise StorageError('Resume journal identity/schema mismatch')
    state = ResumeState(**plan['state'])
    if (state.effective_target != payload['budget_plan']['effective_total']
            or state.current_step % payload['budget_plan']['rollout_size']
            or not re.fullmatch(r'initial|resume-[0-9a-f]{32}', state.resume_attempt_id)):
        raise StorageError('Resume journal counter/attempt mismatch')
    policy = root / 'models' / f'seed_{run}' / f'w_{weight}'
    if state.selected_bundle_path is not None and state.selected_bundle_path not in (
            policy / 'final', policy / 'checkpoints' / f'step_{state.current_step}'):
        raise StorageError('Resume selected bundle path does not match the saved policy')
    archive = root / 'results/superseded' / f'seed_{run}' / f'w_{weight}' / state.resume_attempt_id
    seen = set()
    for item in plan['moves']:
        source, destination = root / item['source'], root / item['destination']
        if source in seen or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise StorageError('Duplicate or invalid resume move entry')
        seen.add(source)
        namespace = None
        for kind in ('evaluations', 'final'):
            parent = root / 'results' / kind / f'seed_{run}' / f'w_{weight}'
            match = re.fullmatch(r'step_([0-9]+)\.(json|npz)', source.name)
            if source.parent == parent and match and int(match[1]) > state.current_step:
                namespace = kind
        match = re.fullmatch(r'step_([0-9]+)', source.name)
        if (source.parent == policy / 'checkpoints' and match and int(match[1]) > state.current_step
                or source == policy / 'final' and state.current_step < state.effective_target):
            namespace = 'bundles'
        if namespace is None or destination != archive / namespace / source.name:
            raise StorageError('Resume move is outside the selected policy future history')
        try:
            source.resolve().relative_to(root.resolve())
            destination.resolve().relative_to(archive.resolve())
        except ValueError as error:
            raise StorageError('Resume move escapes its experiment/archive') from error
    return state


def resolve_resume_state(experiment_dir, config, seed_index, weight_index):
    """Select a validated bundle, quarantine later history and return saved schedules.

    This plans checkpoint continuation; it neither loads/trains PPO nor claims
    exact environment/worker/learner RNG replay. A valid COMPLETE marker alone
    permits skipping. At a restored budget, callers repair a missing evaluation
    before advancing, even when current_step already equals effective_target.
    """
    import re
    import uuid
    from dataclasses import replace
    from .types import PolicyIdentity, ResumeState
    root, payload, fingerprint = _evaluation_context(experiment_dir, config)
    if (type(seed_index) is not int or not 0 <= seed_index < len(config.run_seeds)
            or type(weight_index) is not int or not 0 <= weight_index < config.n_weights):
        raise StorageError('Resume indices must identify a saved policy')
    policy = root / 'models' / f'seed_{seed_index}' / f'w_{weight_index}'
    directory = _resume_directory(root, seed_index, weight_index)
    with _policy_storage_lock(directory):
        if list(directory.glob('*.LOCK')) != [directory / 'RESUME.LOCK']:
            raise StorageError('An evaluation writer is active or interrupted; resolve its lock first')
        journal = directory / 'RESUME.json'
        previous = _read_json(journal) if journal.exists() else None
        if previous is not None:
            _validate_resume_plan(root, previous, payload, fingerprint, seed_index, weight_index)
        if previous is not None and previous.get('status') == 'pending':
            selected = previous['state']['selected_bundle_path']
            if selected is not None:
                record, _, _ = _committed_bundle(selected, fingerprint)
                if record.bundle_sha256 != previous['selected_bundle_sha256']:
                    raise StorageError('Pending resume selected bundle changed')
            return _finish_resume_moves(root, directory, previous)
        checkpoint = discover_latest_checkpoint(policy, fingerprint)
        final = None
        if (policy / 'final').exists():
            try:
                final, _, _ = _committed_bundle(policy / 'final', fingerprint)
            except StorageError:
                pass
        selected = max((b for b in (checkpoint, final) if b is not None),
                       key=lambda b: (b.actual_step, b is final), default=None)
        if (policy / 'COMPLETE.json').exists():
            if final is None:
                raise StorageError('Policy completion marker has no valid final bundle')
            evaluation = directory / f'step_{final.actual_step}.json'
            mark_policy_complete(policy, final, evaluation)  # Existing marker: validate, never overwrite.
            completed = []
            for path in directory.glob('step_*.json'):
                meta = _read_json(path)
                item = PolicyIdentity(**(meta['result']['identity'] if 'result' in meta else meta['identity']))
                if (item.experiment_fingerprint != fingerprint or item.run_index != seed_index
                        or item.run_seed != config.run_seeds[seed_index] or item.weight_index != weight_index
                        or item.weight != final.identity.weight or item.actual_step not in payload['budget_plan']['eval_steps']):
                    raise StorageError('Completed policy contains an incompatible evaluation identity')
                _completion_evaluation(path, replace(final, identity=item), payload)
                completed.append((item.actual_step, _evaluation_id(item)))
            source = previous['state'] if previous is not None else _read_json(final.directory / 'bundle.json')['resume_state']
            return ResumeState(True, final.actual_step, payload['budget_plan']['effective_total'],
                selected_bundle_path=final.directory,
                last_committed_checkpoint=None if checkpoint is None else checkpoint.directory,
                completed_evaluation_ids=tuple(item[1] for item in sorted(completed)),
                resume_attempt_id=source['resume_attempt_id'],
                canonical_attempt_ids=tuple(source['canonical_attempt_ids']),
                superseded_attempt_ids=tuple(source['superseded_attempt_ids']))
        if selected is None and ((policy / 'final').exists() or list((policy / 'checkpoints').glob('step_*'))):
            raise StorageError('Existing policy artifacts have no recoverable committed bundle')
        step = 0 if selected is None else selected.actual_step
        identity = selected.identity if selected is not None else PolicyIdentity(fingerprint,
            config.run_seeds[seed_index], seed_index, weight_index,
            tuple(payload['weights']['values'][weight_index]), 0, 'initial')
        results = load_policy_evaluation_records(root, config, seed_index, weight_index)
        moves, archived = [], []
        attempt = 'resume-' + uuid.uuid4().hex
        archive = root / 'results/superseded' / f'seed_{seed_index}' / f'w_{weight_index}' / attempt
        prior_attempt = 'initial' if selected is None else _read_json(selected.directory / 'bundle.json')['resume_state']['resume_attempt_id']
        for namespace in ('evaluations', 'final'):
            source_dir = root / 'results' / namespace / f'seed_{seed_index}' / f'w_{weight_index}'
            for path in sorted(source_dir.glob('step_*.json')):
                result, meta = _load_evaluation_record(path)
                expected, kind = _evaluation_location(root, payload, fingerprint, result.identity, result)
                if expected != path or kind != namespace:
                    raise StorageError('Cannot resume a misplaced evaluation record')
                if result.actual_step > step:
                    archived.append({'identity': _json_value(result.identity), 'namespace': namespace,
                                     'content_sha256': meta['content_sha256'],
                                     'source_attempt_id': meta.get('resume_attempt_id', prior_attempt)})
                    for member in (path, path.with_suffix('.npz')):
                        moves.append({'source': member.relative_to(root).as_posix(),
                            'destination': (archive / namespace / member.name).relative_to(root).as_posix(),
                            'sha256': _resume_member_hash(member)})
        # Preserve unusable later bundle destinations too, allowing resumed saves
        # to reach those budgets without overwriting old/corrupt checkpoints.
        for member in list((policy / 'checkpoints').glob('step_*')) + [policy / 'final']:
            match = re.fullmatch(r'step_([0-9]+)', member.name)
            member_step = int(match[1]) if match else payload['budget_plan']['effective_total']
            if member.exists() and member_step > step:
                moves.append({'source': member.relative_to(root).as_posix(),
                    'destination': (archive / 'bundles' / member.name).relative_to(root).as_posix(),
                    'sha256': _resume_member_hash(member)})
        if previous is not None and not moves and previous['state']['current_step'] == step:
            old_state = ResumeState(**previous['state'])
            if old_state.selected_bundle_path == (None if selected is None else selected.directory):
                results._resume_attempt_id = old_state.resume_attempt_id
                results._canonical_attempt_ids = old_state.canonical_attempt_ids
                results._superseded_attempt_ids = old_state.superseded_attempt_ids
                state = reconcile_evaluations_after_checkpoint(results, step, identity)
                return replace(state, selected_bundle_path=old_state.selected_bundle_path,
                               last_committed_checkpoint=old_state.last_committed_checkpoint)
        results._resume_attempt_id = 'initial' if selected is None and not moves else attempt
        lineage = previous['state'] if previous is not None else (
            None if selected is None else _read_json(selected.directory / 'bundle.json')['resume_state'])
        results._canonical_attempt_ids = () if lineage is None else tuple(dict.fromkeys(
            tuple(lineage['canonical_attempt_ids']) + (lineage['resume_attempt_id'],)))
        results._superseded_attempt_ids = () if lineage is None else tuple(lineage['superseded_attempt_ids'])
        state = reconcile_evaluations_after_checkpoint(results, step, identity)
        if moves:
            state = replace(state, superseded_attempt_ids=tuple(dict.fromkeys(
                state.superseded_attempt_ids + ('superseded-' + attempt,))))
        state = replace(state, selected_bundle_path=None if selected is None else selected.directory,
                        last_committed_checkpoint=None if checkpoint is None else checkpoint.directory)
        plan = {'resume_schema_version': 1, 'status': 'pending', 'fingerprint': fingerprint,
            'run_index': seed_index, 'weight_index': weight_index, 'state': _json_value(state),
            'selected_bundle_sha256': None if selected is None else selected.bundle_sha256,
            'continuation': 'checkpoint continuation; environment and RNG replay is not exact',
            'archived_evaluations': archived, 'moves': moves}
        _validate_resume_plan(root, plan, payload, fingerprint, seed_index, weight_index)
        atomic_write_json(journal, plan)
        return _finish_resume_moves(root, directory, plan)
