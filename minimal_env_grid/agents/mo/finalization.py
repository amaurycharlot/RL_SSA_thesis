"""Exclusive, restartable complete-grid finalization; importing does no work."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import os

from . import storage
from .experiment import load_experiment
from .ownership import claim_finalizer


# Params: root; returns: None.
def _no_active_writers(root):
    claims = list((root / 'models').rglob('OWNER.LOCK'))
    for namespace in ('evaluations', 'final'):
        claims.extend((root / 'results' / namespace).rglob('*.LOCK'))
    if claims:
        raise storage.StorageError(f'Finalization refuses active/interrupted policy or storage claims: {claims[0]}')


def _complete_grid(saved):
    """Validate every declared marker/bundle before global cache writes."""
    records = []
    target = saved.budget_plan.effective_total
    for seed_index, plan in enumerate(saved.seed_plans):
        for weight_index in range(len(saved.weights)):
            policy = saved.root / 'models' / f'seed_{seed_index}' / f'w_{weight_index}'
            if not (policy / 'COMPLETE.json').is_file():
                raise storage.StorageError(f'Finalization requires every policy complete: seed_{seed_index}/w_{weight_index}')
            record, meta, _ = storage._committed_bundle(policy / 'final', saved.fingerprint)
            expected = (seed_index, plan.run_seed, weight_index, tuple(saved.weights[weight_index].tolist()), target)
            identity = record.identity
            if (meta['kind'] != 'final' or (identity.run_index, identity.run_seed, identity.weight_index,
                    identity.weight, identity.actual_step) != expected):
                raise storage.StorageError('Finalization found an incompatible final policy slot/budget')
            # This call only validates an existing marker; absence was rejected.
            storage.mark_policy_complete(policy, record,
                saved.root / 'results/evaluations' / f'seed_{seed_index}' / f'w_{weight_index}' / f'step_{target}.json')
            records.append(record)
    results = storage.load_evaluation_records(saved.root, saved.config, write_cache=False)
    if not results.validity.all():
        raise storage.StorageError('Finalization requires every declared intermediate policy/budget evaluation')
    return records, results


# Params: root, path; returns: SHA-256 hex str.
def _hash(root, path):
    path.resolve().relative_to(root.resolve())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_experiment(experiment_dir):
    """Validate a complete grid, rebuild caches, test final policies and report.

    Final testing reuses its committed per-policy holdout records after an
    interruption. Reports/aggregates are disposable. FINALIZED.json is published
    last, after logger close; an older marker is archived before derived writes.
    This function performs no training or policy reconciliation/repair.
    """
    saved = load_experiment(experiment_dir)
    root, config = saved.root, saved.config
    with claim_finalizer(root, config) as claim:
        _no_active_writers(root)
        records, results = _complete_grid(saved)
        marker = root / 'FINALIZED.json'
        if marker.exists():
            if not marker.is_file():
                raise storage.StorageError('Finalization marker must be a regular file')
            history = root / 'results/finalization_history'
            history.mkdir(exist_ok=True)
            target = history / (claim.token + '.json')
            marker.resolve().relative_to(root.resolve())
            target.resolve().relative_to(root.resolve())
            os.rename(marker, target)
        from .reporting import make_logger, generate_reports
        from .evaluation import evaluate_final_policy_set
        from .trainer import _cache_policy_results
        with ExitStack() as logs:
            logger = logs.enter_context(make_logger(config, root))
            phase = 'cache rebuild'
            try:
                logger.record('experiment_finalization_start', None,
                    {'experiment_fingerprint': saved.fingerprint, 'n_policies': len(records), 'claim_token': claim.token})
                _cache_policy_results(root, results, config)
                phase = 'final evaluation'
                final = evaluate_final_policy_set(root, config, saved.seed_plans)
                if not final.validity.all():
                    raise storage.StorageError('Final evaluation returned an incomplete policy grid')
                phase = 'report generation'
                manifest = generate_reports(root, config)
                if manifest.experiment_fingerprint != saved.fingerprint:
                    raise storage.StorageError('Final report fingerprint does not match the saved experiment')
                canonical = {}
                for namespace in ('evaluations', 'final'):
                    for path in (root / 'results' / namespace).glob('seed_*/w_*/step_*.json'):
                        for member in (path, path.with_suffix('.npz')):
                            canonical[member.relative_to(root).as_posix()] = _hash(root, member)
                commitments = {}
                for record in records:
                    complete = record.directory.parent / 'COMPLETE.json'
                    commitments[complete.relative_to(root).as_posix()] = _hash(root, complete)
                    commitments[(record.directory / 'bundle.json').relative_to(root).as_posix()] = record.bundle_sha256
                derived_paths = ['results/' + name + '.npy' for name in
                    ('eval_returns_raw', 'eval_returns_scaled', 'eval_valid', 'actual_steps', 'hypervolume')]
                derived_paths += ['results/final_results.json', 'results/final_results.npz',
                                  'reports/report_manifest.json', *manifest.relative_paths]
                derived = {relative: _hash(root, root / relative) for relative in derived_paths}
                logger.record('experiment_finalization_artifacts_ready', None,
                    {'experiment_fingerprint': saved.fingerprint, 'n_policies': len(records),
                     'actual_step': saved.budget_plan.effective_total})
                logger.flush()
                phase = 'logger cleanup'
                logs.close()
                phase = 'finalization commit'
                storage.atomic_write_json(marker, {'schema_version': 1, 'kind': 'finalized_experiment',
                    'experiment_fingerprint': saved.fingerprint,
                    'n_policies': len(records), 'actual_step': saved.budget_plan.effective_total,
                    'run_seeds': config.run_seeds, 'test_seeds': saved.seed_plans[0].final_seeds,
                    'canonical_evaluation_hashes': canonical, 'policy_commitments': commitments,
                    'derived_file_hashes': derived, 'claim_token': claim.token,
                    'created_at': datetime.now(timezone.utc).isoformat()})
                return manifest
            except BaseException as error:
                try:
                    logger.record('experiment_finalization_failure', None,
                        {'phase': phase, 'error_type': type(error).__name__, 'error': str(error)})
                    logger.flush()
                except Exception:
                    pass
                raise
