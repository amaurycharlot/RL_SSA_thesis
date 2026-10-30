"""Evaluate the exact saved MO final-policy set; never train or select checkpoints."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    """Scientific settings are read from disk; alternate seeds require a label."""
    from scripts._ppo_mo_cli import path_argument
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--experiment', type=path_argument, required=True)
    parser.add_argument('--device', help='Policy loading/inference device only; never changes saved config')
    parser.add_argument('--retain-traces', action='store_true', help='Retain full episode traces in new evaluation records')
    parser.add_argument('--dataset-label', help='New alternate dataset under results/alternate_tests/LABEL; never overwrites canonical tests')
    parser.add_argument('--test-seeds', type=int, nargs='+', help='Ordered alternate uint32 episode seeds; requires --dataset-label')
    args = parser.parse_args(argv)
    if args.test_seeds is not None and args.dataset_label is None:
        parser.error('--test-seeds requires --dataset-label; canonical test seeds cannot be overridden')
    if args.dataset_label is not None:
        import re
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.dataset_label):
            parser.error('--dataset-label must be 1-64 ASCII letters, digits, underscores or hyphens, starting with a letter/digit')
    if args.test_seeds is not None and (len(set(args.test_seeds)) != len(args.test_seeds)
            or any(not 0 <= seed <= 2 ** 32 - 1 for seed in args.test_seeds)):
        parser.error('--test-seeds must be distinct uint32 integers in episode order')
    return args


def _loading_config(config, device):
    """Allow only device substitution, refusing SB3's implicit CPU fallback."""
    from dataclasses import replace
    import torch
    from stable_baselines3.common.utils import get_device
    configured = config.ppo.device if device is None else device
    resolved = get_device(configured)
    if configured != 'auto':
        requested = torch.device(configured)
        if requested.type != resolved.type:
            raise ValueError(f'Requested device {configured} is unavailable; CPU fallback refused')
        if requested.type in ('cuda', 'xpu'):
            backend = getattr(torch, requested.type, None)
            if backend is None or not backend.is_available() or (requested.index is not None and requested.index >= backend.device_count()):
                raise ValueError(f'Requested device {configured} is unavailable')
        if requested.type == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('Requested MPS device is unavailable')
    return replace(config, ppo=replace(config.ppo, device=configured))


def _preflight(root, config, plans, payload, fingerprint, retain_traces, alternate):
    """Validate the entire final set and existing records before any model load."""
    from agents.mo import storage
    target = payload['budget_plan']['effective_total']
    slots = []
    for i, plan in enumerate(plans):
        for j in range(config.n_weights):
            record, meta, _ = storage._committed_bundle(root / 'models' / f'seed_{i}' / f'w_{j}' / 'final', fingerprint)
            identity = record.identity
            if (meta['kind'] != 'final' or identity.actual_step != target
                    or (identity.run_index, identity.run_seed, identity.weight_index, list(identity.weight))
                    != (i, plan.run_seed, j, payload['weights']['values'][j])):
                raise storage.StorageError('Final bundle does not match saved seed/preference/final budget')
            journal = root / 'results/evaluations' / f'seed_{i}' / f'w_{j}' / 'RESUME.json'
            if journal.exists():
                saved = storage._read_json(journal)
                storage._validate_resume_plan(root, saved, payload, fingerprint, i, j)
                if saved['status'] != 'ready':
                    raise storage.StorageError('Resolve pending resume reconciliation before final testing')
            path = root / 'results/final' / f'seed_{i}' / f'w_{j}' / f'step_{target}.json'
            previous = None
            if not alternate and path.exists():
                previous, saved = storage._load_evaluation_record(path)
                expected, kind = storage._evaluation_location(root, payload, fingerprint, identity, previous)
                if path != expected or kind != 'final' or saved['kind'] != kind or saved['objective_scales'] != list(config.objective_scales):
                    raise storage.StorageError('Existing final evaluation provenance does not match')
                if retain_traces and not previous.traces:
                    raise ValueError('Existing canonical evaluation has no traces; use --dataset-label LABEL --retain-traces to preserve it')
            slots.append((record, previous))
    return slots


def _evaluate(args):
    """Evaluate new slots with owned bundles and commit canonical or labelled data."""
    import hashlib
    import numpy as np
    from scripts._ppo_mo_cli import saved_experiment
    from agents.mo import evaluation, storage
    from agents.mo.envs import make_raw_env
    root, config, plans, payload, fingerprint = saved_experiment(args.experiment)
    loading = _loading_config(config, args.device)
    seeds = tuple(args.test_seeds) if args.test_seeds is not None else plans[0].final_seeds
    alternate = args.dataset_label is not None
    destination = root / 'results/alternate_tests' / args.dataset_label if alternate else None
    if alternate:
        forbidden = {seed for plan in plans for seed in (*plan.worker_seeds, *plan.eval_seeds)}
        if forbidden.intersection(seeds):
            raise ValueError('Alternate test seeds must be disjoint from saved training and intermediate-evaluation seeds')
        if destination.exists():
            raise ValueError('Alternate dataset already exists; use a new --dataset-label (partial datasets are retained for inspection)')
    slots = _preflight(root, config, plans, payload, fingerprint, args.retain_traces, alternate)
    if alternate:
        destination.mkdir(parents=True, exist_ok=False)  # Reserve this dataset exclusively.
    records = []
    with evaluation.evaluation_state_guard():
        for record, previous in slots:
            if previous is not None:
                continue
            with storage.load_policy_bundle(record.directory, loading) as bundle:
                if bundle.identity != record.identity:
                    raise storage.StorageError('Final bundle identity changed after preflight')
                result = evaluation.evaluate_policy(bundle.model, lambda: make_raw_env(config.env),
                    seeds, loading, bundle.identity,
                    frozen_normalizer=evaluation.make_frozen_obs_normalizer(bundle.vec_normalize),
                    collect_traces=args.retain_traces)
            if not alternate:
                storage.save_evaluation_record(root, record.identity, result)
            else:
                identity = record.identity
                path = destination / f'seed_{identity.run_index}' / f'w_{identity.weight_index}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                body, arrays, digest = storage._encode_evaluation(result)
                numeric = path.with_suffix('.npz')
                storage.atomic_write_npz(numeric, arrays)
                storage.atomic_write_json(path, {'record_schema_version': 1, 'kind': 'alternate_test',
                    'objective_scales': config.objective_scales, 'result': body,
                    'content_sha256': digest, 'npz_sha256': hashlib.sha256(numeric.read_bytes()).hexdigest()})
                records.append({'path': path.relative_to(destination).as_posix(),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bundle_sha256': record.bundle_sha256, 'identity': storage._json_value(identity)})
    if alternate:
        storage.atomic_write_json(destination / 'dataset.json', {
            'schema_version': 1, 'kind': 'alternate_final_test_dataset', 'label': args.dataset_label,
            'experiment_fingerprint': fingerprint, 'test_seeds': seeds,
            'canonical_test_seeds': plans[0].final_seeds, 'objective_scales': config.objective_scales,
            'reference_scaled': config.reference_scaled, 'loading_device': loading.ppo.device,
            'retain_traces': args.retain_traces, 'weights': payload['weights']['values'],
            'policy_records': records, 'canonical_results_replaced': False})
        return {'status': 'alternate_evaluation_complete', 'dataset': str(destination / 'dataset.json')}
    # All records now exist. Reuse the established canonical aggregate logic;
    # it validates/reads them without loading any model with the original device.
    final = evaluation.evaluate_final_policy_set(root, config, plans)
    return {'status': 'final_evaluation_complete', 'experiment': str(root),
        'policies': int(np.count_nonzero(final.validity)), 'test_seeds': list(seeds),
        'results': str(root / 'results/final_results.json')}


def main(argv=None):
    """Return nonzero with an actionable error; bundle/env cleanup is scoped."""
    import json
    import sys
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        args = parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        print(json.dumps(_evaluate(args)))
        return 0
    except KeyboardInterrupt:
        print('Final evaluation interrupted; committed canonical slots can be reused. Use a new label for an incomplete alternate dataset.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'Cannot evaluate MO final policies: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
