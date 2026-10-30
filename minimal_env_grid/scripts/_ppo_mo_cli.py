"""Saved-experiment CLI helpers. Importing this module has no runtime effects."""
from pathlib import Path


# Params: value; returns: Path.
def path_argument(value):
    import argparse
    if not value.strip():
        raise argparse.ArgumentTypeError('path must be nonempty')
    return Path(value)


def saved_experiment(directory):
    """Validate initialization and reconstruct the exact saved scientific config."""
    from agents.mo import config as planning, storage
    from agents.mo.types import ExperimentConfig, SeedPlan
    root, payload, fingerprint = storage._evaluation_context(directory)
    values = dict(payload['config'])
    values['logging'] = storage._read_json(root / 'config.json')['logging']
    for name, record_type in planning._NESTED.items():
        values[name] = record_type(**values[name])
    config = ExperimentConfig(**values, output_dir=root)
    storage._evaluation_context(root, config)
    plans = tuple(SeedPlan(**p) for p in payload['seed_plans'])
    return root, config, plans, payload, fingerprint


def report_destination(root, destination):
    """Preflight an optional empty export root, keeping manifest-relative paths."""
    if destination is None:
        return None
    target = destination.expanduser().resolve()
    if target == root or root in target.parents or target in root.parents:
        raise ValueError('--report-dir must be outside the experiment and its ancestors')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('--report-dir export destination must be empty or absent')
    return target


def export_reports(root, manifest, destination):
    """Export only verified derived files; retain the reports/ manifest layout."""
    import hashlib
    import shutil
    if destination is None:
        return root / 'reports/report_manifest.json'
    # Recheck after report generation before creating any export files.
    report_destination(root, destination)
    for relative in manifest.relative_paths:
        source = (root / relative).resolve()
        source.relative_to(root)
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest.file_hashes[relative]:
            raise ValueError('Report changed before export: ' + relative)
    for relative in (*manifest.relative_paths, 'reports/report_manifest.json'):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    return destination / 'reports/report_manifest.json'
