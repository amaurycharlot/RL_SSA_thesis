"""Regenerate MO reports from committed records, without training or model loading."""
import argparse
from pathlib import Path


def parse_args(argv=None):
    """Accept only an experiment and an optional empty report export location."""
    from scripts._ppo_mo_cli import path_argument
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--experiment', type=path_argument, required=True)
    parser.add_argument('--report-dir', type=path_argument,
        help='Optional empty export root; writes reports/ beneath it, preserving manifest paths. Canonical derived reports are also regenerated.')
    return parser.parse_args(argv)


def main(argv=None):
    """Return 0 for a report, 2 for invalid/missing data, 130 on interruption."""
    import json
    import sys
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        args = parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        from scripts._ppo_mo_cli import saved_experiment, report_destination, export_reports
        from agents.mo import reporting
        root, config, _, _, _ = saved_experiment(args.experiment)
        target = report_destination(root, args.report_dir)
        results, final, _, _ = reporting._load_report_results(root, config)
        if not results.validity.any() and not final.validity.any():
            raise ValueError('No committed evaluations found; evaluate saved final policies or finish a training evaluation first')
        manifest = reporting.generate_reports(root, config)
        path = export_reports(root, manifest, target)
        print(json.dumps({'status': 'reports_complete', 'manifest': str(path),
            'source': 'committed canonical evaluations', 'training_started': False}))
        return 0
    except KeyboardInterrupt:
        print('Report generation interrupted; retry after writers finish.', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'Cannot generate MO reports: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
