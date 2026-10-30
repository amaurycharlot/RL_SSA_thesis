"""Block 31 numerical artist checks and standalone plot export regression tests."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import reporting, storage
from agents.mo.config import make_weights
from agents.mo.pareto import summarize_hv_across_seeds, hypervolume_2d_max, unique_front_indices
from agents.mo.types import FinalResults


# Params: valid; returns: FinalResults.
def final_fixture(valid=None):
    means = np.array([[[1, 5], [4, 2], [1, 5]], [[2, 1], [1, 1], [0, 0]]], np.float64)
    valid = np.ones((2, 3), bool) if valid is None else np.asarray(valid, bool)
    means[~valid] = np.nan
    raw = means * [2., 4.]
    episodes = np.repeat(means[:, :, None, :], 2, axis=2)
    fronts = tuple(tuple(int(j) for j in unique_front_indices(means[i], range(3))) if valid[i].all() else () for i in range(2))
    hv = np.asarray([hypervolume_2d_max(means[i], (0., 0.)) if valid[i].all() else np.nan for i in range(2)])
    return FinalResults(raw, means, episodes * [2., 4.], episodes, valid, (17, 29), fronts, hv)


class RewardPlotIntegration(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    # Params: self, callback, name; returns: captured.
    def capture(self, callback, name='plot.png'):
        captured = {}
        save = reporting._save_figure
        # Params: figure, path; returns: None.
        def record(figure, path):
            figure.canvas.draw()
            captured['axes'] = [
                {'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(), 'title': ax.get_title(),
                 'lines': [(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines],
                 'offsets': [np.asarray(c.get_offsets()).copy() for c in ax.collections],
                 'texts': [text.get_text() for text in ax.texts],
                 'paths': [[p.vertices.copy() for p in c.get_paths()] for c in ax.collections]}
                for ax in figure.axes]
            captured['figure_texts'] = [text.get_text() for text in figure.texts]
            save(figure, path)
        with patch.object(reporting, '_save_figure', side_effect=record):
            self.assertIsNone(callback(self.root / name))
        return captured

    # Params: self; returns: None.
    def test_hv_actual_axis_mean_sample_sd_and_incomplete_gap(self):
        steps = np.array([4, 8, 12, 20], np.int64)
        summary = summarize_hv_across_seeds(np.array([[2., 6., np.nan, 10.], [4., 10., np.nan, 14.]]), steps)
        data = self.capture(lambda p: reporting.plot_hypervolume_curve(summary, steps, p, reference=(-1., -2.)))
        upper, counts = data['axes']
        np.testing.assert_array_equal(upper['lines'][0][0], steps)
        np.testing.assert_allclose(upper['lines'][0][1], [3., 8., np.nan, 12.], equal_nan=True)
        np.testing.assert_array_equal(counts['lines'][0][1], [2, 2, 0, 2])
        band = upper['paths'][0]
        self.assertEqual(len(band), 2)
        self.assertTrue(all(not (path[:, 0].min() < 12 < path[:, 0].max()) for path in band))
        values = band[0][:, 1]
        self.assertAlmostEqual(values.min(), 3. - np.sqrt(2))
        self.assertAlmostEqual(values.max(), 8. + np.sqrt(8))
        self.assertIn('scaled reward', upper['ylabel'])
        self.assertIn('Actual training transitions per policy', counts['xlabel'])
        self.assertTrue(any('(-1.0, -2.0)' in text for text in upper['texts']))

    # Params: self; returns: None.
    def test_hv_all_missing_has_no_fabricated_curve_or_band(self):
        steps = [4, 8]
        summary = summarize_hv_across_seeds(np.full((2, 2), np.nan), steps)
        data = self.capture(lambda p: reporting.plot_hypervolume_curve(summary, steps, p))
        self.assertTrue(np.isnan(data['axes'][0]['lines'][0][1]).all())
        self.assertEqual(data['axes'][0]['paths'][0], [])
        self.assertTrue(any('No budget' in text for text in data['axes'][0]['texts']))

    # Params: self; returns: None.
    def test_hv_single_seed_sd_zero_and_nonuniform_budget_spacing_retained(self):
        steps = [4, 28, 100]
        summary = summarize_hv_across_seeds(np.array([[1., 3., 2.]]), steps)
        data = self.capture(lambda p: reporting.plot_hypervolume_curve(summary, steps, p))
        np.testing.assert_array_equal(data['axes'][0]['lines'][0][0], steps)
        np.testing.assert_array_equal(summary.sd, [0., 0., 0.])

    # Params: self; returns: None.
    def test_hv_bad_budget_axis_or_format_rejected_before_writing(self):
        summary = summarize_hv_across_seeds(np.array([[1., 3.]]), [4, 8])
        for steps in ([4, 9], [4., 8.], [8, 4]):
            with self.assertRaises(ValueError): reporting.plot_hypervolume_curve(summary, steps, self.root / 'invalid/plot.png')
        with self.assertRaises(ValueError): reporting.plot_hypervolume_curve(summary, [4, 8], self.root / 'invalid/plot.txt')
        self.assertFalse((self.root / 'invalid').exists())

    # Params: self; returns: None.
    def test_scaled_final_front_is_one_seed_with_stable_duplicate_weight_labels(self):
        final = final_fixture(); weights = make_weights(3)
        data = self.capture(lambda p: reporting.plot_final_front(final, 0, weights, (0., 0.), p))
        ax = data['axes'][0]
        np.testing.assert_array_equal(ax['offsets'][0], final.means_scaled[0])
        np.testing.assert_array_equal(ax['lines'][0][0], [1., 4.])
        np.testing.assert_array_equal(ax['lines'][0][1], [5., 2.])
        self.assertEqual(len(ax['texts']), 2)
        self.assertTrue(any('w0 =' in text and 'w2 =' in text for text in ax['texts']))
        self.assertIn('training seed index 0', ax['title'])
        self.assertIn('scaled reward space', ax['xlabel'])
        self.assertTrue(any('HV (scaled) = 11' in text for text in data['figure_texts']))
        second = self.capture(lambda p: reporting.plot_final_front(final, 1, weights, (0., 0.), p), 'second.png')
        np.testing.assert_array_equal(second['axes'][0]['offsets'][0], final.means_scaled[1])
        self.assertTrue(any('HV (scaled) = 2' in text for text in second['figure_texts']))

    # Params: self; returns: None.
    def test_raw_front_uses_raw_returns_reference_and_explicit_labels(self):
        final = final_fixture()
        data = self.capture(lambda p: reporting.plot_final_front(final, 0, make_weights(3), (0., 0.), p, space='raw'))
        ax = data['axes'][0]
        np.testing.assert_array_equal(ax['offsets'][0], final.means_raw[0])
        np.testing.assert_array_equal(ax['offsets'][1], [[0., 0.]])
        self.assertIn('raw reward space', ax['xlabel']); self.assertIn('raw reward space', ax['ylabel'])
        self.assertTrue(any('HV (raw) = 88' in text for text in data['figure_texts']))

    # Params: self; returns: None.
    def test_incomplete_front_keeps_available_points_without_main_front_or_hv(self):
        final = final_fixture([[True, False, True], [True, True, True]])
        data = self.capture(lambda p: reporting.plot_final_front(final, 0, make_weights(3), (0., 0.), p))
        self.assertEqual(len(data['axes'][0]['offsets'][0]), 2)
        self.assertEqual(data['axes'][0]['lines'], [])
        self.assertTrue(any('Incomplete set: 2/3' in text and 'unavailable' in text for text in data['figure_texts']))

    # Params: self; returns: None.
    def test_wholly_missing_front_remains_explicitly_incomplete(self):
        final = final_fixture([[False, False, False], [True, True, True]])
        data = self.capture(lambda p: reporting.plot_final_front(final, 0, make_weights(3), (0., 0.), p))
        self.assertEqual(data['axes'][0]['offsets'][0].shape, (0, 2))
        self.assertTrue(any('Incomplete set: 0/3' in text for text in data['figure_texts']))

    # Params: self; returns: None.
    def test_bad_seed_weights_reference_or_space_rejected(self):
        final = final_fixture(); weights = make_weights(3)
        for index in (-1, 2, True):
            with self.assertRaises(ValueError): reporting.plot_final_front(final, index, weights, (0., 0.), self.root / 'invalid/p.png')
        with self.assertRaises(ValueError): reporting.plot_final_front(final, 0, make_weights(2), (0., 0.), self.root / 'invalid/p.png')
        with self.assertRaises(ValueError): reporting.plot_final_front(final, 0, weights, (np.nan, 0.), self.root / 'invalid/p.png')
        with self.assertRaises(ValueError): reporting.plot_final_front(final, 0, weights, (0., 0.), self.root / 'invalid/p.png', space='other')
        self.assertFalse((self.root / 'invalid').exists())

    # Params: self; returns: None.
    def test_png_svg_pdf_exports_are_nonempty_and_leave_no_pyplot_figures(self):
        from matplotlib._pylab_helpers import Gcf
        before = list(Gcf.get_all_fig_managers())
        final = final_fixture()
        for suffix in ('png', 'svg', 'pdf'):
            path = self.root / ('front.' + suffix)
            reporting.plot_final_front(final, 0, make_weights(3), (0., 0.), path)
            self.assertGreater(path.stat().st_size, 1000)
            if suffix == 'png': self.assertEqual(path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            if suffix == 'pdf': self.assertTrue(path.read_bytes().startswith(b'%PDF-'))
            if suffix == 'svg': self.assertIn(b'<svg', path.read_bytes())
        self.assertEqual(list(Gcf.get_all_fig_managers()), before)

    # Params: self; returns: None.
    def test_failed_plot_replace_preserves_previous_image_and_cleans_temp(self):
        final = final_fixture(); path = self.root / 'front.png'
        reporting.plot_final_front(final, 0, make_weights(3), (0., 0.), path)
        original = path.read_bytes()
        with patch.object(storage.os, 'replace', side_effect=OSError('injected plot replace failure')):
            with self.assertRaisesRegex(OSError, 'plot replace failure'):
                reporting.plot_final_front(final, 1, make_weights(3), (0., 0.), path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [path])

    # Params: self; returns: None.
    def test_plotting_preserves_numeric_inputs_and_preferences(self):
        final = final_fixture(); weights = make_weights(3)
        raw, scaled, preferences = final.means_raw.copy(), final.means_scaled.copy(), weights.copy()
        reporting.plot_final_front(final, 0, weights, (0., 0.), self.root / 'front.png')
        np.testing.assert_array_equal(final.means_raw, raw)
        np.testing.assert_array_equal(final.means_scaled, scaled)
        np.testing.assert_array_equal(weights, preferences)

    # Params: self; returns: None.
    def test_eleven_preferences_keep_labels_clear_of_other_labels_and_markers(self):
        from matplotlib.transforms import Bbox
        x = np.linspace(1., 5., 11)
        points = np.stack([x, 7. - 2.2 * np.sqrt(x)], axis=1)
        points[1] = points[0]; points[6] = points[5]
        episodes = np.repeat(points[None, :, None, :], 2, axis=2)
        front = tuple(int(j) for j in unique_front_indices(points, range(11)))
        final = FinalResults(points[None], points[None], episodes, episodes, np.ones((1, 11), bool),
            (17, 29), (front,), np.array([hypervolume_2d_max(points, (0., 0.))]))
        # Params: figure, path; returns: None.
        def inspect(figure, path):
            figure.canvas.draw(); ax = figure.axes[0]; renderer = figure.canvas.get_renderer()
            annotations = [t for t in ax.texts if t.get_text().startswith('w')]
            boxes = [t.get_bbox_patch().get_window_extent(renderer) for t in annotations]
            markers = [Bbox.from_bounds(*(ax.transData.transform(point) - 7), 14, 14) for point in points]
            self.assertEqual(sum(len(t.get_text().splitlines()) for t in annotations), 11)
            self.assertFalse(any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:]))
            self.assertFalse(any(a.overlaps(b) for a in boxes for b in markers))
            self.assertFalse(any(a.overlaps(ax.get_legend().get_window_extent(renderer)) for a in boxes))
        with patch.object(reporting, '_save_figure', side_effect=inspect):
            reporting.plot_final_front(final, 0, make_weights(11), (0., 0.), self.root / 'front.png')


class OperationalReports(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from test_mo_reporting import SummaryTableExports
        self.fixture = SummaryTableExports()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.config = self.fixture.root, self.fixture.config

    # Params: self, on_hv; returns: owned.
    def fake_plots(self, on_hv=None):
        from contextlib import ExitStack
        owned = ExitStack()
        # Params: path; returns: None.
        def write(path):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'synthetic plot test')
        # Params: summary, steps, path, **kwargs; returns: None.
        def hv(summary, steps, path, **kwargs):
            write(path)
            if on_hv: on_hv()
        # Params: results, index, weights, reference, path, **kwargs; returns: None.
        def final(results, index, weights, reference, path, **kwargs): write(path)
        # Params: results, weights, folder, **kwargs; returns: None.
        def operational(results, weights, folder, **kwargs):
            for i in range(results.validity.shape[0]): write(Path(folder) / f'operational_seed_{i}.png')
        owned.enter_context(patch.object(reporting, 'plot_hypervolume_curve', side_effect=hv))
        owned.enter_context(patch.object(reporting, 'plot_final_front', side_effect=final))
        owned.enter_context(patch.object(reporting, 'plot_operational_metrics', side_effect=operational))
        return owned

    # Params: self, results, **kwargs; returns: captured.
    def capture_operational(self, results, **kwargs):
        captured = []
        # Params: figure, path; returns: None.
        def inspect(figure, path):
            captured.append({'title': figure._suptitle.get_text(), 'path': str(path),
                'axes': [{'title': ax.get_title(), 'texts': [t.get_text() for t in ax.texts],
                    'lines': [(line.get_xdata().copy(), line.get_ydata().copy()) for line in ax.lines],
                    'offsets': [np.asarray(c.get_offsets()).copy() for c in ax.collections],
                    'ticks': [t.get_text() for t in ax.get_xticklabels()]} for ax in figure.axes]})
        with patch.object(reporting, '_save_figure', side_effect=inspect):
            reporting.plot_operational_metrics(results, self.fixture.weights, self.root / 'physical', **kwargs)
        return captured

    # Params: self; returns: None.
    def test_physical_values_known_and_detected_labels_and_missing_search_time(self):
        self.fixture.record(budget=1)
        data = self.capture_operational(self.fixture.results())[0]
        axes = data['axes']
        self.assertIn('Initially known targets', axes[6]['title'])
        self.assertIn('Known + detected targets', axes[7]['title'])
        np.testing.assert_allclose(axes[6]['lines'][0][1], [np.nan, 0.], equal_nan=True)
        np.testing.assert_allclose(axes[7]['lines'][0][1], [np.nan, .125], equal_nan=True)
        np.testing.assert_allclose(axes[0]['lines'][0][1], [np.nan, 2.], equal_nan=True)
        np.testing.assert_allclose(axes[1]['lines'][0][1], [np.nan, 3.], equal_nan=True)
        self.assertEqual(axes[0]['offsets'][1].shape, (1, 2))  # Partial-episode availability ring.
        self.assertTrue(any('PPO search time is not plotted' in text for text in axes[8]['texts']))
        self.assertEqual(len(axes[8]['lines']), 0)
        self.assertEqual(axes[0]['ticks'], ['w1\n0', 'w0\n1'])

    # Params: self; returns: None.
    def test_intermediate_plot_defaults_to_final_planned_budget_without_fallback(self):
        self.fixture.record(budget=0); results = self.fixture.results()
        default = self.capture_operational(results)[0]
        self.assertIn('actual budget 4', default['title'])
        self.assertTrue(np.isnan(default['axes'][0]['lines'][0][1]).all())
        explicit = self.capture_operational(results, actual_budget=2)[0]
        self.assertTrue(np.isfinite(explicit['axes'][0]['lines'][0][1]).any())

    # Params: self; returns: None.
    def test_final_operational_plot_uses_final_episode_metrics(self):
        self.fixture.record(final=True)
        data = self.capture_operational(self.fixture.final(), actual_budget=4)
        self.assertEqual(len(data), 2)
        self.assertIn('Final test physical metrics', data[0]['title'])
        self.assertIn('training seed index 1', data[1]['title'])
        self.assertTrue(np.isnan(data[1]['axes'][0]['lines'][0][1]).all())

    # Params: self; returns: None.
    def test_bad_operational_budget_or_weights_rejected_without_output(self):
        results = self.fixture.results()
        for budget in (3, True):
            with self.assertRaises(ValueError): reporting.plot_operational_metrics(results, self.fixture.weights, self.root / 'invalid', actual_budget=budget)
        with self.assertRaises(ValueError): reporting.plot_operational_metrics(results, make_weights(3), self.root / 'invalid')
        self.assertFalse((self.root / 'invalid').exists())

    # Params: self; returns: None.
    def test_complete_report_manifest_paths_hashes_budgets_and_counts(self):
        import hashlib, json
        from agents.mo.types import ReportManifest
        self.fixture.populate()
        with self.fake_plots(): manifest = reporting.generate_reports(self.root, self.config)
        self.assertIsInstance(manifest, ReportManifest)
        self.assertEqual(manifest.represented_budgets, (2, 4)); self.assertEqual(manifest.seed_counts, (2, 2))
        self.assertEqual(len(manifest.relative_paths), 18)
        for name in manifest.relative_paths:
            self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), manifest.file_hashes[name])
        saved = json.loads((self.root / 'reports/report_manifest.json').read_text())
        self.assertEqual(ReportManifest(**saved), manifest)
        inputs = json.loads((self.root / 'reports/report_inputs.json').read_text())
        self.assertEqual(inputs['final_complete_seed_count'], 2)
        self.assertEqual(inputs['objective_scales'], [2., 4.]); self.assertEqual(inputs['reference_scaled'], [-1., -1.])
        self.assertIn('Seed index 0, training seed 7', (self.root / 'reports/report.md').read_text(encoding='utf-8'))

    # Params: self; returns: None.
    def test_missing_final_results_remain_incomplete_without_new_evaluation(self):
        self.fixture.record()
        with self.fake_plots(), patch.object(storage, 'load_policy_bundle', side_effect=AssertionError('no policy loading')):
            manifest = reporting.generate_reports(self.root, self.config)
        self.assertEqual(manifest.seed_counts, (0, 0))
        import json
        inputs = json.loads((self.root / 'reports/report_inputs.json').read_text())
        self.assertEqual(inputs['final_complete_seed_count'], 0)
        self.assertIn('0/2 final evaluations', (self.root / 'reports/report.md').read_text(encoding='utf-8'))

    # Params: self; returns: None.
    def test_report_ignores_corrupt_aggregate_caches_and_preserves_all_sources(self):
        self.fixture.populate()
        for name in ('eval_returns_raw.npy', 'hypervolume.npy', 'final_results.npz', 'final_results.json'):
            (self.root / 'results' / name).write_bytes(b'ignored disposable cache corruption')
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file() and 'reports' not in p.parts}
        with self.fake_plots(), \
             patch.object(storage, 'load_evaluation_records', side_effect=AssertionError('do not rebuild caches')), \
             patch.object(storage, 'load_policy_bundle', side_effect=AssertionError('do not load policies')):
            manifest = reporting.generate_reports(self.root, self.config)
        self.assertEqual(manifest.seed_counts, (2, 2))
        for p, content in before.items(): self.assertEqual(p.read_bytes(), content)

    # Params: self; returns: None.
    def test_corrupt_source_record_rejected_before_report_writes(self):
        self.fixture.record()
        (self.root / 'results/evaluations/seed_0/w_0/step_2.npz').write_bytes(b'corrupt authoritative record')
        with self.assertRaises(storage.StorageError): reporting.generate_reports(self.root, self.config)
        self.assertFalse((self.root / 'reports/tables').exists())

    # Params: self; returns: None.
    def test_pending_resume_reconciliation_blocks_report_publication(self):
        storage.resolve_resume_state(self.root, self.config, 0, 0)
        path = self.root / 'results/evaluations/seed_0/w_0/RESUME.json'
        state = storage._read_json(path); state['status'] = 'pending'; storage.atomic_write_json(path, state)
        with self.assertRaisesRegex(storage.StorageError, 'pending'): reporting.generate_reports(self.root, self.config)
        self.assertFalse((self.root / 'reports/report_manifest.json').exists())

    # Params: self; returns: None.
    def test_changing_inputs_during_plotting_prevents_manifest_commit(self):
        self.fixture.record()
        with self.fake_plots(on_hv=lambda: self.fixture.record(run=1, weight=1)):
            with self.assertRaisesRegex(storage.StorageError, 'inputs changed'): reporting.generate_reports(self.root, self.config)
        self.assertFalse((self.root / 'reports/report_manifest.json').exists())

    # Params: self; returns: None.
    def test_changed_science_rejected_and_logging_destination_is_irrelevant(self):
        from dataclasses import replace
        from agents.mo.types import LoggingConfig
        with self.assertRaises(storage.StorageError):
            reporting.generate_reports(self.root, replace(self.config, error_target_set='initial'))
        changed = replace(self.config, output_dir=self.root / 'unused', logging=LoggingConfig(mode='offline', project='local-only'))
        with self.fake_plots(): manifest = reporting.generate_reports(self.root, changed)
        self.assertEqual(manifest.experiment_fingerprint, self.fixture.fingerprint)
        self.assertFalse((self.root / 'unused').exists())

    # Params: self; returns: None.
    def test_manifest_write_failure_preserves_previous_manifest(self):
        self.fixture.record()
        with self.fake_plots(): reporting.generate_reports(self.root, self.config)
        path = self.root / 'reports/report_manifest.json'; before = path.read_bytes()
        write = reporting.atomic_write_json
        # Params: path, payload; returns: write result.
        def fail(path, payload):
            if Path(path).name == 'report_manifest.json': raise OSError('injected manifest failure')
            return write(path, payload)
        with self.fake_plots(), patch.object(reporting, 'atomic_write_json', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'manifest failure'): reporting.generate_reports(self.root, self.config)
        self.assertEqual(path.read_bytes(), before)

    # Params: self; returns: None.
    def test_standalone_regeneration_imports_no_learning_or_environment_libraries(self):
        import os, subprocess
        code = '''
import sys
from pathlib import Path
from agents.mo import storage
from agents.mo.reporting import generate_reports
from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
from agents.mo.types import ExperimentConfig, PPOConfig
c=ExperimentConfig(output_dir=Path('empty-report'), run_seeds=(7,), n_weights=2, eval_episodes=1, final_episodes=1,
reference_scaled=(-1.,-1.),reference_status='frozen',requested_timesteps_per_policy=2,
ppo=PPOConfig(n_envs=1,n_steps=2,batch_size=2,n_epochs=1))
root=storage.create_experiment_dir(c,make_weights(2),make_seed_plan(c,0),make_budget_plan(c))
manifest=generate_reports(root,c)
assert manifest.seed_counts==(0,)
assert not any(n.split('.')[0] in ('torch','gymnasium','stable_baselines3') for n in sys.modules)
assert not list((root/'results').glob('*.npy'))
'''
        subprocess.run([sys.executable, '-B', '-c', code], cwd=self.fixture.temporary.name,
            env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)), check=True, capture_output=True, text=True)


from dataclasses import replace
import json
import warnings
from types import SimpleNamespace
from agents.mo.types import ExperimentConfig, LoggingConfig, PolicyIdentity


class FakeExternalRun:
    # Params: self, log_error, finish_error; returns: None.
    def __init__(self, *, log_error=None, finish_error=None):
        self.records = []
        self.finishes = []
        self.log_error = log_error
        self.finish_error = finish_error
        self.on_log = None

    # Params: self, data; returns: None.
    def log(self, data):
        if self.on_log:
            self.on_log(data)
        if self.log_error:
            raise self.log_error
        self.records.append(data)

    # Params: self, **kwargs; returns: None.
    def finish(self, **kwargs):
        self.finishes.append(kwargs)
        if self.finish_error:
            raise self.finish_error


class FakeExternalClient:
    # Params: self, run, error; returns: None.
    def __init__(self, run=None, error=None):
        self.run = run or FakeExternalRun()
        self.error = error
        self.calls = []

    # Params: self, **kwargs; returns: self.run.
    def init(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.run


class OptionalExternalLogging(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    # Params: self, mode; returns: replace result.
    def config(self, mode='offline'):
        return replace(ExperimentConfig(), logging=LoggingConfig(mode=mode,
            project='owner-selected-project', entity='owner-selected-account'))

    # Params: self, root; returns: list.
    def events(self, root=None):
        root = self.root if root is None else Path(root)
        return [json.loads(p.read_text()) for p in sorted((root / 'logs/events').glob('*.json'))]

    # Params: self; returns: None.
    def test_off_does_not_import_or_initialize_optional_package(self):
        client = FakeExternalClient(error=AssertionError('must not initialize'))
        with patch('importlib.import_module', side_effect=AssertionError('must not import')):
            with reporting.make_logger(ExperimentConfig(), self.root, external_client=client) as logger:
                self.assertIs(type(logger), reporting.LocalLogger)
                logger.record('local-only', None, {'value': 0})
        self.assertEqual(client.calls, [])
        self.assertFalse((self.root / 'logs/external').exists())
        self.assertEqual(self.events()[0]['event'], 'local-only')

    # Params: self; returns: None.
    def test_modes_and_owner_settings_forwarded_without_environment_defaults(self):
        import os
        for mode in ('offline', 'online'):
            root = self.root / mode
            client = FakeExternalClient()
            with patch.dict(os.environ, {'WANDB_PROJECT': 'unrelated', 'WANDB_ENTITY': 'unrelated', 'WANDB_MODE': 'online'}):
                with reporting.make_logger(self.config(mode), root, external_client=client) as logger:
                    self.assertIsInstance(logger, reporting.CompositeLogger)
                    logger.record('test', {'actual_step': 4}, {'mean_scaled': np.array([1., 2.])})
                    logger.record('test', {'actual_step': 2}, {'mean_scaled': np.array([3., 4.])})
                    logger.flush()
            args = client.calls[0]
            self.assertEqual(args['project'], 'owner-selected-project')
            self.assertEqual(args['entity'], 'owner-selected-account')
            self.assertEqual(args['mode'], mode)
            self.assertEqual(Path(args['dir']), (root / 'logs/external').absolute())
            self.assertEqual(args['reinit'], 'create_new')
            self.assertEqual([r['identity']['actual_step'] for r in client.run.records], [4, 2])
            self.assertEqual(client.run.finishes, [{'exit_code': 0}])

    # Params: self; returns: None.
    def test_local_commit_precedes_remote_and_remote_receives_owned_primitive_copies(self):
        run = FakeExternalRun()
        source = np.array([1., np.nan])
        identity = PolicyIdentity('fingerprint', 7, 0, 1, (1., 0.), 4, 'bundle')
        # Params: data; returns: None.
        def check(data):
            local = self.events()
            self.assertEqual(len(local), 1)
            self.assertEqual(local[0]['payload'], {'values': [1., None]})
            self.assertEqual(data['identity']['bundle_id'], 'bundle')
            data['payload']['values'][0] = 99
        run.on_log = check
        with reporting.make_logger(self.config(), self.root, external_client=FakeExternalClient(run)) as logger:
            logger.record('policy_evaluation', identity, {'values': source})
        np.testing.assert_array_equal(source, [1., np.nan])
        self.assertEqual(self.events()[0]['payload'], {'values': [1., None]})

    # Params: self; returns: None.
    def test_remote_log_failure_preserves_local_and_disables_adapter_once(self):
        run = FakeExternalRun(log_error=OSError('credential-containing remote error'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with reporting.make_logger(self.config('online'), self.root, external_client=FakeExternalClient(run)) as logger:
                logger.record('first', None, {'valid': 1})
                self.assertIsNone(logger.external)
                logger.record('second', None, {'valid': 2})
                logger.flush()
        events = self.events()
        self.assertEqual([e['event'] for e in events], ['first', 'external_logging_warning', 'second'])
        self.assertEqual(events[1]['payload']['operation'], 'record')
        self.assertNotIn('credential-containing', json.dumps(events))
        self.assertEqual(len(caught), 1)
        self.assertEqual(run.finishes, [{'exit_code': 1}])
        self.assertFalse(list(self.root.rglob('COMPLETE.json')))

    # Params: self; returns: None.
    def test_missing_package_and_init_failure_fall_back_without_network(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with patch('importlib.import_module', side_effect=ModuleNotFoundError('wandb unavailable')) as importer:
                with reporting.make_logger(self.config(), self.root / 'missing') as logger:
                    logger.record('kept', None, {})
            importer.assert_called_once_with('wandb')
            client = FakeExternalClient(error=OSError('network unavailable'))
            with reporting.make_logger(self.config('online'), self.root / 'init', external_client=client) as logger:
                logger.record('kept', None, {})
        for name in ('missing', 'init'):
            records = self.events(self.root / name)
            self.assertEqual([e['event'] for e in records], ['external_logging_warning', 'kept'])
            self.assertEqual(records[0]['payload']['operation'], 'initialization')

    # Params: self; returns: None.
    def test_finish_failure_does_not_lose_local_records_and_close_is_idempotent(self):
        run = FakeExternalRun(finish_error=OSError('finish unavailable'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            logger = reporting.make_logger(self.config(), self.root, external_client=FakeExternalClient(run))
            logger.record('retained', None, {})
            logger.close()
            logger.close()
            logger.flush()
        self.assertTrue(logger._closed)
        self.assertTrue(logger.local._closed)
        self.assertEqual(run.finishes, [{'exit_code': 0}])
        self.assertEqual([e['event'] for e in self.events()], ['retained', 'external_logging_warning'])
        with self.assertRaises(RuntimeError):
            logger.record('late', None, {})

    # Params: self; returns: None.
    def test_log_and_cleanup_failures_are_both_persisted(self):
        run = FakeExternalRun(log_error=RuntimeError('remote log'), finish_error=OSError('remote cleanup'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with reporting.make_logger(self.config(), self.root, external_client=FakeExternalClient(run)) as logger:
                logger.record('kept', None, {})
                logger.record('still-kept', None, {})
        self.assertEqual([e['payload']['operation'] for e in self.events() if e['event'] == 'external_logging_warning'],
                         ['record', 'close'])
        self.assertEqual(len(run.finishes), 1)

    # Params: self; returns: None.
    def test_local_failure_is_propagated_without_remote_event_or_false_success(self):
        run = FakeExternalRun()
        logger = reporting.make_logger(self.config(), self.root, external_client=FakeExternalClient(run))
        with patch.object(logger.local, 'record', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                logger.record('not-committed', None, {})
        self.assertEqual(run.records, [])
        logger.close()
        self.assertEqual(run.finishes, [{'exit_code': 1}])

    # Params: self; returns: None.
    def test_warnings_as_errors_do_not_turn_remote_failure_into_training_failure(self):
        client = FakeExternalClient(error=RuntimeError('init failed'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with reporting.make_logger(self.config(), self.root, external_client=client) as logger:
                logger.record('kept', None, {})
        self.assertEqual([e['event'] for e in self.events()], ['external_logging_warning', 'kept'])

    # Params: self; returns: None.
    def test_flush_failure_disables_only_external_and_preserves_journal(self):
        calls = []
        class External:
            # Params: self, *args; returns: None.
            def record(self, *args):
                calls.append('record')
            # Params: self; returns: never (raises).
            def flush(self):
                raise OSError('flush failed')
            # Params: self, **kwargs; returns: None.
            def close(self, **kwargs):
                calls.append('close')
        local = reporting.LocalLogger(self.root / 'logs/events')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with reporting.CompositeLogger(local, External()) as logger:
                logger.record('first', None, {})
                logger.flush()
                logger.record('second', None, {})
        self.assertEqual(calls, ['record', 'close'])
        self.assertEqual([e['event'] for e in self.events()], ['first', 'external_logging_warning', 'second'])

    # Params: self; returns: None.
    def test_missing_owner_or_bad_mode_fails_before_allocating(self):
        for settings in (LoggingConfig(mode='offline', project='owner-only'),
                         LoggingConfig(mode='online', entity='owner-only')):
            with self.assertRaisesRegex(ValueError, 'project and entity'):
                reporting.make_logger(replace(ExperimentConfig(), logging=settings), self.root / 'invalid')
        config = self.config()
        object.__setattr__(config.logging, 'mode', 'unsupported')
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            reporting.make_logger(config, self.root / 'invalid')
        self.assertEqual(list(self.root.iterdir()), [])

    # Params: self; returns: None.
    def test_failed_experiment_context_and_failure_event_finish_as_failed(self):
        for event in ('experiment_failure', 'policy_failure'):
            client = FakeExternalClient()
            with reporting.make_logger(self.config(), self.root / event, external_client=client) as logger:
                logger.record(event, None, {'phase': 'evaluation'})
            self.assertEqual(client.run.finishes, [{'exit_code': 1}])
        client = FakeExternalClient()
        with self.assertRaisesRegex(ValueError, 'caller failure'):
            with reporting.make_logger(self.config(), self.root / 'context', external_client=client):
                raise ValueError('caller failure')
        self.assertEqual(client.run.finishes, [{'exit_code': 1}])

    # Params: self; returns: None.
    def test_invalid_run_initialization_closes_partial_run_and_continues_locally(self):
        finishes = []
        run = SimpleNamespace(finish=lambda **kwargs: finishes.append(kwargs))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with reporting.make_logger(self.config(), self.root, external_client=FakeExternalClient(run)) as logger:
                logger.record('kept', None, {})
        self.assertEqual(finishes, [{'exit_code': 1}])
        self.assertEqual([e['event'] for e in self.events()], ['external_logging_warning', 'kept'])
        class CancelledRun:
            # Params: self; returns: never (raises).
            @property
            def log(self):
                raise KeyboardInterrupt('cancelled during adapter construction')
            # Params: self, **kwargs; returns: None.
            def finish(self, **kwargs):
                finishes.append(kwargs)
        locals_created = []
        factory = reporting.LocalLogger
        # Params: directory; returns: logger.
        def local(directory):
            logger = factory(directory)
            locals_created.append(logger)
            return logger
        with patch.object(reporting, 'LocalLogger', side_effect=local):
            with self.assertRaises(KeyboardInterrupt):
                reporting.make_logger(self.config(), self.root / 'cancelled', external_client=FakeExternalClient(CancelledRun()))
        self.assertEqual(finishes, [{'exit_code': 1}, {'exit_code': 1}])
        self.assertTrue(locals_created[0]._closed)

    # Params: self; returns: None.
    def test_default_and_offline_import_do_not_need_optional_package_or_network(self):
        import os
        import subprocess
        code = '''
import socket, sys
from pathlib import Path
from agents.mo.reporting import make_logger
from agents.mo.types import ExperimentConfig, LoggingConfig
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('no network'))
with make_logger(ExperimentConfig(), Path('local')) as logger:
    logger.record('works', None, {})
assert 'wandb' not in sys.modules
assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'stable_baselines3', 'matplotlib'))
import random
import numpy as np
from types import SimpleNamespace
def consume(*a, **k):
    random.random()
    np.random.random()
def initialize(**kwargs):
    assert kwargs['mode'] == 'offline'
    consume()
    return SimpleNamespace(log=consume, finish=consume)
before_python, before_numpy = random.getstate(), np.random.get_state()
c = ExperimentConfig(logging=LoggingConfig(mode='offline', project='owner-project', entity='owner-account'))
with make_logger(c, Path('offline'), external_client=SimpleNamespace(init=initialize)) as logger:
    logger.record('offline-works', None, {})
assert random.getstate() == before_python
np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
assert np.random.get_state()[2:] == before_numpy[2:]
assert 'wandb' not in sys.modules
assert not any(n in sys.modules for n in ('torch', 'gymnasium', 'stable_baselines3', 'matplotlib'))
'''
        subprocess.run([sys.executable, '-B', '-c', code], cwd=self.root,
            env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)), check=True, capture_output=True, text=True)

    # Params: self; returns: None.
    def test_initialization_warning_disk_failure_closes_local_and_partial_remote(self):
        finishes = []
        run = SimpleNamespace(finish=lambda **kwargs: finishes.append(kwargs))
        locals_created = []
        factory = reporting.LocalLogger
        # Params: directory; returns: logger.
        def local(directory):
            logger = factory(directory)
            logger.record = lambda *args: (_ for _ in ()).throw(OSError('journal full'))
            locals_created.append(logger)
            return logger
        with patch.object(reporting, 'LocalLogger', side_effect=local):
            with self.assertRaisesRegex(OSError, 'journal full'):
                reporting.make_logger(self.config(), self.root, external_client=FakeExternalClient(run))
        self.assertEqual(finishes, [{'exit_code': 1}])
        self.assertTrue(locals_created[0]._closed)

    # Params: self; returns: None.
    def test_external_calls_restore_learner_rng_on_success_and_failure(self):
        import random
        import torch
        # Params: none; returns: None.
        def consume():
            random.random()
            np.random.random()
            torch.rand(1)
        class Run:
            # Params: self, data; returns: None.
            def log(self, data):
                consume()
                if data['event'] == 'failure':
                    raise OSError('failed after consuming RNG')
            # Params: self, **kwargs; returns: None.
            def finish(self, **kwargs):
                consume()
        class Client:
            # Params: self, **kwargs; returns: Run.
            def init(self, **kwargs):
                consume()
                return Run()
        before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with reporting.make_logger(self.config(), self.root, external_client=Client()) as logger:
                logger.record('success', None, {})
                logger.flush()
                logger.record('failure', None, {})
        self.assertEqual(before_python, random.getstate())
        current = np.random.get_state()
        self.assertEqual(before_numpy[0], current[0])
        np.testing.assert_array_equal(before_numpy[1], current[1])
        self.assertEqual(before_numpy[2:], current[2:])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))


class SmallCompleteExperiment(unittest.TestCase):
    """Block 36: one real CLI experiment, validated through its full artifact set.

    All reduced parameters are local to this fixture. Four independent learners
    each execute three updates, using two actual spawn workers per learner.
    """

    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        import hashlib
        import json
        import os
        import subprocess
        import torch
        from agents.mo.types import (ExperimentConfig, EnvConfig, PPOConfig,
                                     ExecutionConfig, NormalizationConfig)
        cls.previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        torch.set_num_threads(1)  # Test runner only; restored after direct reloads.
        cls.temporary = tempfile.TemporaryDirectory(prefix='mo_block36_')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.workspace = Path(cls.temporary.name).resolve()
        cls.root = cls.workspace / 'experiment'
        cls.production_path = PROJECT_ROOT / 'configs/ppo_mo_r6_r2.json'
        cls.production_before = cls.production_path.read_bytes()
        cls.config = ExperimentConfig(experiment_id='block36_synthetic_only', output_dir=cls.root,
            run_seeds=(7, 9), n_weights=2, eval_episodes=2, final_episodes=2,
            objective_scales=(2., 4.), scale_status='frozen',
            reference_scaled=(-10., -10.), reference_status='frozen',
            requested_timesteps_per_policy=9, eval_interval=4, checkpoint_interval=4,
            env=EnvConfig(known_targets=1, unknown_targets=2, grid_shape=(2, 3),
                          episode_steps=4, slew_alpha=0.),
            ppo=PPOConfig(n_envs=2, n_steps=2, batch_size=4, n_epochs=1, device='cpu', verbose=0),
            execution=ExecutionConfig(subproc=True, worker_threads=1),
            normalization=NormalizationConfig(norm_obs=True, norm_reward=True))
        raw = storage._json_value(cls.config)
        raw.pop('output_dir')
        cls.config_path = cls.workspace / 'synthetic_config.json'
        cls.config_path.write_text(json.dumps(raw), encoding='utf-8')
        cls.environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT),
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')

        # Observe real CLI-owned workers while the owning process is still alive.
        # The probe changes no factory, solver, reward, action or learning setting.
        cls.probe_path = cls.workspace / 'observe_cli_cleanup.py'
        cls.cleanup_path = cls.workspace / 'worker_cleanup.json'
        cls.probe_path.write_text('''
import multiprocessing as mp

def main():
    import json, pathlib, runpy, sys
    from unittest.mock import patch
    from agents.mo import envs
    destination = pathlib.Path(sys.argv[1])
    sys.argv = ['train_ppo_mo', *sys.argv[2:]]
    groups = []
    original = envs.TrainingSubprocVecEnv.__init__
    def observe(self, *args, **kwargs):
        original(self, *args, **kwargs)
        groups.append(self)
    code = 1
    try:
        with patch.object(envs.TrainingSubprocVecEnv, '__init__', observe):
            try:
                runpy.run_module('agents.train.train_ppo_mo', run_name='__main__')
            except SystemExit as error:
                code = int(error.code)
    finally:
        destination.write_text(json.dumps({
            'exit_code': code,
            'groups': [{'closed': group.closed,
                'workers': [{'pid': p.pid, 'alive': p.is_alive(), 'exitcode': p.exitcode}
                            for p in group.processes],
                'pipes_closed': all(p.closed for p in (*group.remotes, *group.work_remotes))}
                for group in groups],
            'active_children': [p.pid for p in mp.active_children()]}, indent=2), encoding='utf-8')
    raise SystemExit(code)

if __name__ == '__main__':
    mp.freeze_support()
    main()
''', encoding='utf-8')
        cls.commands = []
        # Params: arguments; returns: CLI result dict.
        def execute(arguments):
            result = subprocess.run([sys.executable, '-B', *arguments], cwd=cls.workspace,
                env=cls.environment, capture_output=True, text=True, timeout=240)
            cls.commands.append({'arguments': arguments, 'returncode': result.returncode,
                                 'stdout': result.stdout, 'stderr': result.stderr})
            if result.returncode != 0:
                raise AssertionError('Block 36 CLI failed:\n' + result.stdout + result.stderr)
            return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')][-1]
        cls.train_status = execute([str(cls.probe_path), str(cls.cleanup_path), '--config',
                                   str(cls.config_path), '--output', str(cls.root)])
        cls.cleanup = storage._read_json(cls.cleanup_path)
        cls.immutable_before = cls._immutable_hashes()
        cls.eval_status = execute(['-m', 'scripts.evaluate_ppo_mo', '--experiment', str(cls.root),
            '--dataset-label', 'block36_fresh_reload', '--retain-traces', '--device', 'cpu'])
        cls.canonical_status = execute(['-m', 'scripts.evaluate_ppo_mo', '--experiment', str(cls.root)])
        cls.plot_status = execute(['-m', 'scripts.plot_ppo_mo_results', '--experiment', str(cls.root)])
        if cls._immutable_hashes() != cls.immutable_before:
            raise AssertionError('Evaluation/report CLIs changed immutable training/evaluation artifacts')
        cls.payload = storage._read_json(cls.root / 'manifest.json')['identity']
        cls.fingerprint = storage._read_json(cls.root / 'manifest.json')['fingerprint']
        cls.weights = np.load(cls.root / 'weights.npy', allow_pickle=False)
        cls.budgets = tuple(cls.payload['budget_plan']['eval_steps'])
        cls.intermediate, cls.final, _, _ = reporting._load_report_results(cls.root, cls.config)
        cls.evidence = {'kind': 'block36_small_complete_experiment',
            'requested_training_transitions': 36, 'effective_training_transitions': 48,
            'training_seeds': list(cls.config.run_seeds), 'preferences': cls.weights.tolist(),
            'actual_budgets': list(cls.budgets), 'worker_cleanup': cls.cleanup,
            'commands': cls.commands, 'production_config_sha256': hashlib.sha256(cls.production_before).hexdigest()}

    @classmethod
    def tearDownClass(cls):
        """Optionally retain diagnostic evidence, never the temporary run itself."""
        import json
        import os
        # A validation runner may request a distinct evidence file per run.
        location = os.environ.get('MO_BLOCK36_EVIDENCE')
        if location and hasattr(cls, 'evidence'):
            Path(location).write_text(json.dumps(cls.evidence, indent=2), encoding='utf-8')

    # Params: cls; returns: dict.
    @classmethod
    def _immutable_hashes(cls):
        import hashlib
        paths = [cls.root / name for name in
                 ('config.json', 'manifest.json', 'weights.npy', 'seed_plan.json', 'budget_plan.json')]
        for folder in ('models', 'results/evaluations', 'results/final', 'logs'):
            paths.extend(p for p in (cls.root / folder).rglob('*') if p.is_file())
        return {p.relative_to(cls.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    @staticmethod
    def _two_rectangle_hv(points, reference):
        """Independent inclusion/exclusion oracle, with no Pareto/HV helper call."""
        spans = np.maximum(np.asarray(points, np.float64) - reference, 0.)
        return float(spans[0, 0] * spans[0, 1] + spans[1, 0] * spans[1, 1]
                     - min(spans[:, 0]) * min(spans[:, 1]))

    # Params: self; returns: None.
    def test_complete_cli_plans_identities_directories_and_worker_cleanup(self):
        from agents.mo.config import make_budget_plan, make_seed_plan
        self.assertEqual(self.production_path.read_bytes(), self.production_before)
        self.assertNotEqual(self.config_path, self.production_path)
        self.assertEqual(self.train_status['status'], 'complete')
        self.assertEqual(self.train_status['completed_policies'], 4)
        self.assertEqual(self.train_status['total_policies'], 4)
        self.assertEqual(self.train_status['effective_training_transitions'], 48)
        self.assertEqual(self.train_status['additional_training_transitions'], 48)
        self.assertEqual(self.canonical_status['policies'], 4)
        self.assertEqual(self.plot_status['status'], 'reports_complete')
        self.assertFalse(self.plot_status['training_started'])
        self.assertEqual(storage._json_value(make_budget_plan(self.config)), self.payload['budget_plan'])
        self.assertEqual(self.budgets, (4, 8, 12))
        self.assertEqual(self.payload['budget_plan']['requested_total'], 9)
        for i in range(2):
            self.assertEqual(storage._json_value(make_seed_plan(self.config, i)), self.payload['seed_plans'][i])
        self.assertEqual(self.weights.dtype, np.float32)
        np.testing.assert_array_equal(self.weights, [[0., 1.], [1., 0.]])
        for folder in ('models', 'logs', 'results/evaluations', 'results/final', 'reports/tables', 'reports/plots'):
            self.assertTrue((self.root / folder).is_dir(), folder)
        self.assertEqual(self.cleanup['exit_code'], 0)
        self.assertEqual(self.cleanup['active_children'], [])
        self.assertEqual(len(self.cleanup['groups']), 4)
        for group in self.cleanup['groups']:
            self.assertTrue(group['closed']); self.assertTrue(group['pipes_closed'])
            self.assertEqual(len(group['workers']), 2)
            for worker in group['workers']:
                self.assertIsInstance(worker['pid'], int)
                self.assertFalse(worker['alive']); self.assertEqual(worker['exitcode'], 0)
        self.assertFalse(any(p.name.endswith(('.tmp', '.LOCK')) or p.name == '.INITIALIZING'
                             for p in self.root.rglob('*')))

    # Params: self; returns: None.
    def test_all_bundle_members_completions_and_committed_evaluation_slots(self):
        import hashlib
        from agents.mo.types import PolicyIdentity
        expected = set()
        for i in range(2):
            plan = self.payload['seed_plans'][i]
            self.assertFalse(set(plan['worker_seeds']) & (set(plan['eval_seeds']) | set(plan['final_seeds'])))
            self.assertFalse(set(plan['eval_seeds']) & set(plan['final_seeds']))
            for j in range(2):
                policy = self.root / 'models' / f'seed_{i}' / f'w_{j}'
                for step in self.budgets:
                    record, metadata, _ = storage._committed_bundle(policy / 'checkpoints' / f'step_{step}', self.fingerprint)
                    self.assertEqual(metadata['kind'], 'checkpoint')
                    self.assertEqual(record.identity.actual_step, step)
                    for name, digest in metadata['member_hashes'].items():
                        self.assertEqual(hashlib.sha256((record.directory / name).read_bytes()).hexdigest(), digest)
                final, metadata, _ = storage._committed_bundle(policy / 'final', self.fingerprint)
                self.assertEqual(metadata['kind'], 'final')
                self.assertEqual(final.actual_step, 12)
                self.assertIn('vecnormalize.pkl', metadata['member_hashes'])
                state = storage.resolve_resume_state(self.root, self.config, i, j)
                self.assertTrue(state.complete); self.assertEqual(state.current_step, 12)
                complete = storage._read_json(policy / 'COMPLETE.json')
                self.assertEqual(PolicyIdentity(**complete['identity']), final.identity)
                for namespace, steps, seeds in (('evaluations', self.budgets, plan['eval_seeds']),
                                               ('final', (12,), plan['final_seeds'])):
                    for step in steps:
                        path = self.root / 'results' / namespace / f'seed_{i}' / f'w_{j}' / f'step_{step}.json'
                        expected.add(path)
                        result, meta = storage._load_evaluation_record(path)
                        self.assertEqual(meta['kind'], namespace)
                        self.assertEqual(result.episode_seeds, tuple(seeds))
                        self.assertEqual(result.identity.experiment_fingerprint, self.fingerprint)
                        self.assertEqual((result.identity.run_index, result.identity.run_seed,
                                          result.identity.weight_index, result.identity.actual_step), (i, plan['run_seed'], j, step))
                        self.assertEqual(result.identity.weight, tuple(self.weights[j]))
                        source = policy / ('final' if namespace == 'final' else f'checkpoints/step_{step}')
                        self.assertEqual(storage._committed_bundle(source)[0].identity, result.identity)
                        np.testing.assert_array_equal(result.returns_scaled, result.returns_raw / self.config.objective_scales)
                        for suffix in ('raw', 'scaled'):
                            returns = getattr(result, 'returns_' + suffix)
                            self.assertEqual(returns.shape, (2, 2)); self.assertEqual(returns.dtype, np.float64)
                            self.assertTrue(np.isfinite(returns).all())
                            np.testing.assert_array_equal(getattr(result, 'mean_' + suffix), returns.mean(axis=0))
                            np.testing.assert_array_equal(getattr(result, 'sd_' + suffix), returns.std(axis=0, ddof=1))
                            np.testing.assert_array_equal(getattr(result, 'se_' + suffix), returns.std(axis=0, ddof=1) / np.sqrt(2))
                        self.assertEqual(len(result.episode_metrics), 2)
                        if namespace == 'evaluations':
                            key = (i, self.budgets.index(step), j)
                            np.testing.assert_array_equal(self.intermediate.means_raw[key], result.mean_raw)
                            np.testing.assert_array_equal(self.intermediate.episode_returns[key], result.returns_raw)
                        else:
                            np.testing.assert_array_equal(self.final.means_raw[i, j], result.mean_raw)
                            np.testing.assert_array_equal(self.final.episode_returns_raw[i, j], result.returns_raw)
        observed = {p for namespace in ('evaluations', 'final')
                    for p in (self.root / 'results' / namespace).glob('seed_*/w_*/step_*.json')}
        self.assertEqual(observed, expected); self.assertEqual(len(observed), 16)
        self.assertEqual(len(list((self.root / 'models').rglob('COMMITTED'))), 16)

    # Params: self; returns: None.
    def test_all_cached_arrays_and_hypervolumes_match_independent_geometry(self):
        import hashlib
        raw = np.load(self.root / 'results/eval_returns_raw.npy', allow_pickle=False)
        scaled = np.load(self.root / 'results/eval_returns_scaled.npy', allow_pickle=False)
        valid = np.load(self.root / 'results/eval_valid.npy', allow_pickle=False)
        steps = np.load(self.root / 'results/actual_steps.npy', allow_pickle=False)
        hv = np.load(self.root / 'results/hypervolume.npy', allow_pickle=False)
        self.assertEqual(raw.shape, (2, 3, 2, 2)); self.assertEqual(hv.shape, (2, 3))
        np.testing.assert_array_equal(steps, self.budgets)
        np.testing.assert_array_equal(raw, self.intermediate.means_raw)
        np.testing.assert_array_equal(scaled, self.intermediate.means_scaled)
        np.testing.assert_array_equal(scaled, raw / self.config.objective_scales)
        self.assertTrue(valid.all()); np.testing.assert_array_equal(valid, self.intermediate.validity)
        independent = np.asarray([[self._two_rectangle_hv(scaled[i, k], self.config.reference_scaled)
                                   for k in range(3)] for i in range(2)])
        np.testing.assert_allclose(hv, independent, rtol=1e-12, atol=1e-12)
        self.assertTrue((independent > 0).all())
        meta = storage._read_json(self.root / 'results/final_results.json')
        numeric = self.root / 'results/final_results.npz'
        self.assertEqual(hashlib.sha256(numeric.read_bytes()).hexdigest(), meta['numeric_sha256'])
        with np.load(numeric, allow_pickle=False) as final:
            self.assertEqual(set(final.files), {'means_raw', 'means_scaled', 'episode_returns_raw',
                'episode_returns_scaled', 'validity', 'test_seeds', 'front_mask', 'hypervolume'})
            self.assertEqual(final['episode_returns_raw'].shape, (2, 2, 2, 2))
            np.testing.assert_array_equal(final['means_raw'], final['episode_returns_raw'].mean(axis=2))
            np.testing.assert_array_equal(final['means_scaled'], final['means_raw'] / self.config.objective_scales)
            np.testing.assert_array_equal(final['episode_returns_scaled'], final['episode_returns_raw'] / self.config.objective_scales)
            np.testing.assert_array_equal(final['test_seeds'], self.payload['seed_plans'][0]['final_seeds'])
            self.assertTrue(final['validity'].all())
            for i in range(2):
                expected_hv = self._two_rectangle_hv(final['means_scaled'][i], self.config.reference_scaled)
                self.assertAlmostEqual(final['hypervolume'][i], expected_hv, places=10)
                self.assertEqual(tuple(np.flatnonzero(final['front_mask'][i])), tuple(meta['front_indices'][i]))
                points = final['means_scaled'][i]
                expected_front = []
                for j in range(2):
                    dominated = any(np.all(points[k] >= points[j]) and np.any(points[k] > points[j])
                                    for k in range(2) if k != j)
                    duplicate = any(np.array_equal(points[k], points[j]) for k in range(j))
                    expected_front.append(not dominated and not duplicate)
                np.testing.assert_array_equal(final['front_mask'][i], expected_front)
                np.testing.assert_array_equal(final['means_raw'][i], self.final.means_raw[i])
        self.assertEqual(len(meta['policy_records']), 4)
        for item in meta['policy_records']:
            path = self.root / item['evaluation_record']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['evaluation_sha256'])
        self.evidence['independent_intermediate_hypervolume'] = independent.tolist()
        self.evidence['final_hypervolume'] = self.final.hypervolume.tolist()

    # Params: self; returns: None.
    def test_fresh_reload_of_every_checkpoint_and_final_policy_agrees(self):
        from agents.mo.evaluation import evaluate_policy, make_frozen_obs_normalizer
        from agents.mo.envs import make_raw_env
        reloaded = 0
        for i in range(2):
            for j in range(2):
                for namespace, steps in (('evaluations', self.budgets), ('final', (12,))):
                    for step in steps:
                        path = self.root / 'results' / namespace / f'seed_{i}' / f'w_{j}' / f'step_{step}.json'
                        original, _ = storage._load_evaluation_record(path)
                        policy = self.root / 'models' / f'seed_{i}' / f'w_{j}'
                        source = policy / ('final' if namespace == 'final' else f'checkpoints/step_{step}')
                        with storage.load_policy_bundle(source, self.config) as bundle:
                            self.assertFalse(bundle.vec_normalize.training)
                            self.assertFalse(bundle.vec_normalize.norm_reward)
                            rms = (bundle.vec_normalize.obs_rms.mean.copy(), bundle.vec_normalize.obs_rms.var.copy(),
                                   bundle.vec_normalize.obs_rms.count)
                            result = evaluate_policy(bundle.model, lambda: make_raw_env(self.config.env),
                                original.episode_seeds, self.config, bundle.identity,
                                frozen_normalizer=make_frozen_obs_normalizer(bundle.vec_normalize))
                            np.testing.assert_array_equal(bundle.vec_normalize.obs_rms.mean, rms[0])
                            np.testing.assert_array_equal(bundle.vec_normalize.obs_rms.var, rms[1])
                            self.assertEqual(bundle.vec_normalize.obs_rms.count, rms[2])
                        self.assertTrue(bundle.closed)
                        self.assertEqual(result.identity, original.identity)
                        np.testing.assert_array_equal(result.returns_raw, original.returns_raw)
                        np.testing.assert_array_equal(result.returns_scaled, original.returns_scaled)
                        self.assertEqual(result.episode_metrics, original.episode_metrics)
                        reloaded += 1
        self.assertEqual(reloaded, 16)
        self.assertEqual(self._immutable_hashes(), self.immutable_before)
        self.evidence['fresh_bundle_reload_evaluations_matched'] = reloaded

    # Params: self; returns: None.
    def test_standalone_evaluation_traces_reconstruct_discounted_returns(self):
        import hashlib
        folder = self.root / 'results/alternate_tests/block36_fresh_reload'
        dataset = storage._read_json(folder / 'dataset.json')
        self.assertEqual(dataset['test_seeds'], self.payload['seed_plans'][0]['final_seeds'])
        self.assertEqual(dataset['canonical_test_seeds'], dataset['test_seeds'])
        self.assertFalse(dataset['canonical_results_replaced'])
        self.assertEqual(len(dataset['policy_records']), 4)
        for record in dataset['policy_records']:
            path = folder / record['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record['sha256'])
            result, meta = storage._load_evaluation_record(path)
            i, j = result.identity.run_index, result.identity.weight_index
            original, _ = storage._load_evaluation_record(self.root / 'results/final' / f'seed_{i}' / f'w_{j}/step_12.json')
            np.testing.assert_array_equal(result.returns_raw, original.returns_raw)
            self.assertEqual(result.episode_metrics, original.episode_metrics)
            self.assertEqual(meta['kind'], 'alternate_test')
            self.assertEqual(len(result.traces), 2)
            for index, trace in enumerate(result.traces):
                discounts = self.config.ppo.gamma ** np.arange(len(trace.vector_rewards))
                expected = (discounts[:, None] * trace.vector_rewards.astype(np.float64)).sum(axis=0)
                np.testing.assert_allclose(result.returns_raw[index], expected, rtol=1e-12, atol=1e-12)
                self.assertTrue(trace.terminated[-1] or trace.truncated[-1])
                self.assertEqual(trace.observations.dtype, np.float32)
                self.assertEqual(trace.vector_rewards.dtype, np.float32)
                self.assertEqual(trace.true_states.shape[1], 3)
        self.evidence['cli_fresh_final_evaluations_matched'] = 4

    # Params: self; returns: None.
    def test_every_report_table_plot_manifest_and_local_event_is_valid(self):
        import csv
        import hashlib
        from PIL import Image
        manifest = storage._read_json(self.root / 'reports/report_manifest.json')
        self.assertEqual(manifest['experiment_fingerprint'], self.fingerprint)
        self.assertEqual(manifest['represented_budgets'], [4, 8, 12])
        self.assertEqual(manifest['seed_counts'], [2, 2, 2])
        self.assertEqual(set(manifest['relative_paths']), set(manifest['file_hashes']))
        for name, digest in manifest['file_hashes'].items():
            path = self.root / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            if path.suffix == '.png':
                with Image.open(path) as image:
                    self.assertGreater(image.width, 100); self.assertGreater(image.height, 100)
                    image.verify()
        inputs = storage._read_json(self.root / 'reports/report_inputs.json')
        for name, digest in inputs['input_hashes'].items():
            self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest)
        tables = storage._read_json(self.root / 'reports/tables/table_metadata.json')
        self.assertEqual(len(tables['evaluation_records']), 16)
        records = {}
        # Params: row; returns: records[key].
        def source_record(row):
            key = (row['phase'], int(row['run_index']), int(row['weight_index']), int(row['actual_budget']))
            if key not in records:
                phase, i, j, step = key
                folder = 'evaluations' if phase == 'intermediate' else 'final'
                records[key] = storage._load_evaluation_record(
                    self.root / 'results' / folder / f'seed_{i}' / f'w_{j}' / f'step_{step}.json')[0]
            return records[key]
        all_rows = {}
        for name, description in tables['tables'].items():
            path = self.root / 'reports/tables' / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), description['sha256'])
            with path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream); rows = list(reader)
                self.assertEqual(reader.fieldnames, description['columns'])
            self.assertEqual(len(rows), description['rows']); self.assertTrue(rows)
            for row in rows:
                self.assertEqual(row['experiment_fingerprint'], self.fingerprint)
                if row.get('run_index', '') != '':
                    self.assertEqual(int(row['run_seed']), self.config.run_seeds[int(row['run_index'])])
                if row.get('weight_index', '') != '':
                    self.assertIn(int(row['weight_index']), (0, 1))
                    record = source_record(row)
                    self.assertEqual(row['bundle_id'], record.identity.bundle_id)
                    self.assertEqual(int(row['actual_step']), record.actual_step)
                    self.assertEqual(float(row['weight_tracking']), record.identity.weight[0])
                    self.assertEqual(float(row['weight_search']), record.identity.weight[1])
                    if 'episode_index' in row:
                        self.assertEqual(int(row['episode_seed']), record.episode_seeds[int(row['episode_index'])])
            all_rows[name] = rows
        self.assertEqual(len(all_rows['policy_results.csv']), 16)
        self.assertEqual(len(all_rows['episode_returns.csv']), 32)
        self.assertEqual(len(all_rows['target_observation_metrics.csv']), 96)
        for row in all_rows['policy_results.csv']:
            self.assertIn(row['valid'].lower(), ('true', '1'))
            self.assertEqual(int(row['completed_episode_count']), 2)
            record = source_record(row)
            for space in ('raw', 'scaled'):
                for stat in ('mean', 'sd', 'se'):
                    for index, objective in enumerate(('tracking', 'search')):
                        self.assertEqual(float(row[f'{objective}_{stat}_{space}']), getattr(record, f'{stat}_{space}')[index])
        for row in all_rows['episode_returns.csv']:
            record = source_record(row)
            for space in ('raw', 'scaled'):
                for index, objective in enumerate(('tracking', 'search')):
                    self.assertEqual(float(row[f'{objective}_return_{space}']),
                                     getattr(record, f'returns_{space}')[int(row['episode_index']), index])
        for row in all_rows['episode_metrics.csv']:
            metric = source_record(row).episode_metrics[int(row['episode_index'])]
            value = getattr(metric, row['metric'])
            if value is None:
                self.assertEqual(row['value'], '')
                self.assertEqual(row['unavailable_reason'], metric.unavailable_reasons[row['metric']])
            else:
                self.assertEqual(float(row['value']), value)
        for row in all_rows['target_observation_metrics.csv']:
            metric = source_record(row).episode_metrics[int(row['episode_index'])]
            target = next(t for t in metric.target_observation_metrics if t.target_id == int(row['target_id']))
            for name in ('observation_count', 'counting_start_timestep', 'eligible_timesteps', 'observation_frequency'):
                expected = getattr(target, name)
                self.assertEqual(row[name], '' if expected is None else str(expected))
        for row in all_rows['operational_summary.csv']:
            metrics = source_record(row).episode_metrics
            values = [getattr(metric, row['metric']) for metric in metrics if getattr(metric, row['metric']) is not None]
            self.assertEqual(int(row['available_episode_count']), len(values))
            self.assertEqual(int(row['missing_episode_count']), 2 - len(values))
            if values:
                self.assertEqual(float(row['mean_available_episodes']), np.mean(values))
                self.assertEqual(float(row['sd_available_episodes']), np.std(values, ddof=1) if len(values) > 1 else 0.)
            else:
                self.assertEqual(row['mean_available_episodes'], '')
        for row in all_rows['hypervolume.csv']:
            step = int(row['actual_budget'])
            values = self.final.means_scaled if row['phase'] == 'final' else self.intermediate.means_scaled[:, self.budgets.index(step)]
            expected = np.asarray([self._two_rectangle_hv(p, self.config.reference_scaled) for p in values])
            if row['row_kind'] == 'training_seed':
                self.assertAlmostEqual(float(row['hypervolume']), expected[int(row['run_index'])], places=10)
            else:
                self.assertAlmostEqual(float(row['mean_hypervolume']), expected.mean(), places=10)
                self.assertAlmostEqual(float(row['sd_hypervolume']), expected.std(ddof=1), places=10)
            self.assertEqual(int(row['complete_seed_count']), 2)
        events = [storage._read_json(path) for path in sorted((self.root / 'logs').rglob('*.json'))]
        self.assertTrue(events)
        self.assertEqual([event['event_index'] for event in events], list(range(len(events))))
        names = [event['event'] for event in events]
        self.assertEqual(names.count('policy_start'), 4)
        self.assertEqual(names.count('policy_evaluation'), 12)
        self.assertEqual(names.count('policy_artifacts_committed'), 4)
        self.assertEqual(names.count('experiment_training_complete'), 1)
        self.assertEqual(names.count('experiment_final_testing_complete'), 1)
        self.assertNotIn('experiment_failure', names); self.assertNotIn('policy_failure', names)
        self.assertEqual(self.production_path.read_bytes(), self.production_before)
        self.evidence['validated_report_files'] = len(manifest['relative_paths'])
        self.evidence['validated_table_rows'] = {name: len(rows) for name, rows in all_rows.items()}


class InterruptionRecoveryCLI(unittest.TestCase):
    """Block 37: real Windows CLI faults followed by checkpoint continuation.

    Each test owns one synthetic two-preference experiment. No post-interrupt
    learning trajectory is compared bit for bit with uninterrupted training.
    """

    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        import torch
        cls.previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        torch.set_num_threads(1)
        cls.evidence = []

    # Params: cls; returns: None.
    @classmethod
    def tearDownClass(cls):
        import json
        import os
        destination = os.environ.get('MO_BLOCK37_EVIDENCE')
        if destination:
            Path(destination).write_text(json.dumps(cls.evidence, indent=2), encoding='utf-8')

    # Params: self; returns: None.
    def setUp(self):
        import json
        import os
        from agents.mo.types import (ExperimentConfig, EnvConfig, PPOConfig,
                                     ExecutionConfig, NormalizationConfig)
        self.temporary = tempfile.TemporaryDirectory(prefix='mo_block37_')
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.root = self.workspace / 'experiment'
        self.config = ExperimentConfig(experiment_id='block37_synthetic_only', output_dir=self.root,
            run_seeds=(7,), n_weights=2, eval_episodes=1, final_episodes=1,
            objective_scales=(2., 4.), scale_status='frozen',
            reference_scaled=(-10., -10.), reference_status='frozen',
            requested_timesteps_per_policy=5, eval_interval=2, checkpoint_interval=2,
            env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3),
                          episode_steps=2, slew_alpha=0.),
            ppo=PPOConfig(n_envs=2, n_steps=1, batch_size=2, n_epochs=1, device='cpu', verbose=0),
            execution=ExecutionConfig(subproc=True, worker_threads=1),
            normalization=NormalizationConfig(norm_obs=True, norm_reward=True))
        self.production = PROJECT_ROOT / 'configs/ppo_mo_r6_r2.json'
        self.production_bytes = self.production.read_bytes()
        self.addCleanup(lambda: self.assertEqual(self.production.read_bytes(), self.production_bytes))
        raw = storage._json_value(self.config); raw.pop('output_dir')
        self.config_path = self.workspace / 'synthetic_config.json'
        self.config_path.write_text(json.dumps(raw), encoding='utf-8')
        self.environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT), PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
        self.probe = self.workspace / 'recover_cli.py'
        self.probe.write_text(self.probe_source(), encoding='utf-8')
        self.calls = []

    # Params: none; returns: str.
    @staticmethod
    def probe_source():
        return '''
import multiprocessing as mp

def main():
    import json, pathlib, runpy, sys
    from contextlib import ExitStack
    from unittest.mock import patch
    from multiprocessing.process import BaseProcess
    from stable_baselines3 import PPO
    from agents.mo import envs, evaluation, reporting, storage, trainer
    mode, destination = sys.argv[1], pathlib.Path(sys.argv[2])
    sys.argv = ['train_ppo_mo', *sys.argv[3:]]
    groups, loggers, timeline, loaded = [], [], [], []
    original_init = envs.TrainingSubprocVecEnv.__init__
    original_start = BaseProcess.start
    original_learn = trainer.learn_one_update
    original_evaluate = evaluation.evaluate_policy
    original_load = storage.load_policy_bundle
    original_logger = reporting.make_logger
    original_json = storage.atomic_write_json
    original_save = PPO.save
    original_atomic = storage._atomic_write
    starts = 0
    def observe_group(self, *args, **kwargs):
        groups.append(self)
        original_init(self, *args, **kwargs)
    def start(self, *args, **kwargs):
        nonlocal starts
        starts += 1
        if mode == 'startup' and starts == 2:
            raise OSError('block37 injected second-worker startup failure')
        return original_start(self, *args, **kwargs)
    def learn(model, *args, **kwargs):
        timeline.append({'event': 'learn', 'step': model.num_timesteps, 'updates': model._n_updates})
        if mode == 'resume_stop':
            raise OSError('block37 injected stop after reconciliation')
        result = original_learn(model, *args, **kwargs)
        timeline.append({'event': 'learn_complete', 'step': model.num_timesteps, 'updates': model._n_updates})
        return result
    def evaluate(*args, **kwargs):
        identity = args[4]
        timeline.append({'event': 'evaluate', 'step': identity.actual_step,
                         'weight_index': identity.weight_index, 'episode_seeds': list(args[2])})
        return original_evaluate(*args, **kwargs)
    def load(*args, **kwargs):
        bundle = original_load(*args, **kwargs)
        if len(args) >= 3 and args[2] is not None:
            loaded.append({'step': bundle.model.num_timesteps, 'updates': bundle.model._n_updates,
                'weight_index': bundle.identity.weight_index,
                'optimizer_steps': sorted(set(int(v['step']) for v in bundle.model.policy.optimizer.state.values() if 'step' in v)),
                'observation_rms_count': float(bundle.vec_normalize.obs_rms.count)})
        return bundle
    def logger(*args, **kwargs):
        result = original_logger(*args, **kwargs); loggers.append(result); return result
    def write_json(path, data):
        path = pathlib.Path(path)
        if mode == 'eval_write' and path.as_posix().endswith('results/evaluations/seed_0/w_0/step_4.json'):
            raise KeyboardInterrupt('block37 injected evaluation marker interruption')
        return original_json(path, data)
    def save(model, path, *args, **kwargs):
        if mode == 'bundle_write' and model.num_timesteps == 4:
            path.write(b'block37 partial model archive')
            raise OSError('block37 injected partial model write')
        return original_save(model, path, *args, **kwargs)
    def atomic(path, writer):
        if mode == 'report' and pathlib.Path(path).as_posix().endswith('reports/plots/hypervolume.png'):
            def partial(stream):
                stream.write(b'block37 partial plot')
                raise OSError('block37 injected report write')
            return original_atomic(path, partial)
        return original_atomic(path, writer)
    def final_copy(*args, **kwargs):
        raise OSError('block37 injected final publication interruption')
    code = 1
    try:
        with ExitStack() as stack:
            for target, name, replacement in (
                (envs.TrainingSubprocVecEnv, '__init__', observe_group),
                (BaseProcess, 'start', start), (trainer, 'learn_one_update', learn),
                (evaluation, 'evaluate_policy', evaluate), (storage, 'load_policy_bundle', load),
                (reporting, 'make_logger', logger), (storage, 'atomic_write_json', write_json),
                (PPO, 'save', save), (storage, '_atomic_write', atomic)):
                stack.enter_context(patch.object(target, name, replacement))
            if mode == 'older':
                stack.enter_context(patch.object(trainer, '_final_bundle_from_checkpoint', final_copy))
            try:
                runpy.run_module('agents.train.train_ppo_mo', run_name='__main__')
            except SystemExit as error:
                code = int(error.code)
    finally:
        destination.write_text(json.dumps({'mode': mode, 'exit_code': code, 'timeline': timeline,
            'training_bundle_loads': loaded, 'loggers_closed': [logger._closed for logger in loggers],
            'groups': [{'closed': getattr(group, 'closed', False),
                'pipes_closed': all(p.closed for p in (*getattr(group, 'remotes', ()), *getattr(group, 'work_remotes', ()))),
                'workers': [{'pid': p.pid, 'alive': p.is_alive(), 'exitcode': p.exitcode}
                            for p in getattr(group, 'processes', ())]} for group in groups],
            'active_children': [p.pid for p in mp.active_children()]}, indent=2), encoding='utf-8')
    raise SystemExit(code)

if __name__ == '__main__':
    mp.freeze_support()
    main()
'''

    # Params: self, mode, resume; returns: probe.
    def cli(self, mode, resume=False):
        import json
        import subprocess
        destination = self.workspace / f'probe_{len(self.calls)}_{mode}.json'
        arguments = ['--resume', str(self.root)] if resume else ['--config', str(self.config_path), '--output', str(self.root)]
        result = subprocess.run([sys.executable, '-B', str(self.probe), mode, str(destination), *arguments],
            cwd=self.workspace, env=self.environment, capture_output=True, text=True, timeout=240)
        self.assertTrue(destination.is_file(), result.stdout + result.stderr)
        probe = storage._read_json(destination)
        probe.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        self.calls.append(probe)
        self.assertEqual(probe['exit_code'], result.returncode)
        self.assertEqual(probe['active_children'], [])
        self.assertTrue(all(probe['loggers_closed']))
        for group in probe['groups']:
            self.assertTrue(group['closed']); self.assertTrue(group['pipes_closed'])
            for worker in group['workers']:
                self.assertFalse(worker['alive'])
                self.assertIsNotNone(worker['exitcode'])
                if mode != 'startup': self.assertEqual(worker['exitcode'], 0)
        documents = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        probe['status_documents'] = documents
        return probe

    # Params: paths; returns: dict.
    @staticmethod
    def snapshot(paths):
        return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths if path.is_file()}

    # Params: self, before; returns: None.
    def assert_preserved(self, before):
        for path, (content, modified) in before.items():
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(path.read_bytes(), content, str(path))
            self.assertEqual(path.stat().st_mtime_ns, modified, str(path))

    # Params: self, expected_step; returns: None.
    def assert_incomplete(self, expected_step):
        self.assertNotEqual(self.calls[-1]['returncode'], 0)
        self.assertFalse(any(doc.get('status') == 'complete' for doc in self.calls[-1]['status_documents']))
        self.assertFalse(list((self.root / 'models').rglob('COMPLETE.json')))
        self.assertFalse((self.root / 'reports/report_manifest.json').exists())
        latest = storage.discover_latest_checkpoint(self.root / 'models/seed_0/w_0',
            storage._read_json(self.root / 'manifest.json')['fingerprint'])
        self.assertEqual(None if latest is None else latest.actual_step, expected_step)

    # Params: self, expected_additional, expected_restore; returns: probe.
    def finish(self, expected_additional, expected_restore=None):
        probe = self.cli('none', resume=True)
        self.assertEqual(probe['returncode'], 0, probe['stdout'] + probe['stderr'])
        summary = probe['status_documents'][-1]
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['completed_policies'], 2)
        self.assertEqual(summary['additional_training_transitions'], expected_additional)
        if expected_restore is not None:
            load = probe['training_bundle_loads'][0]
            self.assertEqual(load['step'], expected_restore)
            self.assertEqual(load['updates'], expected_restore // 2)
            self.assertEqual(load['optimizer_steps'], [expected_restore // 2])
            self.assertGreater(load['observation_rms_count'], 0.)
        results, final, _, _ = reporting._load_report_results(self.root, self.config)
        self.assertTrue(results.validity.all()); self.assertTrue(final.validity.all())
        self.assertEqual(list(results.actual_budgets), [2, 4, 6])
        self.assertTrue(np.isfinite(results.hypervolume).all())
        self.assertTrue(np.isfinite(final.hypervolume).all())
        self.assertEqual(len(list((self.root / 'models').rglob('COMPLETE.json'))), 2)
        self.assertTrue((self.root / 'reports/report_manifest.json').is_file())
        self.assertFalse(any(path.name.endswith(('.tmp', '.LOCK')) for path in self.root.rglob('*')))
        self.assertEqual(self.production.read_bytes(), self.production_bytes)
        self.evidence.append({'case': self.id().split('.')[-1], 'calls': self.calls,
                              'expected_additional_training_transitions': expected_additional})
        return probe

    # Params: self; returns: None.
    def test_partial_worker_startup_releases_first_child_and_resumes_from_zero(self):
        probe = self.cli('startup')
        self.assertEqual(probe['returncode'], 1)
        self.assertIn('second-worker startup', probe['stderr'])
        self.assertEqual(len(probe['groups']), 1)
        self.assertEqual(len(probe['groups'][0]['workers']), 1)
        self.assert_incomplete(None)
        self.finish(12)

    # Params: self; returns: None.
    def test_evaluation_marker_interrupt_ignores_orphan_and_repairs_before_learning(self):
        probe = self.cli('eval_write')
        self.assertEqual(probe['returncode'], 130)
        self.assert_incomplete(4)
        directory = self.root / 'results/evaluations/seed_0/w_0'
        self.assertTrue((directory / 'step_4.npz').exists())
        self.assertFalse((directory / 'step_4.json').exists())
        results = storage.load_evaluation_records(self.root, self.config)
        np.testing.assert_array_equal(results.validity[0, :, 0], [True, False, False])
        early = self.snapshot(list((self.root / 'models/seed_0/w_0/checkpoints').rglob('*'))
                              + [directory / 'step_2.json', directory / 'step_2.npz'])
        resumed = self.finish(8, expected_restore=4)
        events = resumed['timeline']
        self.assertEqual((events[0]['event'], events[0]['step']), ('evaluate', 4))
        first_learn = next(event for event in events if event['event'] == 'learn')
        self.assertEqual(first_learn['step'], 4)
        self.assert_preserved(early)

    # Params: self; returns: None.
    def test_partial_model_write_keeps_previous_checkpoint_and_resumes_remaining_budget(self):
        probe = self.cli('bundle_write')
        self.assertEqual(probe['returncode'], 1)
        self.assertIn('partial model write', probe['stderr'])
        self.assert_incomplete(2)
        policy = self.root / 'models/seed_0/w_0'
        self.assertFalse((policy / 'checkpoints/step_4').exists())
        self.assertFalse(list(policy.glob('checkpoints/*.tmp')))
        early = self.snapshot(list((policy / 'checkpoints/step_2').rglob('*'))
                              + list((self.root / 'results/evaluations/seed_0/w_0').glob('step_2.*')))
        resumed = self.finish(10, expected_restore=2)
        self.assertEqual((resumed['timeline'][0]['event'], resumed['timeline'][0]['step']), ('learn', 2))
        self.assert_preserved(early)

    # Params: self; returns: None.
    def test_report_write_failure_has_no_manifest_and_resume_adds_no_training(self):
        probe = self.cli('report')
        self.assertEqual(probe['returncode'], 1)
        self.assertIn('report generation failed', probe['stderr'])
        self.assertFalse(any(doc.get('status') == 'complete' for doc in probe['status_documents']))
        self.assertEqual(len(list((self.root / 'models').rglob('COMPLETE.json'))), 2)
        self.assertFalse((self.root / 'reports/report_manifest.json').exists())
        self.assertFalse((self.root / 'reports/plots/hypervolume.png').exists())
        self.assertFalse(list((self.root / 'reports').rglob('*.tmp')))
        immutable = self.snapshot([path for folder in ('models', 'results/evaluations', 'results/final')
                                   for path in (self.root / folder).rglob('*')])
        resumed = self.finish(0)
        self.assertEqual(resumed['groups'], [])
        self.assertEqual(resumed['timeline'], [])
        self.assertEqual(resumed['training_bundle_loads'], [])
        self.assert_preserved(immutable)

    # Params: self; returns: None.
    def test_older_valid_checkpoint_quarantines_later_records_before_continuing(self):
        from agents.mo.evaluation import evaluate_policy, make_frozen_obs_normalizer
        from agents.mo.envs import make_raw_env
        probe = self.cli('older')
        self.assertEqual(probe['returncode'], 1)
        self.assert_incomplete(6)
        policy = self.root / 'models/seed_0/w_0'
        directory = self.root / 'results/evaluations/seed_0/w_0'
        plans = storage._read_json(self.root / 'seed_plan.json')
        with storage.load_policy_bundle(policy / 'checkpoints/step_6', self.config) as bundle:
            result = evaluate_policy(bundle.model, lambda: make_raw_env(self.config.env),
                tuple(plans[0]['final_seeds']), self.config, bundle.identity,
                frozen_normalizer=make_frozen_obs_normalizer(bundle.vec_normalize))
        storage.save_evaluation_record(self.root, result.identity, result)
        early = self.snapshot(list((policy / 'checkpoints/step_2').rglob('*'))
                              + [directory / 'step_2.json', directory / 'step_2.npz'])
        future = self.snapshot([directory / f'step_{step}.{extension}' for step in (4, 6) for extension in ('json', 'npz')]
                              + list((self.root / 'results/final/seed_0/w_0').glob('step_6.*')))
        # Corrupt only this owned temporary experiment: retain unusable later
        # bundles so reconciliation must quarantine them, rather than deleting.
        for step in (4, 6):
            path = (policy / f'checkpoints/step_{step}/model.zip').resolve()
            path.relative_to(self.workspace)
            path.write_bytes(b'block37 damaged later model')
        (policy / 'final').mkdir()
        (policy / 'final/partial.zip').write_bytes(b'block37 interrupted final publication')
        self.assertEqual(storage.discover_latest_checkpoint(policy,
            storage._read_json(self.root / 'manifest.json')['fingerprint']).actual_step, 2)
        stopped = self.cli('resume_stop', resume=True)
        self.assertEqual(stopped['returncode'], 1)
        self.assertEqual(stopped['training_bundle_loads'][0]['step'], 2)
        self.assertFalse(any(event['event'] == 'learn_complete' for event in stopped['timeline']))
        self.assert_incomplete(2)
        plan = storage._read_json(directory / 'RESUME.json')
        self.assertEqual(plan['status'], 'ready')
        self.assertEqual(plan['state']['current_step'], 2)
        self.assertIn('not exact', plan['continuation'])
        self.assertTrue(plan['state']['superseded_attempt_ids'])
        moves = {item['source']: item for item in plan['moves']}
        for path, (content, _) in future.items():
            self.assertFalse(path.exists())
            move = moves[path.relative_to(self.root).as_posix()]
            self.assertEqual((self.root / move['destination']).read_bytes(), content)
        for step in (4, 6):
            self.assertFalse((policy / f'checkpoints/step_{step}').exists())
        self.assertFalse((policy / 'final').exists())
        rebuilt = storage.load_evaluation_records(self.root, self.config)
        np.testing.assert_array_equal(rebuilt.validity[0, :, 0], [True, False, False])
        self.assertTrue(np.isnan(rebuilt.means_raw[0, 1:, 0]).all())
        self.assertTrue(np.isnan(rebuilt.hypervolume).all())
        archive = self.root / 'results/superseded/seed_0/w_0' / plan['state']['resume_attempt_id']
        archived_before = self.snapshot(list(archive.rglob('*')))
        self.assertTrue((archive / 'bundles/final/partial.zip').exists())
        resumed = self.finish(10, expected_restore=2)
        self.assertEqual((resumed['timeline'][0]['event'], resumed['timeline'][0]['step']), ('learn', 2))
        self.assert_preserved(early); self.assert_preserved(archived_before)
        for step in (4, 6):
            current, metadata = storage._load_evaluation_record(directory / f'step_{step}.json')
            self.assertEqual(metadata['resume_attempt_id'], plan['state']['resume_attempt_id'])
            self.assertTrue(current.identity.bundle_id.startswith(plan['state']['resume_attempt_id']))


if __name__ == '__main__':
    unittest.main()
