"""Block 30 table round trips, missing semantics and committed-record provenance."""
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import storage
from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
from agents.mo.reporting import write_summary_tables
from agents.mo.types import (ExperimentConfig, EnvConfig, PPOConfig, ExecutionConfig, NormalizationConfig,
    EpisodeMetrics, TargetObservationMetrics, PolicyIdentity, EvaluationResult, FinalResults)


class SummaryTableExports(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ExperimentConfig(output_dir=Path(self.temporary.name) / 'experiment',
            run_seeds=(7, 9), n_weights=2, eval_episodes=2, final_episodes=2,
            reference_scaled=(-1., -1.), reference_status='frozen',
            objective_scales=(2., 4.), scale_status='frozen',
            requested_timesteps_per_policy=3, eval_interval=1, checkpoint_interval=1,
            env=EnvConfig(known_targets=1, unknown_targets=2, grid_shape=(2, 3), episode_steps=4, slew_alpha=0.),
            ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0),
            execution=ExecutionConfig(subproc=False), normalization=NormalizationConfig(norm_obs=True, norm_reward=True))
        self.initialize()

    # Params: self; returns: None.
    def initialize(self):
        self.weights = make_weights(self.config.n_weights)[::-1].copy()
        self.plans = tuple(make_seed_plan(self.config, i) for i in range(len(self.config.run_seeds)))
        self.budget = make_budget_plan(self.config)
        self.root = storage.create_experiment_dir(self.config, self.weights, self.plans, self.budget)
        self.fingerprint = storage._read_json(self.root / 'manifest.json')['fingerprint']
        self.final_records = {}

    # Params: self, run, weight, budget, final; returns: record.
    def record(self, run=0, weight=0, budget=0, final=False):
        seeds = self.plans[run].final_seeds if final else self.plans[run].eval_seeds
        step = self.budget.effective_total if final else self.budget.eval_steps[budget]
        identity = PolicyIdentity(self.fingerprint, self.config.run_seeds[run], run, weight,
                                  tuple(self.weights[weight].tolist()), step, f'policy-{run}-{weight}-{step}-{final}')
        raw = np.array([[2. * e + (10 if final else 0), 4. * (e + 1)] for e in range(len(seeds))], np.float64)
        scaled = raw / self.config.objective_scales
        metrics = []
        for e in range(len(seeds)):
            reasons = {'search_time_percentage': 'PPO has no explicit search/track action mode'}
            if e:
                reasons.update({name: 'No valid target pairs' for name in
                    ('mean_positional_error', 'mean_final_positional_error', 'mean_covariance_trace')})
            targets = (TargetObservationMetrics(0, True, 0, 0, 4, 0.),
                       TargetObservationMetrics(1, False, 1, 3, 2, .5) if not e
                       else TargetObservationMetrics(1, False, 0, None, None, None),
                       TargetObservationMetrics(2, False, 0, None, None, None))
            metrics.append(EpisodeMetrics(mean_positional_error=2. if not e else None,
                mean_final_positional_error=3. if not e else None, mean_covariance_trace=4. if not e else None,
                final_coverage=5. if not e else 0., mean_coverage=4. if not e else 0., unique_new_detections=1 if not e else 0,
                mean_known_observation_frequency=0., mean_observation_frequency=.25 if not e else 0.,
                decisions=2, elapsed_timesteps=4, target_observation_metrics=targets, unavailable_reasons=reasons))
        summaries = {}
        for space, values in (('raw', raw), ('scaled', scaled)):
            sd = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(2)
            summaries.update({f'mean_{space}': values.mean(axis=0), f'sd_{space}': sd, f'se_{space}': sd / np.sqrt(len(seeds))})
        record = EvaluationResult(seeds, raw, scaled, episode_metrics=tuple(metrics), deterministic=True,
                                  actual_step=step, identity=identity, **summaries)
        storage.save_evaluation_record(self.root, identity, record)
        if final: self.final_records[(run, weight)] = record
        return record

    # Params: self; returns: storage.load_evaluation_records result.
    def results(self):
        return storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: FinalResults.
    def final(self):
        from agents.mo.pareto import hypervolume_2d_max, unique_front_indices
        s, w, e = len(self.plans), self.config.n_weights, self.config.final_episodes
        raw = np.full((s, w, e, 2), np.nan, np.float64); scaled = np.full_like(raw, np.nan)
        valid = np.zeros((s, w), bool); metrics = [[() for _ in range(w)] for _ in range(s)]
        for (i, j), record in self.final_records.items():
            raw[i, j] = record.returns_raw; scaled[i, j] = record.returns_scaled
            valid[i, j] = True; metrics[i][j] = record.episode_metrics
        # Avoid warnings from intentionally missing slots.
        mean_raw = np.full((s, w, 2), np.nan); mean_scaled = np.full_like(mean_raw, np.nan)
        mean_raw[valid] = raw[valid].mean(axis=1); mean_scaled[valid] = scaled[valid].mean(axis=1)
        fronts = tuple(tuple(int(j) for j in unique_front_indices(mean_scaled[i], range(w))) if valid[i].all() else () for i in range(s))
        hv = np.array([hypervolume_2d_max(mean_scaled[i], self.config.reference_scaled) if valid[i].all() else np.nan for i in range(s)])
        return FinalResults(mean_raw, mean_scaled, raw, scaled, valid, self.plans[0].final_seeds,
                            fronts, hv, tuple(tuple(row) for row in metrics))

    # Params: self; returns: None.
    def populate(self):
        for i in range(len(self.plans)):
            for j in range(self.config.n_weights):
                for k in range(len(self.budget.eval_steps)): self.record(i, j, k)
                self.record(i, j, final=True)

    # Params: self, name; returns: list.
    def rows(self, name):
        with (self.root / 'reports/tables' / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    # Params: self; returns: None.
    def test_round_trip_identity_raw_scaled_uncertainty_and_final_separation(self):
        self.populate()
        self.assertIsNone(write_summary_tables(self.results(), self.final(), self.root))
        rows = self.rows('policy_results.csv')
        self.assertEqual(len(rows), 12)
        first = rows[0]
        self.assertEqual((first['phase'], int(first['run_seed']), int(first['weight_index']), int(first['actual_step'])), ('intermediate', 7, 0, 2))
        self.assertEqual(float(first['weight_tracking']), float(self.weights[0, 0]))
        self.assertEqual(float(first['tracking_mean_raw']), 1.)
        self.assertAlmostEqual(float(first['tracking_sd_raw']), np.sqrt(2))
        self.assertEqual(float(first['tracking_se_raw']), 1.)
        self.assertEqual(float(first['tracking_mean_scaled']), .5)
        self.assertEqual(float(first['search_mean_scaled']), 1.5)
        final = next(r for r in rows if r['phase'] == 'final')
        self.assertEqual(float(final['tracking_mean_raw']), 11.)
        episodes = self.rows('episode_returns.csv')
        self.assertEqual(len(episodes), 24)
        for row in episodes:
            i, e = int(row['run_index']), int(row['episode_index'])
            seeds = self.plans[i].final_seeds if row['phase'] == 'final' else self.plans[i].eval_seeds
            self.assertEqual(int(row['episode_seed']), seeds[e])

    # Params: self; returns: None.
    def test_nullable_metrics_and_real_zero_remain_distinct(self):
        self.record()
        write_summary_tables(self.results(), None, self.root)
        rows = self.rows('episode_metrics.csv')
        missing = next(r for r in rows if r['metric'] == 'mean_positional_error' and r['episode_index'] == '1')
        self.assertEqual(missing['value'], '')
        self.assertEqual(missing['unavailable_reason'], 'No valid target pairs')
        zero = next(r for r in rows if r['metric'] == 'mean_known_observation_frequency')
        self.assertEqual(float(zero['value']), 0.)
        self.assertEqual(zero['unavailable_reason'], '')
        error = next(r for r in rows if r['metric'] == 'mean_positional_error' and r['episode_index'] == '0')
        self.assertEqual(error['within_episode_sample_count'], '')
        self.assertIn('not recorded', error['sample_count_unavailable_reason'])

    # Params: self; returns: None.
    def test_target_exposure_known_discovered_and_never_detected_round_trip(self):
        self.record(); write_summary_tables(self.results(), None, self.root)
        rows = [r for r in self.rows('target_observation_metrics.csv') if r['episode_index'] == '0']
        self.assertEqual(len(rows), 3)
        known, discovered, never = rows
        self.assertEqual((known['initially_known'], known['counting_start_timestep'], known['eligible_timesteps']), ('true', '0', '4'))
        self.assertEqual(float(known['observation_frequency']), 0.)
        self.assertEqual((discovered['counting_start_timestep'], discovered['eligible_timesteps']), ('3', '2'))
        self.assertEqual(float(discovered['observation_frequency']), int(discovered['observation_count']) / int(discovered['eligible_timesteps']))
        self.assertEqual(never['observation_count'], '0')
        self.assertEqual((never['counting_start_timestep'], never['eligible_timesteps'], never['observation_frequency']), ('', '', ''))

    # Params: self; returns: None.
    def test_operational_available_episode_counts_and_unavailable_search_time(self):
        self.record(); write_summary_tables(self.results(), None, self.root)
        rows = self.rows('operational_summary.csv')
        first = next(r for r in rows if r['metric'] == 'mean_positional_error')
        self.assertEqual((first['available_episode_count'], first['missing_episode_count'], first['all_episodes_available']), ('1', '1', 'false'))
        self.assertEqual(float(first['mean_available_episodes']), 2.)
        self.assertEqual(float(first['sd_available_episodes']), 0.)
        search = next(r for r in rows if r['metric'] == 'search_time_percentage')
        self.assertEqual((search['available_episode_count'], search['mean_available_episodes']), ('0', ''))
        self.assertEqual(json.loads(search['unavailable_reason_counts']), {'PPO has no explicit search/track action mode': 2})

    # Params: self; returns: None.
    def test_missing_policy_slots_are_explicit_and_have_no_attained_step(self):
        self.record(); write_summary_tables(self.results(), None, self.root)
        rows = self.rows('policy_results.csv')
        self.assertEqual(len(rows), 8)
        missing = next(r for r in rows if r['valid'] == 'false')
        self.assertEqual((missing['actual_step'], missing['bundle_id'], missing['tracking_mean_raw']), ('', '', ''))
        self.assertIn(missing['actual_budget'], ('2', '4'))
        self.assertEqual(missing['completed_episode_count'], '0')

    # Params: self; returns: None.
    def test_hv_counts_require_all_weights_and_all_declared_seeds(self):
        self.record(0, 0); self.record(0, 1); self.record(1, 0)
        write_summary_tables(self.results(), None, self.root)
        rows = self.rows('hypervolume.csv')
        aggregate = next(r for r in rows if r['row_kind'] == 'across_training_seeds' and r['actual_budget'] == '2')
        self.assertEqual((aggregate['complete_seed_count'], aggregate['declared_seed_count']), ('1', '2'))
        self.assertEqual((aggregate['mean_hypervolume'], aggregate['sd_hypervolume']), ('', ''))
        incomplete = next(r for r in rows if r['run_index'] == '1' and r['actual_budget'] == '2')
        self.assertEqual((incomplete['complete_policy_count'], incomplete['declared_policy_count'], incomplete['hypervolume']), ('1', '2', ''))

    # Params: self; returns: None.
    def test_incomplete_final_results_have_missing_hv_and_explicit_final_slots(self):
        self.record(final=True)
        write_summary_tables(self.results(), self.final(), self.root)
        final = [r for r in self.rows('policy_results.csv') if r['phase'] == 'final']
        self.assertEqual(len(final), 4)
        self.assertEqual(sum(r['valid'] == 'true' for r in final), 1)
        self.assertTrue(all(not r['hypervolume'] and not r['mean_hypervolume']
                            for r in self.rows('hypervolume.csv') if r['phase'] == 'final'))

    # Params: self; returns: None.
    def test_metadata_preserves_scales_normalization_reference_and_resume_state(self):
        state = storage.resolve_resume_state(self.root, self.config, 0, 0)
        self.record(); write_summary_tables(self.results(), None, self.root)
        folder = self.root / 'reports/tables'; meta = json.loads((folder / 'table_metadata.json').read_text())
        self.assertEqual(meta['objective_scales'], [2., 4.]); self.assertEqual(meta['reference_scaled'], [-1., -1.])
        self.assertTrue(meta['normalization']['norm_obs']); self.assertTrue(meta['normalization']['norm_reward'])
        self.assertEqual(meta['resume_attempts'][0]['state'], storage._json_value(state))
        self.assertEqual(meta['evaluation_records'][0]['resume_attempt_id'], 'initial')
        self.assertEqual(meta['error_target_set'], self.config.error_target_set)
        for name, info in meta['tables'].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), info['sha256'])
            self.assertEqual(len(self.rows(name)), info['rows'])
        self.assertNotIn('NaN', (folder / 'table_metadata.json').read_text())

    # Params: self; returns: None.
    def test_changed_means_rejected_before_any_report_output(self):
        self.record(); results = self.results(); results.means_raw[0, 0, 0, 0] += 1.
        with self.assertRaises(storage.StorageError): write_summary_tables(results, None, self.root)
        self.assertFalse((self.root / 'reports/tables').exists())

    # Params: self; returns: None.
    def test_stale_validity_and_wrong_final_seed_order_rejected(self):
        results = self.results(); self.record()
        with self.assertRaises(storage.StorageError): write_summary_tables(results, None, self.root)
        final = replace(self.final(), test_seeds=self.plans[0].final_seeds[::-1])
        with self.assertRaises(storage.StorageError): write_summary_tables(self.results(), final, self.root)
        self.assertFalse((self.root / 'reports/tables').exists())

    # Params: self; returns: None.
    def test_corrupt_committed_payload_rejected_without_report_output(self):
        self.record(); results = self.results()
        (self.root / 'results/evaluations/seed_0/w_0/step_2.npz').write_bytes(b'corrupt')
        with self.assertRaises(storage.StorageError): write_summary_tables(results, None, self.root)
        self.assertFalse((self.root / 'reports/tables').exists())

    # Params: self; returns: None.
    def test_atomic_write_failure_preserves_existing_csv_and_cleans_temporary_file(self):
        self.record(); results = self.results(); write_summary_tables(results, None, self.root)
        folder = self.root / 'reports/tables'; before = {p: p.read_bytes() for p in folder.iterdir()}
        with patch.object(storage.os, 'replace', side_effect=OSError('injected replace failure')):
            with self.assertRaisesRegex(OSError, 'replace failure'): write_summary_tables(results, None, self.root)
        self.assertEqual(set(folder.iterdir()), set(before))
        for p, content in before.items(): self.assertEqual(p.read_bytes(), content)

    # Params: self; returns: None.
    def test_repeatable_exports_preserve_inputs_and_authoritative_files(self):
        self.populate(); results, final = self.results(), self.final()
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file() and 'reports' not in p.parts}
        raw = results.means_raw.copy(); final_raw = final.episode_returns_raw.copy()
        write_summary_tables(results, final, self.root)
        tables = {p: p.read_bytes() for p in (self.root / 'reports/tables').iterdir()}
        write_summary_tables(results, final, self.root)
        for p, content in before.items(): self.assertEqual(p.read_bytes(), content)
        for p, content in tables.items(): self.assertEqual(p.read_bytes(), content)
        np.testing.assert_array_equal(results.means_raw, raw); np.testing.assert_array_equal(final.episode_returns_raw, final_raw)

    # Params: self; returns: None.
    def test_import_and_empty_export_do_not_import_learning_or_plotting_libraries(self):
        code = '''
import sys
from pathlib import Path
from agents.mo.reporting import write_summary_tables
from agents.mo import storage
from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
from agents.mo.types import ExperimentConfig, PPOConfig
c=ExperimentConfig(output_dir=Path('empty'), run_seeds=(1,), n_weights=2, eval_episodes=1, final_episodes=1,
reference_scaled=(-1.,-1.), reference_status='frozen', requested_timesteps_per_policy=2,
ppo=PPOConfig(n_envs=1,n_steps=2,batch_size=2,n_epochs=1))
root=storage.create_experiment_dir(c,make_weights(2),make_seed_plan(c,0),make_budget_plan(c))
write_summary_tables(storage.load_evaluation_records(root,c),None,root)
assert not any(n.split('.')[0] in ('torch','gymnasium','stable_baselines3','matplotlib') for n in sys.modules)
'''
        import os
        env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
        subprocess.run([sys.executable, '-B', '-c', code], cwd=self.temporary.name,
                       env=env, check=True, capture_output=True, text=True)

    # Params: self; returns: None.
    def test_real_normalized_training_and_final_records_export_without_evaluation(self):
        import torch
        import warnings
        from agents.mo.trainer import run_experiment
        from agents.mo.evaluation import evaluate_final_policy_set
        config = replace(self.config, output_dir=Path(self.temporary.name) / 'actual',
                         requested_timesteps_per_policy=2, env=replace(self.config.env, episode_steps=2))
        threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                summary = run_experiment(config)
            root = summary.experiment_path
            plans = tuple(make_seed_plan(config, i) for i in range(2))
            results = storage.load_evaluation_records(root, config)
            final = evaluate_final_policy_set(root, config, plans)
            with patch('agents.mo.evaluation.evaluate_policy', side_effect=AssertionError('report must not evaluate')):
                write_summary_tables(results, final, root)
            with (root / 'reports/tables/policy_results.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8); self.assertTrue(all(r['valid'] == 'true' for r in rows))
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
