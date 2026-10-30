"""Numerical tests, independent of the optional plotting dependency.

Run from the project root: python -B -m unittest discover -s tests
Override METRICS_SCRIPT to validate a staged copy before installation.
"""
import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


script = Path(os.environ.get("METRICS_SCRIPT", str(
    Path(__file__).resolve().parents[1] / "scripts" / "plot_combined_threshold_tradeoff.py")))
spec = importlib.util.spec_from_file_location("surveillance_metrics_under_test", script)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
try:
    import matplotlib.pyplot
    spec.loader.exec_module(m)
except ModuleNotFoundError as error:
    if not error.name.startswith("matplotlib"):
        raise
    # None of these tests call plotting code. Do not install dependencies or
    # leave mock modules in sys.modules for unrelated tests.
    colors = types.ModuleType("matplotlib.colors")
    colors.Normalize = object
    with patch.dict(sys.modules, {
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "matplotlib.colors": colors,
    }):
        spec.loader.exec_module(m)


class SurveillanceMetricsTests(unittest.TestCase):
    def test_window_boundaries_and_slew(self):
        # Observations at 2 and 4, disjoint 9-cell footprints in 100 cells.
        p = [[5, 5], [55, 55]]
        self.assertAlmostEqual(m.compute_mean_window_coverage(p, [2, 2], [10, 10], 100, 2), .09)
        self.assertAlmostEqual(m.compute_mean_window_coverage(p, [2, 2], [10, 10], 100, 3), .135)
        self.assertTrue(np.isnan(m.compute_mean_window_coverage(p, [2, 2], [10, 10], 100, 5)))
        # No measurement until t=4; slew does not cover the destination early.
        self.assertAlmostEqual(m.compute_mean_window_coverage([[5, 5]], [4], [10, 10], 100, 1), .09 / 4)

    # Params: self; returns: None.
    def test_wrapping_and_small_grid_no_double_count(self):
        masks = m._fov_masks([[5, 5]], [10, 10], 100).reshape(10, 10)
        self.assertEqual(masks.sum(), 9)
        self.assertTrue(masks[9, 9])
        self.assertEqual(m.compute_mean_window_coverage([[5, 5]], [1], [2, 2], 100, 1), 1)

    def test_grid_age_and_horizon_before_first_measurement(self):
        # Disjoint resets at 2,4; mean ages: 1,1.82,2.82,3.46.
        self.assertAlmostEqual(m.compute_mean_grid_observation_age(
            [[5, 5], [55, 55]], [2, 2], [10, 10], 100), 2.275)
        self.assertEqual(m.compute_mean_grid_observation_age(
            [[5, 5]], [4], [10, 10], 100, horizon=2), 1.5)

    # Params: self; returns: None.
    def test_known_frequency_and_age_include_terminal_gap(self):
        observed = [[1, 0, 0], [0, 0, 1]]
        self.assertEqual(m.compute_mean_known_observation_frequency(observed, [2, 3], 2), .1)
        # Known target 0 ages: 1,0,1,2,3; target 1: 1,2,3,4,5.
        self.assertEqual(m.compute_mean_known_observation_age(observed, [2, 3], 2), 2.2)
        self.assertEqual(m.compute_mean_known_observation_age(observed, [2, 3], 2, horizon=3), 4 / 3)
        self.assertTrue(np.isnan(m.compute_mean_known_observation_age(observed, [2, 3], 0)))
        self.assertTrue(np.isnan(m.compute_mean_known_observation_frequency(observed, [2, 3], 0)))

    # Params: self; returns: None.
    def test_discovery_censoring_not_detected_only_average(self):
        observed = [[1, 1, 0, 0], [1, 0, 1, 0]]
        self.assertEqual(m.compute_restricted_mean_discovery_time(observed, [2, 3], 1), 4)
        self.assertAlmostEqual(m.compute_restricted_mean_discovery_time(
            observed, [2, 3], 1, horizon=4), 10 / 3)
        self.assertTrue(np.isnan(m.compute_restricted_mean_discovery_time([[1]], [1], 1)))

    # Params: self; returns: None.
    def test_dense_wrapped_known_error_and_sparse_unavailability(self):
        truth = np.zeros((2, 2, 4))
        estimate = np.zeros((2, 2, 9))
        estimate[:, :, 4:6] = 1
        truth[:, 0, 0] = 1
        estimate[:, 0, 0] = [99, 4]  # errors 2 and 3
        estimate[:, 1, :2] = 45  # unknown excluded
        self.assertEqual(m.compute_time_averaged_known_positional_error(
            truth, estimate, [1, 1], 1, 100), 2.5)
        self.assertTrue(np.isnan(m.compute_time_averaged_known_positional_error(
            truth, estimate, [1, 2], 1, 100)))
        self.assertEqual(m.compute_time_averaged_known_positional_error(
            truth, estimate, [1, 2], 1, 100, horizon=1), 2)
        estimate[1, 0, 4:6] = 0
        self.assertTrue(np.isnan(m.compute_time_averaged_known_positional_error(
            truth, estimate, [1, 1], 1, 100)))

    # Params: self; returns: None.
    def test_validation(self):
        for elapsed in ([0], [1.5], [np.nan], [1, 2]):
            with self.assertRaises(ValueError):
                m.compute_mean_known_observation_age([[1]], elapsed, 1)
        for horizon in (0, 3, 1.5):
            with self.assertRaises(ValueError):
                m.compute_restricted_mean_discovery_time([[1]], [2], 0, horizon)
        with self.assertRaises(ValueError):
            m.compute_mean_known_observation_frequency([[1]], [1], 2)
        with self.assertRaises(ValueError):
            m.compute_mean_window_coverage([[0, 0]], [1], [10, 10], 100, 0)

    # Params: self; returns: None.
    def test_run_loader_aggregation_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = np.zeros((2, 2, 4))
            estimate = np.zeros((2, 2, 9))
            estimate[:, :, 4:6] = 1
            estimate[:, 0, 0] = 3
            for name, values in {
                "true_states": truth, "est_states": estimate,
                "sensor_pointings": [[5, 5], [55, 55]], "elapsed_steps": [2, 3],
                "observed_targets": [[1, 0], [0, 1]], "policy_modes": ["track", "search"],
            }.items():
                np.save(root / (name + ".npy"), values)
            (root / "meta.json").write_text('{"known_targets": 1, "grid_shape": [10,10]}')
            run = m.load_run_metrics(m.RunSpec(1, 1, root), 100, "initial", 2, 4)
            self.assertEqual(run.known_target_observation_frequency, .25)
            self.assertEqual(run.mean_known_observation_frequency, .2)  # legacy full run
            self.assertEqual(run.restricted_mean_discovery_time, 4)
            self.assertTrue(np.isnan(run.time_averaged_known_positional_error))
            self.assertEqual(run.endpoint_mean_known_positional_error, 3)
            self.assertEqual(run.metric_horizon, 4)
            from dataclasses import replace
            run2 = replace(run, seed=2, mean_window_coverage=run.mean_window_coverage + .2)
            aggregated = m.aggregate_by_threshold([run, run2])
            self.assertAlmostEqual(aggregated[0].mean_window_coverage_std, np.sqrt(.02))
            self.assertEqual(aggregated[0].time_averaged_known_positional_error_n_valid, 0)
            self.assertEqual(aggregated[0].known_target_observation_frequency_n_valid, 2)
            m.save_csv_files([run, run2], aggregated, root)
            for filename, aggregate in [("per_seed_metrics.csv", False), ("threshold_metrics.csv", True)]:
                with (root / filename).open() as f:
                    row = next(csv.DictReader(f))
                self.assertNotIn(None, row)  # CSV header/value lengths agree
                self.assertEqual(row["metric_horizon"], "4")
                for name in m.EXTRA_SCALAR_METRICS:
                    self.assertIn(name + ("_mean" if aggregate else ""), row)
            with self.assertRaises(ValueError):
                m.aggregate_by_threshold([run, replace(run2, metric_horizon=5)])
            with self.assertRaises(ValueError):
                m.aggregate_by_threshold([run, replace(run2, coverage_window=3)])


if __name__ == "__main__":
    unittest.main()
