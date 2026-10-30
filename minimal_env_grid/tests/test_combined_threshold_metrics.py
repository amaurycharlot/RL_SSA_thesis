"""Run from the project root: python -B tests/test_combined_threshold_metrics.py."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts import plot_combined_threshold_tradeoff as metrics


class CombinedMetricsTests(unittest.TestCase):
    # Params: self; returns: None.
    def test_frequency_starts_at_detection_and_weights_targets_equally(self):
        observed = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ], dtype=bool)
        result = metrics.compute_observation_frequency_per_target(observed, [1, 1, 1, 1], 2)
        self.assertEqual(result[0].observation_frequency, 0.5)
        self.assertEqual(result[0].counting_start_timestep, 0)
        self.assertEqual(result[1].observation_frequency, 0)  # known, never observed
        self.assertEqual(result[2].counting_start_timestep, 3)
        self.assertEqual(result[2].eligible_timesteps, 2)
        self.assertEqual(result[2].observation_frequency, 1)  # discovery counts
        self.assertEqual(result[3].observation_frequency, 1)  # discovered at final time
        self.assertIsNone(result[4].observation_frequency)  # never detected
        all_mean, _ = metrics.finite_mean_and_std([
            r.observation_frequency for r in result if r.observation_frequency is not None
        ])
        self.assertEqual(all_mean, 0.625)
        known_mean, _ = metrics.finite_mean_and_std([
            r.observation_frequency for r in result if r.initially_known
        ])
        self.assertEqual(known_mean, 0.25)

    # Params: self; returns: None.
    def test_frequency_excludes_slew_time_before_discovery(self):
        observed = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
        result = metrics.compute_observation_frequency_per_target(observed, [4, 5, 3], 1)
        self.assertAlmostEqual(result[0].observation_frequency, 2 / 12)
        self.assertEqual(result[1].counting_start_timestep, 9)
        self.assertEqual(result[1].eligible_timesteps, 4)
        self.assertEqual(result[1].observation_frequency, 0.5)
        with self.assertRaises(ValueError):
            metrics.compute_observation_frequency_per_target(observed, [4, 0, 3], 1)
        with self.assertRaises(ValueError):
            metrics.compute_observation_frequency_per_target(observed, [4, 5, 3], 3)

    # Params: self; returns: None.
    def test_legacy_observations_use_post_action_truth_and_wrapped_fov(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = np.zeros((2, 3, 4), dtype=np.float32)
            truth[0, :, :2] = [[95, 5], [15, 15], [25, 5]]
            truth[1, :, :2] = [[55, 55], [45, 45], [5, 5]]
            np.save(root / 'sensor_pointings.npy', [[5, 5], [55, 55]])
            recovered = metrics.load_target_observations(root, truth, {'grid_shape': [10, 10]}, 100)
            np.testing.assert_array_equal(recovered, [[1, 1, 0], [1, 1, 0]])
            explicit = np.array([[0, 0, 1], [0, 1, 1]], dtype=bool)
            np.save(root / 'observed_targets.npy', explicit)
            np.testing.assert_array_equal(metrics.load_target_observations(root, truth, {}, 100), explicit)

    # Params: self; returns: None.
    def test_frequency_aggregation_export_and_third_plot(self):
        target_metrics = metrics.compute_observation_frequency_per_target(np.array([[1], [0]]), [1, 1], 1)
        runs = [metrics.RunMetrics(1, 1, 2, 10, 3, 20, 0.4, 0.3, tuple(target_metrics)),
                metrics.RunMetrics(1, 2, 4, 20, 5, 80, 0.8, 0.5)]
        aggregated = metrics.aggregate_by_threshold(runs)
        item = aggregated[0]
        self.assertAlmostEqual(item.observation_frequency_mean, 0.6)
        self.assertAlmostEqual(item.known_observation_frequency_mean, 0.4)
        self.assertAlmostEqual(item.known_observation_frequency_std, np.sqrt(0.02))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics.save_csv_files(runs, aggregated, root)
            with (root / 'per_target_observation_metrics.csv').open() as f:
                row = next(csv.DictReader(f))
                self.assertEqual(float(row['observation_frequency']), 0.5)
                self.assertEqual(int(row['eligible_timesteps']), 2)
            with (root / 'threshold_metrics.csv').open() as f:
                row = next(csv.DictReader(f))
                self.assertAlmostEqual(float(row['known_observation_frequency_mean']), 0.4)
            metrics.plot_observation_frequency_tradeoff(aggregated, root / 'third.png', True)
            self.assertGreater((root / 'third.png').stat().st_size, 0)
            metrics.plot_observation_frequency_tradeoff(
                metrics.aggregate_by_threshold([metrics.RunMetrics(1, 1, 0, 1, 1, 0)]),
                root / 'no_known.png', True)

    # Params: self; returns: None.
    def test_final_error_wraps_and_averages_targets_only_at_final_time(self):
        truth = np.zeros((2, 3, 4), dtype=np.float32)
        estimate = np.zeros((2, 3, 9), dtype=np.float32)
        truth[-1, 0, :2] = [1, 1]
        estimate[0, :, :2] = 30
        estimate[:, :, 4:6] = 1
        estimate[-1, 0, :2] = [99, 1]  # wrapped distance 2, not 98
        estimate[-1, 1, :2] = [3, 4]   # Euclidean distance 5
        estimate[-1, 2, :2] = [0, 0]   # valid estimate at the origin
        self.assertAlmostEqual(metrics.compute_mean_final_positional_error(
            truth, estimate, 100, "catalogue", {}), 7 / 3, places=6)
        self.assertEqual(metrics.compute_mean_final_positional_error(
            truth, estimate, 100, "initial", {"known_targets": 1}), 2)
        estimate[-1, 2, 4:6] = 0  # undetected target has no estimate
        self.assertEqual(metrics.compute_mean_final_positional_error(
            truth, estimate, 100, "catalogue", {}), 3.5)
        with self.assertRaises(ValueError):
            metrics.compute_mean_final_positional_error(truth[:1], estimate, 100, "catalogue", {})
        estimate[-1, :, 4:6] = 0
        with self.assertRaises(ValueError):
            metrics.compute_mean_final_positional_error(truth, estimate, 100, "catalogue", {})

    # Params: self; returns: None.
    def test_search_time_uses_pre_action_observation_and_elapsed_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = np.zeros((3, 1, 9), dtype=np.float32)
            states[:, 0, 4:6] = np.array([10, 1, 10])[:, None] / 2
            observations = np.zeros((3, 1, 1, 3), dtype=np.float32)
            layer = np.log1p(np.array([10, 1, 10], dtype=np.float32))
            layer /= np.log1p(100)
            observations[:, 0, 0, 0] = layer
            np.save(root / "observations.npy", observations)
            np.save(root / "elapsed_steps.npy", [1, 8, 1])
            # Pre-action uncertainties are 2 (initial), 10, 1.
            self.assertEqual(metrics.compute_search_time(root, 5, states, {"known_targets": 1}), 20)
            modes = np.array(["track", "search", "track"])
            np.save(root / "policy_modes.npy", modes)
            self.assertEqual(metrics.compute_search_time(root, 5, states, {}), 80)
            np.save(root / "policy_modes.npy", np.full(3, "search"))
            self.assertEqual(metrics.compute_search_time(root, 5, states, {}), 100)
            np.save(root / "policy_modes.npy", np.full(3, "track"))
            self.assertEqual(metrics.compute_search_time(root, 5, states, {}), 0)
            np.save(root / "policy_modes.npy", np.full(3, "combined"))
            with self.assertRaises(ValueError):
                metrics.compute_search_time(root, 5, states, {})
            (root / "policy_modes.npy").unlink()
            (root / "elapsed_steps.npy").unlink()
            self.assertAlmostEqual(metrics.compute_search_time(root, 5, states, {"known_targets": 1}), 200 / 3)
            np.save(root / "elapsed_steps.npy", [1, 0, 1])
            with self.assertRaises(ValueError):
                metrics.compute_search_time(root, 5, states, {"known_targets": 1})

    # Params: self; returns: None.
    def test_aggregation_csv_and_second_plot(self):
        runs = [metrics.RunMetrics(1, 1, 2, 10, 3, 20),
                metrics.RunMetrics(1, 2, 4, 20, 5, 80)]
        aggregated = metrics.aggregate_by_threshold(runs)
        item = aggregated[0]
        self.assertEqual(item.final_positional_error_mean, 4)
        self.assertAlmostEqual(item.final_positional_error_std, np.sqrt(2))
        self.assertEqual(item.search_time_mean, 50)
        self.assertAlmostEqual(item.search_time_std, np.sqrt(1800))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics.save_csv_files(runs, aggregated, root)
            with (root / "per_seed_metrics.csv").open() as f:
                row = next(csv.DictReader(f))
                self.assertEqual(float(row["mean_final_positional_error"]), 3)
                self.assertEqual(float(row["search_time_percent"]), 20)
            with (root / "threshold_metrics.csv").open() as f:
                row = next(csv.DictReader(f))
                self.assertEqual(float(row["mean_final_positional_error_mean"]), 4)
                self.assertEqual(float(row["search_time_percent_mean"]), 50)
            metrics.plot_search_time_tradeoff(aggregated, root / "plot.png", True)
            self.assertGreater((root / "plot.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
