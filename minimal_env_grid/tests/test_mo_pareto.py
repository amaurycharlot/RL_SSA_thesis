"""Block 08 exact maximization/front contracts; no training or HV computation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo.pareto import (
    non_dominated_mask, unique_front_indices, hypervolume_2d_max,
    hypervolume_reference_diagnostics,
    compute_hv_curve, summarize_hv_across_seeds,
)
from agents.mo import ExperimentResults


class MaximizationDominance(unittest.TestCase):
    # Params: self; returns: None.
    def test_tradeoff_front_and_dominated_points(self):
        points = np.array([[1., 3.], [2., 2.], [3., 1.], [1., 1.], [0., 0.]])
        np.testing.assert_array_equal(non_dominated_mask(points), [True, True, True, False, False])
        np.testing.assert_array_equal(unique_front_indices(points, [10, 20, 30, 40, 50]), [0, 1, 2])

    # Params: self; returns: None.
    def test_one_strict_coordinate_is_sufficient_and_rewards_are_maximized(self):
        points = np.array([[1., 1.], [1., 2.], [2., 2.]])
        np.testing.assert_array_equal(non_dominated_mask(points), [False, False, True])
        negative = np.array([[-3., -3.], [-2., -2.], [-1., -4.]])
        np.testing.assert_array_equal(non_dominated_mask(negative), [False, True, True])

    # Params: self; returns: None.
    def test_equal_rows_do_not_strictly_dominate_and_lowest_id_represents(self):
        points = np.array([[2., 2.], [2., 2.], [2., 2.]])
        np.testing.assert_array_equal(non_dominated_mask(points), [True, True, True])
        np.testing.assert_array_equal(unique_front_indices(points, [30, 10, 20]), [1])
        np.testing.assert_array_equal(unique_front_indices(points, ["bundle_c", "bundle_a", "bundle_b"]), [1])

    # Params: self; returns: None.
    def test_dominated_duplicates_stay_out_of_front(self):
        points = np.array([[1., 1.], [1., 1.], [2., 2.], [2., 2.]])
        np.testing.assert_array_equal(non_dominated_mask(points), [False, False, True, True])
        np.testing.assert_array_equal(unique_front_indices(points, [1, 2, 9, 3]), [3])

    # Params: self; returns: None.
    def test_empty_and_singleton_shapes_and_dtypes(self):
        empty = np.empty((0, 2), dtype=np.float32)
        mask, indices = non_dominated_mask(empty), unique_front_indices(empty, [])
        self.assertEqual(mask.shape, (0,))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(indices.shape, (0,))
        self.assertEqual(indices.dtype, np.intp)
        singleton = np.array([[1., -1.]], dtype=np.float32)
        np.testing.assert_array_equal(non_dominated_mask(singleton), [True])
        np.testing.assert_array_equal(unique_front_indices(singleton, [0]), [0])

    # Params: self; returns: None.
    def test_representatives_are_deterministic_under_input_permutation(self):
        points = np.array([[3., 1.], [1., 3.], [2., 2.], [1., 3.], [2., 2.], [0., 0.]])
        ids = np.array([30, 19, 29, 10, 20, 0])
        expected_points = np.array([[1., 3.], [2., 2.], [3., 1.]])
        for permutation in (np.arange(6), np.array([5, 4, 3, 2, 1, 0]), np.array([2, 0, 5, 1, 4, 3])):
            reordered, reordered_ids = points[permutation], ids[permutation]
            indices = unique_front_indices(reordered, reordered_ids)
            np.testing.assert_array_equal(reordered[indices], expected_points)
            np.testing.assert_array_equal(reordered_ids[indices], [10, 20, 30])

    # Params: self; returns: None.
    def test_close_points_remain_distinct_and_precision_is_not_downcast(self):
        points = np.array([[1., 2.], [1. + 1e-12, 2. - 1e-12]], dtype=np.float64)
        np.testing.assert_array_equal(non_dominated_mask(points), [True, True])
        np.testing.assert_array_equal(unique_front_indices(points, [1, 0]), [0, 1])
        larger = np.array([[1., 2.], [1. + 1e-12, 2.]], dtype=np.float64)
        np.testing.assert_array_equal(non_dominated_mask(larger), [False, True])

    # Params: self; returns: None.
    def test_input_buffers_and_all_policy_id_associations_are_preserved(self):
        points = np.array([[1., 3.], [2., 2.], [1., 3.], [0., 0.]])[::-1]
        expected = points.copy()
        points.setflags(write=False)
        ids = [(0, 8), (0, 7), (0, 6), (0, 5)]
        previous_ids = list(ids)
        mask, representatives = non_dominated_mask(points), unique_front_indices(points, ids)
        np.testing.assert_array_equal(points, expected)
        self.assertEqual(ids, previous_ids)
        self.assertEqual(np.count_nonzero(mask), 3)
        self.assertEqual(len(representatives), 2)
        self.assertEqual([ids[index] for index in representatives], [(0, 5), (0, 6)])
        self.assertFalse(np.shares_memory(points, mask))
        self.assertFalse(np.shares_memory(points, representatives))

    # Params: self; returns: None.
    def test_malformed_nonfinite_and_nonfloating_points_fail(self):
        invalid = [np.zeros((2, 3)), np.zeros(2), np.zeros((1, 1, 2)),
                   np.array([[np.nan, 0.]]), np.array([[0., np.inf]]), np.array([[-np.inf, 1.]])]
        for points in invalid:
            for function in (lambda: non_dominated_mask(points), lambda: unique_front_indices(points, list(range(len(points))))):
                with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                    function()
        for points in ([[1., 2.]], np.array([[1, 2]]), np.array([[True, False]]),
                       np.array([[1+1j, 2+0j]]), np.array([[1., 2.]], dtype=object)):
            with self.subTest(type=type(points)), self.assertRaises(TypeError):
                non_dominated_mask(points)

    # Params: self; returns: None.
    def test_stable_ids_must_match_be_unique_and_be_orderable(self):
        points = np.array([[1., 2.], [2., 1.]])
        for ids, error in (([0], ValueError), ([1, 1], ValueError), (np.array([[0, 1]]), ValueError),
                           ([0, "one"], TypeError), ([0, None], TypeError), ([True, 1], TypeError),
                           ([0., 1.], TypeError), ([0, -1], ValueError), ("01", TypeError),
                           (["", "one"], TypeError)):
            with self.subTest(ids=ids), self.assertRaises(error):
                unique_front_indices(points, ids)

    # Params: self; returns: None.
    def test_import_is_passive_in_a_fresh_process(self):
        script = '''
import sys
import agents.mo.pareto
assert not any(name in sys.modules for name in
               ("numpy", "torch", "gymnasium", "scipy", "stable_baselines3", "matplotlib"))
print("passive geometry import")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        with tempfile.TemporaryDirectory(prefix="mo_pareto_import_") as directory:
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=directory,
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


class ExactHypervolume(unittest.TestCase):
    # Params: self; returns: None.
    def test_three_point_reference_example_is_six(self):
        points = np.array([[1., 3.], [2., 2.], [3., 1.]])
        area = hypervolume_2d_max(points, [0, 0])
        self.assertIs(type(area), float)
        self.assertEqual(area, 6.0)

    # Params: self; returns: None.
    def test_duplicates_dominated_rows_and_order_do_not_change_union(self):
        points = np.array([[1., 3.], [2., 2.], [3., 1.], [1., 1.], [2., 2.]])
        for order in (np.arange(5), np.arange(5)[::-1], np.array([3, 1, 0, 4, 2])):
            self.assertEqual(hypervolume_2d_max(points[order], [0, 0]), 6.)
        self.assertEqual(hypervolume_2d_max(np.array([[2., 3.], [2., 3.]]), [0, 0]), 6.)

    # Params: self; returns: None.
    def test_outside_reference_and_boundary_rows_add_zero_area(self):
        points = np.array([[1., 3.], [2., 2.], [3., 1.], [-1., 100.], [100., -1.],
                           [0., 100.], [100., 0.], [-1., -1.]])
        self.assertEqual(hypervolume_2d_max(points, [0, 0]), 6.)
        self.assertEqual(hypervolume_2d_max(points[3:], [0, 0]), 0.)
        self.assertEqual(hypervolume_2d_max(np.array([[0., 0.]]), [0, 0]), 0.)

    # Params: self; returns: None.
    def test_empty_and_single_rectangle(self):
        self.assertEqual(hypervolume_2d_max(np.empty((0, 2)), [0., 0.]), 0.)
        self.assertEqual(hypervolume_2d_max(np.array([[4., 5.]]), [1., 2.]), 9.)

    # Params: self; returns: None.
    def test_reference_sign_and_translation_preserve_geometry(self):
        points = np.array([[-2., 0.], [-1., -1.], [0., -2.]])
        self.assertEqual(hypervolume_2d_max(points, [-3., -3.]), 6.)
        self.assertEqual(hypervolume_2d_max(points + 10, [7., 7.]), 6.)
        self.assertEqual(hypervolume_2d_max(np.array([[1., 3.], [2., 2.], [3., 1.]]), [1., 1.]), 1.)

    # Params: self; returns: None.
    def test_reference_diagnostics_preserve_original_row_associations(self):
        points = np.array([[2., 2.], [2., 2.], [1., 1.], [-1., 5.], [0., 4.]])
        diagnostics = hypervolume_reference_diagnostics(points, [0, 0])
        self.assertEqual(diagnostics, {"n_points": 5, "n_reference_contributors": 3,
            "n_noncontributing_points": 2, "contributing_indices": [0, 1, 2],
            "noncontributing_indices": [3, 4]})
        ids = ["a", "b", "c", "outside", "boundary"]
        self.assertEqual([ids[i] for i in diagnostics["noncontributing_indices"]], ["outside", "boundary"])
        self.assertEqual(hypervolume_2d_max(points, [0, 0]), 4.)

    # Params: self; returns: None.
    def test_fractional_float32_area_uses_wider_arithmetic(self):
        points = np.array([[.5, 1.5], [1., 1.], [1.5, .5]], dtype=np.float32)
        self.assertEqual(hypervolume_2d_max(points, [0, 0]), 1.5)
        large = np.array([[1e20, 1e20]], dtype=np.float32)
        expected = float(large[0, 0]) * float(large[0, 1])
        self.assertEqual(hypervolume_2d_max(large, [0, 0]), expected)
        with self.assertRaises(OverflowError):
            hypervolume_2d_max(np.array([[1e308, 1e308]]), [0, 0])

    def test_union_agrees_with_independent_grid_cell_area(self):
        # Brute-force unit-cell occupancy is independent of sorted-front logic.
        points = np.array([[1., 4.], [3., 2.], [2., 3.], [2., 1.], [0., 6.]])
        occupied = sum(any(x < px and y < py for px, py in points) for x in range(4) for y in range(6))
        self.assertEqual(occupied, 9)
        self.assertEqual(hypervolume_2d_max(points, [0, 0]), float(occupied))

    # Params: self; returns: None.
    def test_inputs_reference_and_read_only_buffers_are_unchanged(self):
        points = np.array([[3., 1.], [1., 3.], [2., 2.]])[::-1]
        reference = np.array([0., 0.])
        before_points, before_ref = points.copy(), reference.copy()
        points.setflags(write=False)
        reference.setflags(write=False)
        self.assertEqual(hypervolume_2d_max(points, reference), 6.)
        hypervolume_reference_diagnostics(points, reference)
        np.testing.assert_array_equal(points, before_points)
        np.testing.assert_array_equal(reference, before_ref)

    # Params: self; returns: None.
    def test_malformed_or_nonfinite_reference_and_points_fail(self):
        points = np.array([[1., 2.]])
        for reference in ([], [0], [0, 0, 0], [[0, 0]], [np.nan, 0], [0, np.inf]):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                hypervolume_2d_max(points, reference)
        for reference in ([True, 0], np.array([True, False]), ["0", "0"], [1j, 0]):
            with self.subTest(reference=reference), self.assertRaises(TypeError):
                hypervolume_2d_max(points, reference)
        for invalid in (np.array([[np.nan, 1.]]), np.zeros((1, 3))):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                hypervolume_2d_max(invalid, [0, 0])
        with self.assertRaises(ValueError):
            hypervolume_2d_max(np.empty((0, 2)), [np.nan, 0])


class CompleteSetHypervolumeCurves(unittest.TestCase):
    # Params: self, results, seed, budget, points, raw; returns: None.
    def put(self, results, seed, budget, points, raw=None):
        points = np.asarray(points, dtype=np.float64)
        results.means_scaled[seed, budget] = points
        results.means_raw[seed, budget] = points if raw is None else raw
        results.validity[seed, budget] = True

    # Params: self; returns: None.
    def test_complete_union_by_seed_and_budget_not_pooled_or_policy_average(self):
        results = ExperimentResults.empty(2, [4096, 8192], 2)
        self.put(results, 0, 0, [[1, 3], [3, 1]])  # Union=5; mean individual area=3.
        self.put(results, 1, 0, [[2, 2], [0, 0]])  # Union=4; pooled seed union=6.
        self.put(results, 0, 1, [[2, 4], [4, 2]])  # Union=12.
        self.put(results, 1, 1, [[3, 3], [0, 0]])  # Union=9.
        curve = compute_hv_curve(results, [0, 0])
        np.testing.assert_array_equal(curve, [[5, 12], [4, 9]])
        self.assertNotEqual(curve[0, 0], 3.)
        pooled = hypervolume_2d_max(results.means_scaled[:, 0].reshape(-1, 2), [0, 0])
        self.assertEqual(pooled, 6.)
        summary = summarize_hv_across_seeds(curve, results.actual_budgets)
        np.testing.assert_array_equal(summary.mean, [4.5, 10.5])
        np.testing.assert_allclose(summary.sd, [np.sqrt(.5), np.sqrt(4.5)])
        np.testing.assert_array_equal(summary.complete_seed_counts, [2, 2])

    # Params: self; returns: None.
    def test_incomplete_policy_sets_and_main_seed_aggregates_stay_nan(self):
        results = ExperimentResults.empty(3, [4096, 8192], 2)
        self.put(results, 0, 0, [[1, 3], [3, 1]])
        results.means_raw[1, 0, 0] = [2, 2]
        results.means_scaled[1, 0, 0] = [2, 2]
        results.validity[1, 0, 0] = True  # A partial set, not a valid main HV.
        self.put(results, 0, 1, [[1, 1], [0, 0]])
        self.put(results, 1, 1, [[2, 2], [0, 0]])
        self.put(results, 2, 1, [[3, 3], [0, 0]])
        curve = compute_hv_curve(results, [0, 0])
        np.testing.assert_array_equal(curve[:, 0], [5, np.nan, np.nan])
        summary = summarize_hv_across_seeds(curve, results.actual_budgets)
        self.assertTrue(np.isnan(summary.mean[0]))
        self.assertTrue(np.isnan(summary.sd[0]))
        self.assertEqual(summary.mean[1], 14/3)
        self.assertAlmostEqual(summary.sd[1], np.std([1., 4., 9.], ddof=1))
        np.testing.assert_array_equal(summary.complete_seed_counts, [1, 3])
        np.testing.assert_array_equal(summary.completeness, [False, True])
        self.assertEqual(summary.declared_seed_count, 3)

    # Params: self; returns: None.
    def test_genuine_zero_returns_are_valid_and_can_contribute(self):
        results = ExperimentResults.empty(2, [4096], 2)
        self.put(results, 0, 0, [[0, 0], [0, 0]])
        self.put(results, 1, 0, [[0, 0], [0, 0]])
        curve = compute_hv_curve(results, [-1, -1])
        np.testing.assert_array_equal(curve, [[1], [1]])
        boundary = compute_hv_curve(results, [0, 0])
        np.testing.assert_array_equal(boundary, [[0], [0]])
        summary = summarize_hv_across_seeds(boundary, results.actual_budgets)
        np.testing.assert_array_equal(summary.mean, [0])
        np.testing.assert_array_equal(summary.sd, [0])
        self.assertTrue(summary.completeness[0])

    # Params: self; returns: None.
    def test_scaled_returns_are_used_and_cached_hv_and_inputs_are_untouched(self):
        results = ExperimentResults.empty(1, [4096], 2)
        self.put(results, 0, 0, [[1, 3], [3, 1]], raw=np.array([[10., 30.], [30., 10.]]))
        results.hypervolume[0, 0] = 999
        before_raw, before_scaled = results.means_raw.copy(), results.means_scaled.copy()
        curve = compute_hv_curve(results, [0, 0])
        np.testing.assert_array_equal(curve, [[5]])
        self.assertEqual(results.hypervolume[0, 0], 999)
        np.testing.assert_array_equal(results.means_raw, before_raw)
        np.testing.assert_array_equal(results.means_scaled, before_scaled)
        self.assertFalse(np.shares_memory(curve, results.hypervolume))

    # Params: self; returns: None.
    def test_single_seed_zero_sd_all_missing_and_original_actual_budget_labels(self):
        values = np.array([[0., np.nan, 7.]])
        budgets = np.array([20480, 40960, 503808])
        summary = summarize_hv_across_seeds(values, budgets)
        np.testing.assert_array_equal(summary.mean, [0, np.nan, 7])
        np.testing.assert_array_equal(summary.sd, [0, np.nan, 0])
        np.testing.assert_array_equal(summary.actual_steps, budgets)
        budgets[0] = 123
        self.assertEqual(summary.actual_steps[0], 20480)
        empty = summarize_hv_across_seeds(np.full((2, 2), np.nan), [4096, 8192])
        self.assertTrue(np.isnan(empty.mean).all())
        self.assertTrue(np.isnan(empty.sd).all())
        np.testing.assert_array_equal(empty.complete_seed_counts, [0, 0])

    # Params: self; returns: None.
    def test_large_finite_summary_avoids_intermediate_overflow(self):
        summary = summarize_hv_across_seeds(np.array([[0.], [1e308]]), [4096])
        self.assertEqual(summary.mean[0], 5e307)
        self.assertAlmostEqual(summary.sd[0] / 1e308, 1/np.sqrt(2))

    # Params: self; returns: None.
    def test_invalid_curve_results_and_reference_fail_even_without_complete_sets(self):
        with self.assertRaises(TypeError):
            compute_hv_curve(np.zeros((1, 1)), [0, 0])
        results = ExperimentResults.empty(1, [4096], 2)
        with self.assertRaises(ValueError):
            compute_hv_curve(results, [np.nan, 0])
        results.validity[0, 0, 0] = True
        with self.assertRaises(ValueError):
            compute_hv_curve(results, [0, 0])

    # Params: self; returns: None.
    def test_invalid_hv_matrices_and_budget_axes_fail(self):
        for hv in (np.zeros(2), np.zeros((0, 2)), np.zeros((2, 0)),
                   np.array([[-1.]]), np.array([[np.inf]])):
            with self.subTest(shape=hv.shape), self.assertRaises(ValueError):
                summarize_hv_across_seeds(hv, [4096])
        for hv in ([[1.]], np.array([[1]]), np.array([[True]])):
            with self.subTest(type=type(hv)), self.assertRaises(TypeError):
                summarize_hv_across_seeds(hv, [4096])
        for budgets, error in (([4096], ValueError), ([4096, 4096], ValueError), ([8192, 4096], ValueError),
                               ([0, 4096], ValueError), ([4096., 8192.], TypeError), ([True, 8192], TypeError)):
            with self.subTest(budgets=budgets), self.assertRaises(error):
                summarize_hv_across_seeds(np.ones((2, 2)), budgets)

    # Params: self; returns: None.
    def test_summary_does_not_mutate_hv_or_replace_missing_seed_values(self):
        values = np.array([[2., np.nan], [4., 1.]])
        before = values.copy()
        values.setflags(write=False)
        summarize_hv_across_seeds(values, [4096, 8192])
        np.testing.assert_array_equal(values, before)


if __name__ == "__main__":
    unittest.main()
