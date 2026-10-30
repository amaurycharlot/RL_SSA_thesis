"""Synthetic Block 03 data/ownership tests; never start workers or train PPO."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import (
    EpisodeTrace, EpisodeMetrics, TargetObservationMetrics, EpisodeEvaluation,
    EvaluationResult, ExperimentResults, ObsNormalizer, TrainingEnvHandle,
    PolicyBundle, PolicyIdentity, ResourceCleanupError, ResumeState,
    PolicyTrainingSummary, ExperimentSummary, ReportManifest, FinalResults, HVSummary,
)


# Params: step, weight, run; returns: PolicyIdentity.
def identity(step=4096, weight=0, run=0):
    return PolicyIdentity("test-fingerprint", run, run, weight, (0.5, 0.5), step, f"step_{step}")


# Params: **changes; returns: dict.
def trace_values(**changes):
    values = dict(initial_observation=np.zeros((2, 3, 3), np.float32),
                  observations=np.ones((2, 2, 3, 3), np.float32),
                  vector_rewards=np.array([[1, 2], [3, 4]], np.float32),
                  true_states=np.zeros((2, 3, 4), np.float64),
                  estimated_states=np.zeros((2, 3, 9), np.float32),
                  actions=np.array([[0, 1], [1, 2]], np.int64),
                  observed_targets=np.array([[True, False, False], [False, True, False]]),
                  elapsed_steps=np.array([1, 2], np.int64), coverage_values=np.array([0.5, 0.7]),
                  terminated=np.array([False, True]), truncated=np.array([False, False]),
                  seed=10, initial_known_count=1)
    return {**values, **changes}


# Params: **changes; returns: EpisodeMetrics.
def metrics(**changes):
    values = dict(mean_positional_error=0.2, mean_final_positional_error=0.1,
                  mean_covariance_trace=0.4, final_coverage=0.7, mean_coverage=0.6,
                  unique_new_detections=1, mean_known_observation_frequency=0.5,
                  mean_observation_frequency=0.5, decisions=2, elapsed_timesteps=3,
                  unavailable_reasons={"search_time_percentage": "PPO has no explicit action mode"})
    return EpisodeMetrics(**{**values, **changes})


class TraceAndEvaluationTypes(unittest.TestCase):
    # Params: self; returns: None.
    def test_initial_and_post_action_axes_and_precision(self):
        values = trace_values()
        trace = EpisodeTrace(**values)
        self.assertEqual(trace.initial_observation.shape, (2, 3, 3))
        self.assertEqual(trace.observations.shape, (2, 2, 3, 3))
        self.assertEqual(trace.decisions, 2)
        self.assertEqual(trace.true_states.dtype, np.float64)
        self.assertEqual(trace.estimated_states.dtype, np.float32)
        for name, value in values.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(trace, name), value)
                value.flat[0] = 0
                self.assertFalse(getattr(trace, name).flags.writeable)
                with self.assertRaises(ValueError):
                    getattr(trace, name).setflags(write=True)
        self.assertEqual(trace.observations.flat[0], 1)

    # Params: self; returns: None.
    def test_mismatched_lengths_objective_shapes_and_dtypes_fail(self):
        cases = [(dict(vector_rewards=np.zeros((2, 3), np.float32)), ValueError),
                 (dict(observations=np.zeros((3, 2, 3, 3), np.float32)), ValueError),
                 (dict(initial_observation=np.zeros((2, 3, 2), np.float32)), ValueError),
                 (dict(initial_observation=np.zeros((2, 3, 3), np.float64)), TypeError),
                 (dict(vector_rewards=np.ones((2, 2), np.float64)), TypeError),
                 (dict(true_states=np.zeros((2, 4, 4))), ValueError),
                 (dict(observed_targets=np.zeros((2, 3), np.int64)), TypeError),
                 (dict(actions=np.zeros((2, 2), np.float32)), TypeError),
                 (dict(elapsed_steps=np.array([1, 0])), ValueError),
                 (dict(actions=np.array([[0, 1], [2, 1]])), ValueError)]
        for changes, error in cases:
            with self.subTest(changes=tuple(changes)), self.assertRaises(error):
                EpisodeTrace(**trace_values(**changes))

    # Params: self; returns: None.
    def test_terminal_alignment_and_missing_estimates(self):
        for changes in (dict(terminated=np.array([True, False])),
                        dict(terminated=np.array([False, False])), dict(initial_known_count=4)):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                EpisodeTrace(**trace_values(**changes))
        trace = EpisodeTrace(**trace_values(terminated=np.array([False, False]),
                                           truncated=np.array([False, True]),
                                           estimated_states=np.full((2, 3, 9), np.nan)))
        self.assertTrue(trace.truncated[-1])
        self.assertTrue(np.isnan(trace.estimated_states).all())

    # Params: self; returns: None.
    def test_unavailable_metrics_have_null_and_explicit_reason(self):
        record = metrics(mean_positional_error=None, unavailable_reasons={
            "mean_positional_error": "No catalogue estimates", "search_time_percentage": "No explicit mode"})
        payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
        self.assertIsNone(payload["mean_positional_error"])
        self.assertEqual(payload["unavailable_reasons"]["mean_positional_error"], "No catalogue estimates")
        with self.assertRaises(TypeError):
            record.unavailable_reasons["mean_positional_error"] = "changed"
        for changes in (dict(mean_positional_error=None), dict(mean_positional_error=float("nan")),
                        dict(mean_observation_frequency=1.2), dict(decisions=0),
                        dict(search_time_percentage=101, unavailable_reasons={})):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                metrics(**changes)

    # Params: self; returns: None.
    def test_target_frequency_exposure_and_undefined_unknowns(self):
        known = TargetObservationMetrics(0, True, 0, 0, 10, 0.0)
        discovered = TargetObservationMetrics(1, False, 2, 8, 4, 0.5)
        never = TargetObservationMetrics(2, False, 0, None, None, None)
        self.assertEqual(known.observation_frequency, 0)
        self.assertEqual(discovered.counting_start_timestep, 8)
        self.assertIsNone(never.observation_frequency)
        for args in ((2, False, 0, 0, 10, 0.0), (0, True, 0, None, None, None),
                     (1, False, 1, 3, 4, 0.5), (0, True, 1, 1, 10, 0.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TargetObservationMetrics(*args)

    # Params: self; returns: None.
    def test_episode_evaluation_return_shape_and_trace_alignment(self):
        record = EpisodeEvaluation(np.array([1., 2.]), np.array([0.5, 1.]), metrics(),
                                   EpisodeTrace(**trace_values()))
        self.assertEqual(record.return_raw.dtype, np.float64)
        with self.assertRaises(ValueError):
            replace(record, return_raw=np.zeros(3))
        with self.assertRaises(ValueError):
            replace(record, metrics=metrics(elapsed_timesteps=4))

    # Params: self, raw, **changes; returns: EvaluationResult.
    def result(self, raw=None, **changes):
        if raw is None:
            raw = np.array([[1., 2.], [3., 4.]])
        scaled = raw / 2
        sd = raw.std(axis=0, ddof=1) if len(raw) > 1 else np.zeros(2)
        values = dict(episode_seeds=tuple(range(10, 10 + len(raw))), returns_raw=raw,
                      returns_scaled=scaled, mean_raw=raw.mean(axis=0), mean_scaled=scaled.mean(axis=0),
                      sd_raw=sd, sd_scaled=sd/2, se_raw=sd/np.sqrt(len(raw)), se_scaled=sd/2/np.sqrt(len(raw)),
                      episode_metrics=tuple(metrics() for _ in raw), deterministic=True,
                      actual_step=4096, identity=identity())
        return EvaluationResult(**{**values, **changes})

    # Params: self; returns: None.
    def test_policy_returns_preserve_episode_means_and_sample_uncertainty(self):
        record = self.result()
        np.testing.assert_array_equal(record.mean_raw, [2, 3])
        np.testing.assert_allclose(record.sd_raw, [np.sqrt(2), np.sqrt(2)])
        np.testing.assert_allclose(record.se_raw, [1, 1])
        single = self.result(raw=np.array([[1., 2.]]))
        np.testing.assert_array_equal(single.sd_raw, [0, 0])
        np.testing.assert_array_equal(single.se_raw, [0, 0])
        for changes in (dict(mean_raw=np.array([999., 999.])), dict(sd_raw=np.zeros(2)),
                        dict(episode_seeds=[10, 10]), dict(actual_step=8192),
                        dict(episode_metrics=(metrics(),)), dict(returns_raw=np.ones((2, 3)))):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                self.result(**changes)

    # Params: self; returns: None.
    def test_optional_traces_have_matching_ordered_seeds(self):
        traces = (EpisodeTrace(**trace_values(seed=10)), EpisodeTrace(**trace_values(seed=11)))
        self.result(traces=traces)
        with self.assertRaises(ValueError):
            self.result(traces=tuple(reversed(traces)))


class ResultContainersAndNormalization(unittest.TestCase):
    # Params: self; returns: None.
    def test_empty_results_are_nan_and_false_with_independent_owned_buffers(self):
        result = ExperimentResults.empty(2, [4096, 8192], 3, 4)
        self.assertEqual(result.means_raw.shape, (2, 2, 3, 2))
        self.assertEqual(result.episode_returns.shape, (2, 2, 3, 4, 2))
        self.assertTrue(np.isnan(result.means_raw).all())
        self.assertTrue(np.isnan(result.means_scaled).all())
        self.assertTrue(np.isnan(result.hypervolume).all())
        self.assertTrue(np.isnan(result.episode_returns).all())
        self.assertFalse(result.validity.any())
        self.assertFalse(np.shares_memory(result.means_raw, result.means_scaled))
        result.means_raw[0, 0, 0] = [1, 2]
        with self.assertRaises(ValueError):
            result.validate()  # Do not accidentally turn a missing slot into zero/finite data.
        result.means_scaled[0, 0, 0] = [1, 2]
        result.episode_returns[0, 0, 0] = [1, 2]
        result.validity[0, 0, 0] = True
        result.identities[(0, 0, 0)] = identity()
        result.validate()
        result.hypervolume[0, 0] = 5
        with self.assertRaises(ValueError):
            result.validate()  # A partial policy set has no main HV.

    # Params: self; returns: None.
    def test_result_constructor_copies_and_rejects_mismatched_axes(self):
        raw = np.full((1, 1, 1, 2), np.nan)
        result = ExperimentResults(raw, raw, np.zeros((1, 1, 1), bool), np.array([4096]), np.full((1, 1), np.nan))
        raw.fill(999)
        self.assertTrue(np.isnan(result.means_raw).all())
        with self.assertRaises(ValueError):
            ExperimentResults(raw, raw, np.zeros((1, 1, 2), bool), np.array([4096]), np.full((1, 1), np.nan))
        for budgets in ([8192, 4096], [4096, 4096], [0]):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                ExperimentResults.empty(1, budgets, 1)

    # Params: self; returns: None.
    def test_final_result_axes_missing_slots_and_complete_policy_front(self):
        empty = FinalResults.empty(2, 3, [20, 21])
        self.assertEqual(empty.episode_returns_raw.shape, (2, 3, 2, 2))
        self.assertTrue(np.isnan(empty.means_raw).all())
        self.assertFalse(empty.validity.any())
        self.assertEqual(empty.front_indices, ((), ()))
        values = np.array([[[[1., 3.], [3., 5.]], [[2., 2.], [4., 4.]]]])
        result = FinalResults(values.mean(axis=2), values.mean(axis=2), values, values,
                              np.ones((1, 2), bool), [20, 21], [(0, 1)], np.array([7.]))
        self.assertEqual(result.front_indices, ((0, 1),))
        with self.assertRaises(ValueError):
            replace(result, front_indices=[(0, 0)])
        with self.assertRaises(ValueError):
            replace(result, means_raw=np.zeros((1, 2, 2)))
        with self.assertRaises(ValueError):
            replace(empty, front_indices=[(0,), ()])

    # Params: self; returns: None.
    def test_hv_summary_requires_all_declared_seeds_and_single_seed_sd_zero(self):
        summary = HVSummary(np.array([4096, 8192]), np.array([np.nan, 7.]),
                            np.array([np.nan, 1.]), np.array([1, 2]), 2, np.array([False, True]))
        self.assertTrue(np.isnan(summary.mean[0]))
        with self.assertRaises(ValueError):
            replace(summary, mean=np.array([6., 7.]))
        with self.assertRaises(ValueError):
            replace(summary, completeness=np.array([True, True]))
        HVSummary(np.array([4096]), np.array([7.]), np.array([0.]), np.array([1]), 1, np.array([True]))

    # Params: self; returns: None.
    def test_normalizer_uses_copied_immutable_observation_statistics_only(self):
        mean, variance = np.ones((2, 3, 3)), np.full((2, 3, 3), 4.)
        normalizer = ObsNormalizer(mean, variance, 100., epsilon=1., clip_obs=1.)
        obs = np.full((2, 3, 3), 3., dtype=np.float32)
        mean.fill(999)
        variance.fill(999)
        output = normalizer.normalize(obs)
        np.testing.assert_allclose(output, np.float32(2/np.sqrt(5)))
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(normalizer.count, 100)
        np.testing.assert_array_equal(normalizer.mean, np.ones((2, 3, 3)))
        clipped = normalizer.normalize(np.full((2, 3, 3), 100, np.float32))
        np.testing.assert_array_equal(clipped, np.ones((2, 3, 3)))
        with self.assertRaises(ValueError):
            normalizer.mean.setflags(write=True)
        disabled = replace(normalizer, enabled=False)
        unnormalized = disabled.normalize(obs)
        np.testing.assert_array_equal(unnormalized, obs)
        self.assertFalse(np.shares_memory(unnormalized, obs))
        self.assertNotIn("reward", normalizer.__dict__)
        for changes in (dict(variance=np.full((2, 3, 3), -1.)), dict(epsilon=0), dict(count=-1)):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                replace(normalizer, **changes)


class Child:
    # Params: self, venv, fail; returns: None.
    def __init__(self, venv=None, fail=False):
        self.venv, self.fail, self.calls = venv, fail, 0

    # Params: self; returns: None.
    def close(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected cleanup failure")
        if self.venv is not None:
            self.venv.close()


class Worker:
    # Params: self, fail; returns: None.
    def __init__(self, fail=False):
        self.alive, self.joins, self.terminations, self.fail = True, 0, 0, fail

    # Params: self, timeout; returns: None.
    def join(self, timeout):
        self.joins += 1

    # Params: self; returns: self.alive.
    def is_alive(self):
        return self.alive

    # Params: self; returns: None.
    def terminate(self):
        self.terminations += 1
        if self.fail:
            raise RuntimeError("injected worker cleanup failure")
        self.alive = False


class ResourceOwnership(unittest.TestCase):
    # Params: self; returns: None.
    def test_normal_wrapper_chain_and_repeated_close_are_safe(self):
        raw = Child()
        normalization = Child(raw)
        outer = Child(normalization)
        handle = TrainingEnvHandle(outer, normalization, raw)
        with handle:
            self.assertFalse(handle.closed)
        handle.close()
        self.assertEqual((outer.calls, normalization.calls, raw.calls), (1, 1, 1))
        self.assertTrue(handle.closed)
        with self.assertRaises(RuntimeError):
            with handle:
                pass

    # Params: self; returns: None.
    def test_partial_construction_closes_registered_children_on_failure(self):
        raw, worker = Child(), Worker()
        handle = TrainingEnvHandle()
        with self.assertRaisesRegex(RuntimeError, "construction failure"):
            with handle:
                handle.raw_vec_env = raw
                handle.workers = (worker,)
                raise RuntimeError("construction failure")
        self.assertEqual(raw.calls, 1)
        self.assertFalse(worker.alive)
        self.assertEqual(worker.terminations, 1)
        handle.close()
        self.assertEqual(worker.terminations, 1)

    # Params: self; returns: None.
    def test_failed_outer_close_still_attempts_all_children_and_workers(self):
        raw, worker = Child(), Worker()
        outer = Child(raw, fail=True)
        handle = TrainingEnvHandle(outer, None, raw, (worker,))
        with self.assertRaises(ResourceCleanupError) as caught:
            handle.close()
        self.assertEqual(len(caught.exception.errors), 1)
        self.assertEqual(raw.calls, 1)
        self.assertFalse(worker.alive)
        handle.close()
        self.assertEqual(outer.calls, 1)

    # Params: self; returns: None.
    def test_one_failed_worker_does_not_prevent_other_cleanup(self):
        failed, healthy = Worker(fail=True), Worker()
        with self.assertRaises(ResourceCleanupError):
            TrainingEnvHandle(workers=(failed, healthy)).close()
        self.assertEqual(healthy.terminations, 1)
        self.assertFalse(healthy.alive)

    # Params: self; returns: None.
    def test_bundle_closes_owned_environment_and_preserves_borrowed_normalizer(self):
        borrowed, owned = Child(), Child()
        bundle = PolicyBundle(model=object(), vec_normalize=borrowed, identity=identity(),
                              config_fingerprint="test-fingerprint", normalization_env=owned,
                              metadata={"scales": [1, 1]})
        with bundle:
            pass
        bundle.close()
        self.assertEqual(owned.calls, 1)
        self.assertEqual(borrowed.calls, 0)
        self.assertEqual(bundle.metadata["scales"], (1, 1))
        partial_child = Child()
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with PolicyBundle() as partial:
                partial.normalization_env = partial_child
                raise RuntimeError("load failed")
        self.assertEqual(partial_child.calls, 1)


class SummaryAndResumeTypes(unittest.TestCase):
    # Params: self; returns: None.
    def test_resume_state_pending_budgets_and_attempt_identity(self):
        state = ResumeState(False, 4096, 8192, "checkpoint", "checkpoint", ("eval1",),
                            (8192,), (8192,), "attempt2", ("attempt1",), ("superseded",))
        self.assertEqual(state.selected_bundle_path, Path("checkpoint"))
        json.dumps(state.to_dict(), allow_nan=False)
        for changes in (dict(next_evaluation_budgets=(4096,)), dict(complete=True),
                        dict(superseded_attempt_ids=("attempt1",))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(state, **changes)
        ResumeState(True, 8192, 8192)

    # Params: self; returns: None.
    def test_policy_and_experiment_summary_completion_and_metadata(self):
        summary = PolicyTrainingSummary(identity(), 4000, 4096, 4096, True, "final_bundle",
                                        ("eval1",), resume_metadata={"attempts": ["first"]})
        experiment = ExperimentSummary("test-fingerprint", "test_dir", (summary,), 1, 1, 4096)
        json.dumps(experiment.to_dict(), allow_nan=False)
        self.assertEqual(summary.resume_metadata["attempts"], ("first",))
        with self.assertRaises(ValueError):
            replace(summary, actual_transitions=0)
        with self.assertRaises(ValueError):
            replace(summary, final_bundle_path=None)
        with self.assertRaises(ValueError):
            replace(experiment, completed_policies=0)

    # Params: self; returns: None.
    def test_report_manifest_relative_paths_and_immutable_hashes(self):
        hashes = {"plots/hv.png": "a" * 64}
        manifest = ReportManifest(("plots/hv.png",), hashes, "test-fingerprint", (4096,), (2,), "2026-10-14T00:00:00Z")
        hashes["plots/hv.png"] = "b" * 64
        self.assertEqual(manifest.file_hashes["plots/hv.png"], "a" * 64)
        json.dumps(manifest.to_dict(), allow_nan=False)
        for changes in (dict(relative_paths=("../escape",)), dict(file_hashes={"plots/hv.png": "bad"}),
                        dict(seed_counts=()), dict(represented_budgets=(0,))):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                replace(manifest, **changes)


if __name__ == "__main__":
    unittest.main()
