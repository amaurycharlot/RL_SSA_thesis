"""Block 02 record contracts: no environment allocation or PPO training."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from agents.mo import (
    BudgetPlan, EnvConfig, ExecutionConfig, ExperimentConfig, LoggingConfig,
    NormalizationConfig, PolicyIdentity, PPOConfig, SeedPlan,
)
from agents.mo.config import (
    ConfigError, load_config, validate_config, make_weights, validate_weights,
    make_seed_plan, make_budget_plan, next_scheduled_step, training_transition_cost,
)


class ConfigurationRecords(unittest.TestCase):
    # Params: self; returns: None.
    def test_preserved_defaults_and_invariants(self):
        config = ExperimentConfig()
        self.assertEqual(config.env.reward_pair, ("R6", "R2"))
        self.assertEqual(config.reward_pair_id, "R6_R2")
        self.assertEqual(config.env.observation_shape, (10, 10, 3))
        self.assertEqual(config.env.observation_dtype, "float32")
        self.assertEqual(config.env.action_nvec, (10, 10))
        self.assertEqual(config.env.action_type, "MultiDiscrete")
        self.assertEqual(config.env.mode, "combined")
        self.assertFalse(config.env.simple_obs)
        self.assertEqual(config.num_objectives, 2)
        self.assertEqual(config.reward_order, ("tracking", "search"))
        self.assertEqual(config.ppo.rollout_size, 4096)
        self.assertEqual(config.ppo.n_epochs, 10)
        self.assertEqual(config.ppo.features_dim, 128)
        self.assertEqual(config.ppo.policy, "MlpPolicy")
        self.assertTrue(config.execution.subproc)
        self.assertEqual(config.execution.worker_threads, 1)
        self.assertFalse(config.normalization.norm_obs)
        self.assertFalse(config.normalization.norm_reward)
        self.assertEqual(config.normalization.gamma, config.ppo.gamma)
        self.assertEqual(config.logging.mode, "off")
        self.assertIsNone(config.reference_scaled)
        self.assertEqual(config.reference_status, "unselected")
        self.assertEqual(config.run_seeds, tuple(range(10)))
        self.assertEqual(config.n_weights, 11)

    # Params: self; returns: None.
    def test_reward_pair_is_derived_and_not_an_independent_setting(self):
        config = ExperimentConfig(env=EnvConfig(reward_type="R5", search_reward="R1"))
        self.assertEqual(config.reward_pair, ("R5", "R1"))
        self.assertEqual(config.reward_pair_id, "R5_R1")
        self.assertNotIn("reward_pair", config.to_dict())
        with self.assertRaises(TypeError):
            ExperimentConfig(reward_pair="R6_R2")

    # Params: self; returns: None.
    def test_input_lists_and_nested_settings_are_immutable(self):
        seeds, scales, shape = [0, 1], [1.0, 2.0], [8, 9]
        config = ExperimentConfig(run_seeds=seeds, objective_scales=scales,
                                  env=EnvConfig(grid_shape=shape))
        seeds[0] = 7
        scales[0] = 4
        shape[0] = 6
        self.assertEqual(config.run_seeds, (0, 1))
        self.assertEqual(config.objective_scales, (1, 2))
        self.assertEqual(config.env.grid_shape, (8, 9))
        with self.assertRaises(FrozenInstanceError):
            config.env.known_targets = 99
        with self.assertRaises(FrozenInstanceError):
            config.ppo.gamma = 0.5
        with self.assertRaises(FrozenInstanceError):
            config.execution.subproc = False
        payload = config.to_dict()
        payload["env"]["grid_shape"][0] = 123
        self.assertEqual(config.env.grid_shape, (8, 9))

    # Params: self; returns: None.
    def test_json_serialization_is_primitive_and_lossless(self):
        config = ExperimentConfig(output_dir=Path("outputs/test"),
                                  reference_scaled=[-123.5, 2], reference_status="frozen")
        payload = json.loads(json.dumps(config.to_dict(), allow_nan=False))
        self.assertEqual(payload["env"]["grid_shape"], [10, 10])
        self.assertEqual(payload["output_dir"], str(Path("outputs/test")))
        self.assertEqual(payload["normalization"]["gamma"], 0.99)
        self.assertEqual(payload["reference_scaled"], [-123.5, 2])
        self.assertEqual(payload["ppo"]["normalize_advantage"], True)
        self.assertEqual(payload["ppo"]["target_kl"], None)
        self.assertNotIn("mode", payload["env"])
        self.assertNotIn("num_objectives", payload)
        restored = ExperimentConfig(**{
            **payload, "env": EnvConfig(**payload["env"]), "ppo": PPOConfig(**payload["ppo"]),
            "execution": ExecutionConfig(**payload["execution"]),
            "normalization": NormalizationConfig(**payload["normalization"]),
            "logging": LoggingConfig(**payload["logging"]),
        })
        self.assertEqual(restored, config)

    # Params: self; returns: None.
    def test_normalization_gamma_resolves_without_mutating_input(self):
        normalization = NormalizationConfig()
        config = ExperimentConfig(ppo=PPOConfig(gamma=0.9), normalization=normalization)
        self.assertEqual(config.normalization.gamma, 0.9)
        self.assertIsNone(normalization.gamma)
        with self.assertRaises(ValueError):
            ExperimentConfig(normalization=NormalizationConfig(gamma=0.8))

    # Params: self; returns: None.
    def test_basic_invalid_environment_fields(self):
        cases = [({"known_targets": True}, TypeError), ({"known_targets": -1}, ValueError),
                 ({"known_targets": 0, "unknown_targets": 0}, ValueError),
                 ({"grid_shape": [10, 0]}, ValueError), ({"grid_shape": [10]}, ValueError),
                 ({"grid_shape": [10, 2.5]}, TypeError), ({"episode_steps": 0}, ValueError),
                 ({"dwell_steps": False}, TypeError), ({"zero_noise": "false"}, TypeError),
                 ({"measurement_noise_std": float("nan")}, ValueError),
                 ({"measurement_noise_std": -1}, ValueError), ({"slew_alpha": -1}, ValueError),
                 ({"uncertainty_norm_scale": 0}, ValueError), ({"reward_type": ""}, ValueError)]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                EnvConfig(**kwargs)
        for kwargs in ({"mode": "search"}, {"simple_obs": True}, {"omega0": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                EnvConfig(**kwargs)

    # Params: self; returns: None.
    def test_basic_invalid_ppo_fields(self):
        cases = [({"n_envs": True}, TypeError), ({"n_epochs": 0}, ValueError),
                 ({"n_steps": 0}, ValueError), ({"batch_size": 257}, ValueError),
                 ({"learning_rate": lambda _: 0.01}, TypeError),
                 ({"clip_range": lambda _: 0.2}, TypeError), ({"gamma": 0}, ValueError),
                 ({"gamma": 1.1}, ValueError), ({"gae_lambda": -0.1}, ValueError),
                 ({"clip_range": 1}, ValueError), ({"max_grad_norm": 0}, ValueError),
                 ({"ent_coef": -1}, ValueError), ({"target_kl": float("inf")}, ValueError),
                 ({"normalize_advantage": 1}, TypeError), ({"use_sde": True}, ValueError),
                 ({"device": None}, TypeError), ({"verbose": 3}, ValueError)]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                PPOConfig(**kwargs)

    # Params: self; returns: None.
    def test_basic_invalid_experiment_and_nested_fields(self):
        cases = [({"schema_version": 2}, ValueError), ({"experiment_seed": -1}, ValueError),
                 ({"run_seeds": [0, 0]}, ValueError), ({"run_seeds": []}, ValueError),
                 ({"run_seeds": [True]}, TypeError), ({"eval_seed_root": 2**32}, ValueError),
                 ({"n_weights": 1}, ValueError), ({"reward_order": ["search", "tracking"]}, ValueError),
                 ({"objective_scales": [1, 2, 3]}, ValueError),
                 ({"objective_scales": [1, 0]}, ValueError),
                 ({"reference_scaled": [-1, -2]}, ValueError),
                 ({"reference_status": "frozen"}, ValueError),
                 ({"requested_timesteps_per_policy": 0}, ValueError),
                 ({"eval_episodes": 0}, ValueError), ({"error_target_set": "all"}, ValueError),
                 ({"env": {}}, TypeError), ({"ppo": {}}, TypeError), ({"output_dir": 123}, TypeError)]
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                ExperimentConfig(**kwargs)
        for factory, kwargs, error in [(ExecutionConfig, {"subproc": 1}, TypeError),
                                       (ExecutionConfig, {"worker_threads": 0}, ValueError),
                                       (NormalizationConfig, {"epsilon": 0}, ValueError),
                                       (NormalizationConfig, {"norm_reward": "false"}, TypeError),
                                       (LoggingConfig, {"mode": "bad"}, ValueError)]:
            with self.subTest(factory=factory, kwargs=kwargs), self.assertRaises(error):
                factory(**kwargs)


class PlanAndIdentityRecords(unittest.TestCase):
    # Params: self, **changes; returns: SeedPlan.
    def seed_plan(self, **changes):
        values = dict(run_seed=0, run_index=0, learner_seed=100, base_worker_seed=100,
                      worker_seeds=[100, 101], eval_seeds=[1000, 1001], final_seeds=[2000, 2001])
        return SeedPlan(**{**values, **changes})

    # Params: self, **changes; returns: BudgetPlan.
    def budget_plan(self, **changes):
        values = dict(rollout_size=4096, requested_total=10000, effective_total=12288,
                      eval_steps=[4096, 8192, 12288], checkpoint_steps=[12288],
                      eval_schedule=[[2000, 4096], [4000, 4096], [6000, 8192], [10000, 12288]],
                      checkpoint_schedule=[[10000, 12288]])
        return BudgetPlan(**{**values, **changes})

    # Params: self, **changes; returns: PolicyIdentity.
    def identity(self, **changes):
        values = dict(experiment_fingerprint="captured-test-fingerprint", run_seed=0, run_index=0,
                      weight_index=3, weight=[0.3, 0.7], actual_step=4096, bundle_id="step_4096")
        return PolicyIdentity(**{**values, **changes})

    # Params: self; returns: None.
    def test_seed_plan_serialization_and_domains(self):
        plan = self.seed_plan()
        self.assertEqual(plan.worker_seeds, (100, 101))
        self.assertEqual(SeedPlan(**json.loads(json.dumps(plan.to_dict()))), plan)
        for changes in ({"worker_seeds": [100, 102]}, {"eval_seeds": [100, 1000]},
                        {"final_seeds": [1000, 2000]}, {"learner_seed": 101},
                        {"worker_seeds": [100, 2**32]}, {"final_seeds": [2, 2]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.seed_plan(**changes)
        with self.assertRaises(TypeError):
            self.seed_plan(run_index=True)

    # Params: self; returns: None.
    def test_budget_mapping_and_serialization(self):
        plan = self.budget_plan()
        self.assertEqual(plan.eval_schedule[0:2], ((2000, 4096), (4000, 4096)))
        self.assertEqual(BudgetPlan(**json.loads(json.dumps(plan.to_dict()))), plan)
        with self.assertRaises(FrozenInstanceError):
            plan.effective_total = 999
        for changes in ({"effective_total": 10000}, {"eval_steps": [8192, 4096, 12288]},
                        {"eval_steps": [4096, 4096, 12288]}, {"checkpoint_steps": [4096]},
                        {"eval_schedule": [[2000, 8192], [10000, 12288]]},
                        {"eval_schedule": [[2000, 4096], [10000, 12288]]},
                        {"checkpoint_schedule": []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.budget_plan(**changes)
        with self.assertRaises(TypeError):
            self.budget_plan(eval_schedule=[123])

    # Params: self; returns: None.
    def test_identity_exact_weights_and_serialization(self):
        weights = [0.3, 0.7]
        identity = self.identity(weight=weights)
        weights[0] = 0.9
        self.assertEqual(identity.weight, (0.3, 0.7))
        self.assertEqual(PolicyIdentity(**json.loads(json.dumps(identity.to_dict()))), identity)
        self.assertEqual(self.identity(weight=[1, 0]).weight, (1, 0))
        for changes in ({"weight": [0.3, 0.8]}, {"weight": [-0.1, 1.1]},
                        {"weight": [float("nan"), 1]}, {"weight": [1, 0, 0]},
                        {"actual_step": -1}, {"weight_index": -1}, {"bundle_id": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.identity(**changes)


class PassiveImports(unittest.TestCase):
    # Params: self; returns: None.
    def test_fresh_import_and_default_construction_have_no_external_side_effects(self):
        script = '''
import builtins, dataclasses, math, multiprocessing, pathlib, socket, subprocess, sys, typing
from unittest.mock import patch
def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden side effect during import or record construction")
with patch.object(builtins, "open", forbidden), \\
     patch.object(pathlib.Path, "open", forbidden), \\
     patch.object(pathlib.Path, "mkdir", forbidden), \\
     patch.object(multiprocessing.process.BaseProcess, "start", forbidden), \\
     patch.object(subprocess, "Popen", forbidden), \\
     patch.object(socket.socket, "connect", forbidden):
    import agents.mo
    config = agents.mo.ExperimentConfig()
    payload = config.to_dict()
assert not any(name in sys.modules for name in
               ("numpy", "torch", "scipy", "gymnasium", "stable_baselines3"))
assert payload["normalization"]["gamma"] == 0.99
print("passive import verified")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        with tempfile.TemporaryDirectory(prefix="mo_passive_import_") as output:
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=output,
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("passive import verified", result.stdout)
            self.assertEqual(list(Path(output).iterdir()), [])


class ConfigLoadingAndValidation(unittest.TestCase):
    # Params: self, data, overrides; returns: load_config result.
    def load_payload(self, data, overrides=None):
        with tempfile.TemporaryDirectory(prefix="mo_config_") as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path, overrides)

    # Params: self; returns: None.
    def test_example_matches_specification_and_loads_only_for_preflight(self):
        path = PROJECT_ROOT / "configs/ppo_mo_r6_r2.json"
        config = load_config(path)
        self.assertEqual(config, ExperimentConfig())
        validate_config(config, require_frozen_objectives=False)
        with self.assertRaisesRegex(ConfigError, "selected reference_scaled"):
            validate_config(config)
        self.assertIsNone(config.reference_scaled)
        self.assertEqual(config.objective_scales, (1, 1))
        self.assertEqual(config.reward_pair_id, "R6_R2")

    # Params: self; returns: None.
    def test_frozen_reference_accepts_either_sign_without_inventing_values(self):
        for reference in ([-10., -20.], [1., 2.]):
            with self.subTest(reference=reference):
                config = self.load_payload({"reference_scaled": reference, "reference_status": "frozen"})
                validate_config(config)
                self.assertEqual(config.reference_scaled, tuple(reference))

    # Params: self; returns: None.
    def test_defaults_resolve_all_record_fields_and_normalization_gamma(self):
        config = self.load_payload({"ppo": {"gamma": 0.9}, "normalization": {"norm_obs": True}})
        self.assertEqual(config.normalization.gamma, 0.9)
        payload = config.to_dict()
        json.dumps(payload, allow_nan=False)
        for key, expected in {"n_epochs": 10, "clip_range_vf": None, "normalize_advantage": True,
                              "target_kl": None, "use_sde": False, "sde_sample_freq": -1,
                              "stats_window_size": 100, "verbose": 1, "device": "auto"}.items():
            self.assertIn(key, payload["ppo"])
            self.assertEqual(payload["ppo"][key], expected)

    # Params: self; returns: None.
    def test_unknown_keys_are_rejected_at_every_level(self):
        for section in (None, "env", "ppo", "execution", "normalization", "logging"):
            data = {"typo": 1} if section is None else {section: {"typo": 1}}
            with self.subTest(section=section), self.assertRaisesRegex(ConfigError, "Unknown"):
                self.load_payload(data)
        for data in ({"num_objectives": 3}, {"reward_pair": "R6_R2"}, {"env": {"mode": "search"}},
                     {"env": {"simple_obs": True}}, {"env": {"omega0": 0.1}},
                     {"ppo": {"features_dim": 64}}, {"output_dir": "runs"}):
            with self.subTest(data=data), self.assertRaisesRegex(ConfigError, "Unknown"):
                self.load_payload(data)

    # Params: self; returns: None.
    def test_input_types_and_numerical_bounds_fail_with_field_context(self):
        cases = [("env", "known_targets", True), ("env", "known_targets", -1),
                 ("env", "episode_steps", 0), ("env", "episode_steps", 1.5),
                 ("env", "measurement_noise_std", -1), ("env", "grid_shape", [10, 0]),
                 ("env", "grid_shape", [10, 10, 3]), ("env", "zero_noise", "false"),
                 ("ppo", "n_envs", 0), ("ppo", "batch_size", 257), ("ppo", "n_epochs", 0),
                 ("ppo", "learning_rate", "linear"), ("ppo", "learning_rate", 0),
                 ("ppo", "clip_range", {"schedule": "linear"}), ("ppo", "clip_range", 1),
                 ("ppo", "gamma", 0), ("ppo", "gae_lambda", 1.1), ("ppo", "ent_coef", -1),
                 ("ppo", "max_grad_norm", 0), ("execution", "worker_threads", 0),
                 ("normalization", "epsilon", 0), ("normalization", "clip_reward", -1)]
        for section, key, value in cases:
            with self.subTest(section=section, key=key), self.assertRaises(ConfigError) as caught:
                self.load_payload({section: {key: value}})
            self.assertIn(section, str(caught.exception))
        for data in ({"run_seeds": [1, 1]}, {"run_seeds": [-1]}, {"reward_order": ["search", "tracking"]},
                     {"objective_scales": [0, 1]}, {"objective_scales": [1, 2, 3]},
                     {"reference_scaled": [1, 2, 3], "reference_status": "frozen"},
                     {"requested_timesteps_per_policy": 0}, {"eval_interval": 0},
                     {"eval_episodes": 0}, {"schema_version": 2}):
            with self.subTest(data=data), self.assertRaises(ConfigError):
                self.load_payload(data)

    # Params: self; returns: None.
    def test_object_roots_and_nested_sections_are_required(self):
        for data in ([], None, 123, {"env": []}, {"normalization": None}):
            with self.subTest(data=data), self.assertRaises(ConfigError):
                self.load_payload(data)

    # Params: self; returns: None.
    def test_duplicate_and_nonfinite_json_and_malformed_files_fail(self):
        invalid = ('{"schema_version":1,"schema_version":2}',
                   '{"ppo":{"gamma":0.9,"gamma":0.8}}',
                   '{"objective_scales":[NaN,1]}', '{"ppo":{"gamma":Infinity}}',
                   '{"ppo":{"learning_rate":1e999}}', '{')
        with tempfile.TemporaryDirectory(prefix="mo_bad_json_") as directory:
            path = Path(directory) / "bad.json"
            for raw in invalid:
                with self.subTest(raw=raw), self.assertRaises(ConfigError):
                    path.write_text(raw, encoding="utf-8")
                    load_config(path)
            with self.assertRaisesRegex(ConfigError, "Cannot read"):
                load_config(Path(directory) / "missing.json")

    # Params: self; returns: None.
    def test_permitted_overrides_are_copied_and_paths_resolve_without_creation(self):
        with tempfile.TemporaryDirectory(prefix="mo_output_") as directory:
            output = Path(directory) / "not_created"
            overrides = {"output_dir": output, "ppo": {"device": "cpu"}}
            config = self.load_payload({}, overrides)
            self.assertEqual(config.output_dir, output.resolve())
            self.assertEqual(config.ppo.device, "cpu")
            self.assertFalse(output.exists())
            self.assertEqual(overrides, {"output_dir": output, "ppo": {"device": "cpu"}})
        config = self.load_payload({}, {"output_dir": "relative_output"})
        self.assertEqual(config.output_dir, (Path.cwd() / "relative_output").resolve())

    # Params: self; returns: None.
    def test_scientific_and_invalid_overrides_are_rejected(self):
        for overrides in ({"env": {"known_targets": 2}}, {"run_seeds": [9]},
                          {"reference_scaled": [-1, -2]}, {"objective_scales": [2, 3]},
                          {"ppo": {"gamma": 0.8}}, {"ppo": {"device": "cpu", "typo": 1}},
                          {"normalization": {"norm_reward": True}}, {"device": "cpu"},
                          {"output_dir": ""}, {"output_dir": 123}, {"ppo": None}, []):
            with self.subTest(overrides=overrides), self.assertRaises(ConfigError):
                self.load_payload({}, overrides)
        with self.assertRaisesRegex(ConfigError, "Unknown"):
            self.load_payload({"ppo": {"typo": 1}}, {"ppo": {"device": "cpu"}})

    # Params: self; returns: None.
    def test_reward_names_match_safe_combined_implementations(self):
        for tracking in ("R1", "R2", "R3", "R6", "R7"):
            for searching in ("R1", "R2", "R3", "R4", "R5"):
                with self.subTest(tracking=tracking, searching=searching):
                    config = self.load_payload({"env": {"reward_type": tracking, "search_reward": searching}})
                    self.assertEqual(config.reward_pair, (tracking, searching))
        for data in ({"env": {"reward_type": "R4"}}, {"env": {"reward_type": "R5"}},
                     {"env": {"reward_type": "R99"}}, {"env": {"search_reward": "R6"}}):
            with self.subTest(data=data), self.assertRaises(ConfigError):
                self.load_payload(data)

    # Params: self; returns: None.
    def test_normalization_gamma_and_seed_roots_are_consistent(self):
        config = self.load_payload({"ppo": {"gamma": 0.9}, "normalization": {"norm_reward": True, "gamma": 0.9}})
        self.assertEqual(config.normalization.gamma, config.ppo.gamma)
        for data in ({"ppo": {"gamma": 0.9}, "normalization": {"gamma": 0.8}},
                     {"eval_seed_root": 100, "final_seed_root": 100}):
            with self.subTest(data=data), self.assertRaises(ConfigError):
                self.load_payload(data)

    # Params: self; returns: None.
    def test_scale_labels_are_truthful_and_strict_training_requires_freezing(self):
        with self.assertRaisesRegex(ConfigError, "identity_baseline_frozen"):
            self.load_payload({"objective_scales": [2, 3]})
        config = self.load_payload({"objective_scales": [2, 3], "scale_status": "unselected",
                                   "reference_scaled": [-1, -2], "reference_status": "frozen"})
        with self.assertRaisesRegex(ConfigError, "frozen objective scales"):
            validate_config(config)
        validate_config(replace(config, scale_status="frozen"))
        with self.assertRaises(ConfigError):
            self.load_payload({"scale_status": "typo"})

    # Params: self; returns: None.
    def test_device_syntax_is_checked_without_loading_torch(self):
        for device in ("auto", "cpu", "cuda", "cuda:0", "mps", "xpu:0"):
            self.load_payload({"ppo": {"device": device}})
        for device in ("banana", "cuda:-1", "cuda:0junk", ""):
            with self.subTest(device=device), self.assertRaises(ConfigError):
                self.load_payload({"ppo": {"device": device}})

    # Params: self; returns: None.
    def test_external_logging_requires_an_owner_supplied_project(self):
        with self.assertRaisesRegex(ConfigError, "supplied project"):
            self.load_payload({"logging": {"mode": "online"}})
        config = self.load_payload({"logging": {"mode": "offline", "project": "local-test"}})
        self.assertEqual(config.logging.project, "local-test")

    # Params: self; returns: None.
    def test_validator_boundary_rejects_invalid_record_and_argument_types(self):
        with self.assertRaises(TypeError):
            validate_config({})
        with self.assertRaises(TypeError):
            validate_config(ExperimentConfig(), require_frozen_objectives=1)
        malformed = ExperimentConfig()
        object.__setattr__(malformed.ppo, "gamma", 0)
        with self.assertRaisesRegex(ConfigError, "ppo"):
            validate_config(malformed, require_frozen_objectives=False)

    # Params: self; returns: None.
    def test_loading_and_preflight_have_no_training_imports_or_file_writes(self):
        script = '''
import builtins, json, pathlib, sys
from unittest.mock import patch
from agents.mo.config import load_config, validate_config
def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected output write")
with patch.object(pathlib.Path, "mkdir", forbidden), patch.object(pathlib.Path, "write_text", forbidden):
    config = load_config(sys.argv[1])
    validate_config(config, require_frozen_objectives=False)
assert not any(name in sys.modules for name in
               ("numpy", "torch", "scipy", "gymnasium", "stable_baselines3"))
assert config.reference_scaled is None
print("preflight loading is passive")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run([sys.executable, "-B", "-c", script,
                                 str(PROJECT_ROOT / "configs/ppo_mo_r6_r2.json")],
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("preflight loading is passive", result.stdout)


class PreferenceGenerationAndValidation(unittest.TestCase):
    # Params: self; returns: None.
    def test_eleven_ordered_float32_weights_and_extreme_policies(self):
        import numpy as np
        weights = make_weights()
        self.assertEqual(weights.shape, (11, 2))
        self.assertEqual(weights.dtype, np.float32)
        np.testing.assert_array_equal(weights[0], [0, 1])
        np.testing.assert_array_equal(weights[-1], [1, 0])
        np.testing.assert_array_equal(weights[:, 0], np.asarray(
            [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.], dtype=np.float32))
        np.testing.assert_allclose(weights[:, 1], [1., .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.],
                                   rtol=0, atol=6e-8)
        self.assertTrue(np.all(np.diff(weights[:, 0]) > 0))
        self.assertTrue(np.all(np.diff(weights[:, 1]) < 0))
        np.testing.assert_allclose(weights.sum(axis=1, dtype=np.float64), 1., rtol=0, atol=1e-6)
        config = load_config(PROJECT_ROOT / "configs/ppo_mo_r6_r2.json")
        np.testing.assert_array_equal(weights, make_weights(config.n_weights))

    # Params: self; returns: None.
    def test_generation_is_exactly_repeatable_and_does_not_consume_rng(self):
        import numpy as np
        state = np.random.get_state()
        first, second = make_weights(), make_weights()
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertFalse(np.shares_memory(first, second))
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        first[0] = [1, 0]
        np.testing.assert_array_equal(second[0], [0, 1])

    # Params: self; returns: None.
    def test_generation_accepts_other_sizes_and_rejects_invalid_counts(self):
        import numpy as np
        np.testing.assert_array_equal(make_weights(2), [[0, 1], [1, 0]])
        np.testing.assert_array_equal(make_weights(3), [[0, 1], [.5, .5], [1, 0]])
        for count in (0, 1, -1):
            with self.subTest(count=count), self.assertRaises(ConfigError):
                make_weights(count)
        for count in (True, 11., "11", None, np.int64(11)):
            with self.subTest(count=count), self.assertRaises(TypeError):
                make_weights(count)

    # Params: self; returns: None.
    def test_validation_preserves_saved_order_values_and_owns_copy(self):
        import numpy as np
        source = make_weights()[::-1]  # Noncontiguous input in saved order.
        expected = source.copy()
        source.setflags(write=False)
        result = validate_weights(source)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertFalse(np.shares_memory(source, result))
        self.assertTrue(result.flags.c_contiguous)
        result[0] = [0, 1]
        np.testing.assert_array_equal(source[0], [1, 0])

    # Params: self; returns: None.
    def test_sum_tolerance_is_absolute_and_never_renormalizes(self):
        import numpy as np
        source = np.asarray([[.5, .5000005]], dtype=np.float32)
        self.assertNotEqual(float(source.sum(dtype=np.float64)), 1.)
        result = validate_weights(source)
        self.assertEqual(result.tobytes(), source.tobytes())
        for source in (np.asarray([[.5, .500002]], dtype=np.float32),
                       np.asarray([[.5, .499998]], dtype=np.float32)):
            before = source.tobytes()
            with self.assertRaises(ConfigError):
                validate_weights(source)
            self.assertEqual(before, source.tobytes())

    # Params: self; returns: None.
    def test_duplicate_negative_nonfinite_and_malformed_weights_fail(self):
        import numpy as np
        invalid = [np.asarray([[.5, .5], [.5, .5]], dtype=np.float32),
                   np.asarray([[-1e-8, 1.]], dtype=np.float32),
                   np.asarray([[np.nan, 1.]], dtype=np.float32),
                   np.asarray([[np.inf, 0.]], dtype=np.float32),
                   np.asarray([[0., -np.inf]], dtype=np.float32),
                   np.zeros((0, 2), np.float32), np.zeros((2, 3), np.float32),
                   np.asarray([0., 1.], dtype=np.float32), np.zeros((1, 1, 2), np.float32),
                   np.asarray([[0., 0.]], dtype=np.float32)]
        for source in invalid:
            before = source.tobytes()
            with self.subTest(shape=source.shape), self.assertRaises(ConfigError):
                validate_weights(source)
            self.assertEqual(before, source.tobytes())
        for source in ([[0, 1]], np.asarray([[0, 1]], dtype=np.int64),
                       np.asarray([[.5, .5]], dtype=np.float64), np.asarray([[True, False]])):
            with self.subTest(type=type(source)), self.assertRaises(TypeError):
                validate_weights(source)

    # Params: self; returns: None.
    def test_distinct_nearby_preferences_remain_distinct(self):
        import numpy as np
        source = np.asarray([[.5, .5], [.5000001, .4999999]], dtype=np.float32)
        np.testing.assert_array_equal(validate_weights(source), source)
        self.assertEqual(validate_weights(source[:1]).shape, (1, 2))

    # Params: self; returns: None.
    def test_saved_roundtrip_keeps_exact_policy_index_values(self):
        import numpy as np
        weights = make_weights()
        with tempfile.TemporaryDirectory(prefix="mo_weight_roundtrip_") as directory:
            path = Path(directory) / "weights.npy"
            np.save(path, weights, allow_pickle=False)
            loaded = np.load(path, allow_pickle=False)
            validated = validate_weights(loaded)
            self.assertEqual(validated.dtype, np.float32)
            self.assertEqual(validated.tobytes(), weights.tobytes())


class ReproducibleSeedPlanning(unittest.TestCase):
    # Params: self; returns: None.
    def test_all_replicates_use_disjoint_workers_and_shared_holdouts(self):
        config = ExperimentConfig()
        plans = [make_seed_plan(config, index) for index in range(len(config.run_seeds))]
        training = set()
        for index, plan in enumerate(plans):
            self.assertEqual((plan.run_index, plan.run_seed), (index, config.run_seeds[index]))
            self.assertEqual(plan.learner_seed, plan.base_worker_seed)
            self.assertEqual(plan.worker_seeds, tuple(plan.base_worker_seed + rank for rank in range(8)))
            self.assertFalse(training.intersection(plan.worker_seeds))
            training.update(plan.worker_seeds)
            self.assertEqual(plan.eval_seeds, plans[0].eval_seeds)
            self.assertEqual(plan.final_seeds, plans[0].final_seeds)
        self.assertEqual(len(training), 80)
        self.assertEqual(len(set(plans[0].eval_seeds)), 20)
        self.assertEqual(len(set(plans[0].final_seeds)), 100)
        self.assertFalse(set(plans[0].eval_seeds).intersection(plans[0].final_seeds))
        self.assertFalse(training.intersection(plans[0].eval_seeds + plans[0].final_seeds))
        for seed in training | set(plans[0].eval_seeds + plans[0].final_seeds):
            self.assertIs(type(seed), int)
            self.assertTrue(0 <= seed <= 2**32 - 1)

    # Params: self; returns: None.
    def test_weight_count_and_run_order_do_not_change_paired_seed_values(self):
        config = ExperimentConfig()
        first = make_seed_plan(config, 3)
        self.assertEqual(first, make_seed_plan(replace(config, n_weights=21), 3))
        reordered = replace(config, run_seeds=tuple(reversed(config.run_seeds)))
        other = make_seed_plan(reordered, reordered.run_seeds.index(first.run_seed))
        self.assertEqual(first.worker_seeds, other.worker_seeds)
        self.assertEqual(first.eval_seeds, other.eval_seeds)
        self.assertEqual(first.final_seeds, other.final_seeds)

    # Params: self; returns: None.
    def test_explicit_root_changes_are_isolated_and_holdout_prefixes_stable(self):
        config = ExperimentConfig()
        original = make_seed_plan(config, 0)
        training = make_seed_plan(replace(config, experiment_seed=43), 0)
        self.assertNotEqual(original.worker_seeds, training.worker_seeds)
        self.assertEqual(original.eval_seeds, training.eval_seeds)
        self.assertEqual(original.final_seeds, training.final_seeds)
        evaluation = make_seed_plan(replace(config, eval_seed_root=100001), 0)
        self.assertNotEqual(original.eval_seeds, evaluation.eval_seeds)
        self.assertEqual(original.worker_seeds, evaluation.worker_seeds)
        self.assertEqual(original.final_seeds, evaluation.final_seeds)
        final = make_seed_plan(replace(config, final_seed_root=200001), 0)
        self.assertNotEqual(original.final_seeds, final.final_seeds)
        extended = make_seed_plan(replace(config, eval_episodes=21, final_episodes=101), 0)
        self.assertEqual(original.eval_seeds, extended.eval_seeds[:20])
        self.assertEqual(original.final_seeds, extended.final_seeds[:100])

    # Params: self; returns: None.
    def test_repeatability_and_no_global_random_state_consumption(self):
        import random
        import numpy as np
        before_numpy, before_python = np.random.get_state(), random.getstate()
        first = make_seed_plan(ExperimentConfig(), 0)
        self.assertEqual(first, make_seed_plan(ExperimentConfig(), 0))
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertEqual(before_python, random.getstate())

    # Params: self; returns: None.
    def test_plans_reproduce_in_fresh_processes_with_different_python_hash_seeds(self):
        expected = make_seed_plan(ExperimentConfig(), 2).to_dict()
        script = "import json; from agents.mo import ExperimentConfig; from agents.mo.config import make_seed_plan; print(json.dumps(make_seed_plan(ExperimentConfig(), 2).to_dict()))"
        for hash_seed in ("1", "987654"):
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            environment["PYTHONHASHSEED"] = hash_seed
            result = subprocess.run([sys.executable, "-B", "-c", script], env=environment,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)

    # Params: self; returns: None.
    def test_json_persistence_contains_exact_plain_integer_lists(self):
        plan = make_seed_plan(ExperimentConfig(), 0)
        with tempfile.TemporaryDirectory(prefix="mo_seed_roundtrip_") as directory:
            path = Path(directory) / "seed_plan.json"
            path.write_text(json.dumps(plan.to_dict(), allow_nan=False), encoding="utf-8")
            restored = SeedPlan(**json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(restored, plan)
        for name in ("worker_seeds", "eval_seeds", "final_seeds"):
            self.assertIsInstance(plan.to_dict()[name], list)

    # Params: self; returns: None.
    def test_collision_resolution_and_uint32_upper_boundary(self):
        import numpy as np
        from unittest.mock import patch, Mock
        sequence = Mock()
        sequence.generate_state.return_value = np.array([2**32 - 1], dtype=np.uint32)
        config = ExperimentConfig(run_seeds=(2**32 - 1, 0), experiment_seed=2**32 - 1,
                                  eval_seed_root=2**32 - 1, final_seed_root=2**32 - 2,
                                  eval_episodes=3, final_episodes=3)
        with patch.object(np.random, "SeedSequence", return_value=sequence):
            first, second = make_seed_plan(config, 0), make_seed_plan(config, 1)
        self.assertEqual(len(set(first.eval_seeds)), 3)
        self.assertEqual(len(set(first.final_seeds)), 3)
        self.assertFalse(set(first.worker_seeds).intersection(second.worker_seeds))
        self.assertEqual(second.worker_seeds[-1], 2**31 - 1)
        self.assertEqual(first.eval_seeds, (3*2**30 - 1, 2**31, 2**31 + 1))
        self.assertEqual(first.final_seeds, (2**32 - 1, 3*2**30, 3*2**30 + 1))

    # Params: self; returns: None.
    def test_invalid_index_and_namespace_capacity_fail_before_allocation(self):
        config = ExperimentConfig()
        for index in (-1, 10):
            with self.subTest(index=index), self.assertRaises(ConfigError):
                make_seed_plan(config, index)
        for index in (True, 0.0, "0"):
            with self.subTest(index=index), self.assertRaises(TypeError):
                make_seed_plan(config, index)
        with self.assertRaises(TypeError):
            make_seed_plan({}, 0)
        with self.assertRaisesRegex(ConfigError, "capacity"):
            make_seed_plan(replace(config, ppo=PPOConfig(n_envs=2**31)), 0)
        with self.assertRaisesRegex(ConfigError, "capacity"):
            make_seed_plan(replace(config, eval_episodes=2**30 + 1), 0)
        with self.assertRaisesRegex(ConfigError, "capacity"):
            make_seed_plan(replace(config, final_episodes=2**30 + 1), 0)


class CompletedUpdateBudgetPlanning(unittest.TestCase):
    # Params: self; returns: None.
    def test_standard_budget_and_known_actual_schedule(self):
        plan = make_budget_plan(ExperimentConfig())
        self.assertEqual(plan.rollout_size, 4096)
        self.assertEqual(plan.requested_total, 500000)
        self.assertEqual(plan.effective_total, 503808)
        self.assertEqual(plan.eval_steps[:5], (20480, 40960, 61440, 81920, 102400))
        self.assertEqual(plan.checkpoint_steps[:3], (53248, 102400, 151552))
        self.assertEqual(len(plan.eval_steps), 25)
        self.assertEqual(len(plan.checkpoint_steps), 10)
        self.assertEqual(plan.eval_schedule[-1], (500000, 503808))
        self.assertEqual(plan.checkpoint_schedule[-1], (500000, 503808))
        for steps in (plan.eval_steps, plan.checkpoint_steps):
            self.assertEqual(steps, tuple(sorted(set(steps))))
            self.assertEqual(steps.count(plan.effective_total), 1)
            self.assertTrue(all(step % plan.rollout_size == 0 for step in steps))

    # Params: self; returns: None.
    def test_standard_complete_experiment_cost_excludes_evaluation(self):
        self.assertEqual(training_transition_cost(ExperimentConfig()), {
            "n_policies": 110, "requested_timesteps_per_policy": 500000,
            "effective_timesteps_per_policy": 503808, "updates_per_policy": 123,
            "requested_training_transitions": 55000000,
            "effective_training_transitions": 55418880,
        })
        altered = replace(ExperimentConfig(), eval_episodes=1, final_episodes=1)
        self.assertEqual(training_transition_cost(altered), training_transition_cost(ExperimentConfig()))
        smaller = replace(ExperimentConfig(), run_seeds=(0, 1), n_weights=3)
        self.assertEqual(training_transition_cost(smaller)["effective_training_transitions"], 6 * 503808)

    # Params: self; returns: None.
    def test_many_thresholds_mapping_to_one_update_keep_all_requested_values(self):
        config = ExperimentConfig(ppo=PPOConfig(n_envs=1, n_steps=8, batch_size=4),
                                  requested_timesteps_per_policy=20, eval_interval=2, checkpoint_interval=3)
        plan = make_budget_plan(config)
        self.assertEqual(plan.effective_total, 24)
        self.assertEqual(plan.eval_steps, (8, 16, 24))
        self.assertEqual(plan.checkpoint_steps, (8, 16, 24))
        self.assertEqual(plan.eval_schedule[:4], ((2, 8), (4, 8), (6, 8), (8, 8)))
        self.assertEqual(plan.checkpoint_schedule, ((3, 8), (6, 8), (9, 16), (12, 16),
                                                   (15, 16), (18, 24), (20, 24)))
        self.assertEqual(len(plan.eval_schedule), 10)
        self.assertEqual(plan.eval_steps.count(24), 1)

    # Params: self; returns: None.
    def test_short_exact_and_noninterval_final_budgets(self):
        for requested, expected in ((1, 4096), (4096, 4096), (4097, 8192), (8192, 8192)):
            with self.subTest(requested=requested):
                plan = make_budget_plan(replace(ExperimentConfig(), requested_timesteps_per_policy=requested))
                self.assertEqual(plan.effective_total, expected)
                self.assertEqual(plan.eval_steps, (expected,))
                self.assertEqual(plan.checkpoint_steps, (expected,))
                self.assertEqual(plan.eval_schedule, ((requested, expected),))
        plan = make_budget_plan(replace(ExperimentConfig(), requested_timesteps_per_policy=45000))
        self.assertEqual(plan.eval_schedule, ((20000, 20480), (40000, 40960), (45000, 45056)))
        self.assertEqual(plan.checkpoint_schedule, ((45000, 45056),))

    # Params: self; returns: None.
    def test_rounding_above_float_exact_integer_range(self):
        requested = 2**53 + 1
        plan = make_budget_plan(replace(ExperimentConfig(), requested_timesteps_per_policy=requested,
                                        eval_interval=requested + 1, checkpoint_interval=requested + 1))
        self.assertEqual(plan.effective_total, 2**53 + 4096)
        self.assertEqual(plan.eval_schedule, ((requested, 2**53 + 4096),))

    # Params: self; returns: None.
    def test_next_event_is_strict_for_initial_resumed_and_completed_steps(self):
        steps = (4096, 8192, 12288)
        for current, expected in ((0, 4096), (4095, 4096), (4096, 8192), (5000, 8192),
                                  (8192, 12288), (12288, None), (99999, None)):
            with self.subTest(current=current):
                self.assertEqual(next_scheduled_step(steps, current), expected)
        self.assertIsNone(next_scheduled_step((), 0))
        self.assertEqual(next_scheduled_step([1, 2], 1), 2)

    # Params: self; returns: None.
    def test_invalid_schedules_and_current_steps_are_rejected(self):
        for steps in ((0, 1), (2, 1), (1, 1), (-1, 1)):
            with self.subTest(steps=steps), self.assertRaises(ConfigError):
                next_scheduled_step(steps, 0)
        for steps in ((True, 2), (1., 2), "123", None, {1, 2}):
            with self.subTest(steps=steps), self.assertRaises(TypeError):
                next_scheduled_step(steps, 0)
        with self.assertRaises(ConfigError):
            next_scheduled_step((1,), -1)
        for current in (True, 0., "0"):
            with self.subTest(current=current), self.assertRaises(TypeError):
                next_scheduled_step((1,), current)

    # Params: self; returns: None.
    def test_budget_plan_is_shared_across_replicates_and_preferences(self):
        config = ExperimentConfig()
        original = make_budget_plan(config)
        alternative = replace(config, n_weights=3, run_seeds=(999,), experiment_seed=999)
        self.assertEqual(make_budget_plan(alternative), original)
        self.assertEqual(BudgetPlan(**json.loads(json.dumps(original.to_dict()))), original)
        with self.assertRaises(TypeError):
            make_budget_plan({})

    # Params: self; returns: None.
    def test_budget_generation_is_passive_in_a_fresh_process(self):
        script = '''
import sys
from agents.mo import ExperimentConfig
from agents.mo.config import make_budget_plan, training_transition_cost
assert make_budget_plan(ExperimentConfig()).effective_total == 503808
assert training_transition_cost(ExperimentConfig())["effective_training_transitions"] == 55418880
assert not any(name in sys.modules for name in ("numpy", "torch", "gymnasium", "scipy", "stable_baselines3"))
print("passive budget planning")
'''
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        with tempfile.TemporaryDirectory(prefix="mo_budget_passive_") as directory:
            result = subprocess.run([sys.executable, "-B", "-c", script], env=environment, cwd=directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
