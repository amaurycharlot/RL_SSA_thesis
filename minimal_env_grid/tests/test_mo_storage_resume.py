"""Block 20 identity/atomic persistence tests; no learners or workers."""
from dataclasses import replace
import hashlib
import json
import os
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
from agents.mo.config import make_weights, make_seed_plan, make_budget_plan, ConfigError
from agents.mo.types import ExperimentConfig, LoggingConfig


# Params: config; returns: (config, make_weights result, tuple, make_budget_plan result).
def inputs(config=None):
    config = config or ExperimentConfig(run_seeds=(0, 1), n_weights=3,
        reference_scaled=(-1., -2.), reference_status='frozen')
    return (config, make_weights(config.n_weights),
            tuple(make_seed_plan(config, i) for i in range(len(config.run_seeds))), make_budget_plan(config))


class ExperimentIdentity(unittest.TestCase):
    # Params: self; returns: None.
    def test_identity_is_deterministic_and_preserves_inputs(self):
        args = inputs()
        original = args[1].copy()
        fingerprint = storage.experiment_fingerprint(*args)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(fingerprint, storage.experiment_fingerprint(*args))
        np.testing.assert_array_equal(args[1], original)
        self.assertEqual(storage._json_bytes({'b': 2, 'a': 1}), storage._json_bytes({'a': 1, 'b': 2}))

    # Params: self; returns: None.
    def test_science_settings_alter_identity(self):
        args = inputs()
        config = args[0]
        fingerprint = storage.experiment_fingerprint(*args)
        variants = [replace(config, objective_scales=(2., 1.), scale_status='frozen'),
            replace(config, reference_scaled=(-2., -3.)), replace(config, error_target_set='initial'),
            replace(config, env=replace(config.env, measurement_noise_std=2.)),
            replace(config, ppo=replace(config.ppo, n_epochs=11)),
            replace(config, ppo=replace(config.ppo, device='cpu')),
            replace(config, execution=replace(config.execution, subproc=False)),
            replace(config, requested_timesteps_per_policy=600000), replace(config, run_seeds=(2, 3)),
            replace(config, eval_seed_root=123456), replace(config, experiment_seed=123)]
        for variant in variants:
            with self.subTest(config=variant):
                self.assertNotEqual(storage.experiment_fingerprint(*inputs(variant)), fingerprint)

    # Params: self; returns: None.
    def test_exact_weight_values_and_order_alter_identity(self):
        config, weights, seeds, budget = inputs()
        fingerprint = storage.experiment_fingerprint(config, weights, seeds, budget)
        changed = weights.copy()
        changed[1] = [np.nextafter(np.float32(.5), np.float32(1)), np.nextafter(np.float32(.5), np.float32(0))]
        for values in (changed, weights[::-1].copy()):
            self.assertNotEqual(storage.experiment_fingerprint(config, values, seeds, budget), fingerprint)

    # Params: self; returns: None.
    def test_output_and_logging_destinations_do_not_alter_identity(self):
        config, weights, seeds, budget = inputs()
        changed = replace(config, output_dir=Path('another-output'),
                          logging=LoggingConfig(mode='online', project='another-project', entity='another-entity'))
        self.assertEqual(storage.experiment_fingerprint(config, weights, seeds, budget),
                         storage.experiment_fingerprint(changed, weights, seeds, budget))

    # Params: self; returns: None.
    def test_provenance_architecture_and_schema_are_part_of_identity(self):
        args = inputs()
        fingerprint = storage.experiment_fingerprint(*args)
        for name, value in (('_source_hashes', {'changed.py': 'newhash'}),
                            ('_versions', {'python': 'different'}), ('_architecture', {'features_dim': 256})):
            with patch.object(storage, name, return_value=value):
                self.assertNotEqual(storage.experiment_fingerprint(*args), fingerprint)
        with patch.object(storage, 'STORAGE_SCHEMA_VERSION', 2):
            self.assertNotEqual(storage.experiment_fingerprint(*args), fingerprint)

    # Params: self; returns: None.
    def test_invalid_or_inconsistent_plans_fail(self):
        config, weights, seeds, budget = inputs()
        with self.assertRaises(storage.StorageError):
            storage.experiment_fingerprint(config, weights, seeds[::-1], budget)
        with self.assertRaises(storage.StorageError):
            storage.experiment_fingerprint(config, weights, seeds[:1], budget)
        with self.assertRaises(storage.StorageError):
            storage.experiment_fingerprint(config, weights[:2], seeds, budget)
        with self.assertRaises(storage.StorageError):
            storage.experiment_fingerprint(config, weights, seeds, inputs(replace(config, eval_interval=30000))[3])
        changed = replace(seeds[1], eval_seeds=tuple(i + 1 for i in seeds[1].eval_seeds))
        with self.assertRaises(storage.StorageError):
            storage.experiment_fingerprint(config, weights, (seeds[0], changed), budget)

    # Params: self; returns: None.
    def test_single_replicate_and_json_plan_roundtrip(self):
        args = inputs(ExperimentConfig(run_seeds=(0,), n_weights=2))
        self.assertEqual(storage.experiment_fingerprint(*args),
            storage.experiment_fingerprint(args[0], args[1], args[2][0], args[3]))
        from agents.mo.types import SeedPlan, BudgetPlan
        seeds = tuple(SeedPlan(**json.loads(json.dumps(p.to_dict()))) for p in args[2])
        budget = BudgetPlan(**json.loads(json.dumps(args[3].to_dict())))
        self.assertEqual(storage.experiment_fingerprint(*args),
            storage.experiment_fingerprint(args[0], args[1], seeds, budget))

    # Params: self; returns: None.
    def test_fresh_process_import_and_fingerprint_are_passive(self):
        code = '''
import sys
from agents.mo import storage
from agents.mo.config import make_weights, make_seed_plan, make_budget_plan
from agents.mo.types import ExperimentConfig
c = ExperimentConfig(run_seeds=(0, 1), n_weights=3, reference_scaled=(-1., -2.), reference_status='frozen')
print(storage.experiment_fingerprint(c, make_weights(3), tuple(make_seed_plan(c, i) for i in range(2)), make_budget_plan(c)))
assert not any(n.split('.')[0] in ('torch', 'gymnasium', 'stable_baselines3', 'matplotlib') for n in sys.modules)
'''
        completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(PROJECT_ROOT),
                                   check=True, capture_output=True, text=True)
        self.assertEqual(completed.stdout.strip(), storage.experiment_fingerprint(*inputs()))


class AtomicPersistence(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    # Params: self; returns: None.
    def test_json_null_conversion_and_canonical_order(self):
        path = self.root / 'record.json'
        storage.atomic_write_json(path, {'z': np.array([1., np.nan, np.inf, -np.inf]),
            'count': np.int64(3), 'path': Path('local'), 'missing': None})
        self.assertEqual(json.loads(path.read_text()),
            {'z': [1., None, None, None], 'count': 3, 'path': 'local', 'missing': None})
        self.assertNotIn('NaN', path.read_text())
        self.assertNotIn('Infinity', path.read_text())

    # Params: self; returns: None.
    def test_npz_exact_dtypes_shapes_and_missing_nan(self):
        path = self.root / 'record.npz'
        arrays = {'returns': np.array([[0., np.nan]], np.float64), 'valid': np.array([False]),
                  'steps': np.array([4096], np.int64)}
        storage.atomic_write_npz(path, arrays)
        with np.load(path, allow_pickle=False) as loaded:
            self.assertEqual(set(loaded.files), set(arrays))
            for name, array in arrays.items():
                np.testing.assert_array_equal(loaded[name], array)
                self.assertEqual(loaded[name].dtype, array.dtype)
        self.assertEqual(list(self.root.iterdir()), [path])

    # Params: self; returns: None.
    def test_replace_failure_preserves_previous_json_and_npz(self):
        for filename, writer, payload in (
                ('record.json', storage.atomic_write_json, {'v': 1}),
                ('record.npz', storage.atomic_write_npz, {'v': np.array([1])})):
            path = self.root / filename
            writer(path, payload)
            previous = path.read_bytes()
            with patch.object(storage.os, 'replace', side_effect=OSError('interrupted replace')):
                with self.assertRaises(OSError):
                    writer(path, payload)
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(any(p.suffix == '.tmp' for p in self.root.iterdir()))

    # Params: self; returns: None.
    def test_write_and_fsync_failures_preserve_previous_record(self):
        path = self.root / 'record.json'
        storage.atomic_write_json(path, {'v': 1})
        previous = path.read_bytes()
        # Params: stream; returns: never (raises).
        def fail(stream):
            stream.write(b'partial')
            raise OSError('interrupted write')
        with self.assertRaises(OSError):
            storage._atomic_write(path, fail)
        with patch.object(storage.os, 'fsync', side_effect=OSError('interrupted flush')):
            with self.assertRaises(OSError):
                storage.atomic_write_json(path, {'v': 2})
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(list(self.root.iterdir()), [path])

    # Params: self; returns: None.
    def test_serialization_failures_do_not_touch_previous_records(self):
        path = self.root / 'record.json'
        storage.atomic_write_json(path, {'valid': 1})
        previous = path.read_bytes()
        for payload in ({'bad': object()}, {1: 'bad'}):
            with self.assertRaises(TypeError):
                storage.atomic_write_json(path, payload)
        self.assertEqual(path.read_bytes(), previous)
        for arrays in ({'object': np.array([object()])}, {'text': np.array(['bad'])}):
            with self.assertRaises(TypeError):
                storage.atomic_write_npz(self.root / 'bad.npz', arrays)
        self.assertFalse((self.root / 'bad.npz').exists())


class ExperimentInitialization(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = inputs(replace(inputs()[0], output_dir=self.root / 'experiment'))

    # Params: self; returns: None.
    def test_create_records_exact_plans_versions_sources_and_resolved_defaults(self):
        config, weights, seeds, budget = self.args
        path = storage.create_experiment_dir(*self.args)
        manifest = json.loads((path / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'initialized')
        self.assertEqual(manifest['fingerprint'], storage.experiment_fingerprint(*self.args))
        saved = json.loads((path / 'config.json').read_text())
        self.assertEqual(saved['ppo']['n_epochs'], 10)
        self.assertEqual(saved['normalization']['gamma'], config.ppo.gamma)
        self.assertEqual(saved['resolved']['architecture']['features_dim'], 128)
        self.assertEqual(saved['resolved']['seed_plans'], [p.to_dict() for p in seeds])
        self.assertEqual(json.loads((path / 'budget_plan.json').read_text()), budget.to_dict())
        self.assertEqual(json.loads((path / 'seed_plan.json').read_text()), [p.to_dict() for p in seeds])
        np.testing.assert_array_equal(np.load(path / 'weights.npy', allow_pickle=False), weights)
        for package in ('python', 'numpy', 'gymnasium', 'stable-baselines3', 'torch'):
            self.assertTrue(manifest['identity']['versions'][package])
        for name, digest in manifest['identity']['source_hashes'].items():
            self.assertEqual(digest, hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest())
        for directory in ('results/evaluations', 'results/final', 'models', 'logs', 'reports'):
            self.assertTrue((path / directory).is_dir())
        self.assertFalse((path / '.INITIALIZING').exists())
        self.assertFalse(list(path.rglob('COMPLETE.json')))

    # Params: self; returns: None.
    def test_nonempty_existing_destination_is_never_overwritten(self):
        path = storage.create_experiment_dir(*self.args)
        before = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
        with self.assertRaisesRegex(storage.StorageError, 'nonempty'):
            storage.create_experiment_dir(*self.args)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()})

    # Params: self; returns: None.
    def test_explicit_resume_accepts_destinations_and_writes_nothing(self):
        path = storage.create_experiment_dir(*self.args)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir() if p.is_file()}
        changed = replace(self.args[0], logging=LoggingConfig(mode='offline', project='different'))
        self.assertEqual(storage.create_experiment_dir(changed, *self.args[1:], resume=True), path)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir() if p.is_file()})
        import shutil
        moved = self.root / 'moved'
        shutil.copytree(path, moved)
        self.assertEqual(storage.create_experiment_dir(replace(changed, output_dir=moved),
            *self.args[1:], resume=True), moved)

    # Params: self; returns: None.
    def test_resume_rejects_science_and_provenance_mismatch(self):
        storage.create_experiment_dir(*self.args)
        changed = replace(self.args[0], error_target_set='initial')
        with self.assertRaisesRegex(storage.StorageError, 'identity mismatch'):
            storage.create_experiment_dir(changed, *self.args[1:], resume=True)
        with patch.object(storage, '_source_hashes', return_value={'different.py': 'different'}):
            with self.assertRaisesRegex(storage.StorageError, 'identity mismatch'):
                storage.create_experiment_dir(*self.args, resume=True)

    # Params: self; returns: None.
    def test_resume_rejects_corruption_missing_manifest_and_in_progress(self):
        path = storage.create_experiment_dir(*self.args)
        (path / 'weights.npy').write_bytes(b'corrupted')
        with self.assertRaisesRegex(storage.StorageError, 'checksum mismatch'):
            storage.create_experiment_dir(*self.args, resume=True)
        (path / 'manifest.json').unlink()
        with self.assertRaisesRegex(storage.StorageError, 'Cannot resume'):
            storage.create_experiment_dir(*self.args, resume=True)
        (path / '.INITIALIZING').touch()
        with self.assertRaisesRegex(storage.StorageError, 'committed'):
            storage.create_experiment_dir(*self.args, resume=True)

    # Params: self; returns: None.
    def test_interrupted_initialization_has_no_commit_and_cannot_be_reused(self):
        real_writer = storage.atomic_write_json
        # Params: path, payload; returns: real_writer result.
        def interrupted(path, payload):
            if Path(path).name == 'manifest.json':
                raise OSError('manifest commit interrupted')
            return real_writer(path, payload)
        with patch.object(storage, 'atomic_write_json', side_effect=interrupted):
            with self.assertRaises(OSError):
                storage.create_experiment_dir(*self.args)
        path = self.args[0].output_dir
        self.assertFalse((path / 'manifest.json').exists())
        self.assertFalse((path / '.INITIALIZING').exists())
        with self.assertRaises(storage.StorageError):
            storage.create_experiment_dir(*self.args)
        with self.assertRaises(storage.StorageError):
            storage.create_experiment_dir(*self.args, resume=True)

    # Params: self; returns: None.
    def test_unfrozen_creation_and_missing_resume_fail_without_creation(self):
        config = replace(self.args[0], reference_scaled=None, reference_status='unselected')
        with self.assertRaises(ConfigError):
            storage.create_experiment_dir(config, *self.args[1:])
        self.assertFalse(config.output_dir.exists())
        with self.assertRaises(storage.StorageError):
            storage.create_experiment_dir(*self.args, resume=True)
        self.assertFalse(config.output_dir.exists())

    # Params: self; returns: None.
    def test_default_destination_and_existing_empty_directory(self):
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        config = replace(self.args[0], output_dir=None)
        path = storage.create_experiment_dir(config, *self.args[1:])
        self.assertEqual(path, self.root / 'runs_mo' / config.experiment_id)
        empty = self.root / 'empty'
        empty.mkdir()
        storage.create_experiment_dir(replace(config, output_dir=empty), *self.args[1:])


class RecoverablePolicyBundles(unittest.TestCase):
    # Params: cls; returns: None.
    @classmethod
    def setUpClass(cls):
        from stable_baselines3 import PPO
        from agents.mo.envs import make_train_vec_env
        from agents.mo.types import EnvConfig, PPOConfig, ExecutionConfig, NormalizationConfig
        from agents.RL.grid_cnn_extractor import GridCNNExtractor
        cls.resources = []
        cls.base = ExperimentConfig(run_seeds=(7,), n_weights=2,
            reference_scaled=(-1., -2.), reference_status='frozen', requested_timesteps_per_policy=4,
            eval_interval=2, checkpoint_interval=2, eval_episodes=2, final_episodes=2,
            env=EnvConfig(known_targets=1, unknown_targets=0, grid_shape=(2, 3),
                          episode_steps=2, slew_alpha=0., zero_noise=True),
            ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0),
            execution=ExecutionConfig(subproc=False))
        try:
            for normalized in (False, True):
                config = replace(cls.base, normalization=NormalizationConfig(norm_obs=normalized, norm_reward=normalized))
                handle = make_train_vec_env(config, [0., 1.], make_seed_plan(config, 0))
                cls.resources.append(handle)
                ppo = config.ppo.to_dict()
                ppo.pop('n_envs')
                model = PPO('MlpPolicy', handle.vec_env,
                    seed=make_seed_plan(config, 0).learner_seed,
                    policy_kwargs={'features_extractor_class': GridCNNExtractor,
                                   'features_extractor_kwargs': {'features_dim': 128}}, **ppo)
                model.learn(2)
                setattr(cls, 'normalized' if normalized else 'plain', (config, model, handle))
        except BaseException:
            cls.tearDownClass()
            raise

    # Params: cls; returns: None.
    @classmethod
    def tearDownClass(cls):
        for handle in cls.resources:
            handle.close()

    # Params: self; returns: None.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    # Params: self, normalized, final; returns: (model, handle, directory, identity, state).
    def bundle_inputs(self, normalized=False, final=False):
        from agents.mo.types import PolicyIdentity, ResumeState
        config, model, handle = self.normalized if normalized else self.plain
        config = replace(config, output_dir=self.root / ('normalized' if normalized else 'plain'),
                         requested_timesteps_per_policy=2 if final else 4)
        args = inputs(config)
        root = storage.create_experiment_dir(*args)
        identity = PolicyIdentity(storage.experiment_fingerprint(*args), 7, 0, 0, (0., 1.), 2, 'update_2')
        pending = () if final else (4,)
        state = ResumeState(False, 2, args[3].effective_total,
            next_evaluation_budgets=pending, next_checkpoint_budgets=pending,
            resume_attempt_id='test-attempt', canonical_attempt_ids=('test-attempt',))
        directory = root / 'models/seed_0/w_0' / ('final' if final else 'checkpoints/step_2')
        return model, handle, directory, identity, state

    # Params: self, directory, identity, normalized, final; returns: None.
    def assert_committed(self, directory, identity, normalized, final=False):
        bundle = json.loads((directory / 'bundle.json').read_text())
        marker = json.loads((directory / 'COMMITTED').read_text())
        self.assertEqual(marker['bundle_sha256'], hashlib.sha256((directory / 'bundle.json').read_bytes()).hexdigest())
        self.assertEqual(bundle['identity'], identity.to_dict())
        self.assertEqual(bundle['actual_step'], 2)
        self.assertEqual(bundle['counters'], {'num_timesteps': 2, 'n_updates': 1, 'completed_rollouts': 1})
        self.assertEqual(bundle['resume_state']['resume_attempt_id'], 'test-attempt')
        self.assertEqual(bundle['architecture']['features_dim'], 128)
        self.assertEqual(bundle['objective_scales'], [1., 1.])
        expected = {'final_model.zip' if final else 'model.zip'}
        if normalized:
            expected.add('vecnormalize.pkl')
        self.assertEqual(set(bundle['member_hashes']), expected)
        for name, digest in bundle['member_hashes'].items():
            self.assertEqual(digest, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        self.assertEqual({p.name for p in directory.iterdir()}, expected | {'bundle.json', 'COMMITTED'})
        self.assertFalse(list(directory.parent.glob('.*.tmp')))

    # Params: self; returns: None.
    def test_plain_real_ppo_bundle_requires_no_normalizer_pickle(self):
        from stable_baselines3 import PPO
        args = self.bundle_inputs()
        storage.save_policy_bundle(*args)
        self.assert_committed(args[2], args[3], False)
        loaded = PPO.load(args[2] / 'model.zip', device='cpu')
        self.assertEqual(loaded.num_timesteps, 2)
        self.assertEqual(loaded._n_updates, 1)
        self.assertTrue(loaded.policy.optimizer.state_dict()['state'])
        observation = np.zeros((2, 3, 3), np.float32)
        np.testing.assert_array_equal(loaded.predict(observation, deterministic=True)[0],
                                     args[0].predict(observation, deterministic=True)[0])

    # Params: self; returns: None.
    def test_normalized_real_ppo_saves_matching_statistics_without_mutation(self):
        import pickle
        args = self.bundle_inputs(True)
        normalizer = args[1].vec_normalize
        before = pickle.dumps(normalizer)
        storage.save_policy_bundle(*args)
        self.assert_committed(args[2], args[3], True)
        self.assertEqual(pickle.dumps(normalizer), before)
        with (args[2] / 'vecnormalize.pkl').open('rb') as stream:
            saved = pickle.load(stream)
        np.testing.assert_array_equal(saved.obs_rms.mean, normalizer.obs_rms.mean)
        self.assertEqual(saved.ret_rms.count, normalizer.ret_rms.count)
        self.assertFalse(args[1].closed)

    # Params: self; returns: None.
    def test_final_bundle_name_is_distinct_and_does_not_mark_policy_complete(self):
        args = self.bundle_inputs(final=True)
        storage.save_policy_bundle(*args)
        self.assert_committed(args[2], args[3], False, final=True)
        self.assertFalse((args[2].parent / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_existing_bundle_is_never_overwritten(self):
        args = self.bundle_inputs()
        storage.save_policy_bundle(*args)
        previous = {p.name: p.read_bytes() for p in args[2].iterdir()}
        with self.assertRaisesRegex(storage.StorageError, 'already exists'):
            storage.save_policy_bundle(*args)
        self.assertEqual(previous, {p.name: p.read_bytes() for p in args[2].iterdir()})

    # Params: self; returns: None.
    def test_partial_model_write_cannot_publish_checkpoint(self):
        args = self.bundle_inputs()
        # Params: path; returns: never (raises).
        def interrupted(path):
            path.write(b'partial ZIP')
            raise OSError('interrupted model')
        with patch.object(args[0], 'save', side_effect=interrupted):
            with self.assertRaises(OSError):
                storage.save_policy_bundle(*args)
        self.assertFalse(args[2].exists())
        self.assertFalse(list(args[2].parent.glob('.*.tmp')))
        storage.save_policy_bundle(*args)  # failed attempt did not occupy the final name

    # Params: self; returns: None.
    def test_partial_or_missing_normalizer_cannot_publish_checkpoint(self):
        args = self.bundle_inputs(True)
        # Params: path; returns: never (raises).
        def interrupted(path):
            Path(path).write_bytes(b'partial pickle')
            raise OSError('interrupted normalizer')
        for saver in (interrupted, lambda path: None):
            with patch.object(args[1].vec_normalize, 'save', side_effect=saver):
                with self.assertRaises((OSError, EOFError)):
                    storage.save_policy_bundle(*args)
            self.assertFalse(args[2].exists())
            self.assertFalse(list(args[2].parent.glob('.*.tmp')))

    # Params: self; returns: None.
    def test_saved_normalizer_must_match_the_live_policy_statistics(self):
        import pickle
        args = self.bundle_inputs(True)
        clone = pickle.loads(pickle.dumps(args[1].vec_normalize))
        # SB3 removes carrier attributes from pickles; restore them only on this
        # synthetic clone so a deliberately mismatched file can be serialized.
        for name, value in args[1].vec_normalize.__dict__.items():
            clone.__dict__.setdefault(name, value)
        clone.obs_rms.mean = clone.obs_rms.mean + 1
        # Params: path; returns: None.
        def mismatched(path):
            with Path(path).open('wb') as stream:
                pickle.dump(clone, stream)
        with patch.object(args[1].vec_normalize, 'save', side_effect=mismatched):
            with self.assertRaisesRegex(storage.StorageError, 'statistics mismatch'):
                storage.save_policy_bundle(*args)
        self.assertFalse(args[2].exists())

    # Params: self; returns: None.
    def test_metadata_marker_and_rename_failures_do_not_commit(self):
        args = self.bundle_inputs()
        real_writer = storage.atomic_write_json
        for failure in ('bundle.json', 'COMMITTED'):
            # Params: path, payload; returns: None.
            def interrupted(path, payload):
                if Path(path).name == failure:
                    raise OSError('commit interrupted')
                real_writer(path, payload)
            with patch.object(storage, 'atomic_write_json', side_effect=interrupted):
                with self.assertRaises(OSError):
                    storage.save_policy_bundle(*args)
            self.assertFalse(args[2].exists())
            self.assertFalse(list(args[2].parent.glob('.*.tmp')))
        with patch.object(storage.os, 'rename', side_effect=OSError('rename interrupted')):
            with self.assertRaises(OSError):
                storage.save_policy_bundle(*args)
        self.assertFalse(args[2].exists())

    # Params: self; returns: None.
    def test_prior_committed_bundle_survives_later_failed_save(self):
        final = self.bundle_inputs(final=True)
        checkpoint = final[2].parent / 'checkpoints/step_2'
        storage.save_policy_bundle(*final[:2], checkpoint, *final[3:])
        previous = {p.name: p.read_bytes() for p in checkpoint.iterdir()}
        with patch.object(final[0], 'save', side_effect=OSError('interrupted later save')):
            with self.assertRaises(OSError):
                storage.save_policy_bundle(*final)
        self.assertEqual(previous, {p.name: p.read_bytes() for p in checkpoint.iterdir()})
        self.assertFalse(final[2].exists())

    # Params: self; returns: None.
    def test_wrong_identity_step_slot_and_resume_counters_rejected_before_save(self):
        args = self.bundle_inputs()
        for identity in (replace(args[3], experiment_fingerprint='wrong'), replace(args[3], run_seed=0),
                         replace(args[3], weight=(1., 0.)), replace(args[3], actual_step=1)):
            with self.assertRaises(storage.StorageError):
                storage.save_policy_bundle(*args[:3], identity, args[4])
        for state in (replace(args[4], current_step=0),
                      replace(args[4], next_evaluation_budgets=())):
            with self.assertRaises(storage.StorageError):
                storage.save_policy_bundle(*args[:4], state)
        with self.assertRaisesRegex(storage.StorageError, 'directory'):
            storage.save_policy_bundle(*args[:2], args[2].parent / 'step_4', *args[3:])
        self.assertFalse(args[2].exists())

    # Params: self; returns: None.
    def test_rollout_collection_and_foreign_handle_are_rejected(self):
        from agents.mo.types import TrainingEnvHandle
        args = self.bundle_inputs()
        with patch.object(args[0], '_n_updates', 0):
            with self.assertRaises(storage.StorageError):
                storage.save_policy_bundle(*args)
        with patch.object(args[0].rollout_buffer, 'full', False):
            with self.assertRaises(storage.StorageError):
                storage.save_policy_bundle(*args)
        with patch.object(args[0].rollout_buffer, 'generator_ready', False):
            with self.assertRaises(storage.StorageError):
                storage.save_policy_bundle(*args)
        with self.assertRaises(storage.StorageError):
            storage.save_policy_bundle(args[0], TrainingEnvHandle(vec_env=object()), *args[2:])
        self.assertFalse(args[2].exists())

    # Params: self; returns: None.
    def test_model_or_normalizer_changes_during_save_are_not_committed(self):
        args = self.bundle_inputs()
        original = storage._validate_live_bundle
        calls = []
        # Params: *values; returns: result or (dict, result[1]).
        def changed(*values):
            result = original(*values)
            calls.append(1)
            return result if len(calls) == 1 else ({**result[0], 'num_timesteps': 4}, result[1])
        with patch.object(storage, '_validate_live_bundle', side_effect=changed):
            with self.assertRaisesRegex(storage.StorageError, 'changed while saving'):
                storage.save_policy_bundle(*args)
        self.assertFalse(args[2].exists())

    # Params: self; returns: None.
    def test_live_wrong_scalarization_and_missing_normalizer_are_rejected(self):
        args = self.bundle_inputs()
        with patch.object(args[1].vec_env, 'get_attr', return_value=[np.array([1., 0.], np.float32)]):
            with self.assertRaisesRegex(storage.StorageError, 'weights'):
                storage.save_policy_bundle(*args)
        normalized = self.bundle_inputs(True)
        with patch.object(normalized[1], 'vec_normalize', None):
            with self.assertRaisesRegex(storage.StorageError, 'normalization'):
                storage.save_policy_bundle(*normalized)
        self.assertFalse(args[2].exists())
        self.assertFalse(normalized[2].exists())

    # Params: self; returns: None.
    def test_successful_but_invalid_model_serializer_cannot_commit(self):
        import zipfile
        args = self.bundle_inputs()
        with patch.object(args[0], 'save', side_effect=lambda stream: stream.write(b'not a ZIP')):
            with self.assertRaises(zipfile.BadZipFile):
                storage.save_policy_bundle(*args)
        self.assertFalse(args[2].exists())
        self.assertFalse(list(args[2].parent.glob('.*.tmp')))


from unittest.mock import Mock


class CommittedBundleLoading(unittest.TestCase):
    setUpClass = classmethod(RecoverablePolicyBundles.setUpClass.__func__)
    tearDownClass = classmethod(RecoverablePolicyBundles.tearDownClass.__func__)
    setUp = RecoverablePolicyBundles.setUp
    bundle_inputs = RecoverablePolicyBundles.bundle_inputs

    # Params: self, normalized, final; returns: evaluation-record path.
    def save(self, normalized=False, final=False):
        args = self.bundle_inputs(normalized, final)
        storage.save_policy_bundle(*args)
        config = (self.normalized if normalized else self.plain)[0]
        return args, replace(config, output_dir=args[2].parents[4] if not final else args[2].parents[3],
                             requested_timesteps_per_policy=2 if final else 4)

    # Params: self, directory, change; returns: None.
    def rewrite_metadata(self, directory, change):
        meta = json.loads((directory / 'bundle.json').read_text())
        change(meta)
        storage.atomic_write_json(directory / 'bundle.json', meta)
        marker = json.loads((directory / 'COMMITTED').read_text())
        marker['bundle_sha256'] = hashlib.sha256((directory / 'bundle.json').read_bytes()).hexdigest()
        storage.atomic_write_json(directory / 'COMMITTED', marker)

    # Params: self; returns: None.
    def test_plain_reload_inference_optimizer_and_owned_resources(self):
        args, config = self.save()
        with storage.load_policy_bundle(args[2], config) as bundle:
            self.assertEqual(bundle.identity, args[3])
            self.assertIsNone(bundle.vec_normalize)
            self.assertIsNone(bundle.normalization_env)
            self.assertIsNone(bundle.model.get_env())
            self.assertTrue(bundle.model.policy.optimizer.state_dict()['state'])
            for obs in (np.zeros((2, 3, 3), np.float32), np.ones((2, 3, 3), np.float32)):
                np.testing.assert_array_equal(bundle.model.predict(obs, deterministic=True)[0],
                                             args[0].predict(obs, deterministic=True)[0])
        self.assertTrue(bundle.closed)
        self.assertFalse(args[1].closed)

    # Params: self; returns: None.
    def test_normalized_eval_owns_frozen_scalar_carrier_and_matches_predictions(self):
        from agents.mo.evaluation import make_frozen_obs_normalizer, normalize_policy_observation
        args, config = self.save(True)
        with storage.load_policy_bundle(args[2], config) as bundle:
            self.assertFalse(bundle.vec_normalize.training)
            self.assertFalse(bundle.vec_normalize.norm_reward)
            self.assertIsNot(bundle.vec_normalize, args[1].vec_normalize)
            self.assertEqual(bundle.normalization_env.get_attr('weights')[0].tolist(), [0., 1.])
            before = bundle.vec_normalize.obs_rms.count
            raw = np.ones((2, 3, 3), np.float32)
            obs = normalize_policy_observation(raw, make_frozen_obs_normalizer(bundle.vec_normalize))
            expected = normalize_policy_observation(raw, make_frozen_obs_normalizer(args[1].vec_normalize))
            np.testing.assert_array_equal(obs, expected)
            np.testing.assert_array_equal(bundle.model.predict(obs, deterministic=True)[0],
                                         args[0].predict(expected, deterministic=True)[0])
            self.assertEqual(before, bundle.vec_normalize.obs_rms.count)
            from agents.mo.evaluation import evaluate_policy
            from agents.mo.envs import make_raw_env
            seeds = make_seed_plan(config, 0).eval_seeds
            expected_returns = evaluate_policy(args[0], lambda: make_raw_env(config.env), seeds, config,
                args[3], make_frozen_obs_normalizer(args[1].vec_normalize)).returns_raw
            actual_returns = evaluate_policy(bundle.model, lambda: make_raw_env(config.env), seeds, config,
                bundle.identity, make_frozen_obs_normalizer(bundle.vec_normalize)).returns_raw
            np.testing.assert_array_equal(actual_returns, expected_returns)
            carrier = bundle.owned_resources[0]
            with patch.object(carrier, 'close', wraps=carrier.close) as close:
                bundle.close()
                close.assert_called_once()
        self.assertTrue(bundle.closed)
        self.assertTrue(args[1].vec_normalize.training)

    # Params: self; returns: None.
    def test_training_reload_restores_normalizer_and_borrows_handle(self):
        from agents.mo.envs import make_train_vec_env
        for normalized in (False, True):
            args, config = self.save(normalized)
            with make_train_vec_env(config, args[3].weight, make_seed_plan(config, 0)) as handle:
                original = handle.vec_normalize
                with storage.load_policy_bundle(args[2], config, handle) as bundle:
                    self.assertIs(bundle.model.get_env(), handle.vec_env)
                    self.assertIs(bundle.vec_normalize, handle.vec_normalize)
                    self.assertIsNone(bundle.normalization_env)
                    if normalized:
                        self.assertIsNot(handle.vec_normalize, original)
                        np.testing.assert_array_equal(handle.vec_normalize.obs_rms.mean, args[1].vec_normalize.obs_rms.mean)
                        self.assertTrue(handle.vec_normalize.training)
                    bundle.model.learn(2, reset_num_timesteps=False)
                    self.assertEqual(bundle.model.num_timesteps, 4)
                self.assertFalse(handle.closed)

    # Params: self; returns: None.
    def test_wrong_scales_architecture_reference_and_weights_rejected_before_load(self):
        args, config = self.save()
        original = (args[2] / 'bundle.json').read_bytes()
        original_marker = (args[2] / 'COMMITTED').read_bytes()
        changes = [lambda m: m.update(objective_scales=[2., 1.]),
                   lambda m: m.update(reference_scaled=[-99., -99.]),
                   lambda m: m['architecture'].update(features_dim=256),
                   lambda m: m['identity'].update(weight=[1., 0.])]
        for change in changes:
            self.rewrite_metadata(args[2], change)
            with self.assertRaises(storage.StorageError):
                storage.load_policy_bundle(args[2], config)
            (args[2] / 'bundle.json').write_bytes(original)
            (args[2] / 'COMMITTED').write_bytes(original_marker)
        for changed in (replace(config, reference_scaled=(-5., -5.)),
                        replace(config, objective_scales=(2., 1.), scale_status='frozen')):
            with self.assertRaises(storage.StorageError):
                storage.load_policy_bundle(args[2], changed)

    # Params: self; returns: None.
    def test_missing_normalizer_and_corrupt_commit_member_are_rejected(self):
        args, config = self.save(True)
        data = (args[2] / 'vecnormalize.pkl').read_bytes()
        (args[2] / 'vecnormalize.pkl').unlink()
        with self.assertRaises(storage.StorageError):
            storage.load_policy_bundle(args[2], config)
        (args[2] / 'vecnormalize.pkl').write_bytes(data + b'corrupt')
        with self.assertRaises(storage.StorageError):
            storage.load_policy_bundle(args[2], config)
        (args[2] / 'vecnormalize.pkl').write_bytes(data)
        (args[2] / 'COMMITTED').unlink()
        self.assertIsNone(storage.discover_latest_checkpoint(args[2].parents[1], args[3].experiment_fingerprint))

    # Params: self; returns: None.
    def test_partial_load_closes_eval_carrier_and_keeps_borrowed_handle(self):
        from agents.mo import envs
        from agents.mo.envs import make_train_vec_env
        from stable_baselines3 import PPO
        args, config = self.save(True)
        carriers = []
        real = envs._make_dummy_vec_env
        # Params: factories; returns: env.
        def tracked(factories):
            env = real(factories)
            env.close = Mock(wraps=env.close)
            carriers.append(env)
            return env
        with patch.object(envs, '_make_dummy_vec_env', side_effect=tracked), \
                patch.object(PPO, 'load', side_effect=OSError('model reload interrupted')):
            with self.assertRaises(OSError):
                storage.load_policy_bundle(args[2], config)
        carriers[0].close.assert_called_once()
        with make_train_vec_env(config, args[3].weight, make_seed_plan(config, 0)) as handle:
            previous = handle.vec_normalize
            with patch.object(PPO, 'load', side_effect=OSError('reload interrupted')):
                with self.assertRaises(OSError):
                    storage.load_policy_bundle(args[2], config, handle)
            self.assertFalse(handle.closed)
            self.assertIs(handle.vec_normalize, previous)
            self.assertIs(handle.vec_env, previous)
            handle.vec_env.reset()

    # Params: self; returns: None.
    def test_failed_normalizer_loading_closes_partial_carrier(self):
        from agents.mo import envs
        from stable_baselines3.common.vec_env import VecNormalize
        args, config = self.save(True)
        carriers = []
        real = envs._make_dummy_vec_env
        # Params: factories; returns: env.
        def tracked(factories):
            env = real(factories)
            env.close = Mock(wraps=env.close)
            carriers.append(env)
            return env
        with patch.object(envs, '_make_dummy_vec_env', side_effect=tracked), \
                patch.object(VecNormalize, 'load', side_effect=OSError('normalizer interrupted')):
            with self.assertRaises(OSError):
                storage.load_policy_bundle(args[2], config)
        carriers[0].close.assert_called_once()

    # Params: self; returns: None.
    def test_numeric_discovery_and_ignoring_uncommitted_or_corrupt_later_steps(self):
        import zipfile
        from agents.mo.types import PolicyIdentity, ResumeState, PPOConfig
        config = replace(self.base, output_dir=self.root / 'numeric',
            requested_timesteps_per_policy=300000, eval_interval=75000, checkpoint_interval=75000,
            ppo=PPOConfig(n_envs=1, n_steps=25000, batch_size=250, n_epochs=1, device='cpu', verbose=0))
        args = inputs(config)
        root = storage.create_experiment_dir(*args)
        payload = json.loads((root / 'manifest.json').read_text())['identity']
        fingerprint = storage.experiment_fingerprint(*args)
        policy = root / 'models/seed_0/w_0'
        for step in (75000, 300000):
            directory = policy / 'checkpoints' / f'step_{step}'
            directory.mkdir(parents=True)
            identity = PolicyIdentity(fingerprint, 7, 0, 0, (0., 1.), step, f'update_{step}')
            with zipfile.ZipFile(directory / 'model.zip', 'w') as archive:
                archive.writestr('data', json.dumps({'num_timesteps': step, '_n_updates': step // 25000}))
                archive.writestr('policy.pth', b'synthetic geometry-only discovery fixture')
                archive.writestr('policy.optimizer.pth', b'synthetic optimizer fixture')
            pending = tuple(s for s in args[3].eval_steps if s > step)
            state = ResumeState(False, step, 300000, next_evaluation_budgets=pending, next_checkpoint_budgets=pending)
            meta = dict(storage_schema_version=storage.STORAGE_SCHEMA_VERSION, bundle_schema_version=1,
                kind='checkpoint', identity=identity.to_dict(), actual_step=step, config_fingerprint=fingerprint,
                objective_scales=payload['config']['objective_scales'], reference_scaled=payload['config']['reference_scaled'],
                normalization=payload['config']['normalization'], architecture=payload['architecture'],
                versions=payload['versions'], source_hashes=payload['source_hashes'], resume_state=state.to_dict(),
                counters={'num_timesteps': step, 'n_updates': step // 25000, 'completed_rollouts': step // 25000},
                member_hashes={'model.zip': hashlib.sha256((directory / 'model.zip').read_bytes()).hexdigest()})
            storage.atomic_write_json(directory / 'bundle.json', meta)
            storage.atomic_write_json(directory / 'COMMITTED', dict(bundle_schema_version=1,
                bundle_id=identity.bundle_id, actual_step=step,
                bundle_sha256=hashlib.sha256((directory / 'bundle.json').read_bytes()).hexdigest()))
        chosen = storage.discover_latest_checkpoint(policy, fingerprint)
        self.assertEqual(chosen.actual_step, 300000)
        self.assertEqual(chosen.directory.name, 'step_300000')
        (chosen.directory / 'model.zip').write_bytes(b'corrupt')
        self.assertEqual(storage.discover_latest_checkpoint(policy, fingerprint).actual_step, 75000)
        self.assertIsNone(storage.discover_latest_checkpoint(policy, 'different-fingerprint'))
        (policy / 'checkpoints/step_400000').mkdir()
        (policy / 'checkpoints/.step_500000.partial.tmp').mkdir()
        self.assertEqual(storage.discover_latest_checkpoint(policy, fingerprint).actual_step, 75000)

    # Params: self, args, wrong_seed; returns: path.
    def evaluation_record(self, args, wrong_seed=False):
        from agents.mo.types import EpisodeMetrics
        directory, identity = args[2], args[3]
        root = directory.parent.parents[2]
        path = root / 'results/evaluations/seed_0/w_0/step_2.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        seeds = list(make_seed_plan(replace(self.base, requested_timesteps_per_policy=2), 0).eval_seeds)
        if wrong_seed:
            seeds[0] += 1
        metrics = EpisodeMetrics(mean_positional_error=0., mean_final_positional_error=0., mean_covariance_trace=2.,
            final_coverage=.5, mean_coverage=.5, unique_new_detections=0, mean_known_observation_frequency=0.,
            mean_observation_frequency=0., decisions=1, elapsed_timesteps=1,
            unavailable_reasons={'search_time_percentage': 'PPO has no action mode'})
        returns = np.array([[1., 2.], [3., 4.]], np.float64)
        arrays = {}
        for suffix in ('raw', 'scaled'):
            arrays['returns_' + suffix] = returns
            arrays['mean_' + suffix] = returns.mean(axis=0)
            arrays['sd_' + suffix] = returns.std(axis=0, ddof=1)
            arrays['se_' + suffix] = returns.std(axis=0, ddof=1) / np.sqrt(2)
        storage.atomic_write_npz(path.with_suffix('.npz'), arrays)
        storage.atomic_write_json(path, dict(storage_schema_version=storage.STORAGE_SCHEMA_VERSION,
            committed=True, identity=identity.to_dict(), actual_step=2, deterministic=True,
            episode_seeds=seeds, objective_scales=[1., 1.], episode_metrics=[metrics.to_dict()] * 2,
            data_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()))
        return path

    # Params: self; returns: None.
    def test_completion_accepts_block21_saved_evaluation_with_traces(self):
        from agents.mo.evaluation import evaluate_policy
        from agents.mo.envs import make_raw_env
        args, config = self.save(final=True)
        policy = args[2].parent
        root = policy.parents[2]
        seeds = storage._read_json(root / 'seed_plan.json')[0]['eval_seeds']
        result = evaluate_policy(args[0], lambda: make_raw_env(config.env), seeds,
                                 config, args[3], collect_traces=True)
        storage.save_evaluation_record(root, args[3], result)
        path = root / 'results/evaluations/seed_0/w_0/step_2.json'
        storage.mark_policy_complete(policy, args[2], path)
        self.assertEqual(storage._read_json(policy / 'COMPLETE.json')['actual_step'], 2)
        self.assertTrue(storage._load_evaluation_record(path)[0].traces)


    # Params: self; returns: None.
    def test_completion_requires_final_bundle_and_committed_correct_evaluation(self):
        args, _ = self.save(final=True)
        path = self.evaluation_record(args)
        policy = args[2].parent
        storage.mark_policy_complete(policy, args[2], path)
        marker = (policy / 'COMPLETE.json').read_bytes()
        storage.mark_policy_complete(policy, args[2], path)
        self.assertEqual(marker, (policy / 'COMPLETE.json').read_bytes())
        self.assertEqual(json.loads(marker)['actual_step'], 2)
        self.assertEqual(json.loads(marker)['identity'], args[3].to_dict())

    # Params: self; returns: None.
    def test_incomplete_or_wrong_evaluation_cannot_mark_policy_complete(self):
        args, _ = self.save(final=True)
        policy = args[2].parent
        with self.assertRaises(storage.StorageError):
            storage.mark_policy_complete(policy, args[2], self.root / 'missing.json')
        path = self.evaluation_record(args, wrong_seed=True)
        with self.assertRaises(storage.StorageError):
            storage.mark_policy_complete(policy, args[2], path)
        self.assertFalse((policy / 'COMPLETE.json').exists())
        path = self.evaluation_record(args)
        path.with_suffix('.npz').write_bytes(b'corrupted numeric file')
        with self.assertRaises(storage.StorageError):
            storage.mark_policy_complete(policy, args[2], path)
        self.assertFalse((policy / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_checkpoint_and_missing_normalizer_cannot_acquire_completion(self):
        plain, _ = self.save(final=True)
        checkpoint = plain[2].parent / 'checkpoints/step_2'
        storage.save_policy_bundle(*plain[:2], checkpoint, *plain[3:])
        with self.assertRaises(storage.StorageError):
            storage.mark_policy_complete(plain[2].parent, checkpoint, self.evaluation_record(plain))
        self.assertFalse((plain[2].parent / 'COMPLETE.json').exists())
        args, _ = self.save(True, final=True)
        path = self.evaluation_record(args)
        (args[2] / 'vecnormalize.pkl').unlink()
        with self.assertRaises(storage.StorageError):
            storage.mark_policy_complete(args[2].parent, args[2], path)
        self.assertFalse((args[2].parent / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_loading_preserves_global_rng_and_allows_only_eval_device_override(self):
        import random
        import torch
        args, config = self.save()
        before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        evaluation_config = replace(config, ppo=replace(config.ppo, device='auto'))
        with storage.load_policy_bundle(args[2], evaluation_config) as bundle:
            self.assertFalse(bundle.model.policy.training)
        self.assertEqual(random.getstate(), before_python)
        current = np.random.get_state()
        self.assertEqual(current[0], before_numpy[0])
        np.testing.assert_array_equal(current[1], before_numpy[1])
        self.assertEqual(current[2:], before_numpy[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        with self.assertRaises(storage.StorageError):
            storage.load_policy_bundle(args[2], evaluation_config, args[1])

    # Params: self; returns: None.
    def test_loaded_normalizer_settings_are_validated_after_checksum_check(self):
        import pickle
        args, config = self.save(True)
        clone = pickle.loads(pickle.dumps(args[1].vec_normalize))
        for name, value in args[1].vec_normalize.__dict__.items():
            clone.__dict__.setdefault(name, value)
        clone.gamma = .5
        with (args[2] / 'vecnormalize.pkl').open('wb') as stream:
            pickle.dump(clone, stream)
        self.rewrite_metadata(args[2], lambda m: m['member_hashes'].update(
            {'vecnormalize.pkl': hashlib.sha256((args[2] / 'vecnormalize.pkl').read_bytes()).hexdigest()}))
        with self.assertRaisesRegex(storage.StorageError, 'normalizer setting mismatch'):
            storage.load_policy_bundle(args[2], config)
        # Finite values alone are insufficient for reward RMS and observation RMS.
        settings = args[1].vec_normalize
        shape = list(args[0].observation_space.shape)
        with patch.object(settings.ret_rms, 'mean', np.zeros(2)):
            with self.assertRaisesRegex(storage.StorageError, 'shape mismatch'):
                storage._loaded_normalizer(settings, config.normalization.to_dict(), shape)

    # Params: self; returns: None.
    def test_completion_rejects_uncommitted_and_inconsistent_scaled_returns(self):
        args, _ = self.save(final=True)
        path = self.evaluation_record(args)
        meta = json.loads(path.read_text())
        meta['committed'] = False
        storage.atomic_write_json(path, meta)
        with self.assertRaises(storage.StorageError):
            storage.mark_policy_complete(args[2].parent, args[2], path)
        meta['committed'] = True
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays['returns_scaled'] = arrays['returns_scaled'] + .1
        arrays['mean_scaled'] = arrays['returns_scaled'].mean(axis=0)
        arrays['sd_scaled'] = arrays['returns_scaled'].std(axis=0, ddof=1)
        arrays['se_scaled'] = arrays['sd_scaled'] / np.sqrt(2)
        storage.atomic_write_npz(path.with_suffix('.npz'), arrays)
        meta['data_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
        storage.atomic_write_json(path, meta)
        with self.assertRaisesRegex(storage.StorageError, 'raw/scaled'):
            storage.mark_policy_complete(args[2].parent, args[2], path)
        self.assertFalse((args[2].parent / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_completion_commit_failure_and_conflict_preserve_existing_state(self):
        args, _ = self.save(final=True)
        path = self.evaluation_record(args)
        policy = args[2].parent
        with patch.object(storage, 'atomic_write_json', side_effect=OSError('completion commit interrupted')):
            with self.assertRaises(storage.StorageError):
                storage.mark_policy_complete(policy, args[2], path)
        self.assertFalse((policy / 'COMPLETE.json').exists())
        storage.mark_policy_complete(policy, args[2], path)
        marker = json.loads((policy / 'COMPLETE.json').read_text())
        marker['actual_step'] = 0
        storage.atomic_write_json(policy / 'COMPLETE.json', marker)
        previous = (policy / 'COMPLETE.json').read_bytes()
        with self.assertRaisesRegex(storage.StorageError, 'Conflicting'):
            storage.mark_policy_complete(policy, args[2], path)
        self.assertEqual(previous, (policy / 'COMPLETE.json').read_bytes())


class EvaluationPersistence(unittest.TestCase):
    # Params: self; returns: None.
    def setUp(self):
        from agents.mo.types import EnvConfig, PPOConfig
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ExperimentConfig(output_dir=Path(self.temporary.name) / 'experiment',
            run_seeds=(4, 9), n_weights=2, eval_episodes=2, final_episodes=1,
            requested_timesteps_per_policy=5, eval_interval=1, checkpoint_interval=2,
            objective_scales=(2., 4.), scale_status='frozen',
            reference_scaled=(-1., -1.), reference_status='frozen',
            env=EnvConfig(known_targets=1, unknown_targets=1, grid_shape=(2, 3), episode_steps=2, slew_alpha=0.),
            ppo=PPOConfig(n_envs=1, n_steps=2, batch_size=2, n_epochs=1, device='cpu', verbose=0))
        self.args = inputs(self.config)
        self.root = storage.create_experiment_dir(*self.args)
        self.payload = storage._read_json(self.root / 'manifest.json')['identity']
        self.fingerprint = storage.experiment_fingerprint(*self.args)

    # Params: self, run, weight, budget, raw, traces, final; returns: EvaluationResult.
    def result(self, run=0, weight=0, budget=0, raw=None, traces=False, final=False):
        from agents.mo.types import (PolicyIdentity, EvaluationResult, EpisodeMetrics,
                                    EpisodeTrace, TargetObservationMetrics)
        step = self.args[3].effective_total if final else self.args[3].eval_steps[budget]
        seeds = self.args[2][run].final_seeds if final else self.args[2][run].eval_seeds
        identity = PolicyIdentity(self.fingerprint, self.config.run_seeds[run], run, weight,
                                  tuple(self.args[1][weight].tolist()), step, f'policy-{run}-{weight}-{step}')
        raw = np.zeros((len(seeds), 2), np.float64) if raw is None else np.asarray(raw, np.float64)
        scaled = raw / self.config.objective_scales
        metrics = tuple(EpisodeMetrics(mean_positional_error=0., mean_final_positional_error=None,
            mean_covariance_trace=0., final_coverage=0., mean_coverage=0., unique_new_detections=0,
            mean_known_observation_frequency=0., mean_observation_frequency=0., decisions=1,
            elapsed_timesteps=1, target_observation_metrics=(TargetObservationMetrics(0, True, 0, 0, 1, 0.),
                TargetObservationMetrics(1, False, 0, None, None, None)),
            unavailable_reasons={'mean_final_positional_error': 'synthetic unavailable final error',
                                 'search_time_percentage': 'PPO has no explicit search/track action mode'})
            for _ in seeds)
        retained = tuple(EpisodeTrace(initial_observation=np.zeros((2, 3, 3), np.float32),
            observations=np.ones((1, 2, 3, 3), np.float32), vector_rewards=raw[i:i+1].astype(np.float32),
            true_states=np.zeros((1, 2, 4), np.float32), estimated_states=np.full((1, 2, 9), np.nan),
            actions=np.zeros((1, 2), np.int64), observed_targets=np.zeros((1, 2), bool),
            elapsed_steps=np.ones(1, np.int64), coverage_values=np.zeros(1),
            terminated=np.ones(1, bool), truncated=np.zeros(1, bool), seed=seed,
            initial_known_count=1) for i, seed in enumerate(seeds)) if traces else ()
        summary = {}
        for name, array in (('raw', raw), ('scaled', scaled)):
            sd = array.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(2)
            summary.update({f'mean_{name}': array.mean(axis=0), f'sd_{name}': sd,
                            f'se_{name}': sd / np.sqrt(len(seeds))})
        return EvaluationResult(episode_seeds=seeds, returns_raw=raw, returns_scaled=scaled,
            **summary, episode_metrics=metrics, deterministic=True, actual_step=step,
            identity=identity, traces=retained)

    # Params: self, result; returns: evaluation-record path.
    def save(self, result):
        storage.save_evaluation_record(self.root, result.identity, result)
        return storage._evaluation_location(self.root, self.payload, self.fingerprint, result.identity, result)[0]

    # Params: self; returns: None.
    def test_round_trip_metrics_nullable_values_and_trace_precision(self):
        result = self.result(raw=[[2, 8], [4, 16]], traces=True)
        path = self.save(result)
        loaded, meta = storage._load_evaluation_record(path)
        self.assertEqual(loaded.identity, result.identity)
        self.assertEqual(loaded.episode_seeds, result.episode_seeds)
        self.assertEqual(loaded.episode_metrics, result.episode_metrics)
        for name in storage._EVAL_ARRAYS:
            np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
        for left, right in zip(loaded.traces, result.traces):
            self.assertEqual(left.initial_known_count, 1)
            for name in storage._TRACE_ARRAYS:
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
                self.assertEqual(getattr(left, name).dtype, getattr(right, name).dtype)
        self.assertNotIn('NaN', path.read_text())
        self.assertIn('null', path.read_text())
        self.assertEqual(meta['npz_sha256'], hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())

    # Params: self; returns: None.
    def test_actual_budget_mapping_missing_nan_and_real_zero_are_distinct(self):
        result = self.result(run=1, weight=1, budget=2)
        self.save(result)
        loaded = storage.load_evaluation_records(self.root, self.config)
        self.assertEqual(tuple(loaded.actual_budgets), (2, 4, 6))
        self.assertEqual(np.count_nonzero(loaded.validity), 1)
        self.assertTrue(loaded.validity[1, 2, 1])
        np.testing.assert_array_equal(loaded.means_raw[1, 2, 1], [0, 0])
        self.assertTrue(np.isnan(loaded.means_raw[0]).all())
        self.assertTrue(np.isnan(loaded.hypervolume).all())
        self.assertEqual(loaded.identities[(1, 2, 1)], result.identity)
        self.assertEqual(loaded.episode_metric_tables[(1, 2, 1)], result.episode_metrics)

    # Params: self; returns: None.
    def test_identical_write_is_byte_identical_noop_and_conflicts_are_rejected(self):
        result = self.result()
        path = self.save(result)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, path.with_suffix('.npz'))}
        self.save(result)
        for p, expected in before.items():
            self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), expected)
        for changed in (self.result(raw=[[1, 0], [1, 0]]),
                        replace(result, identity=replace(result.identity, bundle_id='other-attempt')),
                        replace(result, episode_metrics=tuple(replace(m, mean_coverage=1.) for m in result.episode_metrics))):
            with self.assertRaisesRegex(storage.StorageError, 'Conflicting'):
                self.save(changed)
        for p, expected in before.items():
            self.assertEqual(p.read_bytes(), expected[0])

    # Params: self; returns: None.
    def test_numeric_orphans_and_temporary_files_never_enter_curves(self):
        result = self.result()
        path = storage._evaluation_location(self.root, self.payload, self.fingerprint, result.identity, result)[0]
        path.parent.mkdir(parents=True)
        path.with_suffix('.npz').write_bytes(b'interrupted numeric write')
        (path.parent / ('.' + path.name + '.tmp')).write_bytes(b'{partial')
        loaded = storage.load_evaluation_records(self.root, self.config)
        self.assertFalse(loaded.validity.any())
        self.assertTrue(np.isnan(loaded.means_scaled).all())
        self.save(result)  # An uncommitted numeric orphan can be retried under the slot lock.
        self.assertTrue(storage.load_evaluation_records(self.root, self.config).validity[0, 0, 0])

    # Params: self; returns: None.
    def test_marker_write_interruption_is_invisible_and_can_be_retried(self):
        result = self.result()
        with patch.object(storage, 'atomic_write_json', side_effect=OSError('injected marker failure')):
            with self.assertRaisesRegex(OSError, 'marker failure'):
                self.save(result)
        self.assertFalse(list((self.root / 'results/evaluations').rglob('*.json')))
        self.assertFalse(list((self.root / 'results/evaluations').rglob('*.LOCK')))
        self.assertFalse(storage.load_evaluation_records(self.root, self.config).validity.any())
        self.save(result)

    # Params: self; returns: None.
    def test_numeric_write_interruption_leaves_other_committed_slots_readable(self):
        first = self.result()
        self.save(first)
        second = self.result(weight=1)
        with patch.object(storage, 'atomic_write_npz', side_effect=OSError('numeric failure')):
            with self.assertRaises(OSError):
                self.save(second)
        loaded = storage.load_evaluation_records(self.root, self.config)
        self.assertTrue(loaded.validity[0, 0, 0])
        self.assertFalse(loaded.validity[0, 0, 1])

    # Params: self; returns: None.
    def test_lock_excludes_competing_or_interrupted_writer(self):
        result = self.result()
        path = storage._evaluation_location(self.root, self.payload, self.fingerprint, result.identity, result)[0]
        path.parent.mkdir(parents=True)
        lock = path.with_suffix('.LOCK')
        lock.write_text('owned elsewhere')
        with self.assertRaisesRegex(storage.StorageError, 'writer'):
            self.save(result)
        self.assertEqual(lock.read_text(), 'owned elsewhere')
        self.assertFalse(path.exists())

    # Params: self; returns: None.
    def test_wrong_identity_budget_seeds_scales_and_inference_are_rejected(self):
        result = self.result(raw=[[2, 8], [4, 16]])
        variants = [replace(result, identity=replace(result.identity, experiment_fingerprint='wrong')),
                    replace(result, identity=replace(result.identity, run_seed=999)),
                    replace(result, identity=replace(result.identity, weight=(1., 0.))),
                    replace(result, actual_step=3, identity=replace(result.identity, actual_step=3)),
                    replace(result, episode_seeds=result.episode_seeds[::-1]),
                    replace(result, episode_seeds=(1, 2)), replace(result, deterministic=False)]
        scaled = result.returns_scaled + 1
        variants.append(replace(result, returns_scaled=scaled, mean_scaled=scaled.mean(axis=0)))
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(storage.StorageError): self.save(variant)
        with self.assertRaises(storage.StorageError):
            storage.save_evaluation_record(self.root, replace(result.identity, bundle_id='wrong'), result)
        self.assertFalse(list((self.root / 'results/evaluations').rglob('*.json')))

    # Params: self; returns: None.
    def test_final_records_remain_separate_and_require_the_final_budget(self):
        final = self.result(final=True)
        path = self.save(final)
        self.assertIn('final', path.parts)
        self.assertFalse(storage.load_evaluation_records(self.root, self.config).validity.any())
        bad = replace(final, actual_step=2, identity=replace(final.identity, actual_step=2))
        with self.assertRaisesRegex(storage.StorageError, 'Final evaluation'):
            self.save(bad)

    # Params: self; returns: None.
    def test_corrupt_or_missing_numeric_payload_is_rejected_without_trusting_caches(self):
        result = self.result()
        path = self.save(result)
        numeric = path.with_suffix('.npz')
        original = numeric.read_bytes()
        numeric.write_bytes(original + b'tampered')
        with self.assertRaisesRegex(storage.StorageError, 'checksum'):
            storage.load_evaluation_records(self.root, self.config)
        numeric.unlink()
        with self.assertRaises(storage.StorageError): storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: None.
    def test_misplaced_commit_and_altered_metadata_are_rejected(self):
        result = self.result()
        path = self.save(result)
        original = path.read_bytes()
        meta = storage._read_json(path)
        meta['result']['episode_metrics'][0]['mean_coverage'] = 9.
        storage.atomic_write_json(path, meta)
        with self.assertRaisesRegex(storage.StorageError, 'semantic checksum'):
            storage.load_evaluation_records(self.root, self.config)
        path.write_bytes(original)
        wrong = path.with_name('step_999.json')
        path.rename(wrong)
        path.with_suffix('.npz').rename(wrong.with_suffix('.npz'))
        with self.assertRaisesRegex(storage.StorageError, 'path'):
            storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: None.
    def test_forged_checksum_cannot_hide_bad_dtype_or_inventory(self):
        result = self.result()
        path = self.save(result)
        _, arrays, _ = storage._encode_evaluation(result)
        arrays['returns_raw'] = arrays['returns_raw'].astype(np.float32)
        storage.atomic_write_npz(path.with_suffix('.npz'), arrays)
        meta = storage._read_json(path)
        meta['npz_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
        storage.atomic_write_json(path, meta)
        with self.assertRaisesRegex(storage.StorageError, 'float64'):
            storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: None.
    def test_changed_config_and_initialization_are_rejected(self):
        with self.assertRaises(storage.StorageError):
            storage.load_evaluation_records(self.root, replace(self.config,
                ppo=replace(self.config.ppo, gamma=.8),
                normalization=replace(self.config.normalization, gamma=.8)))
        (self.root / 'seed_plan.json').write_text('[]')
        with self.assertRaises(storage.StorageError): storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: None.
    def test_rebuilt_caches_and_hv_require_every_weight_and_ignore_old_cache_files(self):
        self.save(self.result(weight=0))
        first = storage.load_evaluation_records(self.root, self.config)
        self.assertTrue(np.isnan(first.hypervolume[0, 0]))
        self.save(self.result(weight=1))
        for name in ('eval_returns_raw', 'eval_returns_scaled', 'eval_valid', 'actual_steps', 'hypervolume'):
            (self.root / 'results' / (name + '.npy')).write_bytes(b'corrupt stale cache')
        rebuilt = storage.load_evaluation_records(self.root, self.config)
        self.assertEqual(rebuilt.hypervolume[0, 0], 1.)
        self.assertTrue(np.isnan(rebuilt.hypervolume[1]).all())
        for name, expected in (('eval_returns_raw', rebuilt.means_raw),
                ('eval_returns_scaled', rebuilt.means_scaled), ('eval_valid', rebuilt.validity),
                ('actual_steps', rebuilt.actual_budgets), ('hypervolume', rebuilt.hypervolume)):
            np.testing.assert_array_equal(np.load(self.root / 'results' / (name + '.npy'), allow_pickle=False), expected)

    # Params: self; returns: None.
    def test_cache_write_interruption_does_not_damage_authoritative_record(self):
        path = self.save(self.result())
        original = path.read_bytes(), path.with_suffix('.npz').read_bytes()
        with patch.object(storage, '_atomic_write', side_effect=OSError('cache failure')):
            with self.assertRaisesRegex(OSError, 'cache failure'):
                storage.load_evaluation_records(self.root, self.config)
        self.assertEqual(original, (path.read_bytes(), path.with_suffix('.npz').read_bytes()))
        self.assertTrue(storage.load_evaluation_records(self.root, self.config).validity[0, 0, 0])

    # Params: self; returns: None.
    def test_store_slot_is_idempotent_and_rejects_conflicting_episode_data(self):
        from agents.mo.types import ExperimentResults
        results = ExperimentResults.empty(2, [2, 4, 6], 2, 2)
        result = self.result(raw=[[1, 3], [3, 1]])
        storage.store_eval_slot(results, 0, 0, 0, result)
        storage.store_eval_slot(results, 0, 0, 0, result)
        self.assertTrue(results.validity[0, 0, 0])
        # Same vector means, different episodes: still a conflicting immutable record.
        with self.assertRaisesRegex(storage.StorageError, 'Conflicting'):
            storage.store_eval_slot(results, 0, 0, 0, self.result(raw=[[3, 1], [1, 3]]))
        np.testing.assert_array_equal(results.episode_returns[0, 0, 0], result.returns_raw)
        results.validate()

    # Params: self; returns: None.
    def test_slot_validation_is_before_mutation_including_actual_step_and_episode_count(self):
        from agents.mo.types import ExperimentResults
        results = ExperimentResults.empty(2, [2, 4, 6], 2, 1)
        for key in ((0, 0, 0), (0, 1, 0), (True, 0, 0), (-1, 0, 0), (2, 0, 0), (0, 0, 1)):
            with self.assertRaises(storage.StorageError):
                storage.store_eval_slot(results, *key, self.result())
        self.assertFalse(results.validity.any())
        self.assertTrue(np.isnan(results.means_raw).all())
        self.assertEqual(results.identities, {})

    # Params: self; returns: None.
    def test_slot_without_episode_tensor_still_rejects_same_mean_different_episodes(self):
        from agents.mo.types import ExperimentResults
        results = ExperimentResults.empty(2, [2, 4, 6], 2)
        first = self.result(raw=[[1, 3], [3, 1]])
        storage.store_eval_slot(results, 0, 0, 0, first)
        with self.assertRaisesRegex(storage.StorageError, 'Conflicting'):
            storage.store_eval_slot(results, 0, 0, 0, self.result(raw=[[3, 1], [1, 3]]))
        results.means_raw[0, 0, 0] = [99, 99]
        with self.assertRaisesRegex(storage.StorageError, 'Conflicting'):
            storage.store_eval_slot(results, 0, 0, 0, first)

    # Params: self; returns: None.
    def test_altered_array_inventory_is_rejected_even_with_updated_file_checksum(self):
        result = self.result()
        path = self.save(result)
        _, arrays, _ = storage._encode_evaluation(result)
        arrays['unexpected_array'] = np.zeros(1)
        storage.atomic_write_npz(path.with_suffix('.npz'), arrays)
        meta = storage._read_json(path)
        meta['npz_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
        storage.atomic_write_json(path, meta)
        with self.assertRaisesRegex(storage.StorageError, 'inventory'):
            storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: None.
    def test_incomplete_initialization_is_not_accepted_for_evaluation_storage(self):
        (self.root / '.INITIALIZING').touch()
        with self.assertRaisesRegex(storage.StorageError, 'initialization'):
            self.save(self.result())
        with self.assertRaisesRegex(storage.StorageError, 'initialization'):
            storage.load_evaluation_records(self.root, self.config)

    # Params: self; returns: None.
    def test_actual_policy_evaluation_saves_and_rebuilds_without_precision_changes(self):
        import torch
        from types import SimpleNamespace
        from agents.mo.envs import make_raw_env
        from agents.mo.evaluation import evaluate_policy
        identity = self.result().identity
        model = SimpleNamespace(policy=torch.nn.Linear(1, 1),
            predict=lambda observation, deterministic: (np.array([0, 0], np.int64), None))
        evaluated = evaluate_policy(model, lambda: make_raw_env(self.config.env),
            self.args[2][0].eval_seeds, self.config, identity, collect_traces=True)
        path = self.save(evaluated)
        restored, _ = storage._load_evaluation_record(path)
        rebuilt = storage.load_evaluation_records(self.root, self.config)
        np.testing.assert_array_equal(rebuilt.means_raw[0, 0, 0], evaluated.mean_raw)
        np.testing.assert_array_equal(rebuilt.episode_returns[0, 0, 0], evaluated.returns_raw)
        self.assertEqual(restored.episode_metrics, evaluated.episode_metrics)
        for left, right in zip(restored.traces, evaluated.traces):
            for name in storage._TRACE_ARRAYS:
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


class ResumeRecovery(unittest.TestCase):
    setUpClass = classmethod(RecoverablePolicyBundles.setUpClass.__func__)
    tearDownClass = classmethod(RecoverablePolicyBundles.tearDownClass.__func__)
    setUp = RecoverablePolicyBundles.setUp
    bundle_inputs = RecoverablePolicyBundles.bundle_inputs

    # Params: self, final, save; returns: (root, config, args).
    def case(self, final=False, save=True):
        args = self.bundle_inputs(final=final)
        if save:
            storage.save_policy_bundle(*args)
        root = next(p for p in args[2].parents if (p / 'manifest.json').exists())
        config = replace(self.plain[0], output_dir=root, requested_timesteps_per_policy=2 if final else 4)
        return root, config, args

    # Params: self, root, identity, step, final, weight; returns: (result, path).
    def evaluation(self, root, identity, step=None, final=False, weight=None):
        from agents.mo.types import EvaluationResult, EpisodeMetrics
        if step is not None:
            identity = replace(identity, actual_step=step, bundle_id=f'old-step-{step}')
        if weight is not None:
            identity = replace(identity, weight_index=weight, weight=(1., 0.))
        plan = storage._read_json(root / 'seed_plan.json')[identity.run_index]
        seeds = plan['final_seeds'] if final else plan['eval_seeds']
        raw = np.array([[1., 2.], [3., 4.]], np.float64)
        metric = EpisodeMetrics(
            mean_positional_error=0., mean_final_positional_error=0., mean_covariance_trace=1.,
            final_coverage=0., mean_coverage=0., unique_new_detections=0,
            mean_known_observation_frequency=0., mean_observation_frequency=0., decisions=1,
            elapsed_timesteps=1, unavailable_reasons={'search_time_percentage': 'PPO has no action mode'})
        sd = raw.std(axis=0, ddof=1)
        result = EvaluationResult(episode_seeds=seeds, returns_raw=raw, returns_scaled=raw,
            mean_raw=raw.mean(axis=0), mean_scaled=raw.mean(axis=0), sd_raw=sd, sd_scaled=sd,
            se_raw=sd / np.sqrt(2), se_scaled=sd / np.sqrt(2), episode_metrics=(metric, metric),
            deterministic=True, actual_step=identity.actual_step, identity=identity)
        storage.save_evaluation_record(root, identity, result)
        path = root / 'results' / ('final' if final else 'evaluations') / f'seed_{identity.run_index}' / f'w_{identity.weight_index}' / f'step_{identity.actual_step}.json'
        return result, path

    # Params: self; returns: None.
    def test_fresh_policy_starts_at_zero_with_saved_schedules(self):
        root, config, args = self.case(save=False)
        state = storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(state.current_step, 0)
        self.assertEqual(state.effective_target, 4)
        self.assertEqual(state.resume_attempt_id, 'initial')
        self.assertIsNone(state.selected_bundle_path)
        self.assertEqual(state.next_evaluation_budgets, (2, 4))
        self.assertEqual(state.next_checkpoint_budgets, (2, 4))
        self.assertFalse(state.complete)
        self.assertEqual(storage.resolve_resume_state(root, config, 0, 0), state)

    # Params: self; returns: None.
    def test_restored_missing_evaluation_is_repaired_before_any_more_training(self):
        root, config, args = self.case()
        state = storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(state.current_step, 2)
        self.assertEqual(state.selected_bundle_path, args[2])
        self.assertEqual(state.last_committed_checkpoint, args[2])
        self.assertEqual(state.completed_evaluation_ids, ())
        self.assertEqual(state.next_evaluation_budgets, (4,))
        self.assertIn(args[4].resume_attempt_id, state.canonical_attempt_ids)
        self.evaluation(root, args[3])
        repaired = storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(repaired.current_step, 2)
        self.assertEqual(repaired.resume_attempt_id, state.resume_attempt_id)
        self.assertEqual(repaired.completed_evaluation_ids, ('seed_0/w_0/step_2',))

    # Params: self; returns: None.
    def test_earlier_bytes_preserved_and_later_intermediate_final_and_bad_bundle_quarantined(self):
        root, config, args = self.case()
        _, early = self.evaluation(root, args[3])
        old, future = self.evaluation(root, args[3], step=4)
        _, final = self.evaluation(root, args[3], step=4, final=True)
        early_before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (early, early.with_suffix('.npz'))}
        future_before = {p.name: p.read_bytes() for p in (future, future.with_suffix('.npz'))}
        bad_bundle = args[2].parent / 'step_4'
        bad_bundle.mkdir(); (bad_bundle / 'broken.zip').write_bytes(b'incomplete bundle')
        state = storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(state.current_step, 2)
        self.assertFalse(future.exists()); self.assertFalse(final.exists()); self.assertFalse(bad_bundle.exists())
        self.assertTrue(state.superseded_attempt_ids)
        for path, expected in early_before.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
        archive = root / 'results/superseded/seed_0/w_0' / state.resume_attempt_id
        for name, expected in future_before.items():
            self.assertEqual((archive / 'evaluations' / name).read_bytes(), expected)
        self.assertTrue((archive / 'final/step_4.json').is_file())
        self.assertEqual((archive / 'bundles/step_4/broken.zip').read_bytes(), b'incomplete bundle')
        rebuilt = storage.load_evaluation_records(root, config)
        self.assertTrue(rebuilt.validity[0, 0, 0]); self.assertFalse(rebuilt.validity[0, 1, 0])
        history = storage._read_json(root / 'results/resume_history/seed_0/w_0' / (state.resume_attempt_id + '.json'))
        self.assertEqual(history['archived_evaluations'][0]['identity'], old.identity.to_dict())
        self.assertEqual(history['archived_evaluations'][0]['source_attempt_id'], 'initial')
        self.assertIn('not exact', history['continuation'])
        self.assertEqual(storage.resolve_resume_state(root, config, 0, 0), state)

    # Params: self; returns: None.
    def test_in_memory_reconciliation_clears_only_the_selected_future_slots(self):
        root, config, args = self.case()
        self.evaluation(root, args[3])
        old, _ = self.evaluation(root, args[3], step=4)
        self.evaluation(root, args[3], step=4, weight=1)
        results = storage.load_evaluation_records(root, config)
        other = results.means_raw[0, 1, 1].copy()
        state = storage.reconcile_evaluations_after_checkpoint(results, 2, args[3])
        self.assertTrue(results.validity[0, 0, 0]); self.assertFalse(results.validity[0, 1, 0])
        np.testing.assert_array_equal(results.means_raw[0, 1, 1], other)
        self.assertTrue(np.isnan(results.episode_returns[0, 1, 0]).all())
        self.assertNotIn((0, 1, 0), results.identities)
        self.assertEqual(results._superseded_evaluations[0]['identity'], old.identity)
        np.testing.assert_array_equal(results._superseded_evaluations[0]['episode_returns'], old.returns_raw)
        self.assertEqual(state.completed_evaluation_ids, ('seed_0/w_0/step_2',))
        results.validate()

    # Params: self; returns: None.
    def test_different_checkpoint_and_eval_schedules_are_not_conflated(self):
        from agents.mo.types import PolicyIdentity
        config = replace(self.plain[0], output_dir=self.root / 'schedules',
                         requested_timesteps_per_policy=6, eval_interval=1, checkpoint_interval=3)
        args = inputs(config); root = storage.create_experiment_dir(*args)
        identity = PolicyIdentity(storage.experiment_fingerprint(*args), 7, 0, 0, (0., 1.), 0, 'initial')
        results = storage.load_evaluation_records(root, config)
        state = storage.reconcile_evaluations_after_checkpoint(results, 0, identity)
        self.assertEqual(state.next_evaluation_budgets, (2, 4, 6))
        self.assertEqual(state.next_checkpoint_budgets, (4, 6))

    # Params: self; returns: None.
    def test_invalid_reconciliation_is_rejected_before_mutating_arrays(self):
        root, config, args = self.case()
        self.evaluation(root, args[3], step=4)
        results = storage.load_evaluation_records(root, config)
        before = results.means_raw.copy()
        for step, identity in ((3, args[3]), (2, replace(args[3], weight=(1., 0.))),
                (2, replace(args[3], experiment_fingerprint='wrong')), (True, args[3])):
            with self.assertRaises(storage.StorageError):
                storage.reconcile_evaluations_after_checkpoint(results, step, identity)
        np.testing.assert_array_equal(results.means_raw, before)

    # Params: self; returns: None.
    def test_final_budget_missing_evaluation_can_finish_without_learning(self):
        root, config, args = self.case(final=True)
        with patch.object(args[0], 'learn') as learn:
            state = storage.resolve_resume_state(root, config, 0, 0)
            self.assertEqual(state.current_step, state.effective_target)
            self.assertEqual(state.next_evaluation_budgets, ())
            self.assertFalse(state.complete)
            _, path = self.evaluation(root, args[3])
            storage.mark_policy_complete(args[2].parent, args[2], path)
            complete = storage.resolve_resume_state(root, config, 0, 0)
            self.assertTrue(complete.complete)
            self.assertEqual(complete.resume_attempt_id, state.resume_attempt_id)
            self.assertEqual(complete.canonical_attempt_ids, state.canonical_attempt_ids)
            learn.assert_not_called()

    # Params: self; returns: None.
    def test_a_checkpoint_at_final_budget_without_final_bundle_is_not_complete(self):
        root, config, args = self.case(final=True)
        checkpoint = args[2].parent / 'checkpoints/step_2'
        checkpoint.parent.mkdir(parents=True)
        # Keep a real compatible checkpoint, not the final directory.
        storage.save_policy_bundle(*args[:2], checkpoint, *args[3:])
        quarantine = self.root / 'preserved-final'
        args[2].resolve().relative_to(self.root.resolve())
        quarantine.resolve().relative_to(self.root.resolve())
        args[2].rename(quarantine)
        state = storage.resolve_resume_state(root, config, 0, 0)
        self.assertFalse(state.complete)
        self.assertEqual(state.current_step, 2)
        self.assertEqual(state.selected_bundle_path, checkpoint)
        self.assertFalse((checkpoint.parent.parent / 'COMPLETE.json').exists())

    # Params: self; returns: None.
    def test_corrupt_completion_or_unrecoverable_bundle_never_silently_restarts(self):
        root, config, args = self.case()
        marker = args[2].parents[1] / 'COMPLETE.json'
        marker.write_text('{}')
        with self.assertRaises(storage.StorageError): storage.resolve_resume_state(root, config, 0, 0)
        marker.unlink()
        (args[2] / 'model.zip').write_bytes(b'broken')
        with self.assertRaisesRegex(storage.StorageError, 'no recoverable'):
            storage.resolve_resume_state(root, config, 0, 0)

    # Params: self; returns: None.
    def test_interrupted_move_blocks_read_write_and_retries_from_journal(self):
        root, config, args = self.case()
        old, path = self.evaluation(root, args[3], step=4)
        rename = os.rename
        calls = []
        # Params: source, destination; returns: rename result.
        def fail_second(source, destination):
            calls.append(str(source))
            if len(calls) == 2: raise OSError('move interrupted')
            return rename(source, destination)
        with patch.object(storage.os, 'rename', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'move interrupted'):
                storage.resolve_resume_state(root, config, 0, 0)
        self.assertFalse(path.exists()); self.assertTrue(path.with_suffix('.npz').exists())
        with self.assertRaisesRegex(storage.StorageError, 'pending'):
            storage.load_evaluation_records(root, config)
        with self.assertRaisesRegex(storage.StorageError, 'pending'):
            storage.save_evaluation_record(root, old.identity, old)
        recovered = storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(recovered.current_step, 2)
        self.assertFalse(path.with_suffix('.npz').exists())
        self.assertFalse(storage.load_evaluation_records(root, config).validity.any())

    # Params: self; returns: None.
    def test_interrupted_ready_marker_is_idempotently_repaired(self):
        root, config, args = self.case()
        self.evaluation(root, args[3], step=4)
        write = storage.atomic_write_json
        # Params: path, payload; returns: write result.
        def fail_ready(path, payload):
            if Path(path).name == 'RESUME.json' and payload.get('status') == 'ready':
                raise OSError('ready interrupted')
            return write(path, payload)
        with patch.object(storage, 'atomic_write_json', side_effect=fail_ready):
            with self.assertRaisesRegex(OSError, 'ready interrupted'):
                storage.resolve_resume_state(root, config, 0, 0)
        journal = root / 'results/evaluations/seed_0/w_0/RESUME.json'
        pending = storage._read_json(journal)
        recovered = storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(recovered.resume_attempt_id, pending['state']['resume_attempt_id'])
        self.assertEqual(storage._read_json(journal)['status'], 'ready')

    # Params: self; returns: None.
    def test_failed_plan_publication_does_not_move_any_record(self):
        root, config, args = self.case()
        _, path = self.evaluation(root, args[3], step=4)
        before = path.read_bytes(), path.with_suffix('.npz').read_bytes()
        with patch.object(storage, 'atomic_write_json', side_effect=OSError('journal failure')):
            with self.assertRaises(OSError): storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(before, (path.read_bytes(), path.with_suffix('.npz').read_bytes()))
        self.assertTrue(storage.load_evaluation_records(root, config).validity[0, 1, 0])

    # Params: self; returns: None.
    def test_journal_tampering_and_source_changes_are_not_replayed(self):
        root, config, args = self.case()
        _, path = self.evaluation(root, args[3], step=4)
        with patch.object(storage.os, 'rename', side_effect=OSError('paused')):
            with self.assertRaises(OSError): storage.resolve_resume_state(root, config, 0, 0)
        journal = path.parent / 'RESUME.json'
        original = journal.read_bytes()
        plan = storage._read_json(journal)
        plan['moves'][0]['destination'] = '../outside.json'
        storage.atomic_write_json(journal, plan)
        with self.assertRaises(storage.StorageError): storage.resolve_resume_state(root, config, 0, 0)
        journal.write_bytes(original)
        path.with_suffix('.npz').write_bytes(b'changed after planning')
        with self.assertRaisesRegex(storage.StorageError, 'changed'):
            storage.resolve_resume_state(root, config, 0, 0)

    # Params: self; returns: None.
    def test_stale_future_records_cannot_reenter_a_new_trajectory(self):
        root, config, args = self.case()
        old, path = self.evaluation(root, args[3], step=4)
        storage.resolve_resume_state(root, config, 0, 0)
        with self.assertRaisesRegex(storage.StorageError, 'active resume attempt'):
            storage.save_evaluation_record(root, old.identity, old)
        self.assertFalse(path.exists())

    # Params: self; returns: None.
    def test_manually_restored_superseded_files_do_not_enter_curves(self):
        root, config, args = self.case()
        _, path = self.evaluation(root, args[3], step=4)
        old = path.read_bytes(), path.with_suffix('.npz').read_bytes()
        storage.resolve_resume_state(root, config, 0, 0)
        path.write_bytes(old[0]); path.with_suffix('.npz').write_bytes(old[1])
        with self.assertRaisesRegex(storage.StorageError, 'active resumed trajectory'):
            storage.load_evaluation_records(root, config)

    # Params: self; returns: None.
    def test_resolution_does_not_consume_training_rng_or_modify_the_learner(self):
        import random
        import torch
        root, config, args = self.case()
        self.evaluation(root, args[3], step=4)
        rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        parameters = [p.detach().clone() for p in args[0].policy.parameters()]
        storage.resolve_resume_state(root, config, 0, 0)
        self.assertEqual(random.getstate(), rng[0])
        after = np.random.get_state()
        self.assertEqual(after[0], rng[1][0]); np.testing.assert_array_equal(after[1], rng[1][1])
        self.assertEqual(after[2:], rng[1][2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng[2]))
        self.assertEqual(args[0].num_timesteps, 2)
        for original, parameter in zip(parameters, args[0].policy.parameters()):
            self.assertTrue(torch.equal(original, parameter))

    # Params: self; returns: None.
    def test_new_attempt_bundle_can_publish_its_own_later_evaluation(self):
        from agents.mo.types import PolicyIdentity
        root, config, args = self.case()
        self.evaluation(root, args[3])
        self.evaluation(root, args[3], step=4)
        state = storage.resolve_resume_state(root, config, 0, 0)
        with storage.load_policy_bundle(args[2], config, training_handle=args[1]) as bundle:
            self.assertEqual(bundle.model.num_timesteps, 2)
            bundle.model.learn(2, reset_num_timesteps=False)
            self.assertEqual(bundle.model.num_timesteps, 4)
            identity = replace(args[3], actual_step=4, bundle_id='resumed-update-4')
            snapshot = replace(state, current_step=4, next_evaluation_budgets=(), next_checkpoint_budgets=())
            destination = args[2].parent / 'step_4'
            storage.save_policy_bundle(bundle.model, args[1], destination, identity, snapshot)
            _, path = self.evaluation(root, identity)
            final = args[2].parents[1] / 'final'
            storage.save_policy_bundle(bundle.model, args[1], final, identity, snapshot)
            storage.mark_policy_complete(final.parent, final, path)
        self.assertEqual(storage._read_json(path)['resume_attempt_id'], state.resume_attempt_id)
        rebuilt = storage.load_evaluation_records(root, config)
        self.assertTrue(rebuilt.validity[0, 1, 0])
        self.assertEqual(rebuilt.identities[(0, 1, 0)].bundle_id, 'resumed-update-4')
        complete = storage.resolve_resume_state(root, config, 0, 0)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.resume_attempt_id, state.resume_attempt_id)
        self.assertEqual(complete.completed_evaluation_ids, ('seed_0/w_0/step_2', 'seed_0/w_0/step_4'))

    # Params: self; returns: None.
    def test_resume_rejects_invalid_indices_config_and_held_locks(self):
        root, config, args = self.case()
        for run, weight in ((-1, 0), (True, 0), (0, 2)):
            with self.assertRaises(storage.StorageError): storage.resolve_resume_state(root, config, run, weight)
        with self.assertRaises(storage.StorageError):
            storage.resolve_resume_state(root, replace(config, env=replace(config.env, episode_steps=3)), 0, 0)
        directory = root / 'results/evaluations/seed_0/w_0'
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / 'step_2.LOCK'; lock.touch()
        with self.assertRaisesRegex(storage.StorageError, 'writer'):
            storage.resolve_resume_state(root, config, 0, 0)
        self.assertTrue(lock.exists())


if __name__ == '__main__':
    unittest.main()
