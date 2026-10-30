"""Validate spawn execution and benchmark it without touching existing runs.

Run from the project root: python -B tests/validate_parallel_training.py
Use --ppo to also benchmark one complete, unchanged PPO rollout/update per mode.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from agents.train.train_ppo_combined import (
    build_parser, create_combined_env, create_train_env, make_env,
)
from agents.RL.grid_cnn_extractor import GridCNNExtractor
import agents.train.train_ppo_combined as trainer


# Params: left, right, path; returns: None.
def assert_equal(left, right, path="result"):
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            # Monitor's episode wall time is intentionally different.
            if key == "t" and path.endswith(".episode"):
                continue
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{i}]")
    else:
        assert left == right, f"{path}: {left!r} != {right!r}"


# Params: args, actions, disturb; returns: (trajectory, dict).
def collect(args, actions, disturb=False):
    parent_threads = torch.get_num_threads()
    parent_environment = dict(os.environ)
    started = time.perf_counter()
    env = create_train_env(args)
    workers = list(getattr(env, "processes", []))
    evaluation = None
    try:
        if args.subproc:
            assert len(workers) == 8 and all(p.is_alive() for p in workers)
        assert torch.get_num_threads() == parent_threads
        assert dict(os.environ) == parent_environment
        env.seed(args.seed)
        initial = env.reset()
        startup = time.perf_counter() - started
        if disturb:
            evaluation = DummyVecEnv([make_env(10_000, args)])
            evaluation.reset()
        trajectory = [copy.deepcopy((initial, env.reset_infos))]
        dones = np.zeros(8, dtype=int)
        started = time.perf_counter()
        for i, action in enumerate(actions):
            if disturb and i % 8 == 0:
                np.random.seed(i + 765)
                np.random.normal(size=1000)
                evaluation.step(action[:1])
            result = env.step(action)
            dones += result[2]
            trajectory.append(copy.deepcopy(result))
        elapsed = time.perf_counter() - started
        assert np.all(dones > 0), f"Missing automatic episode resets: {dones}"
    finally:
        if evaluation is not None:
            evaluation.close()
        env.close()
    assert all(not p.is_alive() and p.exitcode == 0 for p in workers)
    return trajectory, {
        "mode": "parallel" if args.subproc else "sequential",
        "worker_threads": args.worker_threads if args.subproc else None,
        "startup_and_reset_seconds": startup,
        "collection_seconds": elapsed,
        "transitions_per_second": len(actions) * 8 / elapsed,
        "episode_resets": dones.tolist(),
        "workers_closed": len(workers),
        "disturbed_rng_and_evaluation": disturb,
    }


class RolloutTimer(BaseCallback):
    # Params: self; returns: bool.
    def _on_step(self):
        return True

    # Params: self; returns: None.
    def _on_rollout_start(self):
        self.started = time.perf_counter()

    # Params: self; returns: None.
    def _on_rollout_end(self):
        self.collection_seconds = time.perf_counter() - self.started


# Params: args, output; returns: dict.
def ppo_cycle(args, output):
    parent_threads = torch.get_num_threads()
    started = time.perf_counter()
    raw = create_train_env(args)
    workers = list(getattr(raw, "processes", []))
    env = VecMonitor(raw)
    try:
        timer = RolloutTimer()
        model = PPO(
            policy="MlpPolicy", env=env,
            policy_kwargs={"features_extractor_class": GridCNNExtractor,
                           "features_extractor_kwargs": {"features_dim": 128}},
            learning_rate=args.learning_rate, n_steps=args.n_steps,
            batch_size=args.batch_size, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm, verbose=0, seed=args.seed,
        )
        setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        model.learn(total_timesteps=args.n_envs * args.n_steps, callback=timer)
        learn_seconds = time.perf_counter() - started
        model.save(str(output / ("parallel_model" if args.subproc else "sequential_model")))
        assert model.num_timesteps == 4096
        assert model.n_epochs == 10
        assert torch.get_num_threads() == parent_threads
        return {
            "mode": "parallel" if args.subproc else "sequential",
            "setup_seconds": setup_seconds,
            "collection_seconds": timer.collection_seconds,
            "learn_seconds": learn_seconds,
            "update_and_learn_setup_seconds": learn_seconds - timer.collection_seconds,
            "transitions_per_second": model.num_timesteps / learn_seconds,
            "learner_threads": parent_threads,
            "optimisation_epochs": model.n_epochs,
        }
    finally:
        env.close()
        assert all(not p.is_alive() and p.exitcode == 0 for p in workers)


# Params: none; returns: never (raises).
def fail_worker_startup():
    raise RuntimeError("Expected validation-only worker startup failure")


# Params: output; returns: None.
def validate_failure_cleanup(output):
    previous_children = {p.pid for p in mp.active_children()}
    previous_environment = dict(os.environ)
    args = build_parser().parse_args([])
    try:
        with trainer.worker_thread_environment(1):
            trainer.TrainingSubprocVecEnv(
                [fail_worker_startup, make_env(1, args, 1)], start_method="spawn"
            )
    except EOFError:
        pass
    else:
        raise AssertionError("Failed startup did not propagate")
    assert {p.pid for p in mp.active_children()} == previous_children
    assert dict(os.environ) == previous_environment

    original_ppo, original_argv = trainer.PPO, sys.argv
    # Params: *args, **kwargs; returns: never (raises).
    def fail_model(*args, **kwargs):
        raise RuntimeError("Expected validation-only learner setup failure")
    try:
        trainer.PPO = fail_model
        sys.argv = ["train_ppo_combined", "--outdir", str(output),
                    "--run_name", "expected_failure"]
        try:
            trainer.main()
        except RuntimeError as exc:
            assert "validation-only learner setup failure" in str(exc)
        else:
            raise AssertionError("Failed learner setup did not propagate")
    finally:
        trainer.PPO, sys.argv = original_ppo, original_argv
    assert {p.pid for p in mp.active_children()} == previous_children
    assert dict(os.environ) == previous_environment
    print("Startup-failure and learner-failure cleanup passed.", flush=True)


# Params: none; returns: None.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    output = opts.output or Path(tempfile.mkdtemp(prefix="ppo_parallel_validation_"))
    output.mkdir(parents=True, exist_ok=True)
    if opts.cleanup_only:
        validate_failure_cleanup(output)
        return
    args = build_parser().parse_args([])
    assert args.subproc and args.n_envs == 8
    assert build_parser().parse_args(["--subproc"]).subproc
    assert not build_parser().parse_args(["--no-subproc"]).subproc
    assert not build_parser().parse_args(["--sequential"]).subproc

    # Explicit reset restores the same stream; implicit reset continues it.
    env = create_combined_env(args, args.seed)
    try:
        first = env.reset(seed=123)
        env.step(np.array([0, 0]))
        assert_equal(first, env.reset(seed=123), "explicit_reset")
        assert env.unwrapped.sensor.rng is env.unwrapped.np_random
    finally:
        env.close()

    actions = np.random.default_rng(2026).integers(0, 10, size=(opts.steps, 8, 2))
    report = {"python": sys.executable, "torch_version": torch.__version__,
              "learner_threads": torch.get_num_threads(), "collection": [], "ppo": []}
    baseline = None
    for parallel, threads, disturb in (
        (False, 1, False), (True, 1, False),
        (True, 1, True), (True, 2, False),
    ):
        args.subproc, args.worker_threads = parallel, threads
        trajectory, result = collect(args, actions, disturb)
        if baseline is None:
            baseline = trajectory
        else:
            assert_equal(baseline, trajectory)
        report["collection"].append(result)
        print(json.dumps(result), flush=True)
    report["exact_trajectory_comparisons_passed"] = True
    report["explicit_seed_reset_passed"] = True
    args.worker_threads = 1
    if opts.ppo:
        for parallel in (False, True):
            args.subproc = parallel
            result = ppo_cycle(args, output)
            report["ppo"].append(result)
            print(json.dumps(result), flush=True)
    (output / "results.json").write_text(json.dumps(report, indent=2))
    print(f"Validation passed. Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
