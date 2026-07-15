# scripts/plot_training_run.py

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


def load_eval_mean_return(eval_log_dir: Path):
    """
    Load mean return vs timesteps from SB3 EvalCallback output.

    Expects:
        eval_log_dir / "evaluations.npz"
    """
    eval_file = eval_log_dir / "evaluations.npz"
    if not eval_file.exists():
        raise FileNotFoundError(f"Could not find eval file: {eval_file}")

    data = np.load(eval_file)
    timesteps = data["timesteps"]                    # shape: (n_evals,)
    results = data["results"]                        # shape: (n_evals, n_eval_episodes)
    mean_returns = results.mean(axis=1)

    return timesteps, mean_returns


def find_latest_event_file(tb_dir: Path) -> Path:
    """
    Find the most recent TensorBoard event file in a directory tree.
    """
    event_files = sorted(tb_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event files found under: {tb_dir}\n"
            f"Did you train with use_tensorboard=True?"
        )
    return event_files[-1]


def load_tb_scalars(event_file: Path, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load selected scalar tags from a TensorBoard event file.

    Returns:
        dict[tag] = (steps, values)
    """
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()

    available_tags = set(ea.Tags().get("scalars", []))
    out = {}

    for tag in tags:
        if tag not in available_tags:
            print(f"[warning] TensorBoard tag not found: {tag}")
            continue

        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        values = np.array([e.value for e in events], dtype=np.float32)
        out[tag] = (steps, values)

    return out


def make_plot(x, y, xlabel, ylabel, title, save_path: Path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def extract_step_from_name(path: Path) -> int | None:
    """
    Extract timestep from checkpoint filename like:
    ppo_ssa_50000_steps.zip
    """
    m = re.search(r"_(\d+)_steps", path.stem)
    if m is None:
        return None
    return int(m.group(1))


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for ckpt in checkpoint_dir.glob("*.zip"):
        step = extract_step_from_name(ckpt)
        if step is not None:
            checkpoints.append((step, ckpt))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def find_vecnormalize_for_checkpoint(checkpoint_dir: Path, step: int, run_dir: Path) -> Path | None:
    """
    Try to find a matching VecNormalize checkpoint file.
    Falls back to final vecnormalize.pkl if no per-checkpoint file is found.
    """
    patterns = [
        f"*{step}_steps*.pkl",
        f"*{step}*.pkl",
    ]
    for pattern in patterns:
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            return matches[0]

    final_vecnorm = run_dir / "vecnormalize.pkl"
    if final_vecnorm.exists():
        return final_vecnorm

    return None


def make_eval_env(num_targets: int, measurement_noise_std: float):
    def _make():
        return MinimalSSAEnv(
            num_targets=num_targets,
            measurement_noise_std=measurement_noise_std,
        )
    return _make


def load_model_for_eval(
    checkpoint_path: Path,
    vecnorm_path: Path | None,
    env_fn,
):
    venv = DummyVecEnv([env_fn])
    venv = VecMonitor(venv)

    if vecnorm_path is not None and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(str(checkpoint_path), env=venv)
    return model, venv


def evaluate_checkpoint(
    checkpoint_path: Path,
    vecnorm_path: Path | None,
    num_targets: int,
    measurement_noise_std: float,
    n_episodes: int,
    episode_steps: int,
    seed: int,
):
    """
    Evaluate one checkpoint and compute:
      - mean position uncertainty
      - final uncertainty
      - target selection frequency
    """
    env_fn = make_eval_env(num_targets=num_targets, measurement_noise_std=measurement_noise_std)
    model, venv = load_model_for_eval(checkpoint_path, vecnorm_path, env_fn)

    per_episode_mean_unc = []
    per_episode_final_unc = []
    action_counts = np.zeros(num_targets, dtype=np.int64)
    total_actions = 0

    for ep in range(n_episodes):
        obs = venv.reset()
        done = False
        step_count = 0

        unc_over_steps = []

        while not done and step_count < episode_steps:
            action, _ = model.predict(obs, deterministic=True)

            # action shape for DummyVecEnv is usually (1,)
            action_idx = int(np.asarray(action).reshape(-1)[0])
            if 0 <= action_idx < num_targets:
                action_counts[action_idx] += 1
            total_actions += 1

            # observation is normalized if VecNormalize is active.
            # We want the raw observation to compute physical uncertainty.
            raw_obs = venv.get_original_obs()
            raw_obs = np.asarray(raw_obs).reshape(1, num_targets, 3)
            trace_pos = raw_obs[0, :, 2]
            unc_over_steps.append(float(np.mean(trace_pos)))

            obs, rewards, dones, infos = venv.step(action)
            done = bool(dones[0])
            step_count += 1

        # final raw observation after last step
        raw_obs = venv.get_original_obs()
        raw_obs = np.asarray(raw_obs).reshape(1, num_targets, 3)
        final_trace_pos = raw_obs[0, :, 2]

        if len(unc_over_steps) > 0:
            per_episode_mean_unc.append(float(np.mean(unc_over_steps)))
        else:
            per_episode_mean_unc.append(np.nan)

        per_episode_final_unc.append(float(np.mean(final_trace_pos)))

    venv.close()

    selection_freq = action_counts / max(total_actions, 1)

    return {
        "mean_position_uncertainty": float(np.nanmean(per_episode_mean_unc)),
        "final_uncertainty": float(np.nanmean(per_episode_final_unc)),
        "target_selection_frequency": selection_freq,
    }


def plot_target_selection_frequency(
    timesteps: np.ndarray,
    freq_matrix: np.ndarray,
    save_path: Path,
):
    """
    freq_matrix shape: (n_checkpoints, num_targets)
    """
    plt.figure(figsize=(7, 4.5))
    num_targets = freq_matrix.shape[1]

    for i in range(num_targets):
        plt.plot(timesteps, freq_matrix[:, i], label=f"Target {i}")

    plt.xlabel("Timesteps")
    plt.ylabel("Selection frequency")
    plt.title("Target selection frequency vs timesteps")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def compute_and_plot_ssa_metrics(
    run_dir: Path,
    num_targets: int,
    measurement_noise_std: float,
    n_eval_episodes: int,
    episode_steps: int,
    seed: int,
    out_dir: Path,
):
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"[warning] No checkpoints found in {checkpoint_dir}")
        return

    timesteps = []
    mean_uncs = []
    final_uncs = []
    selection_freqs = []

    for step, ckpt_path in checkpoints:
        vecnorm_path = find_vecnormalize_for_checkpoint(checkpoint_dir, step, run_dir)

        print(f"Evaluating checkpoint {ckpt_path.name} at step {step} ...")
        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            vecnorm_path=vecnorm_path,
            num_targets=num_targets,
            measurement_noise_std=measurement_noise_std,
            n_episodes=n_eval_episodes,
            episode_steps=episode_steps,
            seed=seed,
        )

        timesteps.append(step)
        mean_uncs.append(metrics["mean_position_uncertainty"])
        final_uncs.append(metrics["final_uncertainty"])
        selection_freqs.append(metrics["target_selection_frequency"])

    timesteps = np.asarray(timesteps, dtype=np.int64)
    mean_uncs = np.asarray(mean_uncs, dtype=np.float32)
    final_uncs = np.asarray(final_uncs, dtype=np.float32)
    selection_freqs = np.asarray(selection_freqs, dtype=np.float32)

    make_plot(
        timesteps,
        mean_uncs,
        xlabel="Timesteps",
        ylabel="Mean position uncertainty",
        title="Mean position uncertainty vs timesteps",
        save_path=out_dir / "mean_position_uncertainty_vs_timesteps.png",
    )
    print(f"Saved: {out_dir / 'mean_position_uncertainty_vs_timesteps.png'}")

    make_plot(
        timesteps,
        final_uncs,
        xlabel="Timesteps",
        ylabel="Final uncertainty",
        title="Final uncertainty vs timesteps",
        save_path=out_dir / "final_uncertainty_vs_timesteps.png",
    )
    print(f"Saved: {out_dir / 'final_uncertainty_vs_timesteps.png'}")

    plot_target_selection_frequency(
        timesteps,
        selection_freqs,
        save_path=out_dir / "target_selection_frequency_vs_timesteps.png",
    )
    print(f"Saved: {out_dir / 'target_selection_frequency_vs_timesteps.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="runs/ppo_baseline")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Mean return vs timesteps from EvalCallback
    # ------------------------------------------------------------------
    eval_log_dir = run_dir / "eval_logs"
    try:
        eval_steps, mean_returns = load_eval_mean_return(eval_log_dir)
        make_plot(
            eval_steps,
            mean_returns,
            xlabel="Timesteps",
            ylabel="Mean return",
            title="Mean return vs timesteps",
            save_path=out_dir / "mean_return_vs_timesteps.png",
        )
        print(f"Saved: {out_dir / 'mean_return_vs_timesteps.png'}")
    except FileNotFoundError as e:
        print(f"[warning] {e}")

    # ------------------------------------------------------------------
    # 2) PPO internal metrics from TensorBoard
    # ------------------------------------------------------------------
    tb_dir = run_dir / "tb"
    try:
        event_file = find_latest_event_file(tb_dir)
        print(f"Using TensorBoard file: {event_file}")

        # Common SB3 TensorBoard tags
        tags = [
            "train/entropy_loss",
            "train/approx_kl",
            "train/explained_variance",
        ]
        tb_data = load_tb_scalars(event_file, tags)

        # Policy entropy:
        # SB3 logs entropy_loss, which is usually negative entropy term.
        # For presentation, it is more intuitive to plot positive entropy,
        # so we flip the sign.
        if "train/entropy_loss" in tb_data:
            steps, values = tb_data["train/entropy_loss"]
            policy_entropy = -values
            make_plot(
                steps,
                policy_entropy,
                xlabel="Timesteps",
                ylabel="Policy entropy (approx.)",
                title="Policy entropy vs timesteps",
                save_path=out_dir / "policy_entropy_vs_timesteps.png",
            )
            print(f"Saved: {out_dir / 'policy_entropy_vs_timesteps.png'}")

        if "train/approx_kl" in tb_data:
            steps, values = tb_data["train/approx_kl"]
            make_plot(
                steps,
                values,
                xlabel="Timesteps",
                ylabel="Approximate KL",
                title="Approximate KL vs timesteps",
                save_path=out_dir / "approx_kl_vs_timesteps.png",
            )
            print(f"Saved: {out_dir / 'approx_kl_vs_timesteps.png'}")

        if "train/explained_variance" in tb_data:
            steps, values = tb_data["train/explained_variance"]
            make_plot(
                steps,
                values,
                xlabel="Timesteps",
                ylabel="Explained variance",
                title="Explained variance vs timesteps",
                save_path=out_dir / "explained_variance_vs_timesteps.png",
            )
            print(f"Saved: {out_dir / 'explained_variance_vs_timesteps.png'}")

    except FileNotFoundError as e:
        print(f"[warning] {e}")
        print("[warning] The three PPO-internal plots require TensorBoard logs.")

    print(f"Done. Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()