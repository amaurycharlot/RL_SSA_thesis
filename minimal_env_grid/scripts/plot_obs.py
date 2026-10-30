import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Params: rollout_dir; returns: dict.
def load_meta(rollout_dir: str) -> dict:
    meta_path = os.path.join(rollout_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Could not find meta.json in {rollout_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Params: rollout_dir; returns: np.ndarray.
def load_observations(rollout_dir: str) -> np.ndarray:
    obs_path = os.path.join(rollout_dir, "observations.npy")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(
            f"Could not find observations.npy in {rollout_dir}. "
            "Make sure run_agent.py saves observations."
        )

    return np.load(obs_path)


# Params: meta; returns: tuple[int, int].
def infer_grid_shape(meta: dict) -> tuple[int, int]:
    if "grid_shape" in meta:
        grid_shape = meta["grid_shape"]
        return int(grid_shape[0]), int(grid_shape[1])

    print(
        "Warning: meta.json does not contain 'grid_shape'. ",
        "Add \"grid_shape\": list(env.grid_shape) to the rollout metadata.",
        "Assuming default grid shape of (10, 10) for plotting, but this may be incorrect."
    )
    return (10, 10)  # default fallback, but may be incorrect


# Params: meta; returns: str.
def infer_mode(meta: dict) -> str:
    if "mode" not in meta:
        raise KeyError(
            "meta.json does not contain 'mode'. "
            "Add \"mode\": args.mode to the rollout metadata."
        )

    mode = str(meta["mode"]).lower().strip()
    if mode not in {"track", "search", "combined"}:
        raise ValueError(f"Invalid mode in meta.json: {mode}")

    return mode


def extract_components(obs: np.ndarray, mode: str, grid_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """
    Returns dict:
        component_name -> array of shape (T, N_grid)
    """

    n_grid = int(grid_shape[0] * grid_shape[1])

    if mode in {"track", "search"}:
        obs = obs.reshape(obs.shape[0], n_grid)

        if mode == "track":
            return {"uncertainty": obs}

        return {"recency": obs}

    if mode == "combined":
        obs = obs.reshape(obs.shape[0], n_grid, 2)

        return {
            "uncertainty": obs[:, :, 0],
            "recency": obs[:, :, 1],
        }

    raise ValueError(f"Unsupported mode: {mode}")


def plot_component(values: np.ndarray, name: str, outdir: str, show: bool = False) -> None:
    """
    values shape: (T, N_grid)
    """

    timesteps = np.arange(values.shape[0])
    min_values = values.min(axis=1)
    max_values = values.max(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, min_values, label=f"min {name}")
    plt.plot(timesteps, max_values, label=f"max {name}")

    plt.xlabel("Timestep")
    plt.ylabel(name)
    plt.title(f"Observation component range over grid cells: {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(outdir, f"obs_{name}_min_max.png")
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()

    plt.close()
    print(f"Saved {output_path}")


# Params: none; returns: None.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dir", type=str, help="Path to saved rollout directory")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--mode", type=str, default=None, help="Environment mode: track | search | combined (overrides meta.json)")
    args = parser.parse_args()

    meta = load_meta(args.rollout_dir)
    mode = args.mode if args.mode is not None else infer_mode(meta)
    grid_shape = infer_grid_shape(meta)

    observations = load_observations(args.rollout_dir)

    components = extract_components(
        obs=observations,
        mode=mode,
        grid_shape=grid_shape,
    )

    print(f"Loaded rollout: {args.rollout_dir}")
    print(f"Mode: {mode}")
    print(f"Grid shape: {grid_shape}")
    print(f"Observations shape: {observations.shape}")

    for name, values in components.items():
        plot_component(
            values=values,
            name=name,
            outdir=args.rollout_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()