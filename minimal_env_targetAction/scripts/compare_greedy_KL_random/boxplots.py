import os
import json
import glob
import argparse
import matplotlib.pyplot as plt


METRICS = [
    "mean_reward",
    "mean_pos_error",
    "mean_3sigma",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_greedy_meta(indir: str):
    path = os.path.join(indir, "greedy_kl", "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find greedy meta file: {path}")
    return load_json(path)


def find_random_metas(indir: str):
    paths = sorted(glob.glob(os.path.join(indir, "random_seed_*", "meta.json")))
    if not paths:
        raise FileNotFoundError(
            f"Could not find any random meta files under: {os.path.join(indir, 'random_seed_*', 'meta.json')}"
        )
    return [load_json(p) for p in paths]


def make_boxplot(metric_name: str, greedy_value: float, random_values, outpath: str):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.boxplot(
        [random_values],
        labels=["Random agents"],
        widths=0.45,
        showmeans=True,
    )

    ax.axhline(
        greedy_value,
        linestyle=":",
        linewidth=2.5,
        label="Greedy KL",
    )

    ax.set_title(f"{metric_name}: Greedy KL vs Random")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Directory containing greedy_kl/ and random_seed_*/ subfolders",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory where plots will be saved. Defaults to <indir>/plots_boxplots",
    )
    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots_boxplots")
    os.makedirs(outdir, exist_ok=True)

    greedy_meta = find_greedy_meta(indir)
    random_metas = find_random_metas(indir)

    for metric_name in METRICS:
        if metric_name not in greedy_meta:
            raise KeyError(f"Metric '{metric_name}' not found in greedy meta.json")

        random_values = []
        for meta in random_metas:
            if metric_name not in meta:
                raise KeyError(f"Metric '{metric_name}' not found in one random meta.json")
            random_values.append(float(meta[metric_name]))

        greedy_value = float(greedy_meta[metric_name])
        outpath = os.path.join(outdir, f"boxplot_{metric_name}.png")
        make_boxplot(metric_name, greedy_value, random_values, outpath)
        print(f"Saved {outpath}")


if __name__ == "__main__":
    main()