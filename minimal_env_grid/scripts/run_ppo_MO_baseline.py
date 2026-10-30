import sys
import subprocess

if __name__ == "__main__":
    for seed in range(1,6): # LATER RUN ACROSS MORE SEEDS
        i = 0
        for ppodir in ["runs/ppo_combined_grid_10x10_10known_10unknown_200steps_w0.7_0.3_seed1", "runs/ppo_combined_grid_10x10_10known_10unknown_200steps_w0.85_0.15_seed1"]:
            i += 1
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_agent", "--agent=ppo", "--mode=combined", f"--seed={seed}",
                    "--known_targets=10", "--unknown_targets=90", f"--run_name=ppo_combined_{i}_seed{seed}", 
                    f"--ppodir={ppodir}", "--outdir=rollouts/combined"],
                capture_output=True,
            text=True
            )

            print(result.stdout)
            print(result.stderr)