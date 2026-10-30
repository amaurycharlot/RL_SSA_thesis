import sys
import subprocess

if __name__ == "__main__":
    known = 10
    unknown = 0
    for seed in range(1,6): # LATER RUN ACROSS MORE SEEDS
        for threshold in [0.002, 0.01, 0.04, 0.2, 1.0, 5.0, 50.0]: # [0.002, 0.01, 0.04, 0.2, 1.0, 5.0, 50.0]
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_agent", "--agent=heuristic", "--mode=combined", f"--seed={seed}",
                    f"--known_targets={known}", f"--unknown_targets={unknown}", f"--run_name=heuristic_combined_threshold{threshold}_seed{seed}", 
                    f"--heur_uncertainty_threshold={threshold}", f"--outdir=rollouts/combined_{known}kn_{unknown}uk"],
                capture_output=True,
            text=True
            )

            print(result.stdout)
            print(result.stderr)