import sys
import subprocess

if __name__ == "__main__":
    for seed in range(10):
        for i in range(4):
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_agent", "--agent=ppo", "--mode=search", f"--seed={seed}",
                    f"--num_targets=100", f"--run_name=ppo_search_S{i+1}_seed{seed}", f"--ppodir=runs/s_model{i}",
                    f"--outdir=rollouts/search"],
                capture_output=True,
            text=True
            )

            print(result.stdout)
            print(result.stderr)