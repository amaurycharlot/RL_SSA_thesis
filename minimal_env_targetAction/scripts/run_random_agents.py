import sys
import subprocess


def main():
    for random_seed in range(22,23) :
        for i in range(4,6) :
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_agent", "--agent=random", "--RL", "--outdir=rollouts/random_agents",
                    f"--num_targets=20", f"--run_name=R{i}_random_{random_seed}", f"--random_seed={random_seed}", 
                f"--reward=R{i}", "--outdir=rollouts/random_agents_std0.1", f"--measurement_noise_std=0.1"],
                capture_output=True,
                text=True
            )

            #print(result.stdout)
            #print(result.stderr)
            print(f"Finished random seed {random_seed} for reward R{i}.")

if __name__ == "__main__":
    main()