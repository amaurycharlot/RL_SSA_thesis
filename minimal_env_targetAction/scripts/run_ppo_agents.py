import sys
import subprocess


def main():
    for i in range(1,6) :
        for seed in range(1,6) :
            for train_zero_noise in [True] :
                result = subprocess.run(
                    [sys.executable, "-m", "scripts.run_agent", "--agent=ppo", f"--ppodir=runs/ppo_baseline_5targets_reward_R{i}_zero_noise_{train_zero_noise}", 
                     f"--num_targets=5", "--targets_known", f"--run_name=ppo_5targets_reward_R{i}_seed{seed}", f"--seed={seed}", 
                    f"--reward=R{i}"],
                    capture_output=True,
                    text=True
                )

                print(result.stdout)
                print(result.stderr)

if __name__ == "__main__":
    main()