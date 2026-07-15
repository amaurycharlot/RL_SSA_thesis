import sys
import subprocess


def main():
    for R in range(1,6) :
        for timesteps in [100, 200, 400]:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.plt_mean_return", f"--run_dir=runs/ppo_baseline_20targets_reward_R{R}_{timesteps}timesteps"],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print(result.stderr)  

if __name__ == "__main__":
    main()