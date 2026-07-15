import sys
import subprocess


def main():
    for i in range(1,6):
        for script in ["scripts.plot_xy_errors", "scripts.plot_rewards", "scripts.plot_episode"]:
            for zero_noise in [True]:
                result = subprocess.run(
                    [sys.executable, "-m", script, f"rollouts/ppo_5targets_reward_R{i}_zero_noise_{zero_noise}"],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
                print(result.stderr)    

if __name__ == "__main__":
    main()