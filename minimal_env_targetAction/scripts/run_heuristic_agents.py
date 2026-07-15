import sys
import subprocess


def main():
    for i in range(1,6) :
        for seed in range(1,2) :
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_agent", "--agent=heuristic", "--RL", 
                "--num_targets=5", "--targets_known", f"--run_name=heuristicRL_5targets_reward_R{i}", f"--seed={seed}", 
                f"--reward=R{i}"],
                capture_output=True,
                text=True
            )

            print(result.stdout)
            print(result.stderr)

def main_random():
    for i in range(5,6) :
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_agent", "--agent=heuristic", "--RL", "--outdir=rollouts/random_agents",
                f"--num_targets=5", f"--run_name=R{i}_heuristic", f"--reward=R{i}"],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)

if __name__ == "__main__":
    main()