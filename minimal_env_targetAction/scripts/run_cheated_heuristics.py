# --agent=heuristic --RL --num_targets=5 --run_name=heuristicRL_5targets_R4_nrml --reward=R4
import sys
import subprocess

if __name__ == "__main__":
    for i in range(5,6) :
        for seed in range(1,2) :
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_agent", "--agent=heuristic", "--RL", "--outdir=rollouts/random_agents",
                    f"--num_targets=5", f"--run_name=R{i}_heuristic", f"--reward=R{i}", f"--seed={seed}"],
                capture_output=True,
                text=True
            )

            print(result.stdout)
            print(result.stderr)