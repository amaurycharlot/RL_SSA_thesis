from agents.train.train_ppo import main as train_ppo_main
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import os
import time

"""
def main():
    for num_targets in [2, 3, 4, 5]:
        result = subprocess.run(
            [sys.executable, "-m", "agents.train.train_ppo", f"--num_targets={num_targets}"],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print(result.stderr)
"""

def run_training(args):
    num_targets, total_timesteps, learning_rate, reward_type, train_zero_noise = args
    save_dir = (
        f"runs/ppo_baseline_{num_targets}targets_reward_{reward_type}_zero_noise_{train_zero_noise}"
    )

    train_ppo_main(
        num_targets=num_targets,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        reward_type=reward_type,
        train_zero_noise=train_zero_noise,
        save_dir=save_dir,
        timesteps=200
    )

    return f"Finished: {save_dir}"


if __name__ == "__main__":
    """
    
    configs = [
        (n, ts, lr, rt)
        for n, ts, lr, rt in product(
            [20, 50],
            [250000],
            [1e-4],
            [f"R{i}" for i in range(1, 5)]
        )]

    max_workers = min(len(configs), os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_training, cfg) for cfg in configs]

        for future in as_completed(futures):
            print(future.result())
    """

    
    runtimes = []


    for i in range(1, 6):
        start = time.perf_counter()
        run_training((25, 500000, 1e-4, f"R{i}", True))
        end = time.perf_counter()
        runtimes.append(end - start)

    print(f"Runtimes: {runtimes}")