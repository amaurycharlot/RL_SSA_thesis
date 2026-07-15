import numpy as np

import sys
import subprocess

from scripts.plot_xy_errors import wrap_diff

def main():
    WORLD_SIZE = 100.0
    IDX_X = 0
    IDX_Y = 1
    IDX_VARX = 4
    IDX_VARY = 5
    for R in range(1, 5):
        mean_pos_errors = []
        mean_cov_traces = []
        for seed in range(1, 6):
            dir = f"rollouts/ppo_5targets_reward_R{R}_seed{seed}"
            # Calculate mean position error at end of episode
            est = np.load(f"{dir}/est_states.npy")
            true = np.load(f"{dir}/true_states.npy")
            
            err_x = wrap_diff(est[49, :, IDX_X], true[49, :, 0], WORLD_SIZE)
            err_y = wrap_diff(est[49, :, IDX_Y], true[49, :, 1], WORLD_SIZE)

            var_x = np.maximum(est[49, :, IDX_VARX], 0.0)
            var_y = np.maximum(est[49, :, IDX_VARY], 0.0)

            mean_pos_errors.append(np.mean(np.sqrt(err_x**2 + err_y**2)))
            mean_cov_traces.append(np.mean(var_x + var_y))
            # UNCOMMENT THE FOLLOWING TO PLOT THE RESULTS FOR EACH SEED
            """
            for script in ["scripts.plot_xy_errors", "scripts.plot_rewards", "scripts.plot_episode"]:
                result = subprocess.run(
                    [sys.executable, "-m", script, f"rollouts/ppo_5targets_reward_R{R}_seed{seed}"],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
                print(result.stderr) 
            """ 
            
        mean_pos_error = np.mean(mean_pos_errors)
        mean_cov_trace = np.mean(mean_cov_traces)

        print(f"Reward R{R}: Mean Position Error = {mean_pos_error:.4f}, Mean Covariance Trace = {mean_cov_trace:.4f}")

if __name__ == "__main__":
    main()