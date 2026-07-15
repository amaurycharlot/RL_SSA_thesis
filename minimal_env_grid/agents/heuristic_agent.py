from agents.base_agent import BaseAgent
import numpy as np

class HeuristicAgent(BaseAgent):
    """Agent that selects actions based on a simple heuristic.
    """
    def __init__(self, num_targets: int = 2, world_size: float = 100.0, grid_shape=(10, 10), mode: str = "track", uncertainty_threshold: float = 0.5):
        super().__init__(num_targets=num_targets, world_size=world_size, grid_shape=grid_shape, mode=mode)
        self.og_mode = mode
        self.uncertainty_threshold = uncertainty_threshold

    def select_action(self, obs):
        """
        Observation "track" is a vector of measurements for each grid cell :
            - obs[i] is the trace of positional covariance of the target in grid cell i (or 0 if no target is present).
        Observation "search" is a vector of decayed count of last observation for each grid cell
            - 0 means never observed
            - 1 means observed at the current time step
            - values in (0, 1) mean observed in the past, with decay of *0.99 per time step.
        Observation "combined" is the concatenation of "track" and "search" observations
            - obs[i, 0] is "track" observation for grid cell i
            - obs[i, 1] is "search" observation for grid cell i

        Track heuristic: point to the grid cell with the highest observation (highest uncertainty).
        Search heuristic: point to the grid cell with the lowest observation (least observed).
        """

        if self.og_mode == "combined" : # In combined mode, we switch between track and search
            # compute maximum uncertainty
            max_uncertainty = np.max(obs[:, :, 0])
            #print(f"Max uncertainty: {max_uncertainty}")
            if max_uncertainty > self.uncertainty_threshold : # If there is at least one grid cell with uncertainty above the threshold, we go into track mode
                self.mode = "track"
            else : # Otherwise, we are in search mode
                self.mode = "search"

        best_pointing_idx = np.ndarray([0,0], dtype=np.int32)
        if self.mode == "track" : # Heuristic: point to the grid cell with the highest obervation
            max_measurement = -np.inf
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    if obs[i, j, 0] > max_measurement:
                        max_measurement = obs[i, j, 0]
                        best_pointing_idx = np.array([i, j], dtype=np.int32)
        elif self.mode == "search" : # Heuristic: point to the grid cell with lowest observation sum in neighboring 3x3 area (least observed)
            s_id = 1 if self.og_mode == "combined" else 0 # In combined mode, the search observation is at index 1, otherwise it's at index 0
            min_measurement = np.inf
            sum_obs = np.zeros_like(obs[:, :, s_id])
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    # Compute sum of observations in 3x3 neighborhood
                    for di in range(-1, 2):
                        ni = (i + di) % obs.shape[0]
                        for dj in range(-1, 2):
                            nj = (j + dj) % obs.shape[1]
                            sum_obs[i, j] += obs[ni, nj, s_id]
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    if sum_obs[i, j] < min_measurement:
                        min_measurement = sum_obs[i, j]
                        best_pointing_idx = np.array([i, j], dtype=np.int32)
        return best_pointing_idx