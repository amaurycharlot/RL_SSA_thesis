from agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    """Agent that selects random actions.
    """
    def __init__(self, num_targets: int = 2, world_size: float = 100.0, grid_shape=(10, 10), random_seed: int = 42, mode: str = "track"):
        super().__init__(num_targets=num_targets, world_size=world_size, grid_shape=grid_shape, mode=mode)
        self.seed = random_seed
        np.random.seed(self.seed)

    def select_action(self, obs):
        # Randomly select a pointing index
        N_grid = self.grid_shape[0] * self.grid_shape[1]
        pointing_x = np.random.randint(0, self.grid_shape[0])
        pointing_y = np.random.randint(0, self.grid_shape[1])
        return np.array([pointing_x, pointing_y], dtype=np.int32)