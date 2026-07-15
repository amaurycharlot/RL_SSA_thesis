import numpy as np

from agents.base_agent import BaseAgent


class RandomPointingAgent(BaseAgent):
    """
    Random pointing agent for sanity checking the environment.

    Assumes env observation is a flattened belief vector:
        per target: [x, y, vx, vy, diag(P)(4), trace(P)]  -> 9 dims
        full obs: (num_targets * 9,)
    """

    def __init__(self, num_targets: int, world_size: float):
        super().__init__(num_targets=num_targets, world_size=world_size)

        self.per_target_dim = 9
        self.trace_index = 8  # within each target block
        self.pos_slice = slice(0, 2)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Randomly point somewhere in the world, ignoring the observation.
        """
        x = np.random.uniform(0.0, self.world_size)
        y = np.random.uniform(0.0, self.world_size)
        return np.array([x, y], dtype=np.float32)