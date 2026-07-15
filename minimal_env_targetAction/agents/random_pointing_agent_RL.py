import numpy as np

from agents.base_agent import BaseAgent


class RandomPointingAgent(BaseAgent):
    """
    Random pointing agent for sanity checking the environment.

    Assumes env observation is a flattened belief vector:
        per target: [x, y, vx, vy, diag(P)(4), trace(P)]  -> 9 dims
        full obs: (num_targets * 9,)
    """

    def __init__(self, num_targets: int, world_size: float, seed: int=42):
        super().__init__(num_targets=num_targets, world_size=world_size)
        self.seed = seed
        np.random.seed(self.seed)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> int:
        """
        Randomly point to any target, ignoring the observation.
        """
        target_id = np.random.randint(0, self.num_targets)
        return target_id