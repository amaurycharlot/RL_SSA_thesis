import numpy as np

from agents.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Simple heuristic policy for sanity checking the environment.

    Assumes env observation is a flattened belief vector:
        per target: [x, y, vx, vy, diag(P)(4), trace(P)]  -> 9 dims
        full obs: (num_targets * 9,)
    """

    def __init__(self, num_targets: int, world_size: float):
        super().__init__(num_targets=num_targets, world_size=world_size)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Heuristic policy: point at the target with the largest trace(P).
        """

        # Reshape into (N,)
        X = obs.reshape(self.num_targets, -1)

        # point to the most uncertain target (max trace(P))
        return int(np.argmax(X))