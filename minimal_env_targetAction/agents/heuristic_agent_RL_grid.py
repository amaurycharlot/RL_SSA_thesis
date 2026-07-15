import numpy as np

from agents.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Simple heuristic policy for sanity checking the environment.

    Assumes env observation is a 10*10 grid with :
        empty cell -> 0
        target cell -> uncertainty value (e.g. trace(P))
    """

    def __init__(self, num_targets: int, world_size: float):
        super().__init__(num_targets=num_targets, world_size=world_size)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Heuristic policy: point at the target with the largest trace(P).
        """

        # point to the most uncertain target (max trace(P))
        #print(f"Observation :\n{obs}")
        action = int(np.argmax(obs))
        
        #print("obs :", obs)
        #print(f"HeuristicAgent selected action index: {action}")
        return action