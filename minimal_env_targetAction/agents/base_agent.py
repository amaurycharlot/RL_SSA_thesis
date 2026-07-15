import numpy as np

from .catalogue import CatalogueState


class BaseAgent:
    """Base agent interface.
    """
    def __init__(self, num_targets: int = 2, world_size: float = 100.0):
        self.num_targets = int(num_targets)
        self.world_size = float(world_size)
        

    def reset(self):
        pass

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError