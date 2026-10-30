import numpy as np

from .catalogue import CatalogueState


class BaseAgent:
    """Base agent interface.
    """
    # Params: self, num_targets, world_size, grid_shape, mode; returns: None.
    def __init__(self, num_targets: int = 2, world_size: float = 100.0, grid_shape=(10, 10), mode: str = "track"):
        self.num_targets = int(num_targets)
        self.world_size = float(world_size)
        self.grid_shape = tuple(grid_shape)
        self.mode = mode

    # Params: self; returns: None.
    def reset(self):
        pass

    # Params: self, obs; returns: never (raises).
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError