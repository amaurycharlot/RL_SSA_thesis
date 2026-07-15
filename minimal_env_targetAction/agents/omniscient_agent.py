import numpy as np

from agents.base_agent import BaseAgent


class OmniscientAgent(BaseAgent):
    """
    Debugging policy that uses the true target positions.

    This is not realistic and should not be used for experiments,
    but it is useful to validate:
    - sensor pointing logic
    - reward behavior
    - measurement generation
    - estimator response under ideal tasking

    Modes:
    - "centroid": point at the centroid of all true target positions
    - "target_0", "target_1", ...: point at one specific target
    """

    def __init__(self, num_targets: int, world_size: float, mode: str = "target_0"):
        super().__init__(num_targets=num_targets, world_size=world_size)
        self.mode = str(mode)

    def reset(self) -> None:
        pass

    def select_action(self, true_states: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        true_states : ndarray, shape (num_targets, state_dim)
            True target states from env info["true_states"].
            Positions are expected in columns [0:2].

        Returns
        -------
        action : ndarray, shape (2,)
            Sensor pointing [x, y].
        """
        true_states = np.asarray(true_states, dtype=np.float32)

        if true_states.ndim != 2 or true_states.shape[0] != self.num_targets:
            raise ValueError(
                f"Expected true_states shape ({self.num_targets}, state_dim), "
                f"got {true_states.shape}"
            )

        if self.mode == "centroid":
            pos = true_states[:, 0:2]
            aim = pos.mean(axis=0)

        elif self.mode.startswith("target_"):
            tid = int(self.mode.split("_")[1])
            if tid < 0 or tid >= self.num_targets:
                raise ValueError(f"Invalid target id in mode '{self.mode}'")
            aim = true_states[tid, 0:2]

        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. Use 'centroid' or 'target_k'."
            )

        x = float(np.clip(aim[0], 0.0, self.world_size))
        y = float(np.clip(aim[1], 0.0, self.world_size))
        return np.array([x, y], dtype=np.float32)