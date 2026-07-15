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

        self.per_target_dim = 9
        self.trace_index = 8  # within each target block
        self.pos_slice = slice(0, 2)

    def reset(self) -> None:
        pass

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Heuristic policy:

        - If not all targets are acquired, point randomly.
          (We consider a target acquired if its trace(P) entry is non-zero / non-NaN.)

        - Once all targets are acquired, point at the target with the largest trace(P).
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        # Reshape into (N, 9)
        X = obs.reshape(self.num_targets, self.per_target_dim)

        # Acquisition check: track is "present" if trace(P) is finite and > 0
        traces = X[:, self.trace_index].astype(np.float32)
        acquired = np.isfinite(traces) & (traces > 0.0)

        # 1) Explore until all targets are acquired
        if not np.all(acquired):
            x = np.random.uniform(0.0, self.world_size)
            y = np.random.uniform(0.0, self.world_size)
            return np.array([x, y], dtype=np.float32)

        # 2) Exploit: point to the most uncertain target (max trace(P))
        best_tid = int(np.argmax(traces))
        target_pos = X[best_tid, self.pos_slice]

        x = float(np.clip(target_pos[0], 0.0, self.world_size))
        y = float(np.clip(target_pos[1], 0.0, self.world_size))
        return np.array([x, y], dtype=np.float32)