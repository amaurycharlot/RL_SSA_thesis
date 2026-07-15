import numpy as np
import copy

from agents.base_agent import BaseAgent

class GreedyKLAgent(BaseAgent):
    """
    Myopic greedy agent that chooses the action with the largest
    one-step KL-divergence reward.

    It assumes the environment is the RL version and that:
    - actions are discrete target indices: 0, ..., num_targets-1
    - the environment reward returned by step(action) is the KL reward
      you want to maximize (so in practice use env.reward_type = "R5")

    The live environment is never modified: each candidate action is
    evaluated on a deep copy.
    """

    def __init__(self, num_targets: int = 2, world_size: float = 100.0):
        super().__init__(num_targets=num_targets, world_size=world_size)

    def select_action(self, env):
        best_action = 0
        best_reward = -np.inf

        for action in range(self.num_targets):
            env_copy = copy.deepcopy(env)

            _, reward, terminated, truncated, _ = env_copy.step(action)

            reward = float(reward)

            if reward > best_reward:
                best_reward = reward
                best_action = int(action)

        return best_action