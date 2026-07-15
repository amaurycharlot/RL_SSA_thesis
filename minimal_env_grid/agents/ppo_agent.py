from agents.base_agent import BaseAgent
import numpy as np
from stable_baselines3 import PPO

from pathlib import Path
from typing import Optional, Union

class PPOAgent(BaseAgent):
    def __init__(self, num_targets: int = 2, world_size: float = 100.0, grid_shape=(10, 10), mode: str = "track",
                 model: Optional[PPO] = None, model_path: Optional[Union[str, Path]] = "runs/ppo_baseline/final_model.zip"):
        super().__init__(num_targets=num_targets, world_size=world_size, grid_shape=grid_shape, mode=mode)
        if model is None and model_path is None:
            raise ValueError("Provide either `model` or `model_path`.")

        if model is not None and model_path is not None:
            raise ValueError("Provide only one of `model` or `model_path`.")

        self.model = model if model is not None else PPO.load(model_path)

    def select_action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return action