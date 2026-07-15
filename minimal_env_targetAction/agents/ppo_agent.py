# agents/ppo_agent.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from stable_baselines3 import PPO

from agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    PPO-based sensor pointing agent.

    Wraps a trained Stable-Baselines3 PPO model and impements the BaseAgent interface:
        - reset()
        - select_action(obs)

    Assumes:
        obs is a flattened belief vector of shape (num_targets * 9,)
        action is continuous [sx, sy] in world coordinates
    """

    def __init__(
        self,
        num_targets: int = 2,
        world_size: float = 100.0,
        model: Optional[PPO] = None,
        model_path: Optional[Union[str, Path]] = "runs/ppo_baseline/final_model.zip",
        deterministic: bool = True,
    ):
        super().__init__(num_targets=num_targets, world_size=world_size)

        if model is None and model_path is None:
            raise ValueError("Provide either `model` or `model_path`.")

        if model is not None and model_path is not None:
            raise ValueError("Provide only one of `model` or `model_path`.")

        self.model = model if model is not None else PPO.load(str(model_path))
        self.deterministic = bool(deterministic)

    def reset(self) -> None:
        """
        Kept for interface compatibility.
        PPO policies in SB3 are memoryless unless using recurrent policies.
        """
        pass

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Select an action from the trained PPO policy.

        Parameters
        ----------
        obs : np.ndarray
            Flattened observation vector.

        Returns
        -------
        np.ndarray
            Action [sx, sy] as float32.
        """
        
        obs = np.asarray(obs, dtype=np.float32).reshape(self.num_targets,)
        #print("obs_RL:\n", obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)