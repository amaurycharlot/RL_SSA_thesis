import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridCNNExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for SSA grid observations.

    Expected observation shape:
        (H, W, C)
    Example:
        (10, 10, 2)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape

        if len(obs_shape) != 3:
            raise ValueError(
                f"Expected observation shape (H,W,C), got {obs_shape}"
            )

        h, w, c = obs_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Compute flattened size automatically
        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[None]
            ).float()

            # Convert NHWC -> NCHW
            sample = sample.permute(0, 3, 1, 2)

            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations:
            shape (batch, H, W, C)
        """

        # Convert to channel-first for PyTorch Conv2D
        observations = observations.permute(0, 3, 1, 2)

        x = self.cnn(observations)
        x = self.linear(x)

        return x