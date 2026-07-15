import numpy as np
from typing import List

from .target import Target
from .measurement import Measurement


class Sensor:
    """
    Telescope/radar sensor with a circular FoV.
    """

    def __init__(self, sensor_id: int, fov_radius: float, world_size: float, measurement_noise_std: float = 1.0):
        self.id = int(sensor_id)
        self.world_size = float(world_size)
        self.fov_radius = float(fov_radius)
        self.fov_radius_sq = self.fov_radius ** 2
        self.position = np.zeros(2, dtype=np.float32)  # (sx, sy)

        # Measurement noise
        self.measurement_noise_std = float(measurement_noise_std)
        self.R = (self.measurement_noise_std ** 2) * np.eye(2, dtype=np.float32) # 2-dimensional covariance matrix

    def set_pointing(self, sx: float, sy: float) -> None:
        """
        Point to new coordinates. Clip to [0, world_size]^2.
        """
        sx_clipped = float(np.clip(sx, 0.0, self.world_size))
        sy_clipped = float(np.clip(sy, 0.0, self.world_size))
        self.position[:] = np.array([sx_clipped, sy_clipped], dtype=np.float32)

    def is_in_fov(self, target: Target) -> bool:
        """
        True if target lies within circular FoV.
        """
        dist = self._distance_to_target(target)
        return dist[0] * dist[0] + dist[1] * dist[1] <= self.fov_radius_sq

    def measure_targets(self, targets: List[Target], time: float) -> List[Measurement]:
        """
        Return a measurement for each target in FoV.
        Currently:
            - Gaussian noise measurement of (x, y)
            - true target_id known
        """
        measurements: List[Measurement] = []
        for t in targets:
            if self.is_in_fov(t):
                # Gaussian noise
                noise = np.random.normal(
                    loc=0.0,
                    scale=self.measurement_noise_std,
                    size=2,
                ).astype(np.float32)

                z = t.position + noise
                z[0], z[1] = self._wrap_pos(z[0], z[1])  # wrap around world edges
                meas = Measurement(
                    sensor_id=self.id,
                    target_id=t.id,
                    position=z,  # noisy measurement
                    time=time,
                )
                measurements.append(meas)
        return measurements
    
    #helper function to wrap positions into [0, world_size) using modulo arithmetic
    def _wrap_pos(self, x, y) -> np.ndarray:
            """
            Wrap a 2D position into [0, world_size) using modulo arithmetic.
            """
            return float(x % self.world_size), float(y % self.world_size)
    
    def _distance_to_target(self, target: Target) -> np.ndarray:
        """
        Compute the (dx, dy) distance from the sensor to the target, accounting for world wrapping.
        """
        dx = target.position[0] - self.position[0]
        dy = target.position[1] - self.position[1]

        # Wrap around world edges
        if dx > self.world_size / 2:
            dx -= self.world_size
        elif dx < -self.world_size / 2:
            dx += self.world_size

        if dy > self.world_size / 2:
            dy -= self.world_size
        elif dy < -self.world_size / 2:
            dy += self.world_size

        return np.array([dx, dy], dtype=np.float32)