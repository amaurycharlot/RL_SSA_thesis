import numpy as np
from typing import List

from .target import Target
from .measurement import Measurement
from .grid_geometry import point_to_grid_cell, wrap_position_float32


class Sensor:
    """
    Telescope/radar sensor with a FoV the size of 9 grid cells.
    """

    def __init__(self, sensor_id: int, world_size: float, measurement_noise_std: float = 1.0, grid_shape=(10, 10), rng=None):
        # Standalone sensors retain their legacy RNG unless an environment supplies one.
        self.rng = np.random if rng is None else rng
        self.id = int(sensor_id)
        self.world_size = float(world_size)
        self.grid_shape = grid_shape
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
        True if target lies within grid cell FoV or any adjacent grid cell (so FoV is 3x3 with center at sensor position).
        """
        gx, gy = point_to_grid_cell(self.position, self.world_size, self.grid_shape)
        gx_t, gy_t = point_to_grid_cell(target.position, self.world_size, self.grid_shape)

        # compute wrapped-around distance in grid coordinates
        dx = abs(gx - gx_t)
        dy = abs(gy - gy_t)
        if dx > self.grid_shape[0] // 2:
            dx = self.grid_shape[0] - dx
        if dy > self.grid_shape[1] // 2:
            dy = self.grid_shape[1] - dy
        
        # target is in FoV if it is in the same grid cell or any adjacent grid cell (so dx and dy are both <= 1)
        return (dx <= 1) and (dy <= 1)

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
                noise = self.rng.normal(
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
            wrapped = wrap_position_float32(np.asarray([x, y]), self.world_size)
            return float(wrapped[0]), float(wrapped[1])
