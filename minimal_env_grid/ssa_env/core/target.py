import numpy as np
from .grid_geometry import wrap_position_float32

class Target:
    """
    Physical target with true state [x, y, vx, vy].
    """

    # Params: self, target_id, x, y, vx, vy; returns: None.
    def __init__(self, target_id: int, x: float, y: float, vx: float, vy: float):
        self.id = int(target_id)
        self.state = np.array([x, y, vx, vy], dtype=np.float32)

    def propagate(self, dt: float, world_size: float, omega: float) -> None:
        """
        Propagate state forward in time with coordinated turn and wrap-around.
        """
        x, y, vx, vy = self.state

        if np.abs(omega) > 1e-6:
            theta = omega * dt
            c = np.cos(theta)
            s = np.sin(theta)

            # Rotate velocity
            vx_new = c * vx - s * vy
            vy_new = s * vx + c * vy
        else:
            vx_new = vx
            vy_new = vy

        # Update position using rotated velocity (so the turn affects motion immediately)
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt

        # Wrap-around world
        self.state[:2] = wrap_position_float32(np.asarray([x_new, y_new]), world_size)
        self.state[2] = vx_new  # velocities updated
        self.state[3] = vy_new

    @property
    def position(self) -> np.ndarray:
        """Return (x, y) as a 2D np.array."""
        return self.state[:2]

    @property
    def velocity(self) -> np.ndarray:
        """Return (vx, vy) as a 2D np.array."""
        return self.state[2:4]
