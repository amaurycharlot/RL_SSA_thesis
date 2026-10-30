import numpy as np
from typing import Optional


class Measurement:
    """
    One measurement of a target by a given sensor at a given time.
    """

    # Params: self, sensor_id, target_id, position, time; returns: None.
    def __init__(self, sensor_id: int, target_id: Optional[int],
                 position: np.ndarray, time: float):
        self.sensor_id = int(sensor_id)
        self.target_id = None if target_id is None else int(target_id)
        self.position = np.array(position, dtype=np.float32)  # measured (x, y)
        self.time = float(time)

    def as_array(self) -> np.ndarray:
        """
        Pack into a numeric vector [sensor_id, target_id, x, y, t].
        target_id = -1 if None.
        """
        tid = -1 if self.target_id is None else self.target_id
        return np.array(
            [self.sensor_id, tid, self.position[0], self.position[1], self.time],
            dtype=np.float32,
        )