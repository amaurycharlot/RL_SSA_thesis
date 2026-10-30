"""Toroidal position/grid conversions with half-open floating-point bounds."""
import math
import numpy as np


def wrap_position_float32(point, world_size):
    """Wrap before casting, then enforce 0 <= position < L in stored float32.

    A tiny negative remainder can round to L, either during modulo or the
    float32 cast. Keep it on the upper side of the seam with the greatest
    representable float32 below L. Exact multiples of L still wrap to zero.
    Preserve the input arithmetic dtype until modulo (target propagation may
    supply float64 while catalogue states and measurements use float32).
    """
    values = np.asarray(point)
    size = float(world_size)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f'Position must contain two finite coordinates: {point!r}')
    if not math.isfinite(size) or size <= 0:
        raise ValueError('world_size must be finite and positive')
    upper = np.float32(size)
    if float(upper) >= size:
        upper = np.nextafter(upper, np.float32(-np.inf))
    wrapped = np.asarray(np.remainder(values, size), dtype=np.float32)
    return np.minimum(wrapped, upper)


def point_to_grid_cell(point, world_size, grid_shape):
    """Map finite coordinates periodically, including stale upper-bound values.

    Float64 modulo can itself round a tiny negative value to L. Bound the
    quotient as well as wrapping the position; never let that rounding produce
    a cell equal to the number of cells. Negative coordinates belong on the
    upper side of the torus, rather than being truncated towards zero.
    """
    size = float(world_size)
    if not math.isfinite(size) or size <= 0:
        raise ValueError('world_size must be finite and positive')
    x, y = point
    result = []
    for coordinate, count in zip((x, y), grid_shape):
        value = float(coordinate)
        if not math.isfinite(value):
            raise ValueError(f'Grid position must be finite: {point!r}')
        wrapped = value % size
        cell = int(wrapped // (size / count))
        result.append(min(cell, count - 1))
    return tuple(result)
