"""Fixed float32 objective scaling/scalarization for the existing combined env."""
import gymnasium as gym
import numpy as np


# Params: values, name, positive; returns: read-only pair array.
def _fixed_pair(values, name, positive=False):
    if isinstance(values, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in values):
        raise TypeError(f"{name} must contain real numbers, not bools")
    source = np.asarray(values)
    if source.shape != (2,):
        raise ValueError(f"{name} must have exactly two components")
    if source.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric components")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must be finite")
    invalid_sign = np.any(source <= 0) if positive else np.any(source < 0)
    if invalid_sign:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        copied = source.astype(np.float32)
    if not np.all(np.isfinite(copied)) or (positive and np.any(copied <= 0)):
        raise ValueError(f"{name} must be finite and representable in float32")
    if not positive and (not np.isclose(source.sum(dtype=np.float64), 1., rtol=0, atol=1e-6)
                         or not np.isclose(copied.sum(dtype=np.float64), 1., rtol=0, atol=1e-6)):
        raise ValueError("weights must sum to one within absolute tolerance 1e-6; no renormalization")
    return np.frombuffer(copied.tobytes(), dtype=np.float32)


class LinearScalarizationWrapper(gym.Wrapper):
    """One raw step -> raw/scaled diagnostics + a Python float scalar reward.

    Preferences and scales are copied into immutable arrays. No normalization
    statistics, environment math or episode/action semantics are changed.
    """
    _mo_scalarized_reward = True

    # Params: self, env, weights, scales; returns: None.
    def __init__(self, env, weights, scales=(1.0, 1.0)):
        current, seen = env, set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if getattr(type(current), "_mo_scalarized_reward", False) or type(current).__name__ == "LinearScalarizationWrapper":
                raise ValueError("Do not stack scalarization wrappers, including the legacy wrapper")
            current = current.env if isinstance(current, gym.Wrapper) else None
        fixed_weights = _fixed_pair(weights, "weights")
        fixed_scales = _fixed_pair(scales, "scales", positive=True)
        super().__init__(env)
        self._weights = fixed_weights
        self._scales = fixed_scales

    # Params: self; returns: self._weights.
    @property
    def weights(self):
        return self._weights

    # Params: self; returns: self._scales.
    @property
    def scales(self):
        return self._scales

    # Params: self, action; returns: (observation, scalar, terminated, truncated, diagnostics).
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        source = np.asarray(reward)
        if source.shape != (2,):
            raise ValueError("Combined raw reward must have shape (2,)")
        if source.dtype != np.float32:
            raise TypeError("Combined raw reward must be float32; no precision conversion is performed")
        if not np.all(np.isfinite(source)):
            raise ValueError("Combined raw reward must be finite")
        raw = source.copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = raw / self._scales
            scalar = float(np.dot(self._weights, scaled))
        if not np.all(np.isfinite(scaled)) or not np.isfinite(scalar):
            raise ValueError("Scaled/scalarized reward must remain finite in float32")
        diagnostics = dict(info)
        diagnostics.update(vector_reward=raw.copy(), vector_reward_raw=raw.copy(),
                           vector_reward_scaled=scaled.copy(), tracking_reward=float(raw[0]),
                           search_reward=float(raw[1]), scalarized_reward=scalar)
        return observation, scalar, terminated, truncated, diagnostics
