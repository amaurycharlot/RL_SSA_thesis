"""Read-only training diagnostics, never evaluation or post-update persistence."""
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


# Params: name, value, minimum; returns: int.
def _index(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass
class _Episode:
    raw_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    scalarized_sum: float = 0.
    learner_sum: float = 0.
    decisions: int = 0
    elapsed: int = 0
    detections: int = 0
    measurements: int = 0
    coverage_sum: float = 0.
    coverage_final: float = 0.


class EpisodeDiagnosticsCallback(BaseCallback):
    """Accumulate each live worker episode across repeated learn() chunks.

    Uses only copied step info and scalar rewards. Scalarized rewards before
    VecNormalize and rewards actually received by PPO are logged separately.
    Raw component sums are undiscounted training diagnostics, not evaluation
    returns. Coverage means weight decisions equally. New detections are the
    environment's per-step discovery counts. No observation/reset-info access
    is needed, including on automatic vector-environment resets.

    Reuse this instance only with the same live environments. A resumed process
    starts fresh accumulators, consistent with checkpoint continuation semantics.
    """

    # Params: self, logger, seed_index, weight_index, n_envs; returns: None.
    def __init__(self, logger, seed_index, weight_index, n_envs):
        super().__init__()
        self.event_logger = logger
        self.seed_index = _index('seed_index', seed_index)
        self.weight_index = _index('weight_index', weight_index)
        self.n_envs = _index('n_envs', n_envs, 1)
        self._episodes = [_Episode() for _ in range(self.n_envs)]
        self._completed = [0] * self.n_envs

    # Params: self; returns: None.
    def _on_training_start(self):
        if self.training_env.num_envs != self.n_envs:
            raise ValueError('callback n_envs must match the live training environment')
        # SB3 calls this for every chunk: partial episodes must survive.

    # Params: self; returns: bool.
    def _on_step(self):
        infos = self.locals['infos']
        dones = np.asarray(self.locals['dones'])
        rewards = np.asarray(self.locals['rewards'])
        if len(infos) != self.n_envs or dones.shape != (self.n_envs,) or rewards.shape != (self.n_envs,):
            raise ValueError('step data must align with n_envs')
        # Validate every worker before modifying any accumulator.
        steps = []
        for info, reward in zip(infos, rewards):
            raw = np.asarray(info['vector_reward_raw'], dtype=np.float64)
            scalar = float(info['scalarized_reward'])
            coverage = float(info['coverage_mean'])
            if raw.shape != (2,) or not np.all(np.isfinite(raw)):
                raise ValueError('vector_reward_raw must be a finite pair')
            if not all(np.isfinite(v) for v in (reward, scalar, coverage)):
                raise ValueError('scalar rewards and coverage must be finite')
            steps.append((raw.copy(), scalar, float(reward), coverage,
                          _index('elapsed_steps', info['elapsed_steps'], 1),
                          _index('new_detections', info['new_detections']),
                          _index('num_measurements', info['num_measurements'])))
        for worker, (raw, scalar, reward, coverage, elapsed, detections, measurements) in enumerate(steps):
            episode = self._episodes[worker]
            episode.raw_sum += raw
            episode.scalarized_sum += scalar
            episode.learner_sum += reward
            episode.decisions += 1
            episode.elapsed += elapsed
            episode.detections += detections
            episode.measurements += measurements
            episode.coverage_sum += coverage
            episode.coverage_final = coverage
            if dones[worker]:
                self.event_logger.record('training_episode', {
                    'seed_index': self.seed_index, 'weight_index': self.weight_index,
                    'worker_index': worker, 'episode_index': self._completed[worker],
                    'collection_step': int(self.num_timesteps),
                }, {
                    'diagnostic_kind': 'undiscounted_training_episode',
                    'reward_order': ['tracking', 'search'],
                    'raw_reward_sum': episode.raw_sum.copy(),
                    'scalarized_reward_sum': episode.scalarized_sum,
                    'learner_reward_sum': episode.learner_sum,
                    'decisions': episode.decisions, 'elapsed_timesteps': episode.elapsed,
                    'new_detections': episode.detections,
                    'measurement_count': episode.measurements,
                    'coverage_mean': episode.coverage_sum / episode.decisions,
                    'coverage_final': episode.coverage_final,
                })
                self._completed[worker] += 1
                self._episodes[worker] = _Episode()
        return True
