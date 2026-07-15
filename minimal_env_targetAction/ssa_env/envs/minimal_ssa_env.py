import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List

from ssa_env.core.target import Target
from ssa_env.core.sensor import Sensor
from ssa_env.core.measurement import Measurement

from agents.catalogue import CatalogueState


class MinimalSSAEnv(gym.Env):
    """
    Minimal SSA environment (see previous description).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, num_targets: int = 2, omega0: float = 0.05, measurement_noise_std=1.0, targets_known: bool = False):
        super().__init__()

        self.num_targets = int(num_targets)
        self.targets_known = bool(targets_known)
        self.omega0 = float(omega0)
        self.world_size = 100.0
        self.dt = 1.0
        self.t = 0.0
        self.max_steps = 50
        self.steps = 0
        self.catalogue = CatalogueState(
            num_targets=self.num_targets,
            world_size=self.world_size,
            dt=self.dt,
            omega0=self.omega0,
            sigma_z=1.0,
            sigma_a=0.1,
            init_pos_var=25.0,
            init_vel_var=25.0
        )

        # World objects
        self.targets: List[Target] = []
        self.sensor = Sensor(sensor_id=0, fov_radius=10.0, world_size=self.world_size, measurement_noise_std=measurement_noise_std)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(self.num_targets * 9,), dtype=np.float32
        )
        self.action_space = self._build_action_space()

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.t = 0.0

        self.targets = self._sample_initial_targets()
        self.sensor.set_pointing(self.world_size / 2, self.world_size / 2)
        self.catalogue.reset()

        if self.targets_known:
            # Update catalogue to simulate known initial target states
            for i, tgt in enumerate(self.targets):
                self.catalogue.maybe_create_track_from_measurement(i, tgt.position, self.t)

        measurements = self._compute_measurements()
        obs = self._build_observation(measurements)
        info = {"true_states": self._get_true_states_array()}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.targets is not None, "Call reset() before step()."

        self.steps += 1
        self.t += self.dt

        # Before step : compute total trace for reward logging
        total_trace_before = 0.0
        for tid in range(self.num_targets):
            trk = self.catalogue.tracks[tid]
            total_trace_before += float(np.trace(trk.P[0:2, 0:2]))

        # 1) Apply action
        sx = float(action[0])
        sy = float(action[1])
        self.sensor.set_pointing(sx, sy)

        # 2) Propagate targets
        for tgt in self.targets:
            tgt.propagate(self.dt, self.world_size, self.omega0)

        # 3) Measurements
        measurements = self._compute_measurements()

        # catalogue update: predict + update with new measurements
        #t_now = float(self.t)
        #self.catalogue.predict_all(t_now=t_now)
        self.catalogue.predict_all()
        #self.catalogue.update_from_measurements(measurements, t_now=t_now)
        self.catalogue.update_from_measurements(measurements)

        # 4) Reward
        #reward = self._compute_reward(measurements)
        total_trace_after = 0.0
        for tid in range(self.num_targets):
            trk = self.catalogue.tracks[tid]
            total_trace_after += float(np.trace(trk.P[0:2, 0:2]))
        reward = np.max([np.min([total_trace_before - total_trace_after, 50.0]), -50.0])

        # 5) Termination flags
        terminated = self.steps >= self.max_steps
        truncated = False

        # 6) Observation
        obs = self._build_observation(measurements)
        info = {
            "num_measurements": len(measurements),
            "true_states": self._get_true_states_array(),
        }

        return obs, float(reward), terminated, truncated, info

    # ---------- Helpers: spaces ----------

    def _build_action_space(self) -> spaces.Space:
        sensor_low = np.array([0.0, 0.0], dtype=np.float32)
        sensor_high = np.array([self.world_size, self.world_size], dtype=np.float32)
        return spaces.Box(
            low=sensor_low,
            high=sensor_high,
            shape=(2,),
            dtype=np.float32,
        )

    # ---------- Helpers: world ----------

    def _sample_initial_targets(self) -> List[Target]:
        targets: List[Target] = []
        for i in range(self.num_targets):
            x, y = self.np_random.uniform(0.0, self.world_size, size=2)
            vx, vy = self.np_random.uniform(-1.0, 1.0, size=2)
            omega = self.omega0
            tgt = Target(target_id=i, x=x, y=y, vx=vx, vy=vy)
            targets.append(tgt)
        return targets

    def _compute_measurements(self) -> List[Measurement]:
        return self.sensor.measure_targets(self.targets, time=self.t)

    def _compute_reward(self, measurements: List[Measurement]) -> float:
        """
        RL reward based on belief uncertainty.

        reward = - sum_i trace(P_i)
        """
        penalty_unacquired = 0.0
        if not self.targets_known:
            penalty_unacquired = 100.0  # tune later

        total_trace = 0.0
        unacquired = 0

        for tid in range(self.num_targets):
            #if tid not in self.catalogue.tracks: #THIS IS WRONG, A TRACK CAN EXIST WITHOUT BEING ACQUIRED IN THE CURRENT STEP. NEED TO CHECK IF THE TRACK WAS UPDATED WITH A MEASUREMENT IN THIS STEP. FOR NOW, ASSUME ALL TRACKS ARE ACQUIRED IF THEY EXIST IN THE CATALOGUE
            #    unacquired += 1
            #    continue
            trk = self.catalogue.tracks[tid]
            total_trace += float(np.trace(trk.P[0:2, 0:2]))

        reward = -total_trace
        return float(reward)

    def _get_true_states_array(self) -> np.ndarray:
        arr = np.zeros((self.num_targets, 4), dtype=np.float32)
        for i, tgt in enumerate(self.targets):
            arr[i] = tgt.state
        return arr

    def _build_observation(self, measurements: List[Measurement]) -> dict:
        """
        Build the observation dict from updated catalogue state.
         as_mdp_state returns (N,9) (includes diag(P) and trace(P)), 
         the observation dict is flattened to 1D array.
        """
        belief = self.catalogue.as_mdp_state(include_cov_diag=True).reshape(-1).astype(np.float32)
        return belief
