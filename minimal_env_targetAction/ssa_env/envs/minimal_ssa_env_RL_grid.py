import math

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

    def __init__(self, render_mode=None, num_targets: int = 2, omega0: float = 0.05, measurement_noise_std=1.0, targets_known: bool = False,
                 zero_noise: bool = False, timesteps : int = 50):
        super().__init__()

        self.num_targets = int(num_targets)
        self.targets_known = bool(targets_known)
        self.zero_noise = bool(zero_noise)
        self.omega0 = float(omega0)
        self.world_size = 100.0
        self.dt = 1.0
        self.t = 0.0
        self.max_steps = int(timesteps)
        self.steps = 0
        self.catalogue = CatalogueState(
            num_targets=self.num_targets,
            world_size=self.world_size,
            dt=self.dt,
            omega0=self.omega0,
            sigma_z=0.5 if zero_noise else float(measurement_noise_std),
            sigma_a=0,
            init_pos_var=1.0,
            init_vel_var=1.0
        )

        # World objects
        self.targets: List[Target] = []
        self.sensor = Sensor(sensor_id=0, fov_radius=10.0, world_size=self.world_size, measurement_noise_std=measurement_noise_std)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(int(10*10),), dtype=np.float32
        )
        
        #self.action_space = self._build_action_space()
        self.action_space = spaces.Discrete(int(10*10))
        #self.action_space = spaces.Discrete(self.num_targets)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.t = 0.0

        self.targets = self._sample_initial_targets()
        self.sensor.set_pointing(self.world_size / 2, self.world_size / 2)
        self.catalogue.reset()

        if self.zero_noise:
            for i, tgt in enumerate(self.targets):
                self.catalogue.create_track(i, tgt.position, tgt.velocity, self.t)
        else :
            # To simulate known initial target states :
            # - compute measurements at t=-5 with the sensor pointing at the true positions
            # - update at t=0 to get some velocity information in the catalogue
            measurements = []
            for i, tgt in enumerate(self.targets):
                tgt.propagate(-5.0 * self.dt, self.world_size, self.omega0)
                self.catalogue.maybe_create_track_from_measurement(i, tgt.position, self.t - 5.0 * self.dt)

                tgt.propagate(5.0 * self.dt, self.world_size, self.omega0)
                measurements.append(Measurement(self.sensor.id, tgt.id, tgt.position, self.t))
            self.catalogue.update_from_measurements(measurements, t_now=self.t)

        measurements = self._compute_measurements()
        self.catalogue.predict_all()
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

        tracks_before_prediction = self.catalogue.get_tracks_copy()

        # 1) Apply action
        sensor_point = self._action_to_point(action) # Convert discrete action index to pointing coordinates
        sx = float(sensor_point[0])
        sy = float(sensor_point[1])
        self.sensor.set_pointing(sx, sy)

        # 2) Propagate targets and catalogue prediction
        for tgt in self.targets:
            tgt.propagate(self.dt, self.world_size, self.omega0)
        self.catalogue.predict_all(zero_noise=self.zero_noise) # Propagate catalogue state to get predicted target positions at current time step (before update)

        # 3) Measurements
        measurements = self._compute_measurements()

        # catalogue update: predict + update with new measurements
        #t_now = float(self.t)
        #self.catalogue.predict_all(t_now=t_now)
        #self.catalogue.update_from_measurements(measurements, t_now=t_now)
        self.catalogue.update_from_measurements(measurements)

        # 4) Reward computation
        delta_trace = 0.0
        for m in measurements:
            tid = m.target_id
            trk = self.catalogue.tracks[tid]
            trace_before = float(np.trace(tracks_before_prediction[tid].P[0:2, 0:2]))
            trace_after = float(np.trace(trk.P[0:2, 0:2]))
            delta_trace += trace_before - trace_after

        reward = delta_trace / (total_trace_before + 1e-8)

        # 5) Termination flags
        terminated = self.steps >= self.max_steps
        truncated = False
        # terminate if no target inside fov (no measurement acquired)
        if len(measurements) == 0:
            reward = -10.0  # Large negative reward for losing all targets
            truncated = True
        

        # 7) Observation
        obs = self._build_observation(measurements)
        info = {
            "num_measurements": len(measurements),
            "true_states": self._get_true_states_array(),
            "estimated_states": self.catalogue.as_mdp_state(include_cov_diag=True),
            "sensor_pointing": sensor_point,
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
    
    def _action_to_point(self, action: int) -> np.ndarray:
        """
        Convert discrete action index to sensor pointing coordinates.

        Returns
        -------
        np.ndarray shape (2,)
            [sx, sy] in world coordinates
        """
        i = int(action)

        gx = i % 10          # column index
        gy = i // 10         # row index

        sx = (gx*10.0 + 5.0)
        sy = (gy*10.0 + 5.0)

        return np.array([sx, sy], dtype=np.float32)
    
    """
    def _action_to_point(self, action: int) -> np.ndarray:
        ""Convert discrete action index to sensor pointing coordinates.
        Here, action corresponds to index of target to point at. 
        We point at the target's predicted position in the catalogue.""
        target_id = int(action)
        if target_id >= self.num_targets:
            raise ValueError(f"Invalid action index {target_id}. Must be in [0, {self.num_targets-1}].")

        trk = self.catalogue.tracks[target_id]
        sx = float(trk.x[0])  # Predicted x position
        sy = float(trk.x[1])  # Predicted y position

        return np.array([sx, sy], dtype=np.float32)
    """
    # ---------- Helpers: world ----------

    def _sample_initial_targets(self) -> List[Target]:
        targets: List[Target] = []
        for i in range(self.num_targets):
            x, y = self.np_random.uniform(0.0, self.world_size, size=2)
            vx, vy = self.np_random.uniform(-1.0, 1.0, size=2)
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
        total_trace = 0.0
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
        """
        return self.catalogue.build_obs_RL_grid()
