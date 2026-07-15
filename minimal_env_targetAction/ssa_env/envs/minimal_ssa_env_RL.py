import math
import scipy.linalg

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List

from ssa_env.core.target import Target
from ssa_env.core.sensor import Sensor
from ssa_env.core.measurement import Measurement

from agents.catalogue import CatalogueState, Track


class MinimalSSAEnv(gym.Env):
    """
    Minimal SSA environment (see previous description).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, num_targets: int = 2, omega0: float = 0.05, measurement_noise_std=1.0, 
                 targets_known: bool = False, reward_type: str = "R1", zero_noise: bool = False, timesteps : int = 50,
                 initial_target_vars: List[float] = None):
        super().__init__()

        self.num_targets = int(num_targets)
        self.targets_known = bool(targets_known)
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
            #sigma_z=1.0 if not zero_noise else 0.0,
            sigma_z=0.5 if zero_noise else float(measurement_noise_std),
            sigma_a=0.0,
            init_pos_var=1.0,
            init_vel_var=1.0,
            initial_target_vars=initial_target_vars
        )
        self.reward_type = reward_type
        self.zero_noise = zero_noise

        # World objects
        if zero_noise:
            measurement_noise_std = 0.0
        self.targets: List[Target] = []
        self.sensor = Sensor(sensor_id=0, fov_radius=10.0, world_size=self.world_size, measurement_noise_std=measurement_noise_std)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(self.num_targets,), dtype=np.float32
        )
        
        #self.action_space = self._build_action_space()
        #self.action_space = spaces.Discrete(int(10*10))
        self.action_space = spaces.Discrete(self.num_targets)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.t = 0.0

        self.targets = self._sample_initial_targets()
        self.sensor.set_pointing(self.world_size / 2, self.world_size / 2)
        self.catalogue.reset()

        if self.targets_known:
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
        obs = self._build_observation(measurements)
        info = {"true_states": self._get_true_states_array()}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.targets is not None, "Call reset() before step()."

        self.steps += 1
        self.t += self.dt

        tracks_before_prediction = self.catalogue.get_tracks_copy()

        # 1) Apply action
        """#HEURISTIC THAT CHEATS REWARDS
        if self.steps % 17 == 0:
            action = 0
        elif self.steps % 11 == 0:
            action = 1
        else:
            action = int(self.steps % (self.num_targets-2) + 2)
        """
        

        sensor_point = self._action_to_point(action) # Convert discrete action index to pointing coordinates
        sx = float(sensor_point[0])
        sy = float(sensor_point[1])
        self.sensor.set_pointing(sx, sy)

        # 2) Propagate targets and predict catalogue
        for tgt in self.targets:
            tgt.propagate(self.dt, self.world_size, self.omega0)
        self.catalogue.predict_all(zero_noise=self.zero_noise)

        tracks_before_update = self.catalogue.get_tracks_copy()

        # 3) Measurements
        measurements = self._compute_measurements()

        # catalogue update: predict + update with new measurements
        #t_now = float(self.t)
        #self.catalogue.predict_all(t_now=t_now)
        #self.catalogue.update_from_measurements(measurements, t_now=t_now)
        self.catalogue.update_from_measurements(measurements, zero_noise=self.zero_noise)

        # 4) Reward
        reward = self._compute_reward(tracks_before_prediction, tracks_before_update, action)

        # 5) Termination flags
        terminated = self.steps >= self.max_steps
        truncated = False

        # 6) Observation
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
    
    """
    def _action_to_point(self, action: int) -> np.ndarray:
        ""
        Convert discrete action index to sensor pointing coordinates.

        Returns
        -------
        np.ndarray shape (2,)
            [sx, sy] in world coordinates
        ""
        i = int(action)

        gx = i % 10          # column index
        gy = i // 10         # row index

        sx = (gx + 5.0)
        sy = (gy + 5.0)

        return np.array([sx, sy], dtype=np.float32)
    """
    def _action_to_point(self, action: int) -> np.ndarray:
        """"Convert discrete action index to sensor pointing coordinates.
        Here, action corresponds to index of target to point at. 
        We point at the target's predicted position in the catalogue."""
        target_id = int(action)
        if target_id >= self.num_targets:
            raise ValueError(f"Invalid action index {target_id}. Must be in [0, {self.num_targets-1}].")

        trk = self.catalogue.tracks[target_id]
        sx = float(trk.x[0])  # Predicted x position
        sy = float(trk.x[1])  # Predicted y position

        return np.array([sx, sy], dtype=np.float32)

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

    def _compute_reward(self, tracks_before_prediction: Dict[int, Track], tracks_before_update: Dict[int, Track], action: int, rewardType: str="R1") -> float:
        """
        RL rewards
        """

        total_trace_before = 0.0
        for trk in tracks_before_prediction.values():
            total_trace_before += float(np.trace(trk.P[0:2, 0:2]))
        trace_before = tracks_before_prediction[int(action)].P[0:2, 0:2].trace()

        prior_pos_covs = []
        for trk in tracks_before_update.values():
            prior_pos_covs.append(trk.P[0:2, 0:2].copy())
        
        total_trace_after = 0.0
        post_pos_covs = []
        for tid in range(self.num_targets):
            trk = self.catalogue.tracks[tid]
            post_pos_covs.append(trk.P[0:2, 0:2].copy())
            total_trace_after += float(np.trace(trk.P[0:2, 0:2]))
        trace_after = self.catalogue.tracks[int(action)].P[0:2, 0:2].trace()


        delta_trace = trace_before - trace_after
        delta_total_trace = np.max([np.min([total_trace_before - total_trace_after, 50.0]), -50.0])
        eps = 1e-8
        reward = 0.0

        if self.reward_type == "R1" :
            reward = delta_total_trace / (total_trace_before + eps)
        elif self.reward_type == "R2" :
            reward = delta_trace / (total_trace_before + eps)
        elif self.reward_type == "R3" :
            for P_prior, P_post in zip(prior_pos_covs, post_pos_covs):
                reward += np.log(np.linalg.det(P_prior + eps * np.eye(2))) - np.log(np.linalg.det(P_post + eps * np.eye(2)))
        elif self.reward_type == "R4" :
            mus_prior = [t.x[0:2] for t in tracks_before_update.values()] #tracks_before_prediction[:].x[0:2]
            mus_post = [t.x[0:2] for t in self.catalogue.tracks.values()] #self.catalogue.tracks.values().x[0:2]
            for i in range(len(mus_post)) :
                mus_post[i] = self.catalogue._wrap_measurement_to_prediction(mus_post[i], mus_prior[i]).astype(np.float32)
            #P_prior = prior_pos_covs
            #P_post = post_pos_covs
            
            #for debugging renyi computation :
            logging = False
            if logging:
                print(f"Step {self.steps}:")
            #reward = self._renyi_gaussian_divergence(mu_prior, P_prior, mu_post, P_post, alpha=0.5, logging=logging)
            reward = 0
            #for i in range(self.num_targets) :
            i = int(action)
            reward += self._renyi_gaussian_divergence(mus_prior[i], prior_pos_covs[i], mus_post[i], post_pos_covs[i], 
                                                        alpha=0.5, logging=logging)
            """
            if i == 0 :
                print("Renyi divergence (step {}): {:.4f} (observed)".format(self.steps, reward))
            else :
                print("Renyi divergence (step {}): {:.4f}".format(self.steps, self._renyi_gaussian_divergence(mus_prior[0], prior_pos_covs[0], mus_post[0], post_pos_covs[0], 
                                                        alpha=0.5, logging=logging)))
            """

            if reward > 50.0:
                print(f"WARNING: large reward {reward:.2f} at step {self.steps}.")
        elif self.reward_type == "R5":
            # KL divergence of selected target's position distribution before vs after update
            mus_prior = [t.x[0:2] for t in tracks_before_update.values()]
            mus_post  = [t.x[0:2] for t in self.catalogue.tracks.values()]

            for i in range(len(mus_post)):
                mus_post[i] = self.catalogue._wrap_measurement_to_prediction(
                    mus_post[i], mus_prior[i]
                ).astype(np.float32)

            i = int(action)   # selected target only
            reward = self._kl_gaussian_divergence(
                mus_prior[i],
                prior_pos_covs[i],
                mus_post[i],
                post_pos_covs[i]
            )

        return reward



        #R1 : reward = delta_total_trace / (total_trace_before + 1e-6)
        #R2 : reward = delta_trace / (total_trace_before + 1e-6)
        #R3 : reward = delta_logdet(P) -> H(P-) - H(P+) up to constants
        
        
        #for P_prior, P_post in zip(prior_pos_covs, post_pos_covs):
        #    reward += np.log(np.linalg.det(P_prior + eps * np.eye(2))) - np.log(np.linalg.det(P_post + eps * np.eye(2)))
        
        #R4 : reward = Renyi divergence of selected target's pos
        
        

    def _renyi_gaussian_divergence(self, mu_prior, P_prior, mu_post, P_post, alpha=0.5, eps=1e-8, logging=False):
        """
        Rényi divergence D_alpha(post || prior) 
        Uses position mean/covariance
        """
        #print(type(alpha), alpha)
        if alpha <= 0.0 or np.isclose(alpha, 1.0):
            raise ValueError("alpha must be > 0 and != 1")

        mu0 = np.asarray(mu_prior).reshape(-1, 1)
        mu1 = np.asarray(mu_post).reshape(-1, 1)
        P0 = np.asarray(P_prior)
        P1 = np.asarray(P_post)

        d = P0.shape[0]
        I = np.eye(d)

        P0 = P0 + eps * I
        P1 = P1 + eps * I

        P_alpha = alpha * P1 + (1.0 - alpha) * P0
        P_alpha = P_alpha + eps * I

        sign0, logdet0 = np.linalg.slogdet(P0)
        sign1, logdet1 = np.linalg.slogdet(P1)
        signa, logdeta = np.linalg.slogdet(P_alpha)

        if logging:
            print(f"Rényi divergence computation (alpha={alpha}):")
            print(f"P_prior:\n{P0}")
            print(f"P_post:\n{P1}")
            print(f"P_alpha:\n{P_alpha}")
            print(f"logdet(P_prior)={logdet0}, logdet(P_post)={logdet1}, logdet(P_alpha)={logdeta}")

        if sign0 <= 0 or sign1 <= 0 or signa <= 0:
            raise ValueError("Covariance not positive definite in Rényi computation.")

        diff = mu1 - mu0
        quad = float(diff.T @ np.linalg.inv(P_alpha) @ diff)

        term = (
            logdeta
            - alpha * logdet1
            - (1.0 - alpha) * logdet0
            + alpha * (1.0 - alpha) * quad
        )

        div = 0.5 * term / (1.0 - alpha)

        if logging:
            print(f"quadratic term: {quad}")
            print(f"Rényi divergence: {div}")

        return float(div)
    
    def _kl_gaussian_divergence(self, mu_prior, P_prior, mu_post, P_post, eps=1e-8):
        mu0 = np.asarray(mu_prior).reshape(-1, 1)   # prior
        mu1 = np.asarray(mu_post).reshape(-1, 1)    # post
        P0 = np.asarray(P_prior, dtype=np.float64)
        P1 = np.asarray(P_post, dtype=np.float64)

        d = P0.shape[0]
        I = np.eye(d)

        P0 = P0 + eps * I
        P1 = P1 + eps * I

        invP1 = np.linalg.inv(P1)

        sign0, logdet0 = np.linalg.slogdet(P0)
        sign1, logdet1 = np.linalg.slogdet(P1)
        if sign0 <= 0 or sign1 <= 0:
            raise ValueError("Covariance not positive definite in KL computation.")

        diff = mu1 - mu0

        div = 0.5 * (
            (logdet1 - logdet0)
            - d
            + np.trace(invP1 @ P0)
            + float((diff.T @ invP1 @ diff)[0][0])
        )
        return float(div)

    def _get_true_states_array(self) -> np.ndarray:
        arr = np.zeros((self.num_targets, 4), dtype=np.float32)
        for i, tgt in enumerate(self.targets):
            arr[i] = tgt.state
        return arr

    def _build_observation(self, measurements: List[Measurement]) -> dict:
        """
        Build the observation dict from updated catalogue state.
        """
        return self.catalogue.build_obs_RL()
