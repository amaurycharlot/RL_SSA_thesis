import math
import scipy.linalg

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List

from ssa_env.core.target import Target
from ssa_env.core.sensor import Sensor
from ssa_env.core.measurement import Measurement
from ssa_env.core.grid_geometry import point_to_grid_cell

from agents.catalogue import CatalogueState, Track
from agents.ckf.perfect_ckf import update_covariance


class MinimalSSAEnv(gym.Env):
    """
    Minimal SSA environment (see previous description).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Params: self, render_mode, known_targets, unknown_targets, omega0, measurement_noise_std, reward_type,
    #         search_reward, zero_noise, timesteps, initial_target_vars, grid_shape, mode, dwell_steps, slew_alpha,
    #         simple_obs, uncertainty_norm_scale
    # Returns: None.
    def __init__(self, render_mode=None, known_targets: int = 0, unknown_targets: int = 0, omega0: float = 0.05, measurement_noise_std=1.0, 
                 reward_type: str = "R1", search_reward: str = "R1", zero_noise: bool = False, timesteps : int = 50,
                 initial_target_vars: List[float] = None, grid_shape: tuple[int, int] = (10, 10),
                 mode: str = "track", dwell_steps: int = 1, slew_alpha: float = 1.0, simple_obs: bool = False,
                 uncertainty_norm_scale: float = 100.0):
        super().__init__()

        self.known_targets = int(known_targets)
        self.unknown_targets = int(unknown_targets)
        self.num_targets = self.known_targets + self.unknown_targets
        if mode not in {"track", "search", "combined"}:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'track', 'search', or 'combined'.")
        if mode == "search" and self.known_targets > 0:
            raise ValueError("Search mode does not support known targets. Set known_targets=0.")
        if mode == "track" and self.unknown_targets > 0:
            raise ValueError("Track mode does not support unknown targets. Set unknown_targets=0.")
        self.mode=mode

        self.omega0 = float(omega0)
        self.world_size = 100.0
        self.dt = 1.0
        self.t = 0.0
        self.max_steps = int(timesteps)
        self.steps = 0
        self.dwell_steps = int(dwell_steps)
        self.slew_alpha = float(slew_alpha)
        self.current_grid_cell = None
        self.uncertainty_norm_scale = float(uncertainty_norm_scale)
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
            initial_target_vars=initial_target_vars,
            grid_shape=grid_shape,
            uncertainty_norm_scale=self.uncertainty_norm_scale
        )
        self.reward_type = reward_type
        self.search_reward = search_reward
        self.zero_noise = zero_noise
        self.grid_shape = grid_shape
        self.simple_obs = simple_obs

        # World objects
        if zero_noise:
            measurement_noise_std = 0.0
        self.targets: List[Target] = []
        self.sensor = Sensor(sensor_id=0, world_size=self.world_size, measurement_noise_std=measurement_noise_std, grid_shape=self.grid_shape)

        # Gym spaces

        self.obs_layers = 2 if mode in {"track", "search"} else 3
        if self.simple_obs:
            self.obs_layers -= 1 # remove one layer for sensor pointing if simple_obs

        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(int(grid_shape[0]) , int(grid_shape[1]), self.obs_layers), dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([int(grid_shape[0]), int(grid_shape[1])])
        if self.mode == "combined":
            self.reward_dim = 2
            self.reward_order = ("tracking", "search")
            self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode

    # Params: self, seed, options; returns: (obs, info).
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Target sampling and measurement noise belong to this seeded environment.
        # Rebind after explicit reseeding; seed=None continues the existing stream.
        self.sensor.rng = self.np_random
        self.steps = 0
        self.t = 0.0

        self.catalogue.reset()
        self.targets = self._sample_initial_targets()
        center_cell = (self.grid_shape[0] // 2, self.grid_shape[1] // 2)
        self.current_grid_cell = center_cell
        sensor_point = self._grid_cell_to_point(center_cell)
        self.sensor.set_pointing(float(sensor_point[0]), float(sensor_point[1]))

        measurements = self._compute_measurements()
        obs = self._build_observation(measurements)
        info = {"true_states": self._get_true_states_array()}
        return obs, info

    # Params: self, action; returns: (obs, reward, terminated, truncated, info).
    def step(self, action: np.ndarray):
        assert self.targets is not None, "Call reset() before step()."

        # compute elapsed time based on action (dwell + slew time)
        next_cell = self._action_to_grid_cell(action)
        elapsed_steps = self._compute_elapsed_steps(next_cell)
        elapsed_time = elapsed_steps * self.dt

        self.steps += elapsed_steps
        self.t += elapsed_time

        tracks_before_prediction = self.catalogue.get_tracks_copy()

        # 1) Apply action
        sensor_point = self._grid_cell_to_point(next_cell)
        self.sensor.set_pointing(float(sensor_point[0]), float(sensor_point[1]))
        self.current_grid_cell = next_cell

        # 2) Propagate targets and predict catalogue
        for tgt in self.targets:
            tgt.propagate(elapsed_time, self.world_size, self.omega0)
        self.catalogue.predict_all(t_now=self.t, zero_noise=self.zero_noise)

        tracks_before_update = self.catalogue.get_tracks_copy()

        # 3) Measurements
        measurements = self._compute_measurements()

        # catalogue update
        observation_grid_before_update = self.catalogue.get_observation_grid_copy()
        n_detections = self.catalogue.update_from_measurements(measurements, t_now=self.t, zero_noise=self.zero_noise)
        self.catalogue.update_observation_grid(action, elapsed_steps=elapsed_steps)
        observation_grid_after_update = self.catalogue.get_observation_grid_copy()

        # 4) Observation
        obs = self._build_observation(measurements)
        info = {
            "num_measurements": len(measurements),
            "observed_target_ids": [int(meas.target_id) for meas in measurements],
            "true_states": self._get_true_states_array(),
            "estimated_states": self.catalogue.as_mdp_state(include_cov_diag=True),
            "sensor_pointing": sensor_point,
            "elapsed_steps": elapsed_steps,
            "total_steps": self.steps,
        }

        # 5) Reward
        if self.mode == "track" :
            reward = self._compute_track_reward(tracks_before_prediction, tracks_before_update, action, measurements=measurements, rewardType=self.reward_type)
        elif self.mode == "search" :
            reward = self._compute_search_reward(action, measurements, observation_grid_before_update, observation_grid_after_update, n_detections, elapsed_steps=elapsed_steps)
        else : # combined reward
            reward = self._compute_track_reward(tracks_before_prediction, tracks_before_update, action, measurements=measurements, rewardType=self.reward_type)
            #print(f"Step {self.steps}")
            #print(f"Track reward: {reward:.3f}")
            search_reward = self._compute_search_reward(action, measurements, observation_grid_before_update, observation_grid_after_update, n_detections, elapsed_steps=elapsed_steps)
            #print(f"Search reward: {search_reward:.3f}")
            #print("--------------------------------")
            # Combine rewards into vector
            reward = np.array([reward, search_reward], dtype=np.float32)
            # Diagnostics from the existing update; no reward/observation changes.
            info["coverage_mean"] = float(np.mean(observation_grid_after_update))
            info["new_detections"] = int(n_detections)

        # 6) Termination flags
        terminated = self.steps >= self.max_steps
        truncated = False
        if self.mode == "track" : #early termination if no target observed for 5 steps   
            if len(measurements) == 0:
                self.no_observation_steps = getattr(self, "no_observation_steps", 0) + 1
            else:
                self.no_observation_steps = 0
            #if self.no_observation_steps >= 5:
            #    truncated = True

        return obs, reward, terminated, truncated, info

    # ---------- Helpers: world ----------

    def _action_to_grid_cell(self, action) -> tuple[int, int]:
        gx = int(action[0])
        gy = int(action[1])
        return gx, gy


    # Params: self, cell; returns: np.ndarray.
    def _grid_cell_to_point(self, cell: tuple[int, int]) -> np.ndarray:
        gx, gy = cell
        sx = (gx + 0.5) * (self.world_size / self.grid_shape[0])
        sy = (gy + 0.5) * (self.world_size / self.grid_shape[1])
        return np.array([sx, sy], dtype=np.float32)
    
    # Params: self, point; returns: tuple[int, int].
    def _point_to_grid_cell(self, point: np.ndarray) -> tuple[int, int]:
        return point_to_grid_cell(point, self.world_size, self.grid_shape)

    # Params: self, action; returns: np.ndarray.
    def _action_to_point(self, action) -> np.ndarray:
        return self._grid_cell_to_point(self._action_to_grid_cell(action))
    
    # Params: self, next_cell; returns: int.
    def _compute_elapsed_steps(self, next_cell: tuple[int, int]) -> int:
        if self.current_grid_cell is None:
            return self.dwell_steps

        gx0, gy0 = self.current_grid_cell
        gx1, gy1 = next_cell

        dist = self._wrap_grid_distance(np.array([gx0, gy0]), np.array([gx1, gy1]))
        slew_steps = math.ceil(self.slew_alpha * dist)

        return self.dwell_steps + slew_steps

    # Params: self; returns: List[Target].
    def _sample_initial_targets(self) -> List[Target]:
        targets: List[Target] = []
        for i in range(self.num_targets):
            x, y = self.np_random.uniform(0.0, self.world_size, size=2)
            vx, vy = self.np_random.uniform(-1.0, 1.0, size=2)
            tgt = Target(target_id=i, x=x, y=y, vx=vx, vy=vy)
            targets.append(tgt)
            if i < self.known_targets:
                self.catalogue.create_track(i, tgt.position, tgt.velocity, self.t)
        return targets

    # Params: self; returns: List[Measurement].
    def _compute_measurements(self) -> List[Measurement]:
        return self.sensor.measure_targets(self.targets, time=self.t)
    
    # Params: self; returns: np.ndarray.
    def _get_true_states_array(self) -> np.ndarray:
        arr = np.zeros((self.num_targets, 4), dtype=np.float32)
        for i, tgt in enumerate(self.targets):
            arr[i] = tgt.state
        return arr

    def _build_observation(self, measurements: List[Measurement]) -> np.ndarray:
        """
        Build the observation array from updated catalogue state.
        """
        one_hot_sensor_grid = np.zeros((self.grid_shape[0], self.grid_shape[1]), dtype=np.float32)
        one_hot_sensor_grid[self.current_grid_cell[0], self.current_grid_cell[1]] = 1.0
        if self.mode == "track" :
            combined_obs = np.zeros((self.grid_shape[0], self.grid_shape[1], self.obs_layers), dtype=np.float32)
            combined_obs[:, :, 0] = self.catalogue.build_obs_track().reshape(self.grid_shape[0], self.grid_shape[1])
            if not self.simple_obs:
                combined_obs[:, :, 1] = one_hot_sensor_grid
            return combined_obs
        if self.mode == "search" :
            combined_obs = np.zeros((self.grid_shape[0], self.grid_shape[1], self.obs_layers), dtype=np.float32)
            combined_obs[:, :, 0] = self.catalogue.build_obs_search().reshape(self.grid_shape[0], self.grid_shape[1])
            if not self.simple_obs:
                combined_obs[:, :, 1] = one_hot_sensor_grid
            return combined_obs
        track_obs = self.catalogue.build_obs_track()
        search_obs = self.catalogue.build_obs_search()
        # construct np array of shape (grid_size*grid_size,2) with track_obs followed by search_obs
        combined_obs = np.zeros((self.grid_shape[0], self.grid_shape[1], self.obs_layers), dtype=np.float32)
        combined_obs[:, :, 0] = track_obs
        combined_obs[:, :, 1] = search_obs
        if not self.simple_obs:
            combined_obs[:, :, 2] = one_hot_sensor_grid
        return combined_obs

    def _compute_track_reward(self, tracks_before_prediction: Dict[int, Track], tracks_before_update: Dict[int, Track], 
                        action: np.ndarray, measurements: List[Measurement] =[], rewardType: str="R1") -> float:
        """
        RL rewards
        """
        # assert if any target observed
        target_observed = False
        if len(measurements) > 0:
            target_observed = True
            target_observed_ids = set([meas.target_id for meas in measurements])
            target_observed_id = min(target_observed_ids)

        """NO OBSERVATION --> PENALTY"""
        #if not target_observed:
        #    return -1.0  # penalty for not observing any target
        
        total_trace_before = 0.0
        #for trk in tracks_before_prediction.values():
        #    total_trace_before += float(np.trace(trk.P[0:2, 0:2]))
        #trace_before = tracks_before_prediction[int(target_observed_id)].P[0:2, 0:2].trace()

        prior_pos_covs = []
        nb_tracks_before = len(tracks_before_update)
        for trk in tracks_before_update.values():
            prior_pos_covs.append(trk.P[0:2, 0:2].copy())
            total_trace_before += float(np.trace(trk.P[0:2, 0:2]))
        
        total_trace_after = 0.0
        post_pos_covs = []
        i = 0
        for trk in self.catalogue.tracks.values():
            i += 1
            if i > nb_tracks_before:
                break
            post_pos_covs.append(trk.P[0:2, 0:2].copy())
            total_trace_after += float(np.trace(trk.P[0:2, 0:2]))
        #trace_after = self.catalogue.tracks[int(target_observed_id)].P[0:2, 0:2].trace()


        #delta_trace = trace_before - trace_after
        delta_total_trace = np.max([np.min([total_trace_before - total_trace_after, 50.0]), -50.0])
        eps = 1e-8
        reward = 0.0

        # TODO : FIX trace_after, trace_before for the selected target only
        # The problem comes from storing traces as fixed-sized lists while target ids can be larger than previously known catalogue
        if self.reward_type == "R1" or self.reward_type == "R6" or self.reward_type == "R7":
            reward = delta_total_trace / (total_trace_before + eps)
        elif self.reward_type == "R2" :
            reward = 0.0
            #reward = delta_trace / (total_trace_before + eps)
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
            i = int(target_observed_id)
            reward += self._renyi_gaussian_divergence(mus_prior[i], prior_pos_covs[i], mus_post[i], post_pos_covs[i], 
                                                        alpha=0.5, logging=logging)

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

            i = int(target_observed_id)   # selected target only
            reward = self._kl_gaussian_divergence(
                mus_prior[i],
                prior_pos_covs[i],
                mus_post[i],
                post_pos_covs[i]
            )
        if self.reward_type == "R6":
            # This is based on R1, but with an additional proximity bonus
            # The intuition is to encourage the agent to point close to the target
            # Proximity bonus r_prox = alpha * max(uncertainty_target / (1.0 + distance) for target in known_tracks)
            alpha = 0.1
            proximity_bonus = 0.0
            for trk in tracks_before_update.values():
                target_pos = trk.x[0:2]
                target_uncertainty = float(np.trace(trk.P[0:2, 0:2])) / (total_trace_before + eps)
                grid_distance = self._wrap_grid_distance(self._point_to_grid_cell(target_pos), self.current_grid_cell)
                proximity_bonus = max(proximity_bonus, target_uncertainty / (1.0 + grid_distance))
            reward += alpha * proximity_bonus
        if self.reward_type == "R7":
            beta = 1.0
            probability_bonus = 0.0
            cell_center = np.asarray(self.sensor.position, dtype=np.float32)

            for trk in tracks_before_update.values():
                mu = trk.x[0:2]
                P = trk.P[0:2, 0:2]
                p_cell = self._gaussian_cell_probability_cheap(mu=mu, P=P, cell_center=cell_center)
                target_uncertainty = float(np.trace(P)) / (total_trace_before + eps)
                probability_bonus += p_cell * target_uncertainty
            reward += beta * probability_bonus
            

        return reward



        #R1 : reward = delta_total_trace / (total_trace_before + 1e-6)
        #R2 : reward = delta_trace / (total_trace_before + 1e-6)
        #R3 : reward = delta_logdet(P) -> H(P-) - H(P+) up to constants
        
        
        #for P_prior, P_post in zip(prior_pos_covs, post_pos_covs):
        #    reward += np.log(np.linalg.det(P_prior + eps * np.eye(2))) - np.log(np.linalg.det(P_post + eps * np.eye(2)))
        
        #R4 : reward = Renyi divergence of selected target's pos

    def _compute_search_reward(self, action: int, measurements: List[Measurement], observation_grid_before_update: np.ndarray, 
                               observation_grid_after_update: np.ndarray, n_detections: int, elapsed_steps: int = 0) -> float:
        """
        Reward for searching. 
        R1 : mean coverage metric
        R2 : mean coverage improvement in 3x3 neighborhood (FoV) of selected cell
        R3 : number of new detections
        R4 : number of new detections with diminishing returns over time: reward = n_new_detections * exp(-steps/100)
        """
        if self.search_reward == "R1" :
            obs_grid = self.catalogue.get_observation_grid_copy()
            reward = np.sum(obs_grid)
        elif self.search_reward == "R2" or self.search_reward == "R5" :
            reward = 0
            gx = int(action[0])
            gy = int(action[1])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    gx_i = gx + i
                    gy_j = gy + j
                    gx_i, gy_j = self.catalogue._wrap_grid_index(gx_i, gy_j)
                    reward += observation_grid_after_update[gx_i, gy_j] - observation_grid_before_update[gx_i, gy_j]
            reward /= 9.0
        elif self.search_reward == "R3" :
            # Compute number of NEW detected targets (i.e. observed now but not observed before) and give a reward of +1 for each
            reward = n_detections * 1.0
        elif self.search_reward == "R4" :
            # Compute number of NEW detected targets and give a reward of +1 for each, but with diminishing returns over elapsed time steps: reward = n_new_detections * exp(-steps/100)
            reward = n_detections * np.exp(-self.steps / 100)
        elif self.search_reward == "R5" :
            # Similar to R2 but adding a diminishing return factor based on elapsed steps: reward = (mean coverage improvement in 3x3 neighborhood) * exp(-steps/10)
            reward *= np.exp(-self.steps / 10)

        return reward
        
    # ---------- Helpers: computations for reward ----------
    def _renyi_gaussian_divergence(self, mu_prior, P_prior, mu_post, P_post, alpha=0.5, eps=1e-8, logging=False):
        """
        Rényi divergence D_alpha(post || prior) 
        Uses position mean/covariance
        """
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

        return float(div)
    
    # Params: self, mu_prior, P_prior, mu_post, P_post, eps; returns: float.
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
    
    # Params: self, mu, P, cell_center; returns: float.
    def _gaussian_cell_probability_cheap(self, mu: np.ndarray, P: np.ndarray, cell_center: np.ndarray) -> float:
        eps = 1e-8

        sigma2 = float(np.mean(np.diag(P[0:2, 0:2])))
        sigma2 = max(sigma2, eps)

        diff = cell_center - mu
        d2 = float(diff.T @ diff)

        return float(np.exp(-0.5 * d2 / sigma2))
    
    def _wrap_grid_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """
        Compute distance between two positions in the grid, accounting for wrap-around at edges.
        """
        gx = abs(pos1[0] - pos2[0])
        gy = abs(pos1[1] - pos2[1])
        wrapped_gx = min(gx, self.grid_shape[0] - gx)
        wrapped_gy = min(gy, self.grid_shape[1] - gy)
        return np.sqrt(wrapped_gx ** 2 + wrapped_gy ** 2)
