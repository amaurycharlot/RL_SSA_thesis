import numpy as np
from typing import List, Dict

from ssa_env.core.measurement import Measurement
from agents.ckf.ckf import ckf_predict_step, ckf_update_step
from agents.ckf.gen_H_linear_range import gen_H_linear_range
from agents.ckf.int_coordinated_turn_stm import int_coordinated_turn_stm
#from agents.ckf.kf_predict_update import make_F, make_Q, make_H_R, kf_predict, kf_update

import agents.ckf.perfect_ckf as perfect_ckf



class Track:
    """
    One catalogue entry (belief) for a target.
    Keeps estimator state (x,P) + metadata.
    """
    def __init__(self, target_id: int, x0: np.ndarray, P0: np.ndarray):
        self.id = int(target_id)
        self.x = x0.astype(np.float32).reshape(4,)
        self.P = P0.astype(np.float32).reshape(4,4)
        self.last_update_time = None


class CatalogueState:
    """
    Agent's belief over targets (catalogue).
    """

    def __init__(self, num_targets: int, dt: float = 1.0, omega0: float = 0.05, sigma_z: float = 1.0, sigma_a: float = 0,
        init_pos_var: float = 25.0, init_vel_var: float = 25.0, world_size: float = 100.0, initial_target_vars: List[float] = None):
        self.num_targets = int(num_targets)
        self.dt = float(dt)
        self.world_size = float(world_size)

        # noise hyperparams (used by CKF params)
        self.sigma_z = float(sigma_z)
        self.sigma_a = float(sigma_a)

        # initial uncertainty
        self.init_pos_var = float(init_pos_var)
        self.init_vel_var = float(init_vel_var)

        self.initial_target_vars = initial_target_vars
        if self.initial_target_vars is not None:
            assert len(self.initial_target_vars) == self.num_targets, "initial_target_vars length must match num_targets"
        
        self.omega0 = float(omega0)

        self.tracks: Dict[int, Track] = {}

        # CKF params dict (used by ckf_predict_step / ckf_update_step / H_fcn / intfcn)
        # Rk here is a variance; ckf.py will convert scalar -> Rk * I (for 2D measurement)
        # Q here is a scalar-like placeholder; ckf.py will convert scalar -> Q * I (for n=4)
        # omega is the turn rate used by int_coordinated_turn_stm
        self.params = {
            "Rk": self.sigma_z**2,
            "Q": self.sigma_a**2,
            "omega": self.omega0
        }

        # Wrapped integrator function: CKF expects intfcn(t, X_aug)
        self.intfcn = lambda t, X_aug: int_coordinated_turn_stm(t, X_aug, self.params)

        # Measurement model function: Hk, Gk = H_fcn(t, X, params)
        self.H_fcn = gen_H_linear_range

    def reset(self) -> None:
        self.tracks.clear()

    def predict_all(self, t_now: float = None, zero_noise: bool = False) -> None:
        """
        Predict step for all tracks (calls ckf_predict_step).
        If t_now is not provided, we assume a fixed dt step.
        """
        for trk in self.tracks.values():

            if trk.last_update_time is None:
                continue

            if t_now is None:
                t_prev = float(trk.last_update_time)
                t_k = t_prev + self.dt
            else:
                t_k = float(t_now)
                t_prev = float(trk.last_update_time)
            if zero_noise:
                # Perfect prediction with no noise: x^- = Phi x, P^- = Phi P
                x_minus, P_minus = perfect_ckf.predict_state_and_covariance(trk.x, trk.P, t_k - t_prev, self.omega0)
            else :
                #print(f"Predicting track for target_id={trk.id} from time {t_prev:.2f} to {t_k:.2f}")
                x_minus, P_minus, Phi_k = ckf_predict_step(
                    trk.x,
                    trk.P,
                    t_prev,
                    t_k,
                    self.intfcn,
                    self.params
                )

            trk.x = x_minus.astype(np.float32)
            trk.P = P_minus.astype(np.float32)
            self._wrap_track_position(trk)
            trk.last_update_time = t_k

    def update_from_measurements(self, measurements: List[Measurement], t_now: float = None, zero_noise : bool = False) -> None:
        """
        Update step for any targets that got measured this timestep.
        Measurements are assumed to include target_id for now.
        If t_now is not provided, we use measurement.time for each update.
        """
        for m in measurements:
            if m.target_id is None:
                continue

            tid = int(m.target_id)
            z = m.position.astype(np.float32).reshape(2,)

            # Determine time to use for this update
            if t_now is None:
                t_k = float(m.time)
            else:
                t_k = float(t_now)

            self.maybe_create_track_from_measurement(tid, z, t_k)
            trk = self.tracks[tid]

            # Ensure the track is predicted up to t_k (if needed)
            if trk.last_update_time is None:
                t_prev = t_k
            else:
                t_prev = float(trk.last_update_time)

            if t_k != t_prev:
                #print(f"Predicting track for target_id={tid} from time {t_prev:.2f} to {t_k:.2f} before update")
                x_minus, P_minus, Phi_k = ckf_predict_step(
                    trk.x,
                    trk.P,
                    t_prev,
                    t_k,
                    self.intfcn,
                    self.params
                )
            else:
                x_minus = trk.x
                P_minus = trk.P

            # ---- torus-consistent measurement ----
            # Predicted measurement from x_minus
            _, Gk = self.H_fcn(t_k, x_minus, self.params)
            Gk = np.atleast_1d(np.array(Gk, dtype=np.float32)).reshape(2,)

            # Replace measurement by equivalent wrapped copy closest to prediction
            z_wrapped = self._wrap_measurement_to_prediction(z, Gk).astype(np.float32)

            # CKF update
            if zero_noise:
                x_plus = x_minus.copy()
                P_plus = perfect_ckf.update_covariance(P_minus, r=self.params["Rk"])
            else:
                x_plus, P_plus, yk, S, K, ek = ckf_update_step(
                    x_minus,
                    P_minus,
                    t_k,
                    z_wrapped,
                    self.H_fcn,
                    self.params
                )
            #print(f"Update for target_id={tid} at time {t_k:.2f}")
            #print("trace P before update:", np.trace(P_minus))
            #print("trace P after update :", np.trace(P_plus))

            trk.x = x_plus.astype(np.float32)
            trk.P = P_plus.astype(np.float32)
            self._wrap_track_position(trk)
            trk.last_update_time = t_k

    def maybe_create_track_from_measurement(self, target_id: int, z: np.ndarray, t0: float = 0.0) -> None: 
        """
        If track for target_id does not exist yet, create it from first measurement z.
        """
        if target_id in self.tracks:
            return

        x0 = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float32)
        x0[0] = float(x0[0] % self.world_size)
        x0[1] = float(x0[1] % self.world_size)
        P0 = np.diag([self.init_pos_var, self.init_pos_var,
                      self.init_vel_var, self.init_vel_var]).astype(np.float32)
        self.tracks[target_id] = Track(target_id, x0, P0)
        self.tracks[target_id].last_update_time = float(t0)

    def create_track(self, i, position, velocity, t) -> None:
        """Create a track for target i with initial state from tgt and time self.t.
        """
        x0 = np.array([position[0], position[1], velocity[0], velocity[1]], dtype=np.float32)
        pos_var = self.init_pos_var
        vel_var = self.init_vel_var
        if self.initial_target_vars is not None:
            pos_var = self.initial_target_vars[i]
            vel_var = self.initial_target_vars[i]
        P0 = np.diag([pos_var, pos_var,
                      vel_var, vel_var]).astype(np.float32)
        self.tracks[i] = Track(i, x0, P0)
        self.tracks[i].last_update_time = float(t)

    def as_mdp_state(self, include_cov_diag: bool = False) -> np.ndarray:
        """
        Fixed-size array for RL:
        - (N,4): xhat = [x,y,vx,vy] per target id (or zeros if unknown)
        - (N,9) optionally includes diag(P) and trace(P)
        """
        if not include_cov_diag:
            out = np.zeros((self.num_targets, 4), dtype=np.float32)
            for tid in range(self.num_targets):
                if tid in self.tracks:
                    out[tid, :] = self.tracks[tid].x
            return out

        out = np.zeros((self.num_targets, 9), dtype=np.float32)
        for tid in range(self.num_targets):
            if tid in self.tracks:
                trk = self.tracks[tid]
                out[tid, 0:4] = trk.x
                diagP = np.diag(trk.P).astype(np.float32)
                out[tid, 4:8] = diagP
                out[tid, 8] = float(np.trace(trk.P))
        return out

    def build_obs_RL(self) -> np.ndarray:
        """
        Build the RL observation array from catalogue state.
        shape (N,) 
        N = num_targets ; per-target obs = trace(P[0:2]) as uncertainty measure
        """
        obs = np.zeros((self.num_targets), dtype=np.float32)
        for tid in range(self.num_targets):
            trk = self.tracks[tid]
            obs[tid] = float(np.trace(trk.P[0:2, 0:2]))  # uncertainty measure (trace of position covariance)
        return obs
    
    def build_obs_RL_grid(self) -> np.ndarray:
        """
        Build the RL observation array from catalogue state on a grid.
        shape (10, 10)
        """
        obs = np.zeros((10, 10), dtype=np.float32)
        for tid in range(self.num_targets):
            trk = self.tracks[tid]
            gx = int(trk.x[0] // (self.world_size / 10))
            gy = int(trk.x[1] // (self.world_size / 10))
            obs[gy, gx] = float(np.trace(trk.P[0:2, 0:2]))
        return obs.reshape(-1)

    def _wrap_pos(self, xy: np.ndarray) -> np.ndarray:
        """
        Wrap a 2D position into [0, world_size) using modulo arithmetic.
        """
        xy = xy.astype(np.float32)
        xy[0] = float(xy[0] % self.world_size)
        xy[1] = float(xy[1] % self.world_size)
        return xy

    def _wrap_track_position(self, trk: Track) -> None:
        """
        Apply wrap-around to the track's position estimate.
        """
        trk.x[0:2] = self._wrap_pos(trk.x[0:2])

    def _wrap_diff(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Minimum-image difference on a torus of size world_size.
        Returns the shortest signed difference a - b in each coordinate.
        """
        d = a - b
        L = self.world_size
        return (d + L / 2.0) % L - L / 2.0

    def _wrap_measurement_to_prediction(self, z: np.ndarray, Gk: np.ndarray) -> np.ndarray:
        """
        Return the torus-equivalent measurement closest to predicted measurement Gk.
        """
        return Gk + self._wrap_diff(z, Gk)
    
    def get_tracks_copy(self) -> Dict[int, Track]:
        """
        Return a deep copy of the tracks dict (for logging/debugging).
        """
        tracks_copy = {}
        for tid, trk in self.tracks.items():
            trk_copy = Track(trk.id, trk.x.copy(), trk.P.copy())
            trk_copy.last_update_time = trk.last_update_time
            tracks_copy[tid] = trk_copy
        return tracks_copy


"""
Old versions
------------

def predict_all(self, t_now: float = None) -> None:
        ""
        Predict step for all tracks (calls ckf_predict_step).
        If t_now is not provided, we assume a fixed dt step.
        ""
        for trk in self.tracks.values():

            if trk.last_update_time is None:
                continue

            if t_now is None:
                t_prev = float(trk.last_update_time)
                t_k = t_prev + self.dt
            else:
                t_k = float(t_now)
                t_prev = float(trk.last_update_time)

            x_minus, P_minus, Phi_k = ckf_predict_step(
                trk.x,
                trk.P,
                t_prev,
                t_k,
                self.intfcn,
                self.params
            )

            trk.x = x_minus.astype(np.float32)
            trk.P = P_minus.astype(np.float32)
            self._wrap_track_position(trk)
            trk.last_update_time = t_k



    def update_from_measurements(self, measurements: List[Measurement], t_now: float = None) -> None:
        ""
        Update step for any targets that got measured this timestep.
        Measurements are assumed to include target_id for now.
        If t_now is not provided, we use measurement.time for each update.
        ""
        for m in measurements:
            if m.target_id is None:
                continue

            tid = int(m.target_id)
            z = m.position.astype(np.float32).reshape(2,)

            # Determine time to use for this update
            if t_now is None:
                t_k = float(m.time)
            else:
                t_k = float(t_now)

            self.maybe_create_track_from_measurement(tid, z, t_k)
            trk = self.tracks[tid]

            # Ensure the track is predicted up to t_k (if needed)
            if trk.last_update_time is None:
                t_prev = t_k
            else:
                t_prev = float(trk.last_update_time)

            if t_k != t_prev:
                x_minus, P_minus, Phi_k = ckf_predict_step(
                    trk.x,
                    trk.P,
                    t_prev,
                    t_k,
                    self.intfcn,
                    self.params
                )
            else:
                x_minus = trk.x
                P_minus = trk.P

            # CKF update
            x_plus, P_plus, yk, S, K, ek = ckf_update_step(
                x_minus,
                P_minus,
                t_k,
                z,
                self.H_fcn,
                self.params
            )

            trk.x = x_plus.astype(np.float32)
            trk.P = P_plus.astype(np.float32)
            self._wrap_track_position(trk)
            trk.last_update_time = t_k

"""