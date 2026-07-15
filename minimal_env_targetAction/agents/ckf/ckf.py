import numpy as np
from scipy.integrate import solve_ivp

def _make_Qd(params, n):
    """
    Convert params["Q"] into a discrete-time process noise matrix Qd (n x n).

    Homework provides Q = [1e-8] (scalar-like) -> Q*I
    If user provides (n,n) matrix, we use it directly.
    If user provides length-n vector, we use diag(vector).
    """
    if "Q" not in params:
        return np.zeros((n, n))

    Q = params["Q"]

    Q_arr = np.array(Q)

    # Scalar
    if Q_arr.ndim == 0:
        return float(Q_arr) * np.eye(n)

    # 1-element vector/list (e.g., [1e-8])
    if Q_arr.ndim == 1 and Q_arr.size == 1:
        return float(Q_arr[0]) * np.eye(n)

    # length-n vector -> diagonal
    if Q_arr.ndim == 1 and Q_arr.size == n:
        return np.diag(Q_arr.astype(float))

    # full matrix
    if Q_arr.ndim == 2 and Q_arr.shape == (n, n):
        return Q_arr.astype(float)

    raise ValueError(f"Unsupported Q shape {Q_arr.shape} in params['Q']")

def _make_R(params, m):
    """
    Convert params["Rk"] into a measurement noise matrix R (m x m).

    - scalar -> Rk * I
    - 1-element list -> Rk * I
    - length-m vector -> diag(vector)
    - (m,m) matrix -> as is
    """
    if "Rk" not in params:
        return np.zeros((m, m))

    Rk = params["Rk"]
    R_arr = np.array(Rk)

    if R_arr.ndim == 0:
        return float(R_arr) * np.eye(m)

    if R_arr.ndim == 1 and R_arr.size == 1:
        return float(R_arr[0]) * np.eye(m)

    if R_arr.ndim == 1 and R_arr.size == m:
        return np.diag(R_arr.astype(float))

    if R_arr.ndim == 2 and R_arr.shape == (m, m):
        return R_arr.astype(float)

    raise ValueError(f"Unsupported Rk shape {R_arr.shape} in params['Rk']")

def ckf_predict_step(x_plus, P_plus, t_prev, t_k, intfcn, params):
    """
    One CKF predict step from t_prev -> t_k.
    Returns x_minus, P_minus, Phi_k.
    """
    n = len(x_plus)
    Qd = _make_Qd(params, n)

    if t_k == t_prev:
        return x_plus.copy(), P_plus.copy(), np.eye(n)

    X_aug0 = np.zeros(n + n * n)
    X_aug0[0:n] = x_plus
    X_aug0[n:] = np.eye(n).reshape(-1)

    sol = solve_ivp(
        intfcn,
        (t_prev, t_k),
        X_aug0,
        t_eval=[t_k],
        rtol=1e-9,
        atol=1e-9
    )
    if not sol.success:
        raise RuntimeError("CKF ODE integration failed: " + sol.message)

    X_aug_k = sol.y[:, -1]
    x_minus = X_aug_k[0:n]
    Phi_k = X_aug_k[n:].reshape((n, n))

    P_minus = Phi_k @ P_plus @ Phi_k.T + Qd

    return x_minus, P_minus, Phi_k

def ckf_update_step(x_minus, P_minus, t_k, z_k, H_fcn, params):
    """
    One CKF update step at time t_k with measurement z_k (shape (m,)).
    Returns x_plus, P_plus, yk, S, K, ek.
    """
    n = len(x_minus)

    # Ensure z is a vector (m,)
    z = np.atleast_1d(np.array(z_k, dtype=float)).reshape(-1)
    m = z.shape[0]

    # Build R (m x m) from params["Rk"]
    R = _make_R(params, m)

    # Measurement model
    Hk, Gk = H_fcn(t_k, x_minus, params)
    Hk = np.atleast_2d(Hk)                                   # (m,n) expected
    Gk = np.atleast_1d(np.array(Gk, dtype=float)).reshape(-1)  # (m,)

    # Innovation y = z - h(x-)
    yk = z - Gk                                               # (m,)

    # Innovation covariance
    # S = H P- H^T + R
    S = Hk @ P_minus @ Hk.T + R                               # (m,m)

    # Kalman Gain
    # K = P- H^T S^{-1}
    PHt = P_minus @ Hk.T                                      # (n,m)
    #K = np.linalg.solve(S.T, PHt.T).T                         # (n,m)
    K = PHt @ np.linalg.pinv(S)                                       # (n,m)

    # Update state
    xhat = K @ yk                                             # (n,)
    x_plus = x_minus + xhat

    # Update covariance (Joseph form)
    I = np.eye(n)
    KH = K @ Hk
    P_plus = (I - KH) @ P_minus @ (I - KH).T + K @ R @ K.T

    # Post-fit residual: e = y - H * xhat
    ek = yk - (Hk @ xhat)

    return x_plus, P_plus, yk, S, K, ek

def distance_wrapped(x, y, world_size=100.0) -> np.ndarray:
    """
    Compute the (dx, dy) distance from x to y, accounting for world wrapping.
    """
    dx = y[0] - x[0]
    dy = y[1] - x[1]

    # Wrap around world edges
    if dx > world_size / 2:
        dx -= world_size
    elif dx < -world_size / 2:
        dx += world_size

    if dy > world_size / 2:
        dy -= world_size
    elif dy < -world_size / 2:
        dy += world_size

    return np.array([dx, dy], dtype=np.float32)

def ckf(X0_ref, t_obs, obs_data, intfcn, H_fcn, params):
    """
    Conventional Kalman Filter (CKF) for sequential estimation.

    Inputs:
        X0_ref   : (n,) initial state estimate at t0
        t_obs    : (L,) observation times
        obs_data : (L,) scalar measurements (range/position) at each time
        intfcn   : ODE function for dynamics+STM: f(t, X_aug) -> dX_aug/dt
                   IMPORTANT: should already be wrapped so it only needs (t, X_aug)
        H_fcn    : measurement function: Hk, Gk = H_fcn(t, X, params)
        params   : dict with at least:
                    "Po" : (n,n) initial covariance
                    "Rk" : scalar measurement noise variance
                    "Q"  : process noise (scalar, diag vector, or matrix)

    Outputs:
        Xk_mat : (n, L) updated state estimates at each observation epoch
        P_mat  : (n, n, L) updated covariances at each epoch
        resids : (p, L) post-fit residuals e_k = y_k - H_k * xhat_k
                 Here p = 1, so shape is (1, L)
    """

    # Unpack parameters
    P0 = params["Po"]
    Rk = params["Rk"]

    n = len(X0_ref)
    L = len(t_obs)

    # Process noise matrix
    Qd = _make_Qd(params, n)

    # Standardize obs_data to shape (L, m)
    obs_arr = np.array(obs_data, dtype=float)
    if obs_arr.ndim == 1:
        # (L,) -> (L,1)
        obs_arr = obs_arr.reshape((L, 1))
    elif obs_arr.ndim == 2:
        # Accept either (L,m) or (m,L)
        if obs_arr.shape[0] == L:
            pass
        elif obs_arr.shape[1] == L:
            obs_arr = obs_arr.T
        else:
            raise ValueError(f"obs_data shape {obs_arr.shape} incompatible with L={L}")
    else:
        raise ValueError("obs_data must be 1D or 2D")

    m = obs_arr.shape[1]

    # Outputs
    Xk_mat = np.zeros((n, L))
    P_mat = np.zeros((n, n, L))
    resids = np.zeros((m, L))

    # Initial state/covariance (at t0)
    x_plus = X0_ref.copy()
    P_plus = P0.copy()

    # Helper
    I = np.eye(n)

    # ---- Loop over measurements ----
    for k in range(L):
        t_k = t_obs[k]
        z_k = obs_arr[k, :]  # shape (m,)

        # 1) PREDICT: propagate from previous epoch to current
        if k == 0:
            # No propagation needed at t0
            x_minus = x_plus.copy()
            P_minus = P_plus.copy()
        else:
            t_prev = t_obs[k - 1]

            # Augmented initial condition: [state; Phi_flat] with Phi(t_prev,t_prev)=I
            X_aug0 = np.zeros(n + n * n)
            X_aug0[0:n] = x_plus
            X_aug0[n:] = np.eye(n).reshape(-1)

            # Integrate from t_prev to t_k and extract final state and Phi
            sol = solve_ivp(
                intfcn,
                (t_prev, t_k),
                X_aug0,
                t_eval=[t_k],
                rtol=1e-9,
                atol=1e-9
            )
            if not sol.success:
                raise RuntimeError("CKF ODE integration failed: " + sol.message)

            X_aug_k = sol.y[:, -1]
            x_minus = X_aug_k[0:n]
            Phi_k = X_aug_k[n:].reshape((n, n))   # Phi(t_k, t_prev)

            # Covariance prediction with process noise
            P_minus = Phi_k @ P_plus @ Phi_k.T + Qd

        # 2) MEASUREMENT MODEL (linearized / Jacobian)
        Hk, Gk = H_fcn(t_k, x_minus, params)      # Hk: (m,n), Gk: (m,)
        Hk = np.atleast_2d(Hk)                    # ensure 2D
        Gk = np.atleast_1d(np.array(Gk, dtype=float)).reshape(-1)

        # 3) INNOVATION (pre-fit residual)
        # y_k = z_k - h(x_minus)
        yk = np.atleast_1d(z_k).reshape(-1) - Gk  # (m,)

        # 4) INNOVATION COVARIANCE
        # S = H P- H^T + R
        R = _make_R(params, m)
        S = Hk @ P_minus @ Hk.T + R               # (m,m)

        # 5) KALMAN GAIN
        # K = P- H^T S^{-1}
        PHt = P_minus @ Hk.T                      # (n,m)
        K = np.linalg.solve(S.T, PHt.T).T         # (n,m)

        # 6) UPDATE STATE
        xhat = K @ yk                             # deviation applied at this epoch (n,)
        x_plus = x_minus + xhat

        # 7) UPDATE COVARIANCE (Joseph form)
        KH = K @ Hk                               # (n,n)
        P_plus = (I - KH) @ P_minus @ (I - KH).T + K @ R @ K.T

        # 8) POST-FIT residual: e_k = y_k - H_k * xhat
        # (xhat is the correction applied)
        ek = yk - (Hk @ xhat)
        resids[:, k] = ek

        # 9) Store outputs (updated)
        Xk_mat[:, k] = x_plus
        P_mat[:, :, k] = P_plus

    return Xk_mat, P_mat, resids