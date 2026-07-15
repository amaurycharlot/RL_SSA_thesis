import numpy as np


def make_F(dt: float, omega: float) -> np.ndarray:
    """
    Discrete-time transition for:
        p_{k+1} = p_k + v_k * dt
        v_{k+1} = R(omega) v_k
    State: [x, y, vx, vy]
    """
    dt = float(dt)
    omega = float(omega)

    c = float(np.cos(omega))
    s = float(np.sin(omega))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    F = np.eye(4, dtype=np.float32)
    F[0, 2] = dt
    F[1, 3] = dt
    F[2:4, 2:4] = R
    return F


def make_Q(dt: float, sigma_a: float) -> np.ndarray:
    """
    Simple discrete process noise for constant-velocity model in 2D with white accel noise.

    This is the standard CV(Q) block structure per axis:
        Q1D = sigma_a^2 * [[dt^4/4, dt^3/2],
                          [dt^3/2, dt^2]]

    We apply it independently to x and y axes and assemble into 4x4 for [x,y,vx,vy].
    """
    dt = float(dt)
    sa2 = float(sigma_a) ** 2

    q11 = (dt ** 4) / 4.0
    q12 = (dt ** 3) / 2.0
    q22 = (dt ** 2)

    Q1 = sa2 * np.array([[q11, q12],
                         [q12, q22]], dtype=np.float32)

    Q = np.zeros((4, 4), dtype=np.float32)
    # x/vx block
    Q[np.ix_([0, 2], [0, 2])] = Q1
    # y/vy block
    Q[np.ix_([1, 3], [1, 3])] = Q1
    return Q


def make_H_R(sigma_z: float):
    """
    Measurement model:
        z = [x, y] + noise
    """
    sz2 = float(sigma_z) ** 2
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float32)
    R = sz2 * np.eye(2, dtype=np.float32)
    return H, R


def kf_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray):
    """
    Standard linear KF predict:
        x- = F x
        P- = F P F^T + Q
    """
    x = x.astype(np.float32).reshape(4,)
    P = P.astype(np.float32).reshape(4, 4)

    x_minus = (F @ x).astype(np.float32)
    P_minus = (F @ P @ F.T + Q).astype(np.float32)
    return x_minus, P_minus


def kf_update(x_minus: np.ndarray, P_minus: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray):
    """
    Standard linear KF update (Joseph form).
    z is shape (2,) for [x,y].
    """
    x_minus = x_minus.astype(np.float32).reshape(4,)
    P_minus = P_minus.astype(np.float32).reshape(4, 4)
    z = z.astype(np.float32).reshape(2,)

    # Innovation
    y = z - (H @ x_minus)

    S = H @ P_minus @ H.T + R
    # Kalman gain using solve for stability
    K = np.linalg.solve(S.T, (P_minus @ H.T).T).T  # (4,2)

    x_plus = x_minus + (K @ y)
    I = np.eye(4, dtype=np.float32)
    KH = K @ H
    P_plus = (I - KH) @ P_minus @ (I - KH).T + K @ R @ K.T

    return x_plus.astype(np.float32), P_plus.astype(np.float32), y.astype(np.float32), S.astype(np.float32), K.astype(np.float32)