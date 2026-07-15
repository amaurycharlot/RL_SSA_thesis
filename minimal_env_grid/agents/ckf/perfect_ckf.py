import numpy as np


def position_measurement_matrix() -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float64)


def predict_state_and_covariance(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict step with turning dynamics and Q = 0.
    """
    c = np.cos(omega * dt)
    s = np.sin(omega * dt)

    F = np.array([
        [1.0, 0.0, dt * c, -dt * s],
        [0.0, 1.0, dt * s,  dt * c],
        [0.0, 0.0, c,      -s     ],
        [0.0, 0.0, s,       c     ],
    ], dtype=np.float64)

    x_pred = F @ x
    P_pred = F @ P @ F.T
    P_pred = 0.5 * (P_pred + P_pred.T)

    return x_pred, P_pred


def update_covariance(
    P_pred: np.ndarray,
    r: float = 1e-1,
) -> np.ndarray:
    """
    Covariance update for direct position observation with small R > 0.

    Parameters
    ----------
    P_pred : (4,4) ndarray
        Predicted covariance.
    r : float
        Measurement noise variance for each observed position component.
        Uses R = r * I_2.
    """
    H = position_measurement_matrix()
    R = r * np.eye(2, dtype=np.float64)

    PHt = P_pred @ H.T
    S = H @ P_pred @ H.T + R

    K = PHt @ np.linalg.pinv(S)

    I = np.eye(P_pred.shape[0], dtype=np.float64)

    # Joseph form
    P_post = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T

    P_post = 0.5 * (P_post + P_post.T)

    eigvals, eigvecs = np.linalg.eigh(P_post)
    eigvals = np.clip(eigvals, 0.0, None)
    P_post = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return P_post