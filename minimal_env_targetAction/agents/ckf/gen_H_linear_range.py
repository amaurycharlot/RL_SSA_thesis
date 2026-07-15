import numpy as np

def gen_H_linear_range(t, X, params):
    """
    Generate the measurement sensitivity matrix H for a linear range measurement model.

    Parameters:
    -----------
    t : float
        Current time (not used in this model as it's time-invariant).
    X : ndarray, shape (5,)
        State vector containing:
        [0] - position (x)
        [1] - position (y)
        [2] - velocity (vx)
        [3] - velocity (vy)
    params : dict
        Dictionary of parameters (not used in this model).
    Returns:
    --------
    H : ndarray, shape (2, 4)
        Measurement sensitivity matrix.
    """ 

    x = X[0]
    y = X[1]

    # Expected measurement
    Gk = np.array([x, y])

    # Observation matrix (2x4)
    Hk_til = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0]])

    return Hk_til, Gk
