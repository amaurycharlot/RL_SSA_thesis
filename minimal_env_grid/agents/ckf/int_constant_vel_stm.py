import numpy as np

def int_constant_vel_stm(t, X_aug, params):
    """
    Integrate the state and state transition matrix (STM) for a constant velocity model.

    Parameters:
    -----------
    t : float
        Current time (not used in this model as it's time-invariant).
    X_aug : ndarray, shape (20,)
        Augmented state vector containing:
        [0] - position (x)
        [1] - position (y)
        [2] - velocity (vx)
        [3] - velocity (vy)
        [4:20] - flattened STM (16 elements for 4x4 STM)
    params : dict
        Dictionary of parameters (not used in this model).

    Returns:
    --------
    dXdt_aug : ndarray, shape (20,)
        Time derivative of the augmented state vector.
    """

    # Unpack the augmented state vector
    x = X_aug[0]      # position x
    y = X_aug[1]      # position y
    vx = X_aug[2]     # velocity vx
    vy = X_aug[3]     # velocity vy
    phi_flat = X_aug[4:20]  # flattened STM

    # Reshape the flattened STM into a 4x4 matrix
    phi = phi_flat.reshape((4, 4))

    # Define the system dynamics matrix for constant velocity model
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    # Compute the time derivative of the state
    dxdt = vx
    dydt = vy
    dvxdt = 0.0
    dvydt = 0.0

    # Compute the time derivative of the STM
    dphi_dt = A @ phi

    # Flatten the STM derivative back into a vector
    dphi_dt_flat = dphi_dt.flatten()

    # Combine derivatives into a single augmented state derivative vector
    dXdt_aug = np.zeros_like(X_aug)
    dXdt_aug[0] = dxdt
    dXdt_aug[1] = dydt
    dXdt_aug[2] = dvxdt
    dXdt_aug[3] = dvydt
    dXdt_aug[4:20] = dphi_dt_flat

    return dXdt_aug