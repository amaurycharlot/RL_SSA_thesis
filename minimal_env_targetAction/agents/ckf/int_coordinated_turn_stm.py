import numpy as np

def int_coordinated_turn_stm(t, X_aug, params):
    """
    Integrate the state and state transition matrix (STM) for a coordinated turn model.

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
        Dictionary of parameters (we use the 'omega' parameter).

    Returns:
    --------
    dXdt_aug : ndarray, shape (20,)
        Time derivative of the augmented state vector.
    """
    # Unpack parameters
    omega = params.get("omega", 0.05)  # default turn rate if not provided

    # Unpack the augmented state vector
    x = X_aug[0]      # position x
    y = X_aug[1]      # position y
    vx = X_aug[2]     # velocity vx
    vy = X_aug[3]     # velocity vy
    phi_flat = X_aug[4:20]  # flattened STM

    # Reshape the flattened STM into a 4x4 matrix
    phi = phi_flat.reshape((4, 4))

    # Define the system dynamics matrix for coordinated turn model (Jacobian A(x))
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, -omega],
                  [0, 0, omega, 0]])

    # Compute the time derivative of the state (continuous-time CT model)
    dxdt = vx
    dydt = vy
    dvxdt = -omega * vy
    dvydt =  omega * vx

    # Compute the time derivative of the STM
    dphi_dt = A @ phi

    # Combine derivatives into a single augmented state derivative vector
    dXdt_aug = np.zeros_like(X_aug)
    dXdt_aug[0] = dxdt
    dXdt_aug[1] = dydt
    dXdt_aug[2] = dvxdt
    dXdt_aug[3] = dvydt
    dXdt_aug[4:20] = dphi_dt.flatten()

    return dXdt_aug