"""Script to plot how uncertainty is normalized for RL observation (by the catalogue in method build_obs_track)."""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    

    # Define the range of uncertainty values as a logarithmic scale from 1e-2 to 1e2
    uncertainty_values = np.logspace(-2, 2, num=100)
    

    # Normalize the uncertainty values (assuming normalization is done by dividing by the max value)
    normalized_uncertainty = np.log1p(uncertainty_values)  # Using log1p to avoid log(0) issues
    normalized_uncertainty /= np.log1p(500.0)  # Replace self.uncertainty_norm_scale with a fixed value for plotting
    normalized_uncertainty = np.clip(
        normalized_uncertainty,
        0.0,
        1.0
    )

    plt.figure(figsize=(8, 5))
    plt.plot(uncertainty_values, normalized_uncertainty, label="Normalized Uncertainty")
    plt.xlabel("Uncertainty Value")
    plt.ylabel("Normalized Uncertainty")
    plt.title("Normalization of Uncertainty for RL Observation")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()