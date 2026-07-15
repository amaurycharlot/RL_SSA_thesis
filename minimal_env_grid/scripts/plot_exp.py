"""Script to plot the exponential decay function used in the R4 search reward."""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    steps = np.arange(0, 20, 1)
    reward = np.exp(-steps / 10)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, reward, label=r"$e^{-\frac{steps}{10}}$")
    plt.xlabel("Elapsed time steps")
    plt.ylabel("Reward multiplier")
    plt.title("Exponential decay of search reward over time (R4)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()