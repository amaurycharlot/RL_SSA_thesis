import numpy as np
from ssa_env.envs.minimal_ssa_env import MinimalSSAEnv

def main():
    env = MinimalSSAEnv(num_targets=2)
    obs, info = env.reset(seed=0)

    assert isinstance(obs, np.ndarray), "obs should be a numpy array (belief vector)"
    assert obs.dtype == np.float32, "obs dtype should be float32"
    assert np.all(np.isfinite(obs)), "obs contains NaN/Inf"

    a = env.action_space.sample()
    obs2, r, term, trunc, info2 = env.step(a)

    assert obs2.shape == obs.shape, "obs shape must be constant"
    assert np.isfinite(r), "reward is NaN/Inf"
    assert isinstance(term, (bool, np.bool_))
    assert isinstance(trunc, (bool, np.bool_))

    print("OK: reset/step API + shapes + finite values")

if __name__ == "__main__":
    main()