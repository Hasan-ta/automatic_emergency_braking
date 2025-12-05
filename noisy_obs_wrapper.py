import numpy as np
import gymnasium as gym


class NoisyObsWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise to observations.

    Designed for obs = [gap, v_ego, a_ego, v_npc].

    Args:
        env: Gymnasium env with Box observation_space.
        sigma: scalar or array-like of shape (obs_dim,)
               Standard deviation of Gaussian noise per dimension.
        clip: whether to clip noisy obs to observation_space bounds.
        seed: optional seed for internal RNG.
    """

    def __init__(self, env, sigma=0.5, clip=True, seed=123):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Box), \
            "NoisyObsWrapper requires a Box observation_space."

        self.clip = bool(clip)
        self.rng = np.random.default_rng(seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        sigma_arr = np.asarray(sigma, dtype=np.float64)

        if sigma_arr.ndim == 0:
            sigma_arr = np.full((obs_dim,), float(sigma_arr), dtype=np.float64)
        else:
            sigma_arr = sigma_arr.reshape((obs_dim,)).astype(np.float64)

        self.sigma = sigma_arr

        # Preserve the same bounds / dtype
        low = env.observation_space.low.astype(np.float64)
        high = env.observation_space.high.astype(np.float64)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)

        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=obs.shape).astype(np.float64)
        noisy = obs + noise

        if self.clip:
            low = self.observation_space.low
            high = self.observation_space.high
            noisy = np.minimum(np.maximum(noisy, low), high)

        return noisy.astype(np.float64)
