import gymnasium as gym
import numpy as np


class FoodShareWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """

        :param env: environment to learn
        """
        super().__init__(env)
        dim_obs = 2 if env.unwrapped.enable_empathy else 1

        ub = np.ones(dim_obs, dtype=np.float32)
        self.obs_space = gym.spaces.Box(ub * -1, ub)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    @property
    def observation_space(self):
        return self.obs_space

    def observation(self, observation):
        return self.env.encode_obs(observation)
