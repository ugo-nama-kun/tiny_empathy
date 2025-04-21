import gymnasium as gym
import numpy as np
from tiny_empathy.envs.grid_rooms import GridRoomsEnv


class GridRoomsWrapper(gym.ObservationWrapper):
    def __init__(self, env: GridRoomsEnv):
        """

        :param env: environment to learn
        """
        super().__init__(env)
        dim_obs = 3 + env.size
        if env.enable_empathy and env.enable_inference:
            dim_obs += env.size_emotional_feature
        elif env.enable_empathy:
            dim_obs += 1

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
