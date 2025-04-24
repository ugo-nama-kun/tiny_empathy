import gymnasium as gym
import numpy as np


class FoodShareDecoderLearningWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """

        :param env: environment to learn
        """
        super().__init__(env)
        dim_obs = 1
        if env.decoding_mode == "affect":
            dim_obs += env.dim_emotional_feature
        elif env.decoding_mode == "full":
            dim_obs += 1
        else:
            raise ValueError(f"Invalid decoding mode. {env.decoding_mode}")

        ub = np.ones(dim_obs, dtype=np.float32)
        self.obs_space = gym.spaces.Box(ub * -1, ub)

    def step(
        self, action, emotional_decoder=None
    ):
        return self.env.step(action, emotional_decoder)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    @property
    def observation_space(self):
        return self.obs_space

    def observation(self, observation):
        return self.env.encode_obs(observation)
