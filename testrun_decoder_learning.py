import torch

import gymnasium as gym
import tiny_empathy
from tiny_empathy.envs import FoodShareDecoderLearningEnv, GridRoomsDecoderLearningEnv
from tiny_empathy.wrappers import GridRoomsDecoderLearningWrapper

from tiny_empathy.wrappers.foodshare_decoder_learning import FoodShareDecoderLearningWrapper


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 3)

    def forward(self, x):
        return self.model(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.model(x)


enc = Encoder()
dec = Decoder()

# env = FoodShareDecoderLearningEnv(
#     render_mode="human",
#     dim_emotional_feature=3,
#     decoding_mode="affect",
#     emotional_encoder=enc,
# )
# env = FoodShareDecoderLearningWrapper(env)

env = GridRoomsDecoderLearningEnv(
    render_mode="human",
    dim_emotional_feature=3,
    decoding_mode="full",
    emotional_encoder=enc,
)
env = GridRoomsDecoderLearningWrapper(env)

env.reset(emotional_decoder=dec)
for i in range(1000):
    actions = env.action_space.sample()
    print(actions)
    obs, reward, done, truncate, info = env.step(actions, emotional_decoder=dec)
    env.render()
    print("obs:", obs)
    print(info)
    # print("reward:", reward)
    # print(f"done, truncate, info: {done}, {truncate}, {info}")

env.close()
print("finish.")
