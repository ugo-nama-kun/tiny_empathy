from pprint import pprint

import numpy as np
import gymnasium as gym
import tiny_empathy
from tiny_empathy.wrappers import FoodShareWrapper

enc = np.random.randn(3)
enc = enc / np.linalg.norm(enc)
dec = enc.copy()

env = gym.make(id="tiny_empathy/GridRooms-v0",
               size=2,
               render_mode="human",
               enable_inference=True,
               encoder_weight=enc,
               decoder_weight=dec,
               set_energy_loss_partner=0.01)
# env = gym.make(id="tiny_empathy/GridRooms-v0", render_mode="human", enable_empathy=False)
# env = gym.make(id="tiny_empathy/FoodShare-v0", render_mode="human", enable_empathy=False)
# env = FoodShareWrapper(env)

env.reset()

for t in range(1000):
    action = env.action_space.sample()
    # action = 0
    obs, rew, done, truncate, info = env.step(action)
    print("action: ")
    print(action)
    pprint("obs, rew, done, truncate, info")
    pprint(obs)
    pprint(rew)
    pprint(done)
    pprint(truncate)
    pprint(info)
    print("---")
    if done:
        break

print("steps: ", t)
print("done")