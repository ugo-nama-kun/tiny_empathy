from pprint import pprint

import gymnasium as gym
import tiny_empathy
from tiny_empathy.wrappers import FoodShareWrapper

env = gym.make(id="tiny_empathy/GridRooms-v0", render_mode="human", enable_empathy=False)
# env = gym.make(id="tiny_empathy/FoodShare-v0", render_mode="human", enable_empathy=False)
env = FoodShareWrapper(env)

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