from pprint import pprint

import gymnasium as gym
import tiny_empathy
from tiny_empathy.wrappers import FoodShareWrapper

env = gym.make(id="tiny_empathy/Trap-v0", render_mode="human", enable_empathy=False)

env.reset()

for t in range(1000):
    actions = {i: env.action_space.sample() for i in env.possible_agents}
    # action = 0
    obss, rews, dones, truncates, infos = env.step(actions)
    print("actions: ")
    print(actions)
    pprint("obs, rew, done, truncate, info")
    pprint(obss)
    pprint(rews)
    pprint(dones)
    pprint(truncates)
    pprint(infos)
    print("---")
    if any(dones.values()) is True:
        break

print("steps: ", t)
print("done")