from gymnasium.envs.registration import register

register(
    id="tiny_empathy/GridRooms-v0",
    entry_point="tiny_empathy.envs:GridRoomsEnv",
    max_episode_steps=3000,
)

register(
    id="tiny_empathy/FoodShare-v0",
    entry_point="tiny_empathy.envs:FoodShareEnv",
    max_episode_steps=1000,
)

register(
    id="tiny_empathy/Trap-v0",
    entry_point="tiny_empathy.envs:TrapEnv",
    max_episode_steps=5000,
)