# tiny_empathy
A tiny environment package for the emergence of empathic behaviors from the homeostatic principle

#### Food sharing
![full_empathy](https://github.com/ugo-nama-kun/tiny_empathy/assets/1684732/8a67dcb6-a71d-44df-a93a-54bf6bb1599d)

#### Grid world
![full_empathy_grid](https://github.com/ugo-nama-kun/tiny_empathy/assets/1684732/eb3c178c-1758-4b2b-ab5c-7d44787a391b)

#### 2D vulnerable agents
![Jun-13-2024 07-30-39](https://github.com/ugo-nama-kun/tiny_empathy/assets/1684732/9f798e78-95d9-4e18-8de0-e60ed137ad7f)


### install
```commandline
cd path_to_repository
pip install .
```

### test run
```commandline
python testrun.py
python testrun_trap.py
```

### how to use
```python
import gymnasium as gym
import tiny_empathy
from tiny_empathy.wrappers import FoodShareWrapper, GridRoomsWrapper

env = gym.make(id="tiny_empathy/FoodShare-v0", render_mode="human", enable_empathy=False)
env = FoodShareWrapper(env)

# Grid room env
env = gym.make(id="tiny_empathy/GridRooms-v0", render_mode="human", enable_empathy=False)
env = GridRoomsWrapper(env)

# Trap environment
env = gym.make(id="tiny_empathy/Trap-v0", render_mode="human", enable_empathy=False, p_trap=0.001)

# see testrun.py or testrun_trap.py for the detailed usage :)
```

## customizing environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

#### for wrapper
https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/
