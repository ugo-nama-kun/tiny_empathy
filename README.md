# tiny_empathy
A tiny environment for the emergence of empathic behaviors

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