from __future__ import annotations

import functools
from copy import copy, deepcopy
from typing import Any, SupportsFloat

import numpy as np
import pygame

import gymnasium as gym
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv

from gymnasium import spaces
from gymnasium.core import RenderFrame

""" Petting zoo version of the trap env. """


class AgentActions:
    available = ['give_and_take', 'eat', 'protect']
    # action:
    # 0: share
    # 1: eat
    # 2: protect
    # the priority is the order of id. so agent cannot "take" other's food if other agent is taking 'eat' or 'protect'.

    share = 0
    eat = 1
    protect = 2


class DoubleFoodShareEnvPZ(ParallelEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 200
    }
    def __init__(
            self,
            render_mode=None,
            cognitive_empathy=False,
            weight_affective_empathy=0.0,
    ):
        self.window_size = 512
        agents_index = [0, 1]
        self.possible_agents = self.possible_agents = [f"agent{i}" for i in agents_index]

        # enable or remove empathy channel
        self.cognitive_empathy = cognitive_empathy
        self.weight_affective_empathy = weight_affective_empathy

        # for tentative pettingzoo compatibility
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        # Key parameters
        self.default_energy_loss = 0.01  # default energy loss
        self.food_intake = 0.1  # intake of energy when food is consumed
        self.reward_scale = 100.  # reward scale in the homeostatic reward

        # environment condition

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.max_episode_length = 1000
        self._step = 0

        self.food_owner = np.random.randint(2)  # who-has-food
        self.agent_info = {
            id_: {
                "energy": 0.0,
                "have_food": self.food_owner == i,
            } for i, id_ in enumerate(self.possible_agents)
        }

        self.prev_energy = np.array([self.agent_info[id_]["energy"] for id_ in self.possible_agents])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_dim = 3 if self.cognitive_empathy else 2
        # "energy": spaces.Box(-1, 1),  1 dim
        # "have_food": spaces.Discrete(1), a binary value that the agent has food
        # "empathic_signal": spaces.Box(-np.inf, np.inf),  # cognitive empathy channel
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,)
        )

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(AgentActions.available))

    def set_agent_info(self,
                       id_: str,
                       energy: float,
                       have_food: bool,
                       ):
        self.agent_info[id_]["energy"] = energy
        self.agent_info[id_]["have_food"] = have_food

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Initial implementation: The very simple empathic signals
        """
        self._step = 1
        self.agents = copy(self.possible_agents)

        self.food_owner = np.random.randint(2)  # 0 or 1

        for i, agent_id in enumerate(self.possible_agents):
            have_food = i == self.food_owner
            self.set_agent_info(
                id_=agent_id,
                energy=np.random.uniform(-0.2, 0.2),
                have_food=have_food,
            )

        self.prev_energy = np.array([self.agent_info[a]["energy"] for a in self.possible_agents])

        observations = self.get_obs()
        infos = self.get_infos()

        return observations, infos

    def generate_new_food(self):
        for i, agent_id in enumerate(self.possible_agents):
            have_food = i == self.food_owner
            self.agent_info[agent_id]["have_food"] = have_food

    def step(
        self, actions  # {agent_id: 0~6}
    ):
        self._step += 1
        self.pass_event = False

        # energy update
        self.prev_energy = np.array([self.agent_info[id_]["energy"] for id_ in self.possible_agents])

        # update trap condition
        is_shared = False
        is_food_eaten = False
        agent0, agent1 = self.possible_agents

        action0 = actions[agent0]
        action1 = actions[agent1]

        self.agent_info[agent0]["energy"] -= self.default_energy_loss
        self.agent_info[agent1]["energy"] -= self.default_energy_loss

        # agent 0 update
        if action0 == AgentActions.eat:  # consume food if the agent "has food"
            if self.agent_info[agent0]["have_food"] is True:
                self.agent_info[agent0]["energy"] += self.food_intake
                is_food_eaten = True
            elif self.agent_info[agent1]["have_food"] is True and action1 == AgentActions.share:
                self.agent_info[agent0]["energy"] += self.food_intake
                is_food_eaten = True

        elif action0 == AgentActions.share:  # share dynamics: transfer food
            if self.agent_info[agent0]["have_food"] is True:
                self.agent_info[agent0]["have_food"] = False
                self.agent_info[agent1]["have_food"] = True
                is_shared = True
                # print("food transfer: 0 --> 1")

        # agent 1 update
        if action1 == AgentActions.eat:  # consume food if the agent "has food"
            if self.agent_info[agent1]["have_food"] is True:
                self.agent_info[agent1]["energy"] += self.food_intake
                is_food_eaten = True
            elif self.agent_info[agent0]["have_food"] is True and action0 == AgentActions.share:
                self.agent_info[agent1]["energy"] += self.food_intake
                is_food_eaten = True

        elif action1 == AgentActions.share:  # share dynamics: transfer food
            if self.agent_info[agent1]["have_food"] is True and is_shared is False:
                self.agent_info[agent0]["have_food"] = True
                self.agent_info[agent1]["have_food"] = False
                # print("food transfer: 1 --> 0")
                is_shared = True

        if is_food_eaten:
            self.generate_new_food()

        # terminate if any one of agents is dead
        done = any([np.abs(self.agent_info[a]["energy"]) > 1 for a in self.possible_agents])
        if self.max_episode_length <= self._step:
            done = True

        dones = {a: done for a in self.possible_agents}
        truncateds = {a: False for a in self.possible_agents}

        rewards = self.get_reward()
        observations = self.get_obs()
        infos = self.get_infos()

        return observations, rewards, dones, truncateds, infos

    def get_obs(self):
        obss = dict()
        agent0, agent1 = self.possible_agents
        if self.cognitive_empathy:
            obss[agent0] = np.concatenate([
                np.array([self.agent_info[agent0]["energy"]]),
                np.array([float(self.agent_info[agent0]["have_food"])]),
                np.array([self.agent_info[agent1]["energy"]])
            ])
        else:
            obss[agent0] = np.concatenate([
                np.array([self.agent_info[agent0]["energy"]]),
                np.array([float(self.agent_info[agent0]["have_food"])]),
            ])

        if self.cognitive_empathy:
            obss[agent1] = np.concatenate([
                np.array([self.agent_info[agent1]["energy"]]),
                np.array([float(self.agent_info[agent1]["have_food"])]),
                np.array([self.agent_info[agent0]["energy"]])
            ])
        else:
            obss[agent1] = np.concatenate([
                np.array([self.agent_info[agent1]["energy"]]),
                np.array([float(self.agent_info[agent1]["have_food"])]),
            ])

        return obss

    def get_infos(self):
        info = deepcopy(self.agent_info)
        info["steps"] = self._step
        return info

    def get_reward(self):
        # TODO: cleanup code!
        # homeostatic reward by Keramati & Gutkin 2011
        rewards = dict()

        energy_now = np.array([self.agent_info[id_]["energy"] for id_ in self.possible_agents])

        # agent 0
        prev_drive = self.prev_energy[0] ** 2
        prev_drive += self.weight_affective_empathy * self.prev_energy[1] ** 2  # empathic term

        drive_now = energy_now[0] ** 2
        drive_now += self.weight_affective_empathy * energy_now[1] ** 2  # empathic reward term

        reward0 = self.reward_scale * (prev_drive - drive_now)
        rewards[self.possible_agents[0]] = reward0

        # agent 1
        prev_drive = self.prev_energy[1] ** 2
        prev_drive += self.weight_affective_empathy * self.prev_energy[0] ** 2  # empathic reward term

        drive_now = energy_now[1] ** 2
        drive_now += self.weight_affective_empathy * energy_now[0] ** 2  # empathic reward term

        reward1 = self.reward_scale * (prev_drive - drive_now)
        rewards[self.possible_agents[1]] = reward1

        return rewards

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_scale = self.window_size / 2.0
        offset = np.array([self.window_size, self.window_size]) / 2.0

        def write(screen, text, color, position, size):
            font = pygame.font.Font(pygame.font.get_default_font(), size)  # Defining a font with font and size
            text_surface = font.render(text, True, color)  # Defining the text color which will be rendered
            screen.blit(text_surface, (position[0], position[1]))  # Rendering the font

        def energy_to_color(energy):
            s = np.array(energy)
            if s > 0:
                s = 0
            s = np.clip(np.abs(s), 0, 1)
            return (255 * (1 - s), 0, 255 * s)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["video.frames_per_second"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    env = DoubleFoodShareEnvPZ(render_mode="human")
    env.reset()

    for i in range(1000):
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.possible_agents}
        # print(actions)
        obs, reward, done, truncate, info = env.step(actions)
        # env.render()
        print("obs:", obs)
        # print("reward:", reward)
        # print(f"done, truncate, info: {done}, {truncate}, {info}")

    env.close()
    print("finish.")