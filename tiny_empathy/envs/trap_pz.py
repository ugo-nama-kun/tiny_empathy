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
    available = ['left', 'right', 'up', 'down', 'take', 'consume', 'pass']
    # action:
    # 0: move left (movable agent)
    # 1: move right (movable agent)
    # 2: move up (movable agent)
    # 3: move down (movable agent)
    # 4: take (get food or take food from other agent)
    # 5: consume food if the agent "has food"
    # 6: pass food if the agent "has food" and near to other agent

    left = 0
    right = 1
    up = 2
    down = 3
    take = 4
    consume = 5
    pass_ = 6


class TrapEnvPZ(ParallelEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 200
    }
    def __init__(
            self,
            render_mode=None,
            cognitive_empathy=False,
            weight_affective_empathy=0.0,
            p_trap=0.002
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
        self.default_energy_loss = 0.001  # default energy loss
        self.food_intake = 0.3  # intake of energy when food is consumed
        self.reward_scale = 100.  # reward scale in the homeostatic reward

        # environment condition
        self.p_trap = p_trap
        self.movable_thresh = -0.7
        self.field_area = {"x": (-1, 1), "y": (-1, 1)}
        self.agent_size = 0.2
        self.food_size = 0.1
        self.move_step = 0.15
        self.noise_scale = 0.0

        assert render_mode is None or render_mode in self.metadata["render.modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.max_episode_length = 5000
        self._step = 0
        self.pass_event = False

        self.food_pos = np.zeros(2)
        self.agent_info = {
            id_: {
                "energy": 0.0,
                "position": np.zeros(2),
                "have_food": False,
                "is_movable": True,
            } for id_ in self.possible_agents
        }

        self.prev_energy = np.array([self.agent_info[id_]["energy"] for id_ in self.possible_agents])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_dim = 10 if self.cognitive_empathy else 9
        # "energy": spaces.Box(-1, 1),  1 dim
        # "position": spaces.Box(-1, 1, shape=(2,)),  2 dim (vector from agent to target)
        # "food_pos": spaces.Box(-1, 1, shape=(2,)),  2 dim (vector from agent to target)
        # "have_food": spaces.Discrete(2),  2 dim
        # "is_movable": spaces.Discrete(2),  2 sim
        # "empathic_signal": spaces.Box(-np.inf, np.inf),  # empathy channel
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
                       id_: str, energy: float,
                       have_food: bool = False,
                       position: np.ndarray = np.zeros(2),
                       is_movable: bool = True
                       ):
        self.agent_info[id_]["energy"] = energy
        self.agent_info[id_]["position"] = position
        self.agent_info[id_]["have_food"] = have_food
        self.agent_info[id_]["is_movable"] = is_movable

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
        self.pass_event = False

        self.food_pos = np.random.uniform(low=-0.1, high=0.1, size=2)

        for agent_id in self.possible_agents:
            self.set_agent_info(
                id_=agent_id,
                energy=np.random.uniform(-0.2, 0.2),
                have_food=False,
                position=np.random.uniform(low=-1, high=1, size=2),
                is_movable=True,
            )

        self.prev_energy = np.array([self.agent_info[a]["energy"] for a in self.possible_agents])

        observations = self.get_obs()
        infos = self.get_infos()

        return observations, infos

    def step(
        self, actions  # {agent_id: 0~6}
    ):
        self._step += 1
        self.pass_event = False

        # energy update
        self.prev_energy = np.array([self.agent_info[id_]["energy"] for id_ in self.possible_agents])

        # update trap condition
        for agent_id in self.possible_agents:
            action = actions[agent_id]

            # Agent get accident if both agent is movable
            if np.any([self.movable_thresh < self.agent_info[id_]["energy"] for id_ in self.possible_agents]):
                if np.random.random() < self.p_trap:
                    self.agent_info[agent_id]["energy"] = self.movable_thresh

            self.agent_info[agent_id]["energy"] -= self.default_energy_loss

            prev_pos = self.agent_info[agent_id]["position"]
            if action in {AgentActions.left, AgentActions.right, AgentActions.up, AgentActions.down}:
                if self.agent_info[agent_id]["is_movable"] is True:
                    d_xy = np.zeros(2)
                    if action == AgentActions.left:  # move left (movable agent)
                        d_xy = -np.array([self.move_step, 0])
                    elif action == AgentActions.right:  # move right (movable agent)
                        d_xy = np.array([self.move_step, 0])
                    elif action == AgentActions.up:  # move up (movable agent)
                        d_xy = np.array([0, self.move_step])
                    elif action == AgentActions.down:  # move down (movable agent)
                        d_xy = -np.array([0, self.move_step])

                    new_pos = self.agent_info[agent_id]["position"] + d_xy
                    new_pos += np.random.normal(loc=0, scale=self.noise_scale, size=2)

                    if self.field_area["x"][0] < new_pos[0] < self.field_area["x"][1]:
                        pass
                    else:
                        new_pos[0] = prev_pos[0]
                    if self.field_area["y"][0] < new_pos[1] < self.field_area["y"][1]:
                        pass
                    else:
                        new_pos[1] = prev_pos[1]
                    self.agent_info[agent_id]["position"] = new_pos

            elif action == AgentActions.take:  # take (get food or take food from other agent)
                # TODO: implement food take from other agent
                if np.linalg.norm(self.agent_info[agent_id]["position"] - self.food_pos) < self.agent_size + self.food_size:
                    self.agent_info[agent_id]["have_food"] = True
                    self.food_pos = np.random.uniform(-1, 1, (2,))

            elif action == AgentActions.consume:  # consume food if the agent "has food"
                if self.agent_info[agent_id]["have_food"] is True:
                    self.agent_info[agent_id]["energy"] += self.food_intake
                    self.agent_info[agent_id]["have_food"] = False

            elif action == AgentActions.pass_:  # pass food if the agent "has food" and at rightmost position
                if self.agent_info[agent_id]["have_food"] is True:
                    agent_id0 = self.possible_agents[0]
                    agent_id1 = self.possible_agents[1]
                    pos0 = self.agent_info[agent_id0]["position"]
                    pos1 = self.agent_info[agent_id1]["position"]
                    agent_distance = np.linalg.norm(pos0 - pos1)
                    if agent_distance < 2 * self.agent_size:
                        # food pass only if other agent doesn't have food
                        if agent_id == agent_id0 and self.agent_info[agent_id1]["have_food"] is False:
                            self.agent_info[agent_id0]["have_food"] = False
                            self.agent_info[agent_id1]["have_food"] = True
                            self.pass_event = True
                        if agent_id == agent_id1 and self.agent_info[agent_id0]["have_food"] is False:
                            self.agent_info[agent_id0]["have_food"] = True
                            self.agent_info[agent_id1]["have_food"] = False
                            self.pass_event = True

            else:
                raise ValueError("Invalid action. action", action)

            # update movable condition
            if self.agent_info[agent_id]["energy"] < self.movable_thresh:
                self.agent_info[agent_id]["is_movable"] = False
            else:
                self.agent_info[agent_id]["is_movable"] = True

        # terminate if any one of agents is dead
        done = any([self.agent_info[agent_id]["energy"] < -1 for agent_id in self.possible_agents])

        if self.max_episode_length <= self._step:
            done = True

        dones = {agent_id: done for agent_id in self.possible_agents}
        truncateds = dones.copy()

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
                self.agent_info[agent0]["position"],
                self.agent_info[agent1]["position"] - self.agent_info[agent0]["position"],
                self.food_pos - self.agent_info[agent0]["position"],
                np.float32(np.arange(2) == int(self.agent_info[agent0]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
                np.array([self.agent_info[agent1]["energy"]])
            ])
        else:
            obss[agent0] = np.concatenate([
                np.array([self.agent_info[agent0]["energy"]]),
                self.agent_info[agent0]["position"],
                self.agent_info[agent1]["position"] - self.agent_info[agent0]["position"],
                self.food_pos - self.agent_info[agent0]["position"],
                np.float32(np.arange(2) == int(self.agent_info[agent0]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
            ])

        if self.cognitive_empathy:
            obss[agent1] = np.concatenate([
                np.array([self.agent_info[agent1]["energy"]]),
                self.agent_info[agent1]["position"],
                self.agent_info[agent0]["position"] - self.agent_info[agent1]["position"],
                self.food_pos - self.agent_info[agent1]["position"],
                np.float32(np.arange(2) == int(self.agent_info[agent1]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
                np.array([self.agent_info[agent0]["energy"]])
            ])
        else:
            obss[agent1] = np.concatenate([
                np.array([self.agent_info[agent1]["energy"]]),
                self.agent_info[agent1]["position"],
                self.agent_info[agent0]["position"] - self.agent_info[agent1]["position"],
                self.food_pos - self.agent_info[agent1]["position"],
                np.float32(np.arange(2) == int(self.agent_info[agent1]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
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

        # Now we draw the agent
        for agent_id in self.possible_agents:
            if self.agent_info[agent_id]["is_movable"] is False:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    self.agent_info[agent_id]["position"] * pix_scale + offset,
                    pix_scale * self.agent_size * 1.3,
                )

            pygame.draw.circle(
                canvas,
                energy_to_color(self.agent_info[agent_id]["energy"]),
                self.agent_info[agent_id]["position"] * pix_scale + offset,
                pix_scale * self.agent_size,
            )
            if self.agent_info[agent_id]["have_food"] is True:
                pygame.draw.circle(
                    canvas,
                    (100, 255, 100),
                    self.agent_info[agent_id]["position"] * pix_scale + offset,
                    pix_scale * self.food_size,
                )
            write(canvas,
                  f"agent: {agent_id}",
                  color=(0, 0, 0),
                  position=(self.agent_info[agent_id]["position"]) * pix_scale + offset,
                  size=12
                  )

        if self.pass_event:
            write(canvas,
                  f"PASS",
                  color=(0, 1, 1),
                  position=offset + np.array([self.window_size / 5, self.window_size / 3]),
                  size=50
                  )

        # Draw food
        pygame.draw.circle(
            canvas,
            color=(50, 255, 50),
            center=pix_scale * self.food_pos + offset,
            radius=pix_scale * self.food_size
        )

        write(canvas,
              "Food",
              color=(0, 0, 0),
              position=pix_scale * (self.food_pos + np.array([0.05, -0.05])) + offset,
              size=12)
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
    env = TrapEnvPZ(render_mode="human")
    env.reset()

    for i in range(1000):
        actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.possible_agents}
        print(actions)
        obs, reward, done, truncate, info = env.step(actions)
        env.render()
        print("obs:", obs)
        # print("reward:", reward)
        # print(f"done, truncate, info: {done}, {truncate}, {info}")

    env.close()
    print("finish.")