from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import pygame

import gymnasium as gym

from gymnasium import spaces
from gymnasium.core import RenderFrame


class TrapEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, render_mode=None, enable_empathy=False, weight_empathy=0.0):
        self.window_size = 512
        self.max_episode_length = 5000

        # enable or remove empathy channel
        self.enable_empathy = enable_empathy
        self.weight_empathy = weight_empathy

        # for tentative pettingzoo compatibility
        self.possible_agents = [0, 1]
        self.action_spaces = dict()
        self.observation_spaces = dict()
        for agent_id in self.possible_agents:
            self.action_spaces[agent_id] = spaces.Discrete(7)
            # action:
            # 0: move left (movable agent)
            # 1: move right (movable agent)
            # 2: move up (movable agent)
            # 3: move down (movable agent)
            # 4: get (get food)
            # 5: consume food if the agent "has food"
            # 6: pass food if the agent "has food" and near to other agent

            obs_dim = 10 if self.enable_empathy else 9
            # "energy": spaces.Box(-1, 1),  1 dim
            # "position": spaces.Box(-1, 1, shape=(2,)),  2 dim (vector from agent to target)
            # "food_pos": spaces.Box(-1, 1, shape=(2,)),  2 dim (vector from agent to target)
            # "have_food": spaces.Discrete(2),  2 dim
            # "is_movable": spaces.Discrete(2),  2 sim
            # "empathic_signal": spaces.Box(-np.inf, np.inf),  # empathy channel
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim, )
            )

        self.observation_space = self.observation_spaces[0]
        self.action_space = self.action_spaces[0]

        # Key parameters
        self.default_energy_loss = 0.001  # default energy loss
        self.food_intake = 0.3  # intake of energy when food is consumed
        self.reward_scale = 100.  # reward scale in the homeostatic reward

        # environment condition
        self.p_trap = 0.0005
        self.movable_thresh = -0.7
        self.field_area = {"x": (-1, 1), "y": (-1, 1)}
        self.agent_size = 0.2
        self.food_size = 0.1
        self.move_step = 0.15
        self.noise_scale = 0.0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.steps = 0
        self.pass_event = False

        self.food_pos = np.zeros(2)
        self.agent_info = {
            i: {
                "energy": 0.0,
                "position": np.zeros(2),
                "have_food": False,
                "is_movable": True,
            } for i in self.possible_agents
        }

        self.prev_energy = np.array([v["energy"] for k, v in self.agent_info.items()])

    def set_agent_info(self,
                       id_: int, energy: float,
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
        self.steps = 0
        self.pass_event = False

        self.food_pos = self.np_random.uniform(low=-0.1, high=0.1, size=2)

        for agent_id in self.possible_agents:
            self.set_agent_info(
                id_=agent_id,
                energy=self.np_random.uniform(-0.2, 0.2),
                have_food=False,
                position=self.np_random.uniform(low=-0.1, high=0.1, size=2),
                is_movable=True,
            )

        self.prev_energy = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])

        observation = self.get_obs()
        info = self.get_infos()

        return observation, info

    def step(
        self, actions  # {agent_id: 0~6}
    ):
        self.steps += 1
        self.pass_event = False

        # energy update
        self.prev_energy = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])

        # update trap condition
        for agent_id in self.possible_agents:
            action = actions[agent_id]

            # Agent get accident if both agent is movable
            if np.any([self.movable_thresh < self.agent_info[id_]["energy"] for id_ in self.possible_agents]):
                if self.np_random.random() < self.p_trap:
                    self.agent_info[agent_id]["energy"] = self.movable_thresh

            self.agent_info[agent_id]["energy"] -= self.default_energy_loss

            prev_pos = self.agent_info[agent_id]["position"]
            if action in {0, 1, 2, 3}:
                if self.agent_info[agent_id]["is_movable"] is True:
                    d_xy = np.zeros(2)
                    if action == 0:  # move left (movable agent)
                        d_xy = -np.array([self.move_step, 0])
                    elif action == 1:  # move right (movable agent)
                        d_xy = np.array([self.move_step, 0])
                    elif action == 2:  # move up (movable agent)
                        d_xy = np.array([0, self.move_step])
                    elif action == 3:  # move down (movable agent)
                        d_xy = -np.array([0, self.move_step])

                    new_pos = self.agent_info[agent_id]["position"] + d_xy
                    new_pos += self.np_random.normal(loc=0, scale=self.noise_scale, size=2)

                    if self.field_area["x"][0] < new_pos[0] < self.field_area["x"][1]:
                        pass
                    else:
                        new_pos[0] = prev_pos[0]
                    if self.field_area["y"][0] < new_pos[1] < self.field_area["y"][1]:
                        pass
                    else:
                        new_pos[1] = prev_pos[1]
                    self.agent_info[agent_id]["position"] = new_pos

            elif action == 4:  # get (get food)
                if np.linalg.norm(self.agent_info[agent_id]["position"] - self.food_pos) < self.agent_size + self.food_size:
                    self.agent_info[agent_id]["have_food"] = True
                    self.food_pos = self.np_random.uniform(-1, 1, (2,))

            elif action == 5:  # consume food if the agent "has food"
                if self.agent_info[agent_id]["have_food"] is True:
                    self.agent_info[agent_id]["energy"] += self.food_intake
                    self.agent_info[agent_id]["have_food"] = False

            elif action == 6:  # pass food if the agent "has food" and at rightmost position
                if self.agent_info[agent_id]["have_food"] is True:
                    agent_distance = np.linalg.norm(self.agent_info[0]["position"] - self.agent_info[1]["position"])
                    if agent_distance < 2 * self.agent_size:
                        # food pass only if other agent doesn't have food
                        if agent_id == 0 and self.agent_info[1]["have_food"] is False:
                            self.agent_info[0]["have_food"] = False
                            self.agent_info[1]["have_food"] = True
                            self.pass_event = True
                        if agent_id == 1 and self.agent_info[0]["have_food"] is False:
                            self.agent_info[0]["have_food"] = True
                            self.agent_info[1]["have_food"] = False
                            self.pass_event = True

            else:
                raise ValueError("Invalid action. action", action)

            # update movable condition
            if self.agent_info[agent_id]["energy"] < self.movable_thresh:
                self.agent_info[agent_id]["is_movable"] = False
            else:
                self.agent_info[agent_id]["is_movable"] = True

        # terminate if any one of agents is dead
        done = self.agent_info[0]["energy"] < -1 or self.agent_info[1]["energy"] < -1

        if self.max_episode_length <= self.steps:
            done = True

        rewards = self.get_reward()
        observations = self.get_obs()
        infos = self.get_infos()

        if self.render_mode == "human":
            self._render_frame()

        return (
            observations,
            rewards,
            {i: done for i in self.possible_agents},
            {i: False for i in self.possible_agents},
            infos
        )

    def get_obs(self):
        obs = dict()
        if self.enable_empathy:
            obs[0] = np.concatenate([
                np.array([self.agent_info[0]["energy"]]),
                self.agent_info[0]["position"],
                self.agent_info[1]["position"] - self.agent_info[0]["position"],
                self.food_pos - self.agent_info[0]["position"],
                np.float32(np.arange(2) == int(self.agent_info[0]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
                np.array([self.agent_info[1]["energy"]])
            ])
        else:
            obs[0] = np.concatenate([
                np.array([self.agent_info[0]["energy"]]),
                self.agent_info[0]["position"],
                self.agent_info[1]["position"] - self.agent_info[0]["position"],
                self.food_pos - self.agent_info[0]["position"],
                np.float32(np.arange(2) == int(self.agent_info[0]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
            ])

        if self.enable_empathy:
            obs[1] = np.concatenate([
                np.array([self.agent_info[1]["energy"]]),
                self.agent_info[1]["position"],
                self.agent_info[0]["position"] - self.agent_info[1]["position"],
                self.food_pos - self.agent_info[1]["position"],
                np.float32(np.arange(2) == int(self.agent_info[1]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
                np.array([self.agent_info[0]["energy"]])
            ])
        else:
            obs[1] = np.concatenate([
                np.array([self.agent_info[1]["energy"]]),
                self.agent_info[1]["position"],
                self.agent_info[0]["position"] - self.agent_info[1]["position"],
                self.food_pos - self.agent_info[1]["position"],
                np.float32(np.arange(2) == int(self.agent_info[1]["have_food"])),
                # np.float32(np.arange(2) == int(self.agent_info[agent_id]["is_movable"])),
            ])

        return obs

    def get_infos(self):
        info = self.agent_info.copy()
        info["steps"] = self.steps
        return info

    def get_reward(self):
        # TODO: cleanup code!
        # homeostatic reward by Keramati & Gutkin 2011
        rewards = dict()

        energy_now = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])

        # agent 0
        prev_drive = self.prev_energy[0] ** 2
        prev_drive += self.weight_empathy * self.prev_energy[1] ** 2  # empathic reward term

        drive_now = energy_now[0] ** 2
        drive_now += self.weight_empathy * energy_now[1] ** 2  # empathic reward term

        reward0 = self.reward_scale * (prev_drive - drive_now)
        rewards[0] = reward0

        # agent 1
        prev_drive = self.prev_energy[1] ** 2
        prev_drive += self.weight_empathy * self.prev_energy[0] ** 2  # empathic reward term

        drive_now = energy_now[1] ** 2
        drive_now += self.weight_empathy * energy_now[0] ** 2  # empathic reward term

        reward1 = self.reward_scale * (prev_drive - drive_now)
        rewards[1] = reward1

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
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()