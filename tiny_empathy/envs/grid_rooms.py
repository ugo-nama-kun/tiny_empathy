from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import pygame

import gymnasium as gym

from gymnasium import spaces
from gymnasium.core import RenderFrame


class GridRoomsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 render_mode=None,
                 size=5,
                 enable_empathy=False,
                 weight_empathy=0.0,
                 enable_inference=False,
                 encoder_weight=None,
                 decoder_weight=None,
                 set_energy_loss_partner=None,
                 ):
        self.window_size = 512

        # enable or remove empathy channel
        self.enable_empathy = enable_empathy
        self.weight_empathy = weight_empathy

        # emotional feature experiment settings
        self.enable_inference = enable_inference
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight
        self.size_emotional_feature = encoder_weight.size
        # trapped agent energy --(encoder)--> emotional feature --(decoder)--> inferred energy (inferred by possessor)

        # Key parameters
        self.size = size  # size of the 1D grid world
        self.default_energy_loss_posessor = 0.003  # default energy loss
        self.default_energy_loss_partner = 0.003 if set_energy_loss_partner is None else set_energy_loss_partner # default energy loss
        self.food_intake = 0.1  # intake of energy when food is consumed
        self.reward_scale = 100.  # reward scale in the homeostatic reward

        obs_dict = {
                    "energy": spaces.Box(-1, 1),
                    "have_food": spaces.Discrete(2),
                    "position": spaces.Discrete(self.size),
        }

        if enable_inference and enable_empathy:
            obs_dict = spaces.Dict(
                {
                    "energy": spaces.Box(-1, 1),
                    "have_food": spaces.Discrete(2),
                    "position": spaces.Discrete(self.size),
                    "emotional_feature": spaces.Box(-np.inf, np.inf, shape=(self.encoder_weight.size,)),
                },
            )
            self.observation_space = spaces.Dict(obs_dict)
        elif enable_empathy:
            # add empathic signal in the obs dict
            obs_dict = spaces.Dict(
                {
                    "energy": spaces.Box(-1, 1),
                    "have_food": spaces.Discrete(2),
                    "position": spaces.Discrete(self.size),
                    "empathic_signal": spaces.Box(-np.inf, np.inf),  # empathy channel
                },
            )
            self.observation_space = spaces.Dict(obs_dict)
        else:
            self.observation_space = spaces.Dict(obs_dict)

        self.action_space = spaces.Discrete(5)
        # action:
        # 0: move left (movable agent)
        # 1: move right (movable agent)
        # 2: get (get food)
        # 3: consume food if the agent "has food"
        # 4: pass food if the agent "has food" and at rightmost position

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # there are (self.size + 1) positions. Agent_s stacks at self.size
        self.movable_area = set(range(self.size))
        self.stack_area = {self.size}
        self.food_site = 0

        self.agent_info = {
            0: {  # possessor
                "energy": 0.0,
                "have_food": False,
                "position": 0,
            },
            1: { # partner
                "energy": 0.0,
            }
        }

        self.prev_energy = np.array([v["energy"] for k, v in self.agent_info.items()])

    def set_agent_info(self, id_: int, energy: float, have_food: bool = False, position: int = 0):
        self.agent_info[id_]["energy"] = energy
        if id_ == 0:
            self.agent_info[id_]["have_food"] = have_food
            self.agent_info[id_]["position"] = position

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Initial implementation: The very simple empathic signals
        """

        self.set_agent_info(
            id_=0,
            energy=self.np_random.uniform(-0.3, 0.3),
            have_food=False,
            position=self.np_random.choice(list(self.movable_area)),
        )

        self.set_agent_info(
            id_=1,
            energy=self.np_random.uniform(-0.3, 0.3),
        )

        self.prev_energy = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])

        observation = self.get_obs()
        info = self.get_info()

        return observation, info

    def step(
        self, action  # 0-4
    ):

        # energy update
        self.prev_energy = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])
        self.agent_info[0]["energy"] -= self.default_energy_loss_posessor
        self.agent_info[1]["energy"] -= self.default_energy_loss_partner

        # possessor move
        prev_pos = self.agent_info[0]["position"]
        if action == 0:  # 0: move left (movable agent)
            new_pos = self.agent_info[0]["position"] - 1
            if new_pos not in self.movable_area:
                new_pos = prev_pos
            self.agent_info[0]["position"] = new_pos
        elif action == 1:  # 1: move right (movable agent)
            new_pos = self.agent_info[0]["position"] + 1
            if new_pos not in self.movable_area:
                new_pos = prev_pos
            self.agent_info[0]["position"] = new_pos
        elif action == 2:  # 2: get (get food)
            if prev_pos == self.food_site:
                self.agent_info[0]["have_food"] = True
        elif action == 3:  # 3: consume food if the agent "has food"
            if self.agent_info[0]["have_food"] is True:
                self.agent_info[0]["energy"] += self.food_intake
                self.agent_info[0]["have_food"] = False
                # print("EAT FOOD! POSSESSOR $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        elif action == 4:  # 4: pass food if the agent "has food" and at rightmost position
            if (self.agent_info[0]["position"] + 1) in self.stack_area and self.agent_info[0]["have_food"] is True:
                self.agent_info[1]["energy"] += self.food_intake
                self.agent_info[0]["have_food"] = False
                # print("EAT FOOD! PARTNER ##################################")
        else:
            raise ValueError("Invalid action. action0", action)

        # terminate if any one of agents is dead
        done = self.agent_info[0]["energy"] < -1 or self.agent_info[1]["energy"] < -1

        rewards = self.get_reward()
        observation_dict = self.get_obs()
        info = self.get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation_dict, rewards, done, False, info

    def get_obs(self):
        obs = {
            "energy": [self.agent_info[0]["energy"]],
            "have_food": np.float32(np.arange(2) == int(self.agent_info[0]["have_food"])),
            "position": np.float32(np.arange(self.size) == self.agent_info[0]["position"]),
        }

        if self.enable_inference and self.enable_empathy:  # inference
            obs.update({"emotional_feature": self.encoder_weight * self.agent_info[1]["energy"]})
        elif self.enable_empathy:  # direct pass
            obs.update({"empathic_signal": [self.agent_info[1]["energy"]]})

        return obs

    def encode_obs(self, obs_dict):
        return np.concatenate([v for k, v in obs_dict.items()])

    def get_info(self):
        return self.agent_info.copy()

    def get_reward(self):
        # homeostatic reward by Keramati & Gutkin 2011
        prev_drive = self.prev_energy[0] ** 2
        if self.enable_inference:
            inferred_energy = np.dot(self.decoder_weight.transpose(), self.encoder_weight * self.prev_energy[1])
            prev_drive += self.weight_empathy * inferred_energy ** 2  # empathic reward term
        else:
            prev_drive += self.weight_empathy * self.prev_energy[1] ** 2  # empathic reward term

        energy_now = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])
        drive_now = energy_now[0] ** 2
        if self.enable_inference:
            inferred_energy = np.dot(self.decoder_weight.transpose(), self.encoder_weight * energy_now[1])
            drive_now += self.weight_empathy * inferred_energy ** 2  # empathic reward term
        else:
            drive_now += self.weight_empathy * energy_now[1] ** 2  # empathic reward term

        reward = self.reward_scale * (prev_drive - drive_now)
        return reward

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size / (self.size + 2))
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / (self.size + 2)
        )  # The size of a single grid square in pixels

        def write(screen, text, color, position, size):
            font = pygame.font.Font(pygame.font.get_default_font(), size)  # Defining a font with font and size
            text_surface = font.render(text, True, color)  # Defining the text color which will be rendered
            screen.blit(text_surface, (position[0], position[1]))  # Rendering the font

        # Food Site
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * np.array([0, 0]),
                (pix_square_size, pix_square_size),
            ),
        )
        write(canvas, "Food", color=(0, 0, 0), position=pix_square_size * np.array([0.5, 0.5]), size=12)

        # Stacking Area
        pygame.draw.rect(
            canvas,
            (100, 100, 100),
            pygame.Rect(
                pix_square_size * np.array([self.size + 1, 0]),
                (pix_square_size, pix_square_size),
            ),
        )

        def energy_to_color(energy):
            s = np.array(energy)
            if s > 0:
                s = 0
            s = np.clip(np.abs(s), 0, 1)
            return (255 * (1 - s), 0, 255 * s)

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            energy_to_color(self.agent_info[0]["energy"]),
            ((1 + self.agent_info[0]["position"] + 0.5) * pix_square_size, 0.5 * pix_square_size),
            pix_square_size / 3,
        )
        if self.agent_info[0]["have_food"] is True:
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                ((1 + self.agent_info[0]["position"] + 0.5) * pix_square_size, 0.25 * pix_square_size),
                pix_square_size / 5,
            )
        write(canvas,
              "Possessor",
              color=(0, 0, 0),
              position=((1 + self.agent_info[0]["position"] + 0.2) * pix_square_size, 0.5 * pix_square_size),
              size=12
              )

        pygame.draw.circle(
            canvas,
            energy_to_color(self.agent_info[1]["energy"]),
            ((1 + self.size + 0.5) * pix_square_size, 0.5 * pix_square_size),
            pix_square_size / 3,
        )
        write(canvas,
              "Partner",
              color=(0, 0, 0),
              position=((1 + self.size + 0.2) * pix_square_size, 0.5 * pix_square_size),
              size=12
              )

        # Finally, add some gridlines
        for x in range(self.size + 2):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

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