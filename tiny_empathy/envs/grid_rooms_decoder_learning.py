from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import pygame

import gymnasium as gym
import torch

from gymnasium import spaces
from gymnasium.core import RenderFrame


class GridRoomsDecoderLearningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 render_mode=None,
                 size=5,
                 weight_empathy=0.0,
                 dim_emotional_feature=5,
                 emotional_encoder=None,  # assuming a pytorch model
                 set_energy_loss_partner=None,
                 decoding_mode=None,  # "affect" or "full"
                 ):
        self.window_size = 512

        # enable or remove empathy channel
        self.dim_emotional_feature = dim_emotional_feature
        # cognitive empathy by default
        self.weight_empathy = weight_empathy

        assert decoding_mode in {"affect", "full"}, "Invalid decoding mode. decoding mode is affect or full."
        self.decoding_mode = decoding_mode

        # emotional feature experiment settings (bodily encoding of the internal state)
        self.emotional_encoder = emotional_encoder
        self.emotional_encoder.eval()
        # trapped agent energy --(encoder)--> emotional feature --(decoder)--> inferred energy (inferred by possessor)

        # Key parameters
        self.size = size  # size of the 1D grid world
        self.default_energy_loss_posessor = 0.003  # default energy loss
        self.default_energy_loss_partner = 0.003 if set_energy_loss_partner is None else set_energy_loss_partner # default energy loss
        self.food_intake = 0.1  # intake of energy when food is consumed
        self.reward_scale = 100.  # reward scale in the homeostatic reward

        """ dimensions: energy=1, have_food=1, position=env.size, emotional_featire=dim_emotional_feature"""
        dim_obs = 3 + env.size
        if env.decoding_mode == "affect":
            dim_obs += env.dim_emotional_feature
        elif env.decoding_mode == "full":
            dim_obs += 1
        else:
            raise ValueError(f"Invalid decoding mode. {env.decoding_mode}")

        ub = np.ones(dim_obs, dtype=np.float32)
        self.observation_space = gym.spaces.Box(ub * -1, ub)

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

        self._step = 0
        self.max_episode_steps = 2000
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
        emotional_decoder=None
    ):
        """
        Initial implementation: The very simple empathic signals
        """
        self._step = 0

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

        assert emotional_decoder is not None
        observation = self.get_obs(emotional_decoder)
        # print(observation)
        info = self.get_info()

        return self.encode_obs(observation), info

    def step(self, action, emotional_decoder=None):
        assert emotional_decoder is not None

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

        rewards = self.get_reward(emotional_decoder)
        observation_dict = self.get_obs(emotional_decoder)
        info = self.get_info()

        if self.render_mode == "human":
            self._render_frame()

        self._step += 1
        if self._step > self.max_episode_steps:
            done = True

        return self.encode_obs(observation_dict), rewards, done, False, info

    def get_obs(self, emotional_decoder):
        with torch.no_grad():
            s = torch.FloatTensor([self.agent_info[1]["energy"]])
            if self.decoding_mode == "full":
                emotional_feature = emotional_decoder(self.emotional_encoder(s)).cpu().numpy()
            elif self.decoding_mode == "affect":
                emotional_feature = self.emotional_encoder(s).cpu().numpy()
            else:
                raise ValueError(f"decoding mode is invalid: {self.decoding_mode}")

        obs = {
            "energy": [self.agent_info[0]["energy"]],
            "have_food": np.float32(np.arange(2) == int(self.agent_info[0]["have_food"])),
            "position": np.float32(np.arange(self.size) == self.agent_info[0]["position"]),
            "emotional_feature": emotional_feature
        }

        return obs

    def encode_obs(self, obs_dict):
        return np.concatenate([v for k, v in obs_dict.items()])

    def get_info(self):
        return self.agent_info.copy()

    def get_reward(self, emotional_decoder):
        # homeostatic reward by Keramati & Gutkin 2011
        with torch.no_grad():
            prev_drive = self.prev_energy[0] ** 2
            s = torch.FloatTensor([self.prev_energy[1]])
            inferred_energy_prev = emotional_decoder(self.emotional_encoder(s)).cpu().numpy().item()
            prev_drive += self.weight_empathy * inferred_energy_prev ** 2  # empathic reward term

            energy_now = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])
            drive_now = energy_now[0] ** 2
            s = torch.FloatTensor([energy_now[1]])
            inferred_energy = emotional_decoder(self.emotional_encoder(s)).cpu().numpy().item()
            drive_now += self.weight_empathy * inferred_energy ** 2  # empathic reward term
            # print(inferred_energy, energy_now[1])

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


if __name__ == '__main__':
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 3)

        def forward(self, x):
            return self.model(x)


    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.model(x)


    enc = Encoder()
    dec = Decoder()

    env = GridRoomsDecoderLearningEnv(
        render_mode="human",
        dim_emotional_feature=3,
        decoding_mode="full",
        emotional_encoder=enc,
    )
    env.reset(emotional_decoder=dec)

    for i in range(1000):
        actions = env.action_space.sample()
        print(actions)
        obs, reward, done, truncate, info = env.step(actions, emotional_decoder=dec)
        env.render()
        print("obs:", obs)
        print(info)
        # print("reward:", reward)
        # print(f"done, truncate, info: {done}, {truncate}, {info}")

    env.close()
    print("finish.")
