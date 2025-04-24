from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import pygame

import gymnasium as gym
import torch

from gymnasium import spaces
from gymnasium.core import RenderFrame


class FoodShareDecoderLearningEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self,
                 emotional_feature=False,
                 render_mode=None,
                 weight_empathy=0.0,
                 dim_emotional_feature=5,
                 emotional_encoder=None,  # assuming a pytorch model
                 decoding_mode=None,
                 feed_success=1.0):
        self.window_size = 512

        # enable or remove emotional expression feature
        self.emotional_feature = emotional_feature

        # enable or remove empathy channel
        self.dim_emotional_feature = dim_emotional_feature
        # cognitive empathy by default
        self.weight_empathy = weight_empathy

        assert decoding_mode in {"affect", "full"}, f"Invalid decoding mode. decoding mode is affect or full.: {decoding_mode}"
        self.decoding_mode = decoding_mode

        # emotional feature experiment settings (bodily encoding of the internal state)
        self.emotional_encoder = emotional_encoder
        self.emotional_encoder.eval()

        self.feed_success_rate = feed_success

        # Key parameters
        self.prob_H = 0.95  # target distribution of the energy level (high)
        self.prob_L = 1 - self.prob_H
        self.max_low_steps = 10

        self.prob_low_energy = 0.1  # prob of flipping the energy level to low
        self.reward_scale = 1.  # reward scale in the homeostatic reward

        obs_dict = spaces.Dict(
            {
                "energy": spaces.Discrete(2),
                "emotional_feature": spaces.Box(-np.inf, np.inf, shape=(dim_emotional_feature,)),
            }
        )
        self.observation_space = spaces.Dict(obs_dict)

        self.action_space = spaces.Discrete(2)  # pass or eat
        # action:
        # 0: eat food
        # 1: pass food

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.action_now = None

        self.step_low = [0, 0]

        self.agent_info = {
            0: {  # possessor
                "energy": 1,
            },
            1: {  # partner
                "energy": 1,
            }
        }

        self._step = 0
        self.max_episode_steps = 2000
        self.prev_energy = np.array([v["energy"] for k, v in self.agent_info.items()])

    def set_agent_info(self, id_: int, energy: int):
        self.agent_info[id_]["energy"] = energy

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
        self.step_low = [0, 0]

        self.set_agent_info(
            id_=0,
            energy=self.np_random.integers(0, 2),
        )

        self.set_agent_info(
            id_=1,
            energy=self.np_random.integers(0, 2),
        )

        self.prev_energy = np.array([self.agent_info[i]["energy"] for i in range(2)])

        assert emotional_decoder is not None
        observation = self.get_obs(emotional_decoder)
        info = self.get_info()

        return observation, info

    def step(self, action, emotional_decoder=None):
        assert emotional_decoder is not None

        # copy previous energy
        self.prev_energy = np.array([self.agent_info[i]["energy"] for i in range(2)])

        # agent 0 move
        if action == 0:  # possessor took "eat" action
            if self.np_random.uniform() < self.feed_success_rate:
                self.agent_info[0]["energy"] = 1

            # energy update of the partner
            if self.agent_info[1]["energy"] == 1:
                if self.unwrapped.np_random.random() < self.prob_low_energy:
                    self.agent_info[1]["energy"] = 0

        elif action == 1:  # possessor took "pass" action
            if self.np_random.uniform() < self.feed_success_rate:
                self.agent_info[1]["energy"] = 1

            # energy update of the possessor
            if self.agent_info[0]["energy"] == 1:
                if self.unwrapped.np_random.random() < self.prob_low_energy:
                    self.agent_info[0]["energy"] = 0
        else:
            raise ValueError("Invalid action. action", action)

        for i in range(2):
            if self.agent_info[i]["energy"] == 0:
                self.step_low[i] += 1
            else:
                self.step_low[i] = 0

        observation_dict = self.get_obs(emotional_decoder)
        rewards = self.get_reward(emotional_decoder)
        done = False if all([v < self.max_low_steps for v in self.step_low]) else True
        info = self.get_info()

        self.action_now = action

        if self.render_mode == "human":
            self._render_frame()

        self._step += 1
        if self._step > self.max_episode_steps:
            done = True

        return observation_dict, rewards, done, False, info

    def get_obs(self, emotional_decoder):
        obs = {"energy": np.array([self.agent_info[0]["energy"]])}
        with torch.no_grad():
            s = torch.FloatTensor([self.agent_info[1]["energy"]])
            if self.decoding_mode == "full":
                emotional_feature = emotional_decoder(self.emotional_encoder(s)).cpu().numpy()
            elif self.decoding_mode == "affect":
                emotional_feature = self.emotional_encoder(s).cpu().numpy()
            else:
                raise ValueError(f"decoding mode is invalid: {self.decoding_mode}")
            obs.update({"emotional_feature": emotional_feature})
        return obs

    def encode_obs(self, obs_dict):
        return np.concatenate([v for k, v in obs_dict.items()])

    def get_info(self):
        return self.agent_info.copy()

    def get_reward(self, emotional_decoder):

        def drive(en):
            d = 0
            if en == 0:
                d += -np.log(self.prob_L)
            elif en == 1:
                d += - np.log(self.prob_H)
            else:
                raise ValueError("invalid value for energy: ", en)

            return d

        # homeostatic reward by Keramati & Gutkin 2011
        with torch.no_grad():
            prev_drive = drive(self.prev_energy[0])
            s = torch.FloatTensor([self.prev_energy[1]])
            inferred_energy_prev = emotional_decoder(self.emotional_encoder(s)).cpu().numpy().item() > 0.5  # binary thresholding
            prev_drive += self.weight_empathy * drive(inferred_energy_prev)  # empathic reward term

            energy_now = np.array([self.agent_info[0]["energy"], self.agent_info[1]["energy"]])
            drive_now = drive(energy_now[0])
            s = torch.FloatTensor([energy_now[1]])
            inferred_energy = emotional_decoder(self.emotional_encoder(s)).cpu().numpy().item() > 0.5  # binary thresholding
            drive_now += self.weight_empathy * drive(inferred_energy)  # empathic reward term

        reward = self.reward_scale * (prev_drive - drive_now)
        # reward = - self.reward_scale * drive_now
        return reward

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size / 3)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / 3
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
                pix_square_size * np.array([2, 0]),
                (pix_square_size, pix_square_size),
            ),
        )

        def energy_to_color(energy):
            if energy == 0:
                return 50, 50, 255
            else:
                return 255, 50, 50

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            energy_to_color(self.agent_info[0]["energy"]),
            ((1 + 0 + 0.5) * pix_square_size, 0.5 * pix_square_size),
            pix_square_size / 3,
        )
        write(canvas,
              "Possessor",
              color=(0, 0, 0),
              position=((1 + 0 + 0.2) * pix_square_size, 0.5 * pix_square_size),
              size=12
              )
        if self.action_now is not None:
            write(canvas,
                  "EAT" if self.action_now == 0 else "PASS",
                  color=(0, 0, 0),
                  position=((1 + 0 + 0.2) * pix_square_size, 0.1 * pix_square_size),
                  size=12
                  )

        pygame.draw.circle(
            canvas,
            energy_to_color(self.agent_info[1]["energy"]),
            ((1 + 1 + 0.5) * pix_square_size, 0.5 * pix_square_size),
            pix_square_size / 3,
        )
        write(canvas,
              "Partner",
              color=(0, 0, 0),
              position=((1 + 1 + 0.2) * pix_square_size, 0.5 * pix_square_size),
              size=12
              )

        # Finally, add some gridlines
        for x in range(3):
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

    env = FoodShareDecoderLearningEnv(
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
        print("obs:", env.encode_obs(obs))
        print(info)
        # print("reward:", reward)
        # print(f"done, truncate, info: {done}, {truncate}, {info}")

    env.close()
    print("finish.")