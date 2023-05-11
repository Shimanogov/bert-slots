from abc import ABC, abstractmethod
import random

import numpy as np
from gym import Env

from envs import Push


class Policy(ABC):
    def __init__(self, env: Env):
        self.env = env

    @abstractmethod
    def act(self, observation, reward, done):
        raise NotImplementedError


class RandomPolicy(Policy):
    def act(self, observation, reward, done):
        return self.env.action_space.sample()


class AdHocPolicy(Policy):
    def __init__(self, env: Push, random_action_proba=0.5):
        self.env = None
        self.random_action_proba = random_action_proba
        self.set_env(env)

    def set_env(self, env: Push):
        self.env = env
        assert not env.embodied_agent
        assert env.static_goals
        assert len(env.goal_ids) == 1
        assert len(env.static_box_ids) == 0

    def act(self, observation, reward, done):
        if random.random() < self.random_action_proba:
            return self.env.action_space.sample()

        box_pos_in_game = [(idx, box_pos) for idx, box_pos in enumerate(self.env.box_pos)
                           if idx not in self.env.goal_ids and idx not in self.env.static_box_ids and box_pos[0] != -1]
        idx, box_pos = random.choice(box_pos_in_game)
        goal_pos = self.env.box_pos[next(iter(self.env.goal_ids))]
        delta = goal_pos - box_pos
        if np.abs(delta)[0] >= np.abs(delta)[1]:
            direction = (int(delta[0] > 0) * 2 - 1, 0)
        else:
            direction = (0, int(delta[1] > 0) * 2 - 1)

        return idx * 4 + self.env.direction2action[direction]