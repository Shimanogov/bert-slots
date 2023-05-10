from abc import ABC, abstractmethod
from gym import Env


class Policy(ABC):
    def __init__(self, env: Env):
        self.env = env

    @abstractmethod
    def act(self, observation, reward, done):
        raise NotImplementedError


class RandomPolicy(Policy):
    def act(self, observation, reward, done):
        return self.env.action_space.sample()
