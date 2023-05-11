import numpy as np
import torch

from policy import Policy
from numpy.random import randint


def to_torch(obs, act, rew, normalize=True, device='cpu'):
    obs = torch.tensor(obs)
    if normalize:
        obs = obs / 255
    return (obs.to(device),
            torch.tensor(act, dtype=torch.long).to(device),
            torch.tensor(rew, dtype=torch.float).to(device))


class Collector:
    def __init__(self, policy: Policy, rtg=True):
        self.policy = policy
        self.rtg = rtg

    def collect_trajectory(self) -> (np.array, np.array, np.array):
        reward, done = 0, False
        obs = self.policy.env.reset()
        obses, actions, rewards = [], [], []
        while not done:
            obses.append(obs)
            action = self.policy.act(obs, reward, done)
            actions.append(action)
            obs, reward, done, *_ = self.policy.env.step(action)
            rewards.append(reward)
        if self.rtg:
            rtgs = np.zeros_like(rewards)
            for i in reversed(range(len(rewards))):
                rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < len(rewards) else 0)
            return np.array(obses), np.array(actions, dtype=np.uint8), rtgs
        return np.array(obses), np.array(actions, dtype=np.uint8), np.array(rewards)

    def collect_batch(self, batch_size=256, target_len=8):
        batch_obses = []
        batch_actions = []
        batch_rewards = []
        while len(batch_actions) < batch_size:
            obses, actions, rewards = self.collect_trajectory()
            if len(actions) < target_len:
                continue
            start = randint(0, len(actions) - target_len)
            end = start + target_len
            batch_obses.append(obses[start:end])
            batch_actions.append(actions[start:end])
            batch_rewards.append(rewards[start:end])
        return (np.stack(batch_obses),
                np.stack(batch_actions),
                np.stack(batch_rewards))


class FasterCollector(Collector):
    def collect_batch(self, batch_size=256, target_len=8):
        batch_obses = []
        batch_actions = []
        batch_rewards = []
        while len(batch_actions) < batch_size:
            obses, actions, rewards = self.collect_trajectory()
            if len(actions) < target_len:
                continue
            for i in range(0, len(actions) - target_len):
                start = i
                end = start + target_len
                batch_obses.append(obses[start:end])
                batch_actions.append(actions[start:end])
                batch_rewards.append(rewards[start:end])
        return (np.stack(batch_obses)[:batch_size],
                np.stack(batch_actions)[:batch_size],
                np.stack(batch_rewards)[:batch_size])
