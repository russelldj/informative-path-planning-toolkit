from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
import matplotlib.pyplot as plt
import numpy as np
import gym
import copy


class PerfectAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "Perfect"
        self.action_space = action_space

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        print("Cannot train random agent.")

    def load_model(self, model_dir):
        pass

    def get_action(self, observation, env=None):
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError()
        actions = np.arange(self.action_space.n)
        returns = [copy.deepcopy(env).step(action) for action in actions]
        rewards = [r[1] for r in returns]
        action = np.argmax(rewards)
        return action, None
