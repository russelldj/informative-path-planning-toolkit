from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
import matplotlib.pyplot as plt
import numpy as np
import gym
import copy

###WARNING WARNING WARNING
#modified line 634 in vim /home/frc-ag-3/.anaconda3/envs/ipp-toolkit/lib/python3.10/site-packages/imitation/algorithms/dagger.py
#I am not sure where this points in your env but it was required
#and I cannot determine where it has to be set when dagger is called
class PerfectAgent(BaseAgent):
    def __init__(self, env):
        self.name = "Perfect"
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError()

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        print("Cannot train random agent.")

    def load_model(self, model_dir):
        pass

    def get_action(self, observation, env=None):
        actions = np.arange(self.action_space.n)
        returns = [copy.deepcopy(self.env).step(action) for action in actions]
        rewards = [r[1] for r in returns]
        action = np.argmax(rewards)
        return action, None
    
    def __call__(self, x):
        action, _ = self.get_action(x[0])
        return [action]
