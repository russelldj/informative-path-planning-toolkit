from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
import matplotlib.pyplot as plt
import numpy as np
import gym
from ipp_toolkit.config import VIS


class UCBAgent(BaseAgent):
    def __init__(self, action_space, uncertainty_weighting=2.0):
        self.name = "UCB"
        self.action_space = action_space
        self.uncertainty_weighting = uncertainty_weighting

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        print("Cannot train random agent.")

    def load_model(self, model_dir):
        pass

    def convert_continous_to_discrete(self, action, n):
        sqrt_n = int(np.sqrt(n))
        assert sqrt_n ** 2 == n
        # Scale to (0,1)
        action = (action + 1) / 2
        # Scale to (0,n)
        action = action * sqrt_n
        # Take ints
        action = np.floor(action).astype(int)
        action_ind = action[0] * sqrt_n + action[1]
        return action_ind

    def convert_discrete_observation_to_location(self, action_ind, n_actions):
        action_ind = np.squeeze(action_ind)
        sqrt_n = int(np.sqrt(n_actions))
        assert sqrt_n ** 2 == n_actions
        action_loc = np.array([action_ind % sqrt_n, action_ind // sqrt_n])
        action_loc = action_loc / sqrt_n
        action_loc = action_loc * 2 - 1
        return action_loc

    def get_action(self, observation, vis=VIS):
        # TODO deal with the action space
        if len(observation.shape) == 1:
            num_observations = observation.shape[0]
            num_means = int(num_observations / 2)
            n_size = int(np.sqrt(num_means))
            observation = np.reshape(observation, (2, n_size, n_size))
        weighted = (
            observation[0].astype(float)
            + observation[1].astype(float) * self.uncertainty_weighting
        )
        max_mask = weighted == np.max(weighted)
        max_locs = np.array(np.where(max_mask)).T
        action_ind = np.random.choice(max_locs.shape[0])
        action_loc = max_locs[action_ind]
        action = action_loc / observation.shape[1:]
        action = (action * 2) - 1

        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            n = self.action_space.n
            action = self.convert_continous_to_discrete(action, n)
        if vis and False:
            fig, axs = plt.subplots(1, 4)
            plt.colorbar(axs[0].imshow(observation[0]), ax=axs[0])
            plt.colorbar(axs[1].imshow(observation[1]), ax=axs[1])
            plt.colorbar(axs[2].imshow(weighted), ax=axs[2])
            plt.colorbar(axs[3].imshow(max_mask), ax=axs[3])
            plt.show()
        return action, None
