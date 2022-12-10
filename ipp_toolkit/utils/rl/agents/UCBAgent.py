from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
import matplotlib.pyplot as plt
import numpy as np


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

    def get_action(self, observation, vis=True):
        # TODO deal with the action space
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

        if vis:
            fig, axs = plt.subplots(1, 4)
            plt.colorbar(axs[0].imshow(observation[0]), ax=axs[0])
            plt.colorbar(axs[1].imshow(observation[1]), ax=axs[1])
            plt.colorbar(axs[2].imshow(weighted), ax=axs[2])
            plt.colorbar(axs[3].imshow(max_mask), ax=axs[3])
            plt.show()
        return action, None
