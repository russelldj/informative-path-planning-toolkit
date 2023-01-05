from ipp_toolkit.planners.planners import BasePlanner
import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.planners.utils import add_candidates_and_plan
from ipp_toolkit.config import VIS


class RandomMaskedPlanner(BasePlanner):
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.valid_locs = self.data_manager.get_valid_loc_points()

    def plan(self, visit_n_locations, vis=VIS, savepath=None, **kwargs):
        num_points = self.valid_locs.shape[0]
        random_inds = np.random.choice(num_points, visit_n_locations)
        sampled_points = self.valid_locs[random_inds].astype(int)
        sampled_points = np.concatenate(
            (sampled_points, sampled_points[-1:, :]), axis=0
        )
        plt.close()
        plt.clf()
        if vis:
            plt.imshow(self.data_manager.image[..., :3])
            # Note that the convention is switched for plotting
            plt.plot(sampled_points[:, 1], sampled_points[:, 0])
            if savepath is not None:
                plt.savefig(savepath)
            else:
                plt.show()
        plt.close()
        plt.clf()
        plt.cla()
        return sampled_points


class GridMaskedPlanner(BasePlanner):
    def __init__(self, data_manager):
        self.data_manger = data_manager

    def plan(self):
        pass
