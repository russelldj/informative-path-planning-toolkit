from ipp_toolkit.planners.planners import BasePlanner
import numpy as np


class RandomMaskedPlanner(BasePlanner):
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.valid_locs = self.data_manager.get_valid_loc_points()

    def plan(self, n_visit_locations, vis=False, savepath=None):
        num_points = self.valid_locs.shape[0]
        random_inds = np.random.choice(num_points, n_visit_locations)
        sampled_points = self.valid_locs[random_inds].astype(int)
        sampled_points = np.concatenate(
            (sampled_points, sampled_points[-1:, :]), axis=0
        )
        print(sampled_points.shape)
        return sampled_points


class GridMaskedPlanner(BasePlanner):
    def __init__(self, data_manager):
        self.data_manger = data_manager

    def plan(self):
        pass
