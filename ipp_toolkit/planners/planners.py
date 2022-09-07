from ipp_toolkit.utils.sampling import get_flat_samples, get_flat_samples_start_stop
import matplotlib.pyplot as plt
from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import MEAN_KEY, PLANNING_RESOLUTION, VARIANCE_KEY
import numpy as np


class BasePlanner:
    def plan(self, world_model: BaseWorldModel):
        """
        world_model: our current belief of the world
        """
        raise NotImplementedError()


class GridWorldPlanner(BasePlanner):
    def __init__(self, grid_start, grid_end, grid_resolution=PLANNING_RESOLUTION):
        self.grid_start = grid_start
        self.grid_end = grid_end
        self.grid_resolution = grid_resolution
        self.planning_grid, self.initial_size = get_flat_samples_start_stop(
            world_tl=grid_start, world_br=grid_end, resolution=grid_resolution
        )

    def plan(self, world_model):
        raise NotImplementedError()


class MostUncertainLocationPlanner(GridWorldPlanner):
    def plan(self, world_model: BaseWorldModel, n_steps=1):
        belief = world_model.sample_belief_array(self.planning_grid)
        var = belief[VARIANCE_KEY]
        most_uncertain_indices = np.argsort(var)[-n_steps:]
        most_uncertain_locs = self.planning_grid[most_uncertain_indices]
        most_uncertain_locs = np.flip(most_uncertain_locs, axis=0)
        return most_uncertain_locs


class HighestUpperBoundLocationPlanner(GridWorldPlanner):
    def plan(self, world_model: BaseWorldModel, n_steps=1, variance_scale=100):
        belief = world_model.sample_belief_array(self.planning_grid)
        var = belief[VARIANCE_KEY]
        mean = belief[MEAN_KEY]
        lower_bound = mean + variance_scale * var

        most_uncertain_indices = np.argsort(lower_bound)[-n_steps:]
        most_uncertain_locs = self.planning_grid[most_uncertain_indices]
        most_uncertain_locs = np.flip(most_uncertain_locs, axis=0)
        return most_uncertain_locs
