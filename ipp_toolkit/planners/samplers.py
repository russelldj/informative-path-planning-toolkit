from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import MEAN_KEY, UNCERTAINTY_KEY
from ipp_toolkit.planners.planners import GridWorldPlanner
import numpy as np


class MostUncertainLocationPlanner(GridWorldPlanner):
    def plan(self, world_model: BaseWorldModel, n_steps=1):
        belief = world_model.sample_belief_array(self.planning_grid)
        var = belief[UNCERTAINTY_KEY]
        most_uncertain_indices = np.argsort(var)[-n_steps:]
        most_uncertain_locs = self.planning_grid[most_uncertain_indices]
        most_uncertain_locs = np.flip(most_uncertain_locs, axis=0)
        return most_uncertain_locs


class HighestUpperBoundLocationPlanner(GridWorldPlanner):
    def plan(self, world_model: BaseWorldModel, n_steps=1, variance_scale=100):
        belief = world_model.sample_belief_array(self.planning_grid)
        var = belief[UNCERTAINTY_KEY]
        mean = belief[MEAN_KEY]
        upper_bound = mean + variance_scale * var

        most_uncertain_indices = np.argsort(upper_bound)[-n_steps:]
        most_uncertain_locs = self.planning_grid[most_uncertain_indices]
        most_uncertain_locs = np.flip(most_uncertain_locs, axis=0)
        return most_uncertain_locs


class HighestUpperBoundStochasticPlanner(GridWorldPlanner):
    def plan(self, world_model: BaseWorldModel, n_steps=1, variance_scale=100):
        belief = world_model.sample_belief_array(self.planning_grid)
        var = belief[UNCERTAINTY_KEY]
        mean = belief[MEAN_KEY]
        upper_bound = mean + variance_scale * var
        # Normalize the lowest sample to zero
        upper_bound -= np.min(upper_bound)
        # And make a valid probability distribution
        probabilities = upper_bound / np.sum(upper_bound)
        stochastic_inds = np.random.choice(
            probabilities.size, size=(n_steps,), p=probabilities
        )

        stochastic_locs = self.planning_grid[stochastic_inds]
        return stochastic_locs
