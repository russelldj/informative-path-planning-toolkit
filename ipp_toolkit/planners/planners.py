import numpy as np
from ipp_toolkit.config import PLANNING_RESOLUTION
from ipp_toolkit.utils.sampling import get_flat_samples_start_stop
from ipp_toolkit.world_models.world_models import BaseWorldModel


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
        self.planning_grid_rectangular = np.reshape(
            self.planning_grid, np.hstack((self.initial_size, [2]))
        )

    def plan(self, world_model):
        raise NotImplementedError()


class RandomGridWorldPlanner(GridWorldPlanner):
    def plan(self, world_model, current_location, n_steps):
        """
        Arguments:
            current_location: The location (n,)
            n_steps: How many planning steps to take

        Returns:
            A plan specifying the list of locations
        """
        # Find the initial location on the grid
        diff = np.atleast_2d(current_location) - self.planning_grid
        dist = np.linalg.norm(diff, axis=1)
        best_ind = np.argmin(dist)
        index_loc = np.array(
            (best_ind // self.initial_size[1], best_ind % self.initial_size[1])
        )

        plan = []
        for i in range(n_steps):
            step = np.random.choice([-1, 0, 1], size=(2,))
            new_loc = index_loc + step
            # Repeat until we get a valid sample
            while np.any(new_loc < 0) or np.any(new_loc >= self.initial_size):
                step = np.random.choice([-1, 0, 1], size=(2,))
                new_loc = index_loc + step
            index_loc = new_loc
            plan.append(self.planning_grid_rectangular[index_loc[0], index_loc[1], :])

        return plan
