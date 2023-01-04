from idna import InvalidCodepointContext
import numpy as np
from pyparsing import WordStart
from ipp_toolkit.config import MEAN_KEY, PLANNING_RESOLUTION, UNCERTAINTY_KEY
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
        self.index_loc = None

    def plan(self, world_model: BaseWorldModel, current_location, n_step):
        # Find the initial location on the grid
        diff = np.atleast_2d(current_location) - self.planning_grid
        dist = np.linalg.norm(diff, axis=1)
        best_ind = np.argmin(dist)
        self.index_loc = np.array(
            (best_ind // self.initial_size[1], best_ind % self.initial_size[1])
        )


class RandomGridWorldPlanner(GridWorldPlanner):
    def plan(self, world_model, current_location, n_steps):
        """
        Arguments:
            current_location: The location (n,)
            n_steps: How many planning steps to take

        Returns:
            A plan specifying the list of locations
        """
        super().plan(world_model, current_location, n_steps)

        index_loc = self.index_loc
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


class GreedyGridWorldPlanner(GridWorldPlanner):
    def __init__(self, grid_start, grid_end, grid_resolution=PLANNING_RESOLUTION):
        super().__init__(grid_start, grid_end, grid_resolution)

        self.planning_steps = np.array(
            [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        )

    def plan(
        self, world_model, current_location, n_steps, variance_mean_tradeeoff=1000
    ):
        """
        Arguments:
            world_model: the belief of the world
            current_location: The location (n,)
            n_steps: How many planning steps to take

        Returns:
            A plan specifying the list of locations
        """
        super().plan(world_model, current_location, n_steps)

        grid_index = np.atleast_2d(self.index_loc)

        path_indices = [grid_index]
        plan = []
        for _ in range(n_steps):
            next_indices = grid_index + self.planning_steps

            used_paths = [
                np.any(
                    [np.all(next_index == path_index) for path_index in path_indices]
                )
                for next_index in next_indices
            ]
            valid = np.logical_and.reduce(
                [
                    np.all(next_indices >= 0, axis=1),
                    np.all(next_indices < self.initial_size, axis=1),
                    np.logical_not(used_paths),
                ]
            )
            valid_indices = next_indices[valid]
            if len(valid_indices) == 0:
                return plan

            valid_locs = self.planning_grid_rectangular[
                valid_indices[:, 0], valid_indices[:, 1]
            ]
            samples = world_model.sample_belief_array(valid_locs)
            pred_mean = samples[MEAN_KEY]
            pred_var = samples[UNCERTAINTY_KEY]
            value = pred_mean + pred_var * variance_mean_tradeeoff
            best_index_on_valid = np.argmax(value)
            grid_index = valid_indices[best_index_on_valid]
            path_indices.append(grid_index)
            plan.append(valid_locs[best_index_on_valid])

        return plan
