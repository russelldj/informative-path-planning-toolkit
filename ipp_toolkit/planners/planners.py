from ipp_toolkit.utils.sampling import get_flat_samples_start_stop
from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import PLANNING_RESOLUTION


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

