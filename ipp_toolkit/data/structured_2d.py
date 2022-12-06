import numpy as np
from ipp_toolkit.data.data import GridData2D, BaseData
from ipp_toolkit.config import GRID_RESOLUTION, WORLD_SIZE
from ipp_toolkit.utils.sampling import get_flat_samples


class Uniform2D(BaseData):
    def __init__(
        self, value: float = 0, world_size=WORLD_SIZE, resolution=GRID_RESOLUTION,
    ):
        self.value = value
        self.samples, initial_shape = get_flat_samples(world_size, resolution)
        self.map = np.ones(initial_shape) * value

    def sample(self, location):
        return self.value
