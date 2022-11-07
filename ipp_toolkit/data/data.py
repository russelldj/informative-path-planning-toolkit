from ipp_toolkit.config import (
    FLOAT_EPS,
    GRID_RESOLUTION,
    VIS_RESOLUTION,
)
from scipy.interpolate import RegularGridInterpolator, griddata
import matplotlib.pyplot as plt
import numpy as np

from ipp_toolkit.utils.sampling import get_flat_samples


class BaseData:
    def __init__(self):
        pass

    def sample(self, location):
        """
        Args:
            location: Any
        Returns:
            A vector of observations or scalar
        """
        raise NotImplementedError()

    def show(self):
        """
        Visualize the data
        """
        raise NotImplementedError()


class GridData2D(BaseData):
    def __init__(self, world_size):
        self.world_size = world_size
        self.interpolator = None

    def sample(self, location):
        # Avoid weirdness between tuples and lists
        location = tuple(location)
        value = self.interpolator(location)
        return value

    def show(self, resolution=VIS_RESOLUTION):
        samples, initial_shape = get_flat_samples(self.world_size, resolution)
        interpolated_values = self.interpolator(samples)
        interpolated_values = np.reshape(interpolated_values, initial_shape)
        plt.imshow(interpolated_values)
        plt.colorbar()
        plt.show()

    def _build_interpolator(self):
        """
        Build an interpolator from a rectangular grid of sampled data
        """
        # Indexing crap to get the sampled locations for for each axes
        self.axis_points = (
            self.samples[
                0 : self.map.shape[0] * self.map.shape[1] : self.map.shape[1], 0
            ],
            self.samples[: self.map.shape[1], 1],
        )
        # TODO David do we ever not want extrapolate (bounds_error False, fill_value None)?
        self.interpolator = RegularGridInterpolator(
            self.axis_points, self.map, bounds_error=False, fill_value=None
        )
