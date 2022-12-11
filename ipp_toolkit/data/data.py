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
            # self.axis_points, self.map, bounds_error=False, fill_value=None
            self.axis_points,
            self.map,
        )

    #TODO these two below are duplicated from  grid_regression.py
    def get_which_grid_cell(self, location, grid_cell_size):
        """
        location: (n, 2)
        """
        if len(location.shape) < 2:
            location = np.expand_dims(location, axis=0)
        assert len(location.shape) == 2
        which_cells = np.array(location) / grid_cell_size
        which_cells = np.floor(which_cells).astype(int)
        which_cells_i = which_cells[:, 0]
        which_cells_j = which_cells[:, 1]
        return which_cells_i, which_cells_j

    def sample_subset_array(self, locations, grid_cell_size):
        i_inds, j_inds = self.get_which_grid_cell(locations, grid_cell_size)
        data = self.map[i_inds, j_inds]

        return data
