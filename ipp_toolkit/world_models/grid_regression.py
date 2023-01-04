import math
from statistics import mean, variance
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import GRID_RESOLUTION, MEAN_KEY, UNCERTAINTY_KEY


class GridWorldModel(BaseWorldModel):
    """
    Define a grid, where you record whether you've visited each one
    """

    def __init__(self, world_size, grid_cell_size, fill_value=0):
        """
        world_size: (i size, j size)
        grid_cell_size: (i_size,j_size)
        """
        # Include an epsilon to avoid having boundary cases
        self.world_size = np.expand_dims(np.array(world_size) + 1e-6, axis=0)
        self.grid_cell_size = np.expand_dims(np.array(grid_cell_size), axis=0)

        num_cells = self.world_size / self.grid_cell_size
        num_cells = np.ceil(num_cells).astype(int)
        self.belief = np.ones(num_cells[0]) * fill_value
        self.uncertainty = np.ones(num_cells[0])

        self.values = []
        self.locations = []

    def get_which_grid_cell(self, location):
        """
        location: (n, 2)
        """
        if len(location.shape) < 2:
            location = np.expand_dims(location, axis=0)
        assert len(location.shape) == 2
        which_cells = np.array(location) / self.grid_cell_size
        which_cells = np.floor(which_cells).astype(int)
        which_cells_i = which_cells[:, 0]
        which_cells_j = which_cells[:, 1]
        return which_cells_i, which_cells_j

    def add_observation(self, location, value):
        self.locations.append(location)
        self.values.append(value)

        which_cells_i, which_cells_j = self.get_which_grid_cell(location=location)
        self.belief[which_cells_i, which_cells_j] = value
        self.uncertainty[which_cells_i, which_cells_j] = 0

    def sample_belief(self, location):
        location = np.atleast_2d(location)
        return self.sample_belief_array(location)

    def sample_belief_array(self, locations):
        i_inds, j_inds = self.get_which_grid_cell(locations)
        means = self.belief[i_inds, j_inds]
        variances = self.uncertainty[i_inds, j_inds]

        return {MEAN_KEY: means, UNCERTAINTY_KEY: variances}
