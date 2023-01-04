import math
from statistics import mean, variance
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import GRID_RESOLUTION, MEAN_KEY, UNCERTAINTY_KEY
from scipy.spatial.distance import cdist

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import scipy


class InterpolationWorldModel(BaseWorldModel):
    """
    Predictions are produced using linear interpolation. Uncertainty is the distance from the nearest point
    """

    def __init__(
        self,
        world_size,
        grid_cell_size,
        uncertainty_scale: float = 1.0,
        predict_outside_convex=False,
        mean_fill_value=0,
        uncertainty_fill_scale_with_existing_value=0.1,
        uncertainty_float_scale_fill=10,
    ):
        """
        world_size: (i size, j size)
        grid_cell_size: (i_size,j_size)
        uncertainty_scale: How much to scale the distance by
        """
        # Include an epsilon to avoid having boundary cases
        self.uncertainty_scale = uncertainty_scale
        self.predict_outside_convex = predict_outside_convex
        self.mean_fill_value = mean_fill_value
        self.uncertainty_float_scale_fill = uncertainty_float_scale_fill
        self.uncertainty_fill_scale_with_existing_value = (
            uncertainty_fill_scale_with_existing_value
        )

        self.values = np.empty((0,))
        self.locations = np.empty((0, len(world_size)))

        # Set interpolators
        self.linear_interpolator = None
        self.outside_convex_interpolator = None

    def add_observation(self, location, value):
        self.locations = np.concatenate((self.locations, location))
        self.values = np.concatenate((self.values, value))

        try:
            self.linear_interpolator = LinearNDInterpolator(self.locations, self.values)
        # Fall back on a nearest neighbor interpolator if there aren't enough points to make a
        # linear interpolator work
        except scipy.spatial._qhull.QhullError:
            self.linear_interpolator = None

        if self.predict_outside_convex:
            self.outside_convex_interpolator = NearestNDInterpolator(
                self.locations, self.values
            )

    def sample_belief(self, location):
        location = np.atleast_2d(location)

        return self.sample_belief_array(location)

    def sample_belief_array(self, locations):
        dists = cdist(locations, self.locations)
        # Compute the distance to the closest point and scale
        variances = np.min(dists, axis=1) * self.uncertainty_scale

        if self.linear_interpolator is not None:
            # Predict values within the convex hull with linear interpolation
            means = self.linear_interpolator(locations)
            nan_linear_means = np.logical_not(np.isfinite(means))
            if self.predict_outside_convex:
                outside_points = locations[nan_linear_means]
                # Predict values outside of the convex hull with nearest neighbor
                # TODO consider a way to smoothly blend these together
                outside_values = self.outside_convex_interpolator(outside_points)
                means[nan_linear_means] = outside_values
            else:
                means[nan_linear_means] = self.mean_fill_value
                # Linearly interpolate between old value and fill value
                variances[
                    nan_linear_means
                ] = self.uncertainty_fill_scale_with_existing_value * variances[
                    nan_linear_means
                ] + (
                    1 - self.uncertainty_fill_scale_with_existing_value
                ) * (
                    self.uncertainty_float_scale_fill * self.uncertainty_scale
                )

        else:
            # Predict solely with the nearest neighbor predictor
            means = np.zeros(locations.shape[0])

        return {MEAN_KEY: means, UNCERTAINTY_KEY: variances}
