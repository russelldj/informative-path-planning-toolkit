import numpy as np
from informative_path_planning_toolkit.data.data import BaseData
from informative_path_planning_toolkit.config import (
    VIS_RESOLUTION,
    GRID_RESOLUTION,
    FLOAT_EPS,
)
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


class GassianBlob2D(BaseData):
    def __init__(
        self,
        n_blobs=40,
        world_size=(30, 30),
        blob_size_range=(1, 5),
        resolution=GRID_RESOLUTION,
    ):
        self.world_size = world_size

        samples, flat_samples, initial_shape = self.get_flat_samples(resolution)

        maps = [
            self.create_one_gaussian(
                world_size=world_size, blob_size_range=blob_size_range, samples=samples,
            )
            for i in range(n_blobs)
        ]
        # TODO replace map with non-keyword
        map = np.add.reduce(maps)
        map = np.reshape(map, initial_shape)
        map = map / np.max(map)
        self.map = map
        self.samples = flat_samples

        # Take the locations from the first sample, as they should be identical
        self.interopolator = lambda x: griddata(
            np.array(self.samples).T, self.map.flatten(), x
        )

    def get_flat_samples(self, resolution):
        samples = np.meshgrid(
            *[np.arange(0, s + 1e-6, resolution) for s in self.world_size]
        )
        initial_shape = samples[0].shape
        flat_samples = [s.flatten() for s in samples]
        samples = np.vstack(flat_samples).T
        return samples, flat_samples, initial_shape

    def create_one_gaussian(self, world_size, blob_size_range, samples):
        mean = [np.random.uniform(0, s) for s in world_size]
        cov = np.diag(np.random.uniform(*blob_size_range, size=(2,)))

        g_dist = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        return g_dist

    def sample(self, location):
        # TODO this is costly since it constructs an interpolator on the fly just for this one
        # point. See if this can be reduced
        value = self.interopolator(location)[0]
        return value

    def show(self, resolution=VIS_RESOLUTION):
        _, flat_samples, initial_shape = self.get_flat_samples(resolution)
        flat_samples = np.vstack(flat_samples).T
        interpolated_values = self.interopolator(flat_samples)
        interpolated_values = np.reshape(interpolated_values, initial_shape)
        plt.imshow(interpolated_values)
        plt.colorbar()
        plt.show()
