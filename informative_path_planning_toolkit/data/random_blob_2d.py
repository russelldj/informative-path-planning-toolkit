import numpy as np
from informative_path_planning_toolkit.data.data import BaseData
from informative_path_planning_toolkit.config import (
    VIS_RESOLUTION,
    GRID_RESOLUTION,
    FLOAT_EPS,
)
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def get_flat_samples(world_size, resolution):
    samples = np.meshgrid(*[np.arange(0, s + 1e-6, resolution) for s in world_size])
    initial_shape = samples[0].shape
    flat_samples = [s.flatten() for s in samples]
    samples = np.vstack(flat_samples).T
    return samples, initial_shape


class RandomBlob2D(BaseData):
    def __init__(self, world_size):
        self.world_size = world_size
        self.interpolator = None

    def sample(self, location):
        # TODO this is costly since it constructs an interpolator on the fly just for this one
        # point. See if this can be reduced
        value = self.interpolator(location)[0]
        return value

    def show(self, resolution=VIS_RESOLUTION):
        samples, initial_shape = get_flat_samples(self.world_size, resolution)
        interpolated_values = self.interpolator(samples)
        interpolated_values = np.reshape(interpolated_values, initial_shape)
        plt.imshow(interpolated_values)
        plt.colorbar()
        plt.show()


class RandomGMM2D(RandomBlob2D):
    def __init__(self, world_size=(30, 30), n_points=40, n_components=10):
        super().__init__(world_size)
        self.mixture = self.create_random_mixture(n_points, n_components)
        self.map = self.sample_mixture()
        self.map -= np.min(self.map)
        self.map /= np.max(self.map)
        # Take the locations from the first sample, as they should be identical
        self.interpolator = lambda x: griddata(self.samples, self.map.flatten(), x)

    def create_random_mixture(self, n_points, n_components):
        random_points = [
            np.random.uniform(0, s, size=(n_points,)) for s in self.world_size
        ]
        random_points = np.vstack(random_points).T
        mixture = GaussianMixture(n_components=n_components)
        mixture.fit(random_points)
        return mixture

    def sample_mixture(self, resolution=GRID_RESOLUTION):
        self.samples, initial_shape = get_flat_samples(self.world_size, resolution)
        values = self.mixture.score_samples(self.samples)
        values = np.reshape(values, initial_shape)
        return values


class RandomGaussian2D(RandomBlob2D):
    def __init__(
        self,
        world_size=(30, 30),
        n_blobs=40,
        blob_size_range=(1, 5),
        resolution=GRID_RESOLUTION,
    ):
        super().__init__(world_size)

        samples, initial_shape = get_flat_samples(self.world_size, resolution)

        maps = [
            self.create_one_gaussian(
                world_size=world_size, blob_size_range=blob_size_range, samples=samples,
            )
            for i in range(n_blobs)
        ]
        # TODO replace map with non-keyword
        map = np.add.reduce(maps)
        map = np.reshape(map, initial_shape)
        map /= np.max(map)
        self.map = map
        self.samples = samples

        # Take the locations from the first sample, as they should be identical
        self.interpolator = lambda x: griddata(self.samples, self.map.flatten(), x)

    def create_one_gaussian(self, world_size, blob_size_range, samples):
        mean = [np.random.uniform(0, s) for s in world_size]
        cov = np.diag(np.random.uniform(*blob_size_range, size=(2,)))

        g_dist = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        return g_dist
