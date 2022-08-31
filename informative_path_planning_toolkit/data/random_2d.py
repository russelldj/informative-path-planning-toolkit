import numpy as np
from informative_path_planning_toolkit.config import (
    FLOAT_EPS,
    GRID_RESOLUTION,
    VIS_RESOLUTION,
)
from informative_path_planning_toolkit.data.data import GridData2D
from informative_path_planning_toolkit.utils.sampling import get_flat_samples
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class RandomGMM2D(GridData2D):
    def __init__(self, world_size=(30, 30), n_points=40, n_components=10):
        super().__init__(world_size)
        self.mixture = self.create_random_mixture(n_points, n_components)
        self.map = self.sample_mixture()
        self.map -= np.min(self.map)
        self.map /= np.max(self.map)
        super()._build_interpolator()

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


class RandomGaussian2D(GridData2D):
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

        self.map = np.add.reduce(maps)
        self.map = np.reshape(self.map, initial_shape)
        self.map /= np.max(self.map)
        self.samples = samples

        super()._build_interpolator()

    def create_one_gaussian(self, world_size, blob_size_range, samples):
        mean = [np.random.uniform(0, s) for s in world_size]
        cov = np.diag(np.random.uniform(*blob_size_range, size=(2,)))

        g_dist = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        return g_dist
