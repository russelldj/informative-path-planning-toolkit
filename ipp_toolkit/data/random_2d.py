from telnetlib import GA
import numpy as np
from ipp_toolkit.config import (
    FLOAT_EPS,
    GRID_RESOLUTION,
    MEAN_ERROR_KEY,
    MEAN_KEY,
    VIS_RESOLUTION,
)
from ipp_toolkit.utils.randomness import temp_seed
from ipp_toolkit.data.data import GridData2D
from ipp_toolkit.utils.sampling import get_flat_samples
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcessRegression


class RandomGMM2D(GridData2D):
    def __init__(
        self, world_size=(30, 30), n_points=40, n_components=10, random_seed=None
    ):
        super().__init__(world_size)
        with temp_seed(random_seed):
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
        n_blobs=4,
        blob_size_range=(3, 15),
        resolution=GRID_RESOLUTION,
        lower_offset=0,
        random_seed=None,
    ):
        super().__init__(world_size)

        samples, initial_shape = get_flat_samples(self.world_size, resolution)
        with temp_seed(random_seed):
            maps = [
                self.create_one_gaussian(
                    world_size=world_size,
                    blob_size_range=blob_size_range,
                    samples=samples,
                )
                for i in range(n_blobs)
            ]

        self.map = np.add.reduce(maps)
        self.map = np.reshape(self.map, initial_shape)
        self.map /= np.max(self.map)
        self.map = self.map * (1 - lower_offset) + lower_offset
        self.samples = samples

        super()._build_interpolator()

    def create_one_gaussian(self, world_size, blob_size_range, samples):
        mean = [np.random.uniform(0, s) for s in world_size]
        cov = np.diag(np.random.uniform(*blob_size_range, size=(2,)))

        g_dist = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        return g_dist


class RandomGaussianProcess2D(GridData2D):
    def __init__(
        self,
        world_size=(30, 30),
        n_points=50,
        overlap_ind=0,
        resolution=GRID_RESOLUTION,
        random_seed=None,
        GP_training_iters=50,
    ):

        super().__init__(world_size)

        with temp_seed(random_seed):
            locations = np.vstack(
                (
                    np.random.uniform(0, world_size[0], size=2 * n_points),
                    np.random.uniform(0, world_size[1], size=2 * n_points),
                )
            ).T
            values = np.random.uniform(0, 1, size=2 * n_points)
            locations, values = [
                x[overlap_ind : n_points + overlap_ind] for x in (locations, values)
            ]

            gp = GaussianProcessRegression(training_iters=GP_training_iters)
            gp.fit(locations, values)

        samples, initial_shape = get_flat_samples(world_size, resolution)

        predictions = gp.predict(samples)
        self.map = np.reshape(predictions, initial_shape)
        self.samples, self.initial_shape = get_flat_samples(world_size, resolution)

        if False:
            plt.imshow(self.map)
            plt.savefig(f"vis/GP_{overlap_ind}.png")
            plt.show()
        super()._build_interpolator()

    def create_one_gaussian(self, world_size, blob_size_range, samples):
        mean = [np.random.uniform(0, s) for s in world_size]
        cov = np.diag(np.random.uniform(*blob_size_range, size=(2,)))

        g_dist = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        return g_dist
