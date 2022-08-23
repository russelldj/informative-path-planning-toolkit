import numpy as np
from informative_path_planning_toolkit.data.data import BaseData
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class GassianBlob2D(BaseData):
    def __init__(
        self, n_blobs=40, world_size=(30, 30), blob_size_range=(1, 5), resolution=0.1,
    ):
        maps = [
            self.create_one_gaussian(
                world_size=world_size,
                blob_size_range=blob_size_range,
                resolution=resolution,
            )
            for i in range(n_blobs)
        ]
        map = np.add.reduce(maps)
        map = map / np.max(map)
        plt.imshow(map)
        plt.colorbar()
        plt.show()

    def create_one_gaussian(self, world_size, blob_size_range, resolution):
        mean = [np.random.uniform(0, s) for s in world_size]
        cov = np.diag(np.random.uniform(*blob_size_range, size=(2,)))

        samples = np.meshgrid(*[np.arange(0, s, resolution) for s in world_size])
        initial_shape = samples[0].shape
        samples = [s.flatten() for s in samples]
        samples = np.vstack(samples).T

        g_dist = multivariate_normal.pdf(samples, mean=mean, cov=cov)
        g_dist = np.reshape(g_dist, initial_shape)
        return g_dist
