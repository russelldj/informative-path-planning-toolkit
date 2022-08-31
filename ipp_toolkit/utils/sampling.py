import numpy as np


def get_flat_samples(world_size, resolution):
    samples = np.meshgrid(
        *[np.arange(0, s + 1e-6, resolution) for s in world_size], indexing="ij"
    )
    initial_shape = samples[0].shape
    flat_samples = [s.flatten() for s in samples]
    samples = np.vstack(flat_samples).T
    return samples, initial_shape
