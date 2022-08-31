import numpy as np


def get_flat_samples(world_size, resolution, world_start=(0, 0)):
    samples = np.meshgrid(
        *[
            np.arange(start, start + si + 1e-6, resolution)
            for start, si in zip(world_start, world_size)
        ],
        indexing="ij"
    )
    initial_shape = samples[0].shape
    flat_samples = [s.flatten() for s in samples]
    samples = np.vstack(flat_samples).T
    return samples, initial_shape
