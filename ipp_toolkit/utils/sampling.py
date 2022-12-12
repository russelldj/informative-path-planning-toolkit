import numpy as np
import warnings


def get_flat_samples_start_stop(world_tl, world_br, resolution):
    samples = np.meshgrid(
        *[
            np.arange(start, stop + 1e-6, resolution)
            for start, stop in zip(world_tl, world_br)
        ],
        indexing="ij"
    )
    initial_shape = samples[0].shape
    flat_samples = [s.flatten() for s in samples]
    samples = np.vstack(flat_samples).T
    return samples, initial_shape


def get_flat_samples(world_size, resolution, world_start=(0, 0)):
    world_br = (world_size[0] + world_start[0], world_size[1] + world_start[1])
    return get_flat_samples_start_stop(
        world_br=world_br, world_tl=world_start, resolution=resolution
    )
