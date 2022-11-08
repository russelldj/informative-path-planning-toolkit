from ipp_toolkit.data.random_2d import (
    RandomGaussian2D,
    RandomGMM2D,
    RandomGaussianProcess2D,
)
import numpy as np

for _ in range(2):
    gm = RandomGMM2D(world_size=(100, 100), random_seed=0)
    gm.show(0.1)
    print(f"Sampled value {gm.sample([15, 15])}")
    print(f"Checking that randomness is still random {np.random.rand()}")

for _ in range(2):
    gp = RandomGaussianProcess2D(world_size=(30, 30), random_seed=0)
    gp.show(0.1)
    print(f"Sampled value {gp.sample([15, 15])}")
    print(f"Checking that randomness is still random {np.random.rand()}")

for _ in range(2):
    gb = RandomGaussian2D(world_size=(100, 100), blob_size_range=(1, 20), random_seed=0)
    gb.show(0.1)
    print(f"Sampled value {gb.sample([15, 15])}")
    print(f"Checking that randomness is still random {np.random.rand()}")
