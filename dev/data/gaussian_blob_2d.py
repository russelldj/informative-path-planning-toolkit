import random
from ipp_toolkit.data.random_2d import (
    RandomGaussian2D,
    RandomGMM2D,
    RandomGaussianProcess2D,
)

# gm = RandomGMM2D(world_size=(100, 100))
# gm.show(0.1)
# print(gm.sample([15, 15]))


gm = RandomGaussianProcess2D(world_size=(30, 30))
gm.show(0.1)
print(gm.sample([15, 15]))

gb = RandomGaussian2D(world_size=(100, 100), blob_size_range=(1, 20))
gb.show(0.1)
print(gb.sample([15, 15]))

