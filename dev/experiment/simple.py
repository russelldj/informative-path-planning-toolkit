import math

import gpytorch
import numpy as np
import torch
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)
from matplotlib import pyplot as plt

data = RandomGaussian2D(world_size=(20, 20))


gp = GaussianProcessRegressionWorldModel()
for i in range(10):
    for j in range(10):
        x = np.random.uniform(0, 20, size=(2,))
        y = data.sample(x)
        gp.add_observation(x, y)
    gp.train_model()
    gp.test_model(world_size=(20, 20), gt_data=data.map)
