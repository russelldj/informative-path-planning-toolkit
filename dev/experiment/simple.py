import math

import gpytorch
import numpy as np
import torch
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.planners.planners import MostUncertainLocationPlanner
from matplotlib import pyplot as plt

data = RandomGaussian2D(world_size=(20, 20))
sensor = GaussianNoisyPointSensor(data, noise_sdev=0, noise_bias=0)

planner = MostUncertainLocationPlanner(grid_start=(0, 0), grid_end=(20, 20))

gp = GaussianProcessRegressionWorldModel()
# Intialize with a few random samples
for j in range(3):
    x = np.random.uniform(0, 20, size=(2,))
    y = sensor.sample(x)
    gp.add_observation(x, y)

gp.train_model()
gp.test_model(world_size=(20, 20), gt_data=data.map)

for i in range(200):
    plan = planner.plan(gp)

    for loc in plan:
        y = sensor.sample(loc)
        print(loc)
        gp.add_observation(loc, y)

    gp.train_model()
    gp.test_model(
        world_size=(20, 20), gt_data=data.map, savefile=f"vis/highest_var/{i:03d}.png"
    )

