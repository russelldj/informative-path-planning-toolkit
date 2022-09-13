import math
from tkinter import Image

import gpytorch
import numpy as np
import torch
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.planners.planners import RandomGridWorldPlanner, GreedyGridWorldPlanner
from ipp_toolkit.planners.samplers import (
    HighestUpperBoundLocationPlanner,
    MostUncertainLocationPlanner,
)
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)


class NoiseModelExperiment:
    def __init__(self, num_points=100, shift=10, world_size=(30, 30)):
        self.world_size = world_size
        self.num_points = num_points
        self.shift = shift

        self.data = RandomGaussian2D(world_size=world_size)
        self.sensor = GaussianNoisyPointSensor(self.data, noise_sdev=0)

        self.planner = GreedyGridWorldPlanner(grid_start=(0, 0), grid_end=world_size)

        self.world_model = GaussianProcessRegressionWorldModel()

    def run(self, initial_point):
        last_loc = initial_point
        plan = [last_loc]

        for i in range(50):
            for loc in plan:
                y = self.sensor.sample(loc)
                self.world_model.add_observation(loc, y)

            last_loc = plan[-1]

            self.world_model.train_model()
            self.world_model.test_model(
                world_size=self.world_size, plot=True, gt_data=self.data.map
            )
            plan = self.planner.plan(self.world_model, last_loc, 20)
