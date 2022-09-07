import math
from tkinter import Image

import gpytorch
import numpy as np
import torch
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.planners.planners import (
    HighestUpperBoundLocationPlanner,
    MostUncertainLocationPlanner,
)
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)
from matplotlib import pyplot as plt
from sacred import Experiment
from sacred.observers import MongoObserver
import imageio

ex = Experiment("test")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    n_iters = 50


@ex.automain
def main(video_file, n_iters, _run):
    data = RandomGaussian2D(world_size=(20, 20))
    sensor = GaussianNoisyPointSensor(data, noise_sdev=0, noise_bias=0)

    planner = HighestUpperBoundLocationPlanner(grid_start=(0, 0), grid_end=(20, 20))

    gp = GaussianProcessRegressionWorldModel()
    # Intialize with a few random samples
    for j in range(3):
        x = np.random.uniform(0, 20, size=(2,))
        y = sensor.sample(x)
        gp.add_observation(x, y)

    gp.train_model()
    gp.test_model(world_size=(20, 20), gt_data=data.map)

    writer = imageio.get_writer(video_file, fps=20)

    for i in range(n_iters):
        plan = planner.plan(gp)

        for loc in plan:
            y = sensor.sample(loc)
            print(loc)
            gp.add_observation(loc, y)

        gp.train_model()
        img = gp.test_model(
            world_size=(20, 20),
            gt_data=data.map,
            #    savefile=f"vis/highest_lower_bound/{i:03d}.png",
        )
        writer.append_data(img)
        # plt.imshow(img)
        # plt.show()
    writer.close()
    _run.add_artifact(video_file)

