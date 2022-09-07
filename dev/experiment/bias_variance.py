import math
from statistics import variance
from tkinter import Image

import gpytorch
import numpy as np
import torch
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.planners.planners import (
    HighestUpperBoundStochasticPlanner,
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
from tqdm import tqdm

ex = Experiment("bias_variance")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    n_iters = 200
    noise_sdev = 0.1
    noise_bias = 0
    world_size = (20, 20)
    planner_variance_scale = 100


@ex.automain
def main(
    video_file,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    planner_variance_scale,
    _run,
):
    data = RandomGaussian2D(world_size=world_size)
    sensor = GaussianNoisyPointSensor(
        data, noise_sdev=noise_sdev, noise_bias=noise_bias
    )

    planner = HighestUpperBoundStochasticPlanner(grid_start=(0, 0), grid_end=world_size)

    gp = GaussianProcessRegressionWorldModel()
    # Intialize with a few random samples
    for j in range(3):
        # TODO make this work for non-square worlds
        x = np.random.uniform(0, world_size[0], size=(2,))
        y = sensor.sample(x)
        gp.add_observation(x, y)
    gp.train_model()
    gp.test_model(world_size=world_size, gt_data=data.map)

    writer = imageio.get_writer(video_file, fps=20)

    for _ in tqdm(range(n_iters)):
        plan = planner.plan(gp, variance_scale=planner_variance_scale)

        for loc in plan:
            y = sensor.sample(loc)
            gp.add_observation(loc, y)

        gp.train_model()
        img = gp.test_model(world_size=world_size, gt_data=data.map,)
        writer.append_data(img)
    writer.close()
    _run.add_artifact(video_file)

