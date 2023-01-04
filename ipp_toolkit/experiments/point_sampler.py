import math
from statistics import variance
from tkinter import Image

import gpytorch
import numpy as np
import torch
from ipp_toolkit.config import MEAN_ERROR_KEY, MEAN_UNCERTAINTY_KEY
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.planners.samplers import (
    HighestUpperBoundLocationPlanner,
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


def point_sampler(
    video_file,
    error_file,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    planner_variance_scale,
    n_blobs,
    top_frac,
    vis,
    _run,
    use_tqdm=False,
):
    data = RandomGaussian2D(world_size=world_size, n_blobs=n_blobs)
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

    if vis:
        writer = imageio.get_writer(video_file, fps=20)

    all_metrics = []
    prog_iter = tqdm if use_tqdm else lambda x: x

    for _ in prog_iter(range(n_iters)):
        plan = planner.plan(gp, variance_scale=planner_variance_scale)

        for loc in plan:
            y = sensor.sample(loc)
            gp.add_observation(loc, y)

        gp.train_model()

        if vis:
            img = gp.test_model(world_size=world_size, gt_data=data.map)
            writer.append_data(img)

        metrics = gp.evaluate_metrics(
            world_size=world_size, ground_truth=data.map, top_fraction=top_frac
        )
        # Log metrics to sacred
        for k, v in metrics.items():
            _run.log_scalar(k, v)

        all_metrics.append(metrics)

    # Reshape so it's k: list of values
    keys = all_metrics[0].keys()
    all_metrics = {k: [m[k] for m in all_metrics] for k in keys}
    if vis:
        writer.close()
        _run.add_artifact(video_file)

        plt.plot(all_metrics[MEAN_ERROR_KEY])
        plt.savefig(error_file)
        plt.close()
        _run.add_artifact(error_file)
    return all_metrics
