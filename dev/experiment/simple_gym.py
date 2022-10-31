import math
from tkinter import Image

import gpytorch
import numpy as np
import torch
from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.planners.samplers import (
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
from tqdm import tqdm

import gym
import gym_ipp


ex = Experiment("test")
#ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    reward_file = "vis/reward.png"
    n_iters = 50
    noise_sdev = 0.1
    noise_bias = 0
    world_size = (20, 20)
    movement_scale = 1
    grid_size = (11, 11)
    grid_scale = 0.5

@ex.automain
def main(video_file, reward_file, n_iters, noise_sdev, noise_bias, world_size, movement_scale, grid_size, grid_scale, _run):

    info_dict = {}
    info_dict['world_size'] = world_size
    info_dict['noise_sdev'] = noise_sdev
    info_dict['noise_bias'] = noise_bias
    info_dict['init_x'] = world_size[1] / 2
    info_dict['init_y'] = world_size[0] / 2
    info_dict['max_steps'] = n_iters
    info_dict['movement_scale'] = movement_scale
    info_dict['grid_size'] = grid_size
    info_dict['grid_scale'] = grid_scale

    env = gym.make('ipp-v0', info_dict=info_dict)
    _, _= env.reset()

    done = False
    safety_max = 1000
    safety_count = 0
    writer = imageio.get_writer(video_file, fps=2)
    rewards = []
    while ((not done) and (safety_count < safety_max)):
        safety_count += 1

        random_action = env.action_space.sample()
        new_obs, reward, done, info = env.step(random_action)

        img = env.test_gp()
        writer.append_data(img)

        rewards.append(reward)

    writer.close()
    _run.add_artifact(video_file)

    if safety_count == safety_max:
        raise RuntimeError('Safety limit reached')

    x = np.arange(len(rewards), dtype=np.int)
    y = np.array(rewards)
    #plt.xticks(x)
    plt.plot(x, y)
    plt.ylabel('Rewardt')
    plt.xlabel('Step Number')
    plt.title('Performance of Random Agent')
    plt.savefig(reward_file)

