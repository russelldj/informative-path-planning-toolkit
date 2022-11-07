import os
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


def plot_gt(env, world_size, gt_map_file):
    extent = (0, world_size[1], 0, world_size[0])
    gt_map = env.get_gt_map()
    # all values in gt_map should be between 0 and 1
    plt.imshow(gt_map, extent=extent, vmin=0, vmax=1)
    plt.savefig(gt_map_file)

    plt.clf()


def plot_gp(env, world_size, gp_map_dir):
    if not os.path.exists(gp_map_dir):
        os.mkdir(gp_map_dir)

    filename = f"gp_{env.num_steps}.png"
    gp_map_file = os.path.join(gp_map_dir, filename)

    extent = (0, world_size[1], 0, world_size[0])
    gp_map = env.get_gp_map()
    # all values in gt_map should be between 0 and 1
    plt.imshow(gp_map, extent=extent, vmin=0, vmax=1)
    x = env.agent_x
    y = env.agent_y
    plt.plot(x, y, "r+")
    plt.savefig(gp_map_file)

    plt.clf()


ex = Experiment("test")
# ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    save_dir = "vis/random"
    n_iters = 32
    noise_sdev = 0
    noise_bias = 0
    world_size = (15, 15)
    sensor_size = (3, 3)
    sensor_resolution = 0.5
    grid_sample_size = (7, 7)
    grid_sample_resolution = 0.5
    movement_max = 1.0
    num_prev_positions = 10
    obs_clip = 5.0
    obs_sensor_scale = 1.0
    obs_gp_mean_scale = 1.0
    obs_gp_std_scale = 1.0
    rew_top_frac_scale = 1.0
    rew_out_of_map_scale = 1.0


@ex.automain
def main(
    save_dir,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    sensor_size,
    sensor_resolution,
    grid_sample_size,
    grid_sample_resolution,
    movement_max,
    num_prev_positions,
    obs_clip,
    obs_sensor_scale,
    obs_gp_mean_scale,
    obs_gp_std_scale,
    rew_top_frac_scale,
    rew_out_of_map_scale,
    _run,
):

    video_file = os.path.join(save_dir, "test.mp4")
    reward_file = os.path.join(save_dir, "reward.png")
    gt_map_file = os.path.join(save_dir, "gt_map.png")
    gp_map_dir = os.path.join(save_dir, "gp_maps")

    info_dict = {}
    # world size
    info_dict["world_size"] = world_size
    # sensor noise
    info_dict["noise_sdev"] = noise_sdev
    info_dict["noise_bias"] = noise_bias
    # sensor size
    info_dict["sensor_size"] = sensor_size
    # sensor resolution
    info_dict["sensor_resolution"] = sensor_resolution
    # grid sample size
    info_dict["grid_sample_size"] = grid_sample_size
    # grid sample resolution
    info_dict["grid_sample_resolution"] = grid_sample_resolution
    # starting x and y positions
    info_dict["init_x"] = world_size[1] / 2
    info_dict["init_y"] = world_size[0] / 2
    # movement distance
    info_dict["movement_max"] = movement_max
    # number previous actions
    info_dict["num_prev_positions"] = num_prev_positions
    # max number of steps per episode
    info_dict["max_steps"] = n_iters
    # obs clip and scales
    info_dict["obs_clip"] = obs_clip
    info_dict["obs_sensor_scale"] = obs_sensor_scale
    info_dict["obs_gp_mean_scale"] = obs_gp_mean_scale
    info_dict["obs_gp_std_scale"] = obs_gp_std_scale
    # reward scaling
    info_dict["rew_top_frac_scale"] = rew_top_frac_scale
    info_dict["rew_out_of_map_scale"] = rew_out_of_map_scale

    env = gym.make("ipp-v0", info_dict=info_dict)
    _ = env.reset()

    plot_gt(env, world_size, gt_map_file)
    plot_gp(env, world_size, gp_map_dir)

    done = False
    safety_max = 1000
    safety_count = 0
    # writer = imageio.get_writer(video_file, fps=2)
    rewards = []
    while (not done) and (safety_count < safety_max):
        safety_count += 1

        random_action = env.action_space.sample()
        _, reward, done, _ = env.step(random_action)

        # img = env.test_gp()
        # writer.append_data(img)

        plot_gp(env, world_size, gp_map_dir)

        rewards.append(reward)

    # writer.close()
    # _run.add_artifact(video_file)

    if safety_count == safety_max:
        raise RuntimeError("Safety limit reached")

    x = np.arange(len(rewards), dtype=np.int)
    y = np.array(rewards)
    # plt.xticks(x)
    plt.plot(x, y)
    plt.ylabel("Reward")
    plt.xlabel("Step Number")
    plt.title("Performance of Random Agent")
    plt.savefig(reward_file)
