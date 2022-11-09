import os
import math
from tkinter import Image

import gpytorch
import gym
import gym_ipp
import imageio
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
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm

ex = Experiment("test")


def run_training(env, num_par):
    dummy_env = DummyVecEnv([lambda: env] * num_par)

    # Instantiate the agent
    model = PPO("MlpPolicy", dummy_env, learning_rate=1e-3, n_steps=512, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=1024,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Train the agent
    model.learn(
        total_timesteps=int(50000), progress_bar=True, callback=checkpoint_callback
    )

    # Save the agent
    model.save("PPO_ipp")


# ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    save_dir = "vis/sb3"
    n_iters = 64
    noise_sdev = 0.0001
    noise_bias = 0
    world_size = (10, 10)
    movement_scale = 1
    grid_size = (5, 5)
    grid_scale = 1.0
    mode = "test"


@ex.automain
def main(
    save_dir,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    movement_scale,
    grid_size,
    grid_scale,
    mode,
    _run,
):
    video_file = os.path.join(save_dir, "test.mp4")
    reward_file = os.path.join(save_dir, "ppo_reward.png")
    gt_map_file = os.path.join(save_dir, "gt_map.png")

    info_dict = {}
    info_dict["world_size"] = world_size
    info_dict["noise_sdev"] = noise_sdev
    info_dict["noise_bias"] = noise_bias
    info_dict["init_x"] = world_size[1] / 2
    info_dict["init_y"] = world_size[0] / 2
    info_dict["max_steps"] = n_iters
    info_dict["movement_scale"] = movement_scale
    info_dict["grid_size"] = grid_size
    info_dict["grid_scale"] = grid_scale

    env = gym.make("ipp-v0", info_dict=info_dict)
    # check_env(env)

    if mode == "train":
        run_training(env, 1)

    model = PPO.load("PPO_ipp")

    done = False
    safety_max = 1000
    safety_count = 0
    rewards = []
    obs = env.reset()

    extent = (0, world_size[1], 0, world_size[0])
    gt_map = env.get_gt_map()
    # all values in gt_map should be between 0 and 1
    plt.imshow(gt_map, extent=extent, vmin=0, vmax=1)
    plt.savefig(gt_map_file)
    exit(0)
    while (not done) and (safety_count < safety_max):
        safety_count += 1

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        rewards.append(reward)

    x = np.arange(len(rewards), dtype=np.int)
    y = np.array(rewards)
    # plt.xticks(x)
    plt.plot(x, y)
    plt.ylabel("Reward")
    plt.xlabel("Step Number")
    plt.title("Performance of Trained Agent")
    plt.savefig(reward_file)
