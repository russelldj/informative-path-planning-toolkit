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
from tqdm import tqdm

ex = Experiment("test")


def run_training(env):
    # Instantiate the agent
    dummy_vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", dummy_vec_env, learning_rate=1e-3, verbose=1)
    # Train the agent
    model.learn(total_timesteps=int(2e5))
    # Save the agent
    model.save("PPO_ipp")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    model = PPO.load("PPO_ipp")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10
    )

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


# ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    n_iters = 20
    noise_sdev = 0.1
    noise_bias = 0
    world_size = (20, 20)
    movement_scale = 1
    grid_size = (11, 11)
    grid_scale = 0.5


@ex.automain
def main(
    video_file,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    movement_scale,
    grid_size,
    grid_scale,
    _run,
):

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
    run_training(env)
