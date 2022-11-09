import os

from ipp_toolkit.agents.StableBaselinesAgent import agent_dict

from sacred import Experiment

import gym

# Needed for env instantiation
import gym_ipp


def build_train_cfg(
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    model_dir,
    log_dir,
):
    cfg = {
        "num_par": num_par,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "total_timesteps": total_timesteps,
        "verbose": verbose,
        "save_freq": save_freq,
        "model_dir": model_dir,
        "log_dir": log_dir,
    }

    return cfg


ex = Experiment("test")


@ex.config
def config():
    agent_type = "PPO"
    model_dir = "models"
    log_dir = "logs"
    n_iters = 20
    noise_sdev = 0
    noise_bias = 0
    world_size = (20, 20)
    sensor_size = (1, 1)
    sensor_resolution = 1.0
    world_sample_resolution = 1.0
    obs_clip = 1.0
    obs_gp_mean_scale = 1.0
    obs_gp_std_scale = 1.0
    rew_top_frac_scale = 1.0
    rew_diff_num_visited_scale = 1.0
    map_seed = None
    action_space_discretization = 7  # Or an int specifying how many samples per axis
    # GP details
    # n_gp_fit_iters = 10
    # gp_lengthscale_prior = None
    # gp_lengthscale_var_prior = None

    # training details
    num_par = 1
    learning_rate = 3e-4
    n_steps = 2048
    total_timesteps = 300000
    verbose = 1
    save_freq = 1000


@ex.automain
def main(
    agent_type,
    model_dir,
    log_dir,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    sensor_size,
    sensor_resolution,
    world_sample_resolution,
    obs_clip,
    obs_gp_mean_scale,
    obs_gp_std_scale,
    rew_top_frac_scale,
    rew_diff_num_visited_scale,
    map_seed,
    action_space_discretization,
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    # n_gp_fit_iters,
    # gp_lengthscale_prior,
    # gp_lengthscale_var_prior,
    _run,
):

    model_dir = os.path.join(model_dir, agent_type)
    log_dir = os.path.join(log_dir, agent_type)

    # TODO move this into common class
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
    # grid sample resolution
    info_dict["world_sample_resolution"] = world_sample_resolution
    # starting x and y positions
    info_dict["init_x"] = world_size[1] / 2
    info_dict["init_y"] = world_size[0] / 2
    # max number of steps per episode
    info_dict["max_steps"] = n_iters
    # obs clip and scales
    info_dict["obs_clip"] = obs_clip
    info_dict["obs_gp_mean_scale"] = obs_gp_mean_scale
    info_dict["obs_gp_std_scale"] = obs_gp_std_scale
    # reward scaling
    info_dict["rew_top_frac_scale"] = rew_top_frac_scale
    info_dict["rew_diff_num_visited_scale"] = rew_diff_num_visited_scale
    # map determinism
    info_dict["map_seed"] = map_seed
    # action_space
    info_dict["action_space_discretization"] = action_space_discretization
    # gp params
    # info_dict["n_gp_fit_iters"] = n_gp_fit_iters
    # info_dict["gp_lengthscale_prior"] = gp_lengthscale_prior
    # info_dict["gp_lengthscale_var_prior"] = gp_lengthscale_var_prior

    env = gym.make("ipp-v0", info_dict=info_dict)
    agent = agent_dict[agent_type](env.action_space)

    cfg = build_train_cfg(
        num_par,
        learning_rate,
        n_steps,
        total_timesteps,
        verbose,
        save_freq,
        model_dir,
        log_dir,
    )

    agent.train(env, cfg)
