import os

import gym
# Needed for env instantiation
import gym_ipp

from ipp_toolkit.agents.StableBaselinesAgent import agent_dict


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


def create_info_dict(**kwargs):
    # TODO move this into common class
    info_dict = {}
    # world size
    info_dict["world_size"] = kwargs["world_size"]
    # sensor noise
    info_dict["noise_sdev"] = kwargs["noise_sdev"]
    info_dict["noise_bias"] = kwargs["noise_bias"]
    # sensor size
    info_dict["sensor_size"] = kwargs["sensor_size"]
    # sensor resolution
    info_dict["sensor_resolution"] = kwargs["sensor_resolution"]
    # starting x and y positions
    info_dict["init_x"] = kwargs["world_size"][1] / 2
    info_dict["init_y"] = kwargs["world_size"][0] / 2
    # max number of steps per episode
    info_dict["max_steps"] = kwargs["n_iters"]
    # obs clip and scales
    info_dict["obs_clip"] = kwargs["obs_clip"]
    info_dict["obs_gp_mean_scale"] = kwargs["obs_gp_mean_scale"]
    info_dict["obs_gp_std_scale"] = kwargs["obs_gp_std_scale"]
    # reward scaling
    info_dict["rew_top_frac_scale"] = kwargs["rew_top_frac_scale"]
    info_dict["rew_diff_num_visited_scale"] = kwargs["rew_diff_num_visited_scale"]
    # map determinism
    info_dict["map_seed"] = kwargs["map_seed"]
    # action space
    info_dict["action_space_discretization"] = kwargs["action_space_discretization"]
    # world sample resolution
    info_dict["world_sample_resolution"] = kwargs["world_sample_resolution"]

    return info_dict


def train(
    agent_type,
    model_dir,
    log_dir,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    sensor_size,
    sensor_resolution,
    obs_clip,
    obs_gp_mean_scale,
    obs_gp_std_scale,
    rew_top_frac_scale,
    rew_diff_num_visited_scale,
    map_seed,
    action_space_discretization,
    world_sample_resolution,
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
    info_dict = create_info_dict(**locals())

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
