import copy
import os
import json

import gym
import gym_ipp
import imageio

# Needed for env instantiation
import matplotlib.pyplot as plt
import numpy as np

from ipp_toolkit.utils.rl.agents import agent_dict
from ipp_toolkit.utils.rl.agents.MBAgent import MBAgent
from ipp_toolkit.utils.rl.agents.BehaviorCloningAgent import BehaviorCloningAgent


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
    info_dict["observation_space_discretization"] = kwargs[
        "observation_space_discretization"
    ]

    info_dict["map_lower_offset"] = kwargs["map_lower_offset"]
    info_dict["use_interpolation_model"] = kwargs["use_interpolation_model"]

    info_dict["cnn_encoding"] = kwargs["policy"]
    info_dict["move_on_grid"] = kwargs["move_on_grid"]

    return info_dict


def plot_gt(env, world_size, gt_map_file):
    extent = (0, world_size[1], 0, world_size[0])
    gt_map = env.get_gt_map()
    # all values in gt_map should be between 0 and 1
    plt.imshow(gt_map, extent=extent, vmin=0, vmax=1)
    plt.savefig(gt_map_file)

    plt.clf()


def plot_gp(env, world_size, gp_map_dir, filename=None):
    if not os.path.exists(gp_map_dir):
        os.mkdir(gp_map_dir)

    if filename is None:
        filename = f"gp_{env.num_steps}.png"

    gp_map_file = os.path.join(gp_map_dir, filename)

    extent = (0, world_size[1], 0, world_size[0])
    gp_map = env.get_gp_map()
    # all values in gt_map should be between 0 and 1
    plt.imshow(gp_map, extent=extent, vmin=0, vmax=1)
    x = env.agent_x
    y = env.agent_y
    plt.plot(x, y, "r.", markersize=20)
    plt.savefig(gp_map_file)

    plt.clf()


def plot_gp_full(env, gp_map_dir, filename=None):
    if not os.path.exists(gp_map_dir):
        os.mkdir(gp_map_dir)

    if filename is None:
        filename = f"gp_{env.num_steps}_full.png"

    gp_map_file = os.path.join(gp_map_dir, filename)

    img = env.test_gp()
    plt.imsave(gp_map_file, img)

    plt.clf()


def plot_visited(env, visited_size, visiited_dir, filename=None):
    if not os.path.exists(visiited_dir):
        os.mkdir(visiited_dir)

    if filename is None:
        filename = f"visited_{env.num_steps}.png"

    visited_map_file = os.path.join(visiited_dir, filename)

    extent = (0, visited_size, 0, visited_size)
    visited_map = env.get_visited_map()
    # all values in gt_map should be between 0 and 1
    plt.imshow(visited_map, extent=extent, vmin=0, vmax=255)
    plt.savefig(visited_map_file)

    plt.clf()


def plot_all_rewards(full_rewards, agent_names, reward_file, tag="Reward"):
    means = []
    stds = []
    full_rewards = np.array(full_rewards)
    for i in range(full_rewards.shape[1]):
        agent_rewards = full_rewards[:, i, :]
        means.append(np.mean(agent_rewards, axis=0))
        stds.append(np.std(agent_rewards, axis=0))
    for agent_name, mean, std in zip(agent_names, means, stds):
        plt.plot(mean, label=f"{tag} {agent_name}")
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.legend()
    plt.savefig(reward_file)
    plt.clf()
    plt.close()


def plot_final_errors(errors, agent_names, savefile):
    for i, name in enumerate(agent_names):
        agent_errors = errors[i]
        plt.scatter(np.ones_like(agent_errors) * i, agent_errors, label=name)
    plt.legend()
    plt.ylabel("Final map error")
    plt.xticks = []
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    plt.savefig(savefile)


def plot_reward(rewards, agent_name, reward_file):
    x = np.arange(len(rewards), dtype=np.int)
    y = np.array(rewards)
    plt.plot(x, y)
    plt.ylabel("Reward")
    plt.xlabel("Step Number")
    title = f"Performance of {agent_name} Agent"
    plt.title(title)
    plt.savefig(reward_file)

    plt.clf()


def run_trial(
    agent_types,
    policy,
    vis_dir,
    trial_num,
    model_dir,
    safety_max,
    world_size,
    write_video,
    plot,
    _run,
    **kwargs,
):
    if len(agent_types) == 0:
        raise RuntimeError("More than one agent_type required")

    for agent_type_t in agent_types:
        assert agent_type_t in agent_dict

    vis_dirs = []
    model_dirs = []
    reward_files = []
    video_files = []
    gt_map_files = []
    gp_map_dirs = []
    gp_map_full_dirs = []
    for agent_type_t in agent_types:
        vis_dir_agent_pre_trial = os.path.join(vis_dir, agent_type_t)
        if not os.path.exists(vis_dir_agent_pre_trial):
            os.mkdir(vis_dir_agent_pre_trial)

        vis_dir_agent = os.path.join(vis_dir_agent_pre_trial, "trial_" + str(trial_num))
        if not os.path.exists(vis_dir_agent):
            os.mkdir(vis_dir_agent)

        vis_dirs.append(vis_dir_agent)

        model_dirs.append(os.path.join(model_dir, agent_type_t))

        reward_files.append(os.path.join(vis_dir_agent, "reward.png"))
        video_files.append(
            os.path.join(vis_dir_agent, f"{agent_type_t}_{trial_num}_agent.mp4")
        )
        gt_map_files.append(os.path.join(vis_dir_agent, "gt_map.png"))
        gp_map_dirs.append(os.path.join(vis_dir_agent, "gp_maps"))
        gp_map_full_dirs.append(os.path.join(vis_dir_agent, "gp_maps_full"))

    # Merge kwargs and locals appropriately
    kwargs.update(locals())
    kwargs.pop("kwargs")
    info_dict = create_info_dict(**kwargs)

    envs = [None] * len(agent_types)
    envs[0] = gym.make("ipp-v0", info_dict=info_dict)

    dones = [False] * len(agent_types)
    safety_count = 0
    rewards = [None] * len(agent_types)
    errors = [None] * len(agent_types)

    if write_video:
        video_writers = []
        for i in range(len(agent_types)):
            video_writers.append(imageio.get_writer(video_files[i], fps=2))

    obs = [None] * len(agent_types)
    obs[0] = envs[0].reset()

    for i in range(0, len(agent_types)):
        if i > 0:
            envs[i] = copy.deepcopy(envs[0])
            obs[i] = copy.deepcopy(obs[0])

        if plot:
            plot_gt(envs[i], world_size, gt_map_files[i])
            plot_gp(envs[i], world_size, gp_map_dirs[i])
            plot_gp_full(envs[i], gp_map_full_dirs[i])
            # plot_visited(envs[i], action_space_discretization, gp_map_dirs[i])

    agents = []
    for i in range(len(agent_types)):
        agent = agent_dict[agent_types[i]](envs[i])
        agent.policy = policy
        agent.load_model(model_dirs[i])
        agents.append(agent)

    while (np.sum(dones) < len(agent_types)) and (safety_count < safety_max):
        safety_count += 1

        for i in range(len(agent_types)):
            if dones[i]:
                continue

            action, _ = agents[i].get_action(obs[i], envs[i])
            obs[i], reward, dones[i], _ = envs[i].step(action)
            error = envs[i].latest_total_mean_error

            if plot:
                plot_gp(envs[i], world_size, gp_map_dirs[i])
                plot_gp_full(envs[i], gp_map_full_dirs[i])
            # plot_visited(envs[i], action_space_discretization, gp_map_dirs[i])

            if rewards[i] is None:
                rewards[i] = []

            if errors[i] is None:
                errors[i] = []
            rewards[i].append(reward)
            errors[i].append(error)

            if write_video:
                video_img = envs[i].test_gp()
                video_writers[i].append_data(video_img)

    if safety_count == safety_max:
        raise RuntimeError("Safety limit reached")

    for i in range(len(agent_types)):
        plot_gp(envs[i], world_size, vis_dirs[i], filename="gp_final.png")
        plot_gp_full(envs[i], vis_dirs[i], filename="gp_full_final.png")
        # plot_visited(envs[i], action_space_discretization, gp_map_dirs[i], filename="visited_final.png")
        plot_reward(rewards[i], agents[i].get_name(), reward_files[i])
        if write_video:
            video_writers[i].close()
            _run.add_artifact(video_files[i])

        # print(
        #     "Final cost for "
        #     + agent_types[i]
        #     + " is "
        #     + str(envs[i].latest_top_frac_mean_error)
        # )
    return rewards, errors


def train_agent(
    agent_type,
    policy,
    model_dir,
    log_dir,
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    **kwargs,
):

    model_dir = os.path.join(model_dir, agent_type)
    log_dir = os.path.join(log_dir, agent_type)

    # TODO move this into common class
    kwargs.update(locals())
    kwargs.pop("kwargs")
    info_dict = create_info_dict(**kwargs)

    env = gym.make("ipp-v0", info_dict=info_dict)
    agent = agent_dict[agent_type](env.action_space)
    agent.policy = policy

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
    return agent


def test_agents(
    agent_types, num_trials, vis_dir, **kwargs,  # Unused, for compatability
):
    kwargs.update(locals())
    kwargs.pop("kwargs")

    full_rewards = []
    full_errors = []
    for trial_num in range(num_trials):
        rewards, errors = run_trial(trial_num=trial_num, **kwargs)
        full_rewards.append(rewards)
        full_errors.append(errors)
    # TODO rewards may change if not fixed episode length
    # np.save("vis/all_rewards.npy", full_rewards)
    # np.save("vis/all_final_mean_errors.npy", full_final_mean_errors)
    reward_comparison_file = os.path.join(vis_dir, "reward_comparison.png")
    plot_all_rewards(full_rewards, agent_types, reward_comparison_file)

    error_comparison_file = os.path.join(vis_dir, "errors_comparison.png")
    plot_all_rewards(full_errors, agent_types, error_comparison_file, tag="Map error")
    full_rewards = np.array(full_rewards)
    mean_rewards = np.mean(full_rewards, axis=0)
    full_errors = np.stack(full_errors, axis=2)
    final_errors = full_errors[:, -1, :]
    error_scatters_file = os.path.join(vis_dir, "final_error_scatters.png")
    plot_final_errors(final_errors, agent_types, error_scatters_file)
    mean_final_errors = np.mean(final_errors, axis=1)

    final_error_dict = {}
    for i in range(len(agent_types)):
        final_error_dict[agent_types[i]] = mean_final_errors[i]

    with open("vis/mean_erros.json", "w") as f:
        json.dump(final_error_dict, f)

    for i in range(len(agent_types)):
        reward_file = os.path.join(vis_dir, "mean_reward_" + agent_types[i] + ".png")
        plot_reward(mean_rewards[i, :], agent_types[i], reward_file)
