import os
import numpy as np
import imageio
from matplotlib import pyplot as plt
import copy

from ipp_toolkit.agents.StableBaselinesAgent import agent_dict

from sacred import Experiment

import gym

# Needed for env instantiation
import gym_ipp


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


def plot_all_rewards(full_rewards, agent_names, reward_file):
    means = []
    stds = []
    full_rewards = np.array(full_rewards)
    for i in range(full_rewards.shape[1]):
        agent_rewards = full_rewards[:, i, :]
        means.append(np.mean(agent_rewards, axis=0))
        stds.append(np.std(agent_rewards, axis=0))
    for agent_name, mean, std in zip(agent_names, means, stds):
        plt.plot(mean, label=f"Reward {agent_name}")
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.legend()
    plt.savefig(reward_file)
    plt.clf()
    plt.close()


def plot_reward(rewards, agent_name, reward_file):
    x = np.arange(len(rewards), dtype=int)
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
    vis_dir,
    trial_num,
    model_dir,
    n_iters,
    safety_max,
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
    write_video,
    map_seed,
    action_space_discretization,
    # n_gp_fit_iters,
    # gp_lengthscale_prior,
    # gp_lengthscale_var_prior,
    plot,
    world_sample_resolution,
    _run,
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
        video_files.append(os.path.join(vis_dir_agent, "agent.mp4"))
        gt_map_files.append(os.path.join(vis_dir_agent, "gt_map.png"))
        gp_map_dirs.append(os.path.join(vis_dir_agent, "gp_maps"))
        gp_map_full_dirs.append(os.path.join(vis_dir_agent, "gp_maps_full"))

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
    # action space
    info_dict["action_space_discretization"] = action_space_discretization
    # world sample resolution
    info_dict["world_sample_resolution"] = world_sample_resolution
    # GP params
    # info_dict["n_gp_fit_iters"] = n_gp_fit_iters
    # info_dict["gp_lengthscale_prior"] = gp_lengthscale_prior
    # info_dict["gp_lengthscale_var_prior"] = gp_lengthscale_var_prior

    envs = [None] * len(agent_types)
    envs[0] = gym.make("ipp-v0", info_dict=info_dict)

    agents = []
    for i in range(len(agent_types)):
        agent = agent_dict[agent_types[i]](envs[0].action_space)
        agent.load_model(model_dirs[i])
        agents.append(agent)

    dones = [False] * len(agent_types)
    safety_count = 0
    rewards = [None] * len(agent_types)

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

    while (np.sum(dones) < len(agent_types)) and (safety_count < safety_max):
        safety_count += 1

        for i in range(len(agent_types)):
            if dones[i]:
                continue
            action, _ = agents[i].get_action(obs[i])
            obs[i], reward, dones[i], _ = envs[i].step(action)

            if plot:
                plot_gp(envs[i], world_size, gp_map_dirs[i])
                plot_gp_full(envs[i], gp_map_full_dirs[i])
            # plot_visited(envs[i], action_space_discretization, gp_map_dirs[i])

            if rewards[i] is None:
                rewards[i] = []

            rewards[i].append(reward)

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

    return rewards


ex = Experiment("test")


@ex.config
def config():
    agent_types = ["random", "DQN", "PPO"]
    num_trials = 20
    vis_dir = "vis"
    model_dir = "models"
    n_iters = 20
    safety_max = 100
    noise_sdev = 0
    noise_bias = 0
    world_size = (20, 20)
    sensor_size = (1, 1)
    sensor_resolution = 1.0
    obs_clip = 1.0
    obs_gp_mean_scale = 1.0
    obs_gp_std_scale = 1.0
    rew_top_frac_scale = 1.0
    rew_diff_num_visited_scale = 0.0
    write_video = False
    map_seed = None  # Random seed for the map
    action_space_discretization = 7  # Or an int specifying how many samples per axis
    plot=False
    world_sample_resolution=20/(7 - 1e-6) # only used for continous env
    # GP details
    # n_gp_fit_iters = 1

    # gp_lengthscale_prior = 4
    # gp_lengthscale_var_prior = 0.1


@ex.automain
def main(
    agent_types,
    num_trials,
    vis_dir,
    model_dir,
    n_iters,
    safety_max,
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
    write_video,
    map_seed,
    action_space_discretization,
    plot,
    world_sample_resolution,
    # n_gp_fit_iters,
    # gp_lengthscale_prior,
    # gp_lengthscale_var_prior,
    _run,
):
    # full_rewards = np.load("vis/all_rewards.npy")
    # reward_comparison_file = os.path.join(vis_dir, "reward_comparison.png")
    # plot_all_rewards(full_rewards, agent_types, reward_comparison_file)

    full_rewards = []
    for trial_num in range(num_trials):
        rewards = run_trial(
            agent_types,
            vis_dir,
            trial_num,
            model_dir,
            n_iters,
            safety_max,
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
            write_video,
            map_seed,
            action_space_discretization,
            # n_gp_fit_iters,
            # gp_lengthscale_prior,
            # gp_lengthscale_var_prior,
            plot,
            world_sample_resolution,
            _run,
        )

        full_rewards.append(rewards)

    # TODO rewards may change if not fixed episode length
    np.save("vis/all_rewards.npy", full_rewards)
    reward_comparison_file = os.path.join(vis_dir, "reward_comparison.png")
    plot_all_rewards(full_rewards, agent_types, reward_comparison_file)
    full_rewards = np.array(full_rewards)
    mean_rewards = np.mean(full_rewards, axis=0)

    for i in range(len(agent_types)):
        reward_file = os.path.join(vis_dir, "mean_reward_" + agent_types[i] + ".png")
        plot_reward(mean_rewards[i, :], agent_types[i], reward_file)
