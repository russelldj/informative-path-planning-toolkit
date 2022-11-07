import os
import numpy as np
import imageio
from matplotlib import pyplot as plt

from ipp_toolkit.agents.RandomAgent import RandomAgent
from ipp_toolkit.agents.PPOAgent import PPOAgent

from sacred import Experiment

import gym
import gym_ipp

#TODO move to common file
agent_dict = {
                "random": RandomAgent,
                "PPO": PPOAgent
             }

def plot_gt(env, world_size, gt_map_file):
    extent = (0, world_size[1],0, world_size[0])
    gt_map = env.get_gt_map()
    #all values in gt_map should be between 0 and 1
    plt.imshow(gt_map, extent=extent, vmin=0, vmax=1)
    plt.savefig(gt_map_file)

    plt.clf()

def plot_gp(env, world_size, gp_map_dir, filename=None):
    if not os.path.exists(gp_map_dir):
        os.mkdir(gp_map_dir)

    if filename is None:
        filename = f'gp_{env.num_steps}.png'

    gp_map_file = os.path.join(gp_map_dir, filename)

    extent = (0, world_size[1],0, world_size[0])
    gp_map = env.get_gp_map()
    #all values in gt_map should be between 0 and 1
    plt.imshow(gp_map, extent=extent, vmin=0, vmax=1)
    x = env.agent_x
    y = env.agent_y
    plt.plot(x, y, 'r+')
    plt.savefig(gp_map_file)

    plt.clf()

def plot_reward(rewards, agent_name, reward_file):
    x = np.arange(len(rewards), dtype=np.int)
    y = np.array(rewards)
    plt.plot(x, y)
    plt.ylabel('Reward')
    plt.xlabel('Step Number')
    title = f'Performance of {agent_name} Agent'
    plt.title(title)
    plt.savefig(reward_file)

ex = Experiment("test")

@ex.config
def config():
    agent_type = "PPO"
    vis_dir = "vis"
    model_dir = "models"
    n_iters = 64
    safety_max = 100
    noise_sdev = 0
    noise_bias = 0
    world_size = (15, 15)
    sensor_size = (3, 3)
    sensor_resolution = 1.0
    grid_sample_size = (5, 5)
    grid_sample_resolution = 1.0
    movement_max = 1.0
    num_prev_positions = 6
    obs_clip = 5.0
    obs_sensor_scale = 1.0
    obs_gp_mean_scale = 1.0
    obs_gp_std_scale = 1.0
    rew_top_frac_scale = 1.0
    rew_out_of_map_scale = 1.0
    write_video = False

@ex.automain
def main(
         agent_type,
         vis_dir,
         model_dir,
         n_iters, 
         safety_max,
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
         write_video,
         _run
):
    vis_dir = os.path.join(vis_dir, agent_type)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    model_dir = os.path.join(model_dir, agent_type)

    reward_file = os.path.join(vis_dir, 'reward.png')
    video_file = os.path.join(vis_dir, 'agent.mp4')
    gt_map_file = os.path.join(vis_dir, 'gt_map.png')
    gp_map_dir= os.path.join(vis_dir, 'gp_maps')

    #TODO move this into common class
    info_dict = {}
    #world size
    info_dict['world_size'] = world_size
    #sensor noise
    info_dict['noise_sdev'] = noise_sdev
    info_dict['noise_bias'] = noise_bias
    #sensor size
    info_dict['sensor_size'] = sensor_size
    #sensor resolution
    info_dict['sensor_resolution'] = sensor_resolution
    #grid sample size
    info_dict['grid_sample_size'] = grid_sample_size
        #grid sample resolution
    info_dict['grid_sample_resolution'] = grid_sample_resolution
    #starting x and y positions
    info_dict['init_x'] = world_size[1] / 2
    info_dict['init_y'] = world_size[0] / 2
    #movement distance
    info_dict['movement_max'] = movement_max
    #number previous actions
    info_dict['num_prev_positions'] = num_prev_positions
    #max number of steps per episode
    info_dict['max_steps'] = n_iters
    #obs clip and scales
    info_dict['obs_clip'] = obs_clip
    info_dict['obs_sensor_scale'] = obs_sensor_scale
    info_dict['obs_gp_mean_scale'] = obs_gp_mean_scale
    info_dict['obs_gp_std_scale'] = obs_gp_std_scale
    #reward scaling
    info_dict['rew_top_frac_scale'] = rew_top_frac_scale
    info_dict['rew_out_of_map_scale'] = rew_out_of_map_scale

    env = gym.make('ipp-v0', info_dict=info_dict)
    agent = agent_dict[agent_type](env.action_space)
    agent.load_model(model_dir)

    done = False
    safety_count = 0
    rewards = []
    if write_video:
        video_writer = imageio.get_writer(video_file, fps=2)

    obs = env.reset()
    plot_gt(env, world_size, gt_map_file)
    plot_gp(env, world_size, gp_map_dir)

    while ((not done) and (safety_count < safety_max)):
        safety_count += 1

        action, _ = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)

        plot_gp(env, world_size, gp_map_dir)

        rewards.append(reward)

        if write_video:
            video_img = env.test_gp()
            video_writer.append_data(video_img)

    if safety_count == safety_max:
        raise RuntimeError('Safety limit reached')

    plot_gp(env, world_size, vis_dir, filename='gp_final.png')
    plot_reward(rewards, agent.get_name(), reward_file)

    if write_video:
        video_writer.close()
        _run.add_artifact(video_file)