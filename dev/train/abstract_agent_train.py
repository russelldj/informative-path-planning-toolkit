import os

from ipp_toolkit.agents.RandomAgent import RandomAgent
from ipp_toolkit.agents.PPOAgent import PPOAgent
from ipp_toolkit.agents.PPOLSTMAgent import PPOLSTMAgent

from sacred import Experiment

import gym
import gym_ipp

#TODO move to common file
agent_dict = {
                "random": RandomAgent,
                "PPO": PPOAgent
             }

def build_train_cfg(num_par, learning_rate, n_steps, total_timesteps, verbose, save_freq, model_dir, log_dir):
    cfg = {
            'num_par': num_par,
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'total_timesteps': total_timesteps,
            'verbose': verbose,
            'save_freq': save_freq,
            'model_dir': model_dir,
            'log_dir': log_dir
          }

    return cfg

ex = Experiment("test")

@ex.config
def config():
    agent_type = "PPO"
    model_dir = "models"
    log_dir = "logs"
    n_iters = 64
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
    #training details
    num_par = 1
    learning_rate = 1e-3
    n_steps = 512
    total_timesteps = 8000
    verbose = 1
    save_freq = 1024

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
         num_par,
         learning_rate,
         n_steps,
         total_timesteps,
         verbose,
         save_freq,
         _run
):

    model_dir = os.path.join(model_dir, agent_type)
    log_dir = os.path.join(log_dir, agent_type)

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

    cfg = build_train_cfg(num_par, learning_rate, n_steps, total_timesteps, verbose, save_freq, model_dir, log_dir)

    agent.train(env, cfg)
