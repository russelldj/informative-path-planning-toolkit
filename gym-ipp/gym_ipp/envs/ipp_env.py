import gym
import numpy as np

from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import GaussianProcessRegressionWorldModel
from ipp_toolkit.config import MEAN_KEY, VARIANCE_KEY, TOP_FRAC_MEAN_ERROR

def get_grid_delta(size, resolution):
    delta = np.vstack(s.flatten() for s in np.meshgrid(np.arange(size[0]), 
                                                           np.arange(size[1]), indexing='ij')).astype(float).T
    delta -= (np.array([size[0], size[1]]) - 1) / 2
    delta *= resolution

    return delta

#TODO hardcode something if out of bounds for sampling?
class IppEnv(gym.Env):
    def __init__(self, info_dict):
        super(IppEnv, self).__init__()

        #custom args
        #world size, tuple of (y, x)
        self.world_size = info_dict['world_size']
        #sensor noise
        self.noise_sdev = info_dict['noise_sdev']
        self.noise_bias = info_dict['noise_bias']
        #sensor size, tuple of (y, x)
        self.sensor_size = info_dict['sensor_size']
        #sensor resolution
        self.sensor_resolution = info_dict['sensor_resolution']
        #grid sample size
        self.grid_sample_size = info_dict['grid_sample_size']
        #grid sample resolution
        self.grid_sample_resolution = info_dict['grid_sample_resolution']
        #number previous actions
        self.num_prev_positions = info_dict['num_prev_positions']
        #starting x and y positions #TODO can make random
        self.init_x = info_dict['init_x']
        self.init_y = info_dict['init_y']
        #movement distance
        self.movement_max = info_dict['movement_max']
        #max number of steps per episode
        self.max_steps = info_dict['max_steps']
        #observation clipping and scaling 
        self.obs_clip = info_dict['obs_clip']
        self.obs_sensor_scale = info_dict['obs_sensor_scale']
        self.obs_gp_mean_scale = info_dict['obs_gp_mean_scale']
        self.obs_gp_std_scale = info_dict['obs_gp_std_scale']
        #reward scaling
        self.rew_top_frac_scale = info_dict['rew_top_frac_scale']
        self.rew_out_of_map_scale = info_dict['rew_out_of_map_scale']

        #make sure values are legal
        assert self.max_steps > 0
        assert self.init_y >= 0
        assert self.init_x >= 0
        assert self.init_y <= self.world_size[0]
        assert self.init_x <= self.world_size[1]

        self.sensor_delta = get_grid_delta(self.sensor_size, self.sensor_resolution)
        self.grid_sample_delta = get_grid_delta(self.grid_sample_size, self.grid_sample_resolution)

        #observation consists of:
        #sensor measurements
        num_sensors = self.sensor_size[0]*self.sensor_size[1]
        #gp predictions mean and var
        num_gp_pred = 2*self.grid_sample_size[0]*self.grid_sample_size[1]
        #current position
        num_curr_pos = 2
        #last_n_actions
        num_prev_positions = 2*self.num_prev_positions
        
        obs_size = num_sensors + num_gp_pred + num_curr_pos + num_prev_positions
        self.observation_shape = (obs_size,)
        self.observation_space = gym.spaces.Box(low=np.ones(self.observation_shape)*-np.Inf, 
                                            high=np.ones(self.observation_shape)*np.Inf)
        
        #actions consist of normalized y and x positions (not movement)
        self.action_space = gym.spaces.Box(low=np.ones(2) * 0., high=np.ones(2) * 1.)

    def reset(self):
        self.agent_x = self.init_x
        self.agent_y = self.init_y
        self.num_steps = 0
        self.prev_positions = np.zeros((self.num_prev_positions, 2)) - 1

        self.gp = GaussianProcessRegressionWorldModel()
        self.data = RandomGaussian2D(world_size=self.world_size)
        self.sensor = GaussianNoisyPointSensor(
            self.data, noise_sdev=self.noise_sdev, noise_bias=self.noise_bias
        )

        self._make_observation()
        self._get_reward_metrics()
        self._get_info()
        
        return self.latest_observation

    def step(self, action):
        movement_dist = self.movement_max * action[0]
        movement_angle = 2*np.pi*action[1]

        #TODO should we allow it to leave the world?
        self.agent_x += movement_dist * np.cos(movement_angle)
        self.agent_y += movement_dist * np.sin(movement_angle)

        self.num_steps += 1

        done = (self.num_steps >= self.max_steps)

        self._make_observation()
        obs = self.latest_observation

        self._get_reward_metrics()
        rew_top_frac = -(self.latest_top_frac_mean_error*self.rew_top_frac_scale)
        rew_out_of_map = -(self.out_of_map_error*self.rew_out_of_map_scale)

        reward = rew_top_frac + rew_out_of_map

        self._get_info()
        info = self.latest_info

        return obs, reward, done, info

    def render(self):
        pass

    def _make_observation(self):
        x = self.agent_x
        y = self.agent_y
        
        sensor_pos_to_sample = self.sensor_delta + [y, x]
        sensor_values = self.sensor.sample(sensor_pos_to_sample.T)

        self.gp.add_observation(sensor_pos_to_sample, sensor_values, unsqueeze=False)
        #self.gp.add_observation((y, x), self.sensor.sample((y, x)))
        self.gp.train_model()
        
        gp_pose_to_sample = self.grid_sample_delta + [y, x]

        gp_dict = self.gp.sample_belief_array(gp_pose_to_sample)
        mean = gp_dict[MEAN_KEY]
        var = gp_dict[VARIANCE_KEY]

        #get observations and scale
        sensor_obs = sensor_values.flatten() * self.obs_sensor_scale
        gp_mean_obs = mean.flatten() * self.obs_gp_mean_scale
        gp_var_obs = var.flatten() * self.obs_gp_std_scale

        curr_pos_scale = np.array([[y, x]])
        curr_pos_scale[:, 0] /= self.world_size[0]
        curr_pos_scale[:, 1] /= self.world_size[1]
        curr_pos_scale = curr_pos_scale.flatten()
        prev_pos_scale = self.prev_positions.copy()
        prev_pos_scale[:, 0] /= self.world_size[0]
        prev_pos_scale[:, 1] /= self.world_size[1]
        prev_pos_scale = prev_pos_scale.flatten()

        obs = np.concatenate((sensor_obs, 
                              gp_mean_obs, 
                              gp_var_obs,
                              curr_pos_scale,
                              prev_pos_scale))

        #clip observations
        obs = np.clip(obs, -self.obs_clip, self.obs_clip)

        self.latest_observation = obs

    def _get_info(self):
        info = {}
        self.latest_info = info

    def _get_reward_metrics(self):
        eval_dict = self.gp.evaluate_metrics(self.data.map, world_size=self.world_size)
        self.latest_top_frac_mean_error = eval_dict[TOP_FRAC_MEAN_ERROR]

        if self.agent_y < 0:
            y_out = self.agent_y
        elif self.agent_y > self.world_size[0]:
            y_out = self.agent_y - self.world_size[0]
        else:
            y_out = 0.0

        if self.agent_x < 0:
            x_out = self.agent_x
        elif self.agent_x > self.world_size[1]:
            x_out = self.agent_x - self.world_size[1]
        else:
            x_out = 0.0

        self.out_of_map_error = np.sum(np.square([y_out, x_out]))

    def get_gt_map(self):
        return self.data.map

    def get_gp_map(self):
        gp_dict = self.gp.sample_belief_array(self.data.samples)
        mean = gp_dict[MEAN_KEY]

        gp_map = np.reshape(mean, self.data.map.shape)
        return gp_map

    def test_gp(self):
        img = self.gp.test_model(world_size=self.world_size, gt_data=self.data.map)
        return img
