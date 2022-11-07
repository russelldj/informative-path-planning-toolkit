import gym
import numpy as np

from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import GaussianProcessRegressionWorldModel
from ipp_toolkit.config import MEAN_KEY, VARIANCE_KEY, TOP_FRAC_MEAN_ERROR

class IppEnv(gym.Env):
    def __init__(self, info_dict):
        super(IppEnv, self).__init__()

        #custom args
        world_size = info_dict['world_size']
        noise_sdev = info_dict['noise_sdev']
        noise_bias = info_dict['noise_bias']
        init_x = info_dict['init_x']
        init_y = info_dict['init_y']
        max_steps = info_dict['max_steps']
        movement_scale = info_dict['movement_scale']
        grid_size = info_dict['grid_size']
        grid_scale = info_dict['grid_scale']

        #y, x, mean, var
        self.observation_shape = (grid_size[0]*grid_size[1], 4)
        self.observation_space = gym.spaces.Box(low = -np.ones(self.observation_shape)*np.inf, 
                                            high = np.ones(self.observation_shape)*np.inf,
                                            dtype = np.float64)

        #N, NE, E, SE, S, SW, W, NW; always sample
        self.action_space = gym.spaces.Discrete(8,)

        assert max_steps > 0
        assert movement_scale > 0
        assert init_x >= 0
        assert init_y >= 0
        assert init_x <= world_size[1]
        assert init_y <= world_size[0]

        data = RandomGaussian2D(world_size=world_size)
        self.sensor = GaussianNoisyPointSensor(
            data, noise_sdev=noise_sdev, noise_bias=noise_bias
        )

        self.init_x = init_x
        self.init_y = init_y
        self.max_steps = max_steps
        self.movement_scale = movement_scale
        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.world_size = world_size
        self.data = data
        self.noise_sdev = noise_sdev
        self.noise_bias = noise_bias

    def reset(self):
        self.agent_x = self.init_x
        self.agent_y = self.init_y
        self.num_steps = 0

        self.gp = GaussianProcessRegressionWorldModel()

        data = RandomGaussian2D(world_size=self.world_size)
        self.sensor = GaussianNoisyPointSensor(
            data, noise_sdev=self.noise_sdev, noise_bias=self.noise_bias
        )

        self._make_observation()
        self._get_info()
        
        return self.latest_observation

    def step(self, action):
        #N
        if action == 0:
            self.agent_y = np.min([self.world_size[0], self.agent_y + self.movement_scale])
        #NE
        elif action == 1:
            self.agent_y = np.min([self.world_size[0], self.agent_y + self.movement_scale])
            self.agent_x = np.min([self.world_size[1], self.agent_x + self.movement_scale])
        #E
        elif action == 2:
            self.agent_x = np.min([self.world_size[1], self.agent_x + self.movement_scale])
        #SE
        elif action == 3:
            self.agent_y = np.max([0, self.agent_y - self.movement_scale])
            self.agent_x = np.min([self.world_size[1], self.agent_x + self.movement_scale])
        #S
        elif action == 4:
            self.agent_y = np.max([0, self.agent_y - self.movement_scale])
        #SW
        elif action == 5:
            self.agent_y = np.max([0, self.agent_y - self.movement_scale])
            self.agent_x = np.max([0, self.agent_x - self.movement_scale])
        #W
        elif action == 6:
            self.agent_x = np.max([0, self.agent_x - self.movement_scale])
        #NW
        elif action == 7:
            self.agent_y = np.min([self.world_size[0], self.agent_y + self.movement_scale])
            self.agent_x = np.max([0, self.agent_x - self.movement_scale])
        else:
            raise RuntimeError('Illegal action: ' + str(action))

        self.num_steps += 1

        done = (self.num_steps >= self.max_steps)

        prev_top_frac_mean_error = self.latest_top_frac_mean_error
        self._make_observation()
        obs = self.latest_observation
        new_top_frac_mean_error = self.latest_top_frac_mean_error
        reward = -(new_top_frac_mean_error - prev_top_frac_mean_error)

        self._get_info()
        info = self.latest_info

        return obs, reward, done, info

    def render(self):
        pass

    def _make_observation(self):
        x = self.agent_x
        y = self.agent_y
        
        value = self.sensor.sample((y, x))

        self.gp.add_observation((y, x), value)
        self.gp.train_model()
        
        x_vals = np.arange(-self.grid_size[1] / 2.0 + 0.5, self.grid_size[1] / 2.0)
        y_vals = np.arange(-self.grid_size[0] / 2.0 + 0.5, self.grid_size[0] / 2.0)

        x_vals = x_vals * self.grid_scale
        y_vals = y_vals * self.grid_scale

        x_vals = x_vals + x
        y_vals = y_vals + y

        # ignoring for now
        # x_vals = x_vals[x_vals <= self.world_size[1]]
        # x_vals = x_vals[x_vals >= 0]
        # y_vals = y_vals[y_vals <= self.world_size[0]]
        # y_vals = y_vals[y_vals >= 0]

        grid_points = []
        for y_val in y_vals:
            for x_val in x_vals:
                grid_points.append([y_val, x_val])
        
        grid_points = np.array(grid_points)

        gp_dict = self.gp.sample_belief_array(grid_points)
        mean = gp_dict[MEAN_KEY]
        var = gp_dict[VARIANCE_KEY]

        #only works because not filtering outside of world
        latest_observation = np.zeros((grid_points.shape[0], 4))
        latest_observation[:, 0] = grid_points[:, 0]
        latest_observation[:, 1] = grid_points[:, 1]
        latest_observation[:, 2] = mean
        latest_observation[:, 3] = var

        self.latest_observation = latest_observation

        # get top frac mean error
        eval_dict = self.gp.evaluate_metrics(self.data.map, world_size=self.world_size)
        self.latest_top_frac_mean_error = eval_dict[TOP_FRAC_MEAN_ERROR]

    def _get_info(self):
        info = {}
        self.latest_info = info

    def test_gp(self):
        img = self.gp.test_model(world_size=self.world_size, gt_data=self.data.map)
        return img

