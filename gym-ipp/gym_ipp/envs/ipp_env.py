import gym
import numpy as np

from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import GaussianProcessRegressionWorldModel

class IppEnv(gym.Env):
    def __init__(self, info_dict):
        super(IppEnv, self).__init__()

        #x, y, value, did_observe
        self.observation_shape = (4,)
        self.observation_space = gym.spaces.Box(low = -np.ones(self.observation_shape)*np.inf, 
                                            high = np.ones(self.observation_shape)*np.inf,
                                            dtype = np.float64)

        #up, down, left, right, sample
        self.action_space = gym.spaces.Discrete(5,)

        #custom args
        world_size = info_dict['world_size']
        noise_sdev = info_dict['noise_sdev']
        noise_bias = info_dict['noise_bias']
        init_x = info_dict['init_x']
        init_y = info_dict['init_y']
        max_steps = info_dict['max_steps']
        movement_scale = info_dict['movement_scale']

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
        self.world_size = world_size
        self.data = data

    def reset(self):
        self.agent_x = self.init_x
        self.agent_y = self.init_y
        self.num_steps = 0

        self._get_observation(False)
        self._get_info()
        
        return self.latest_observation, self.latest_info

    def step(self, action):
        should_sample = False
        if action == 0:
            self.agent_y = np.min([self.world_size[0], self.agent_y + self.movement_scale])
        elif action == 1:
            self.agent_y = np.max([0, self.agent_y - self.movement_scale])
        elif action == 2:
            self.agent_x = np.min([self.world_size[1], self.agent_x + self.movement_scale])
        elif action == 3:
            self.agent_x = np.max([0, self.agent_x - self.movement_scale])
        elif action == 4:
            should_sample = True
        else:
            raise RuntimeError('Illegal action: ' + str(action))

        self.num_steps += 1

        done = (self.num_steps >= self.max_steps)
        #how we do this
        reward = 0

        self._get_observation(should_sample)
        obs = self.latest_observation

        self._get_info()
        info = self.latest_info

        return obs, reward, done, info

    def render(self):
        pass

    def get_gt(self):
        return self.data.map

    def _get_observation(self, should_sample):
        x = self.agent_x
        y = self.agent_y
        if should_sample:
            value = self.sensor.sample((y, x))
            sample_val = 1
        else:  
            value = 0
            sample_val = 0

        self.latest_observation = np.array([x, y, value, sample_val])

    def _get_info(self):
        info = {}
        self.latest_info = info
