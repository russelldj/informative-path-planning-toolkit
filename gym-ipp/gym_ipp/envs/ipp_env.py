import gym
import numpy as np

from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)
from ipp_toolkit.config import (
    MEAN_KEY,
    VARIANCE_KEY,
    TOP_FRAC_MEAN_ERROR,
    MEAN_ERROR_KEY,
)
from ipp_toolkit.utils.sampling import get_flat_samples


def get_grid_delta(size, resolution):
    delta = (
        np.vstack(
            [
                s.flatten()
                for s in np.meshgrid(
                    np.arange(size[0]), np.arange(size[1]), indexing="ij"
                )
            ]
        )
        .astype(float)
        .T
    )

    delta -= (np.array([size[0], size[1]]) - 1) / 2
    delta *= resolution

    return delta


# TODO hardcode something if out of bounds for sampling?
class IppEnv(gym.Env):
    def __init__(self, info_dict):
        super(IppEnv, self).__init__()

        # custom args
        # world size, tuple of (y, x)
        self.world_size = info_dict["world_size"]
        # sensor noise
        self.noise_sdev = info_dict["noise_sdev"]
        self.noise_bias = info_dict["noise_bias"]
        # sensor size, tuple of (y, x)
        self.sensor_size = info_dict["sensor_size"]
        # sensor resolution
        self.sensor_resolution = info_dict["sensor_resolution"]
        # world sample resolution
        self.world_sample_resolution = info_dict["world_sample_resolution"]
        # starting x and y positions #TODO can make random
        self.init_x = info_dict["init_x"]
        self.init_y = info_dict["init_y"]
        # max number of steps per episode
        self.max_steps = info_dict["max_steps"]
        # observation clipping and scaling
        self.obs_clip = info_dict["obs_clip"]
        self.obs_gp_mean_scale = info_dict["obs_gp_mean_scale"]
        self.obs_gp_std_scale = info_dict["obs_gp_std_scale"]
        # reward scaling
        self.rew_top_frac_scale = info_dict["rew_top_frac_scale"]
        # map determinism
        self.map_seed = info_dict["map_seed"]
        # action_space
        self.action_space_discretization = info_dict["action_space_discretization"]
        # gaussian process
        #self.n_gp_fit_iters = info_dict["n_gp_fit_iters"]
        #self.gp_lengthscale_prior = info_dict["gp_lengthscale_prior"]
        #self.gp_lengthscale_var_prior = info_dict["gp_lengthscale_var_prior"]
        # make sure values are legal
        assert self.max_steps > 0
        assert self.init_y >= 0
        assert self.init_x >= 0
        assert self.init_y <= self.world_size[0]
        assert self.init_x <= self.world_size[1]

        self.sensor_delta = get_grid_delta(self.sensor_size, self.sensor_resolution)

        self.world_sample_points, self.world_sample_points_size = get_flat_samples(
            self.world_size, self.world_sample_resolution
        )

        #Discretizing action space now
        assert self.action_space_discretization is not None

        # observation consists of:
        # gp predictions mean and var
        # TODO what dim order for CNN?
        self.observation_shape = (
            self.action_space_discretization**2,
        )
        self.observation_space = gym.spaces.Box(
            low=np.ones(self.observation_shape, dtype=np.float32) * -1.0,
            high=np.ones(self.observation_shape, dtype=np.float32) * 1.0,
            dtype=np.float32,
        )

        # actions consist of normalized y and x positions (not movement)
        if self.action_space_discretization is None:
            self.action_space = gym.spaces.Box(
                low=np.ones(2, dtype=np.float32) * -1.0,
                high=np.ones(2, dtype=np.float32),
            )
        else:
            self.action_space = gym.spaces.Discrete(
                self.action_space_discretization**2
            )

    def reset(self):
        self.agent_x = self.init_x
        self.agent_y = self.init_y
        self.num_steps = 0
        # print(f"Mean error on reset {self.latest_top_frac_mean_error}")
        # self.gp = GaussianProcessRegressionWorldModel(
        #     training_iters=self.n_gp_fit_iters,
        #     lengthscale=self.gp_lengthscale_prior,
        #     lengthscale_std=self.gp_lengthscale_var_prior,
        # )
        self.data = RandomGaussian2D(
            world_size=self.world_size, random_seed=self.map_seed
        )
        self.sensor = GaussianNoisyPointSensor(
            self.data, noise_sdev=self.noise_sdev, noise_bias=self.noise_bias
        )

        self.visited = np.zeros((self.action_space_discretization, self.action_space_discretization)) - 1

        self._make_observation()
        self._get_reward_metrics()
        self._get_info()

        return self.latest_observation

    def step(self, action):
        # Continous action space
        if self.action_space_discretization is None:
            y, x = action
        else:
            unscaled_x = action % self.action_space_discretization
            unscaled_y = action // self.action_space_discretization
            x, y = [  # check the order of these
                un / self.action_space_discretization
                + 1
                / (2 * self.action_space_discretization)  # Shift to centered intervals
                for un in (unscaled_x, unscaled_y)
            ]
            x = 2 * x - 1
            y = 2 * y - 1

        # x,y are in the range (0,1)
        self.agent_y = (y + 1) / 2 * self.world_size[0]
        self.agent_x = (x + 1) / 2 * self.world_size[1]


        self.visited[unscaled_y, unscaled_x] = 1.0

        self.num_steps += 1

        done = self.num_steps >= self.max_steps

        self._make_observation()
        obs = self.latest_observation

        # prev_top_frac_mean_error = self.latest_top_frac_mean_error
        # self._get_reward_metrics()
        # curr_top_frac_mean_error = self.latest_top_frac_mean_error
        # diff_top_frac_mean_error = curr_top_frac_mean_error - prev_top_frac_mean_error
        # curr_total_mean_error = self.latest_total_mean_error

        prev_num_visited = self.num_visited
        self._get_reward_metrics()
        curr_num_visited = self.num_visited
        diff_num_visited = curr_num_visited - prev_num_visited

        #rew_top_frac = diff_num_visited * self.rew_top_frac_scale

        reward = diff_num_visited

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

        #self.gp.add_observation(sensor_pos_to_sample, sensor_values, unsqueeze=False)
        #self.gp.train_model()

        #gp_dict = self.gp.sample_belief_array(self.world_sample_points)
        #mean = np.reshape(gp_dict[MEAN_KEY], self.world_sample_points_size)
        #var = np.reshape(gp_dict[VARIANCE_KEY], self.world_sample_points_size)

        # obs = np.stack(
        #     (mean * self.obs_gp_mean_scale, var * self.obs_gp_std_scale), axis=0
        # ).astype(np.float32)

        obs = self.visited.flatten()

        # clip observations
        #obs = np.clip(obs, 0, self.obs_clip)

        #obs = (255 * obs).astype(np.uint8)

        self.latest_observation = obs

    def _get_info(self):
        info = {}
        self.latest_info = info

    def _get_reward_metrics(self):
        #eval_dict = self.gp.evaluate_metrics(self.data.map, world_size=self.world_size)
        #self.latest_top_frac_mean_error = eval_dict[TOP_FRAC_MEAN_ERROR]
        #self.latest_total_mean_error = eval_dict[MEAN_ERROR_KEY]
        self.num_visited = (self.visited + 1).sum() / 2

    # def get_gt_map(self):
    #     return self.data.map

    # def get_gp_map(self):
    #     gp_dict = self.gp.sample_belief_array(self.data.samples)
    #     mean = gp_dict[MEAN_KEY]

    #     gp_map = np.reshape(mean, self.data.map.shape)
    #     return gp_map

    # def test_gp(self):
    #     img = self.gp.test_model(world_size=self.world_size, gt_data=self.data.map)
    #     return img

    def get_visited_map(self):        
        img = (255 * (self.visited + 1) / 2).astype(np.uint8)
        return img
