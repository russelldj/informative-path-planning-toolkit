import gym
import numpy as np

from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.grid_regression import GridWorldModel
from ipp_toolkit.world_models.interpolation import InterpolationWorldModel

# from ipp_toolkit.world_models.gaussian_process_regression import (
#    GaussianProcessRegressionWorldModel,
# )
from ipp_toolkit.config import (
    MEAN_KEY,
    UNCERTAINTY_KEY,
    TOP_FRAC_MEAN_ERROR,
    MEAN_ERROR_KEY,
)
from ipp_toolkit.utils.sampling import get_flat_samples

step_dict = {
    0: np.array([-1, 0]),
    1: np.array([1, 0]),
    2: np.array([0, -1]),
    3: np.array([0, 1]),
}


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
        # TODO hacky for now mbagent
        self.info_dict = info_dict

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
        self.rew_diff_num_visited_scale = info_dict["rew_diff_num_visited_scale"]
        # map determinism
        self.map_seed = info_dict["map_seed"]
        # action_space
        self.action_space_discretization = info_dict["action_space_discretization"]
        #
        self.observation_space_discretization = info_dict[
            "observation_space_discretization"
        ]
        # How to encode observations
        self.cnn_encoding = info_dict["cnn_encoding"]
        #
        self.move_on_grid = info_dict["move_on_grid"]
        assert not (self.move_on_grid and (self.action_space_discretization is None))
        self.map_lower_offset = info_dict["map_lower_offset"]
        self.use_interpolation_model = info_dict["use_interpolation_model"]

        # make sure values are legal
        assert self.max_steps > 0
        assert self.init_y >= 0
        assert self.init_x >= 0
        assert self.init_y <= self.world_size[0]
        assert self.init_x <= self.world_size[1]

        self.sensor_delta = get_grid_delta(self.sensor_size, self.sensor_resolution)

        self.location = np.zeros(2)

        # Currently we assume a square world, but this could be relaxed
        assert self.world_size[0] == self.world_size[1]
        # Chose how to discretize the action space

        # Calculate observation space
        if self.cnn_encoding == "CnnPolicy":
            self.observation_shape = (
                2,
                self.observation_space_discretization,
                self.observation_space_discretization,
            )
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.observation_shape, dtype=np.uint8,
            )
        else:
            self.cnn_encoding = None
            self.observation_shape = (2 * self.observation_space_discretization ** 2,)
            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=self.observation_shape, dtype=np.float32,
            )

        self.observation_sample_resolution = np.array(self.world_size[0]) / (
            self.observation_space_discretization - 1e-6
        )
        self.world_sample_points, self.world_sample_points_size = get_flat_samples(
            self.world_size, self.observation_sample_resolution
        )

        # actions consist of normalized y and x positions (not movement)
        if self.move_on_grid:
            self.action_space = gym.spaces.Discrete(4)
        elif self.action_space_discretization is None:
            self.action_space = gym.spaces.Box(
                low=np.ones(2, dtype=np.float32) * -1.0,
                high=np.ones(2, dtype=np.float32),
            )
        else:
            self.action_space = gym.spaces.Discrete(
                self.action_space_discretization ** 2
            )

    def reset(self):
        self.agent_x = self.init_x
        self.agent_y = self.init_y
        self.num_steps = 0
        # print(f"Mean error on reset {self.latest_top_frac_mean_error}")

        if self.use_interpolation_model:
            self.gp = InterpolationWorldModel(
                world_size=self.world_size,
                grid_cell_size=self.observation_sample_resolution,
                uncertainty_scale=0.1,
            )
        else:
            self.gp = GridWorldModel(
                world_size=self.world_size,
                grid_cell_size=self.observation_sample_resolution,
            )

        self.data = RandomGaussian2D(
            world_size=self.world_size,
            random_seed=self.map_seed,
            lower_offset=self.map_lower_offset,
        )

        self.sensor = GaussianNoisyPointSensor(
            self.data, noise_sdev=self.noise_sdev, noise_bias=self.noise_bias
        )

        # self._draw_random_samples(3)
        self._make_observation()
        self._get_reward_metrics()
        self._get_info()

        return self.latest_observation

    def step(self, action):
        # Continous action space
        if self.move_on_grid:
            movement = step_dict[action]
            step_size = 2 * self.world_sample_resolution / np.array(self.world_size)
            self.location += movement * step_size
            self.location = np.clip(self.location, -1, 1)
            x, y = self.location
        elif self.action_space_discretization is None:
            y, x = action
        else:
            unscaled_x = action % self.action_space_discretization
            unscaled_y = action // self.action_space_discretization
            x, y = [  # check the order of these
                2
                * (
                    un / self.action_space_discretization
                    + 1
                    / (
                        2 * self.action_space_discretization
                    )  # Shift to the center of the square
                )
                - 1  # Shift from index to to centered intervals -1,1
                for un in (unscaled_x, unscaled_y)
            ]

        # x,y are in the range (0,1)
        self.agent_y = (y + 1) / 2 * self.world_size[0]
        self.agent_x = (x + 1) / 2 * self.world_size[1]

        self.num_steps += 1

        done = self.num_steps >= self.max_steps

        self._make_observation()
        obs = self.latest_observation

        prev_top_total_mean_error = self.latest_total_mean_error
        prev_num_visited = self.num_visited
        self._get_reward_metrics()
        curr_top_total_mean_error = self.latest_total_mean_error
        curr_num_visited = self.num_visited

        diff_top_total_mean_error = (
            curr_top_total_mean_error - prev_top_total_mean_error
        )
        diff_num_visited = curr_num_visited - prev_num_visited

        rew_top_frac = -diff_top_total_mean_error * self.rew_top_frac_scale
        rew_num_visited = diff_num_visited * self.rew_diff_num_visited_scale

        reward = rew_top_frac + rew_num_visited

        self._get_info()
        info = self.latest_info

        return obs, reward, done, info

    def render(self):
        pass

    def _draw_random_samples(self, n_samples):
        sensor_pos_to_sample = np.vstack(
            (
                np.random.uniform(0, self.world_size[0], size=n_samples),
                np.random.uniform(0, self.world_size[1], size=n_samples),
            )
        ).T
        sensor_values = self.sensor.sample(sensor_pos_to_sample.T)
        self.gp.add_observation(sensor_pos_to_sample, sensor_values)

    def _make_observation(self):
        x = self.agent_x
        y = self.agent_y

        sensor_pos_to_sample = self.sensor_delta + [y, x]
        sensor_values = self.sensor.sample(sensor_pos_to_sample.T)

        # self.gp.add_observation(sensor_pos_to_sample, sensor_values, unsqueeze=False)
        # self.gp.train_model()
        self.gp.add_observation(sensor_pos_to_sample, sensor_values)

        gp_dict = self.gp.sample_belief_array(self.world_sample_points)
        mean = np.reshape(gp_dict[MEAN_KEY], self.world_sample_points_size)
        var = np.reshape(gp_dict[UNCERTAINTY_KEY], self.world_sample_points_size)

        self.latest_var = var

        obs = np.stack(
            (mean * self.obs_gp_mean_scale, var * self.obs_gp_std_scale,), axis=0,
        ).astype(np.float32)

        obs = np.clip(obs, 0, self.obs_clip)
        if not self.cnn_encoding:
            obs = obs * 2 - 1
            obs = obs.flatten()
            # clip observations
        else:
            obs = (obs * 255).astype(np.uint8)
            # import matplotlib.pyplot as plt

            # fig, axs = plt.subplots(1, 2)
            # plt.colorbar(axs[0].imshow(obs[0]), ax=axs[0])
            # plt.colorbar(axs[1].imshow(obs[1]), ax=axs[1])
            # plt.show()

        # commenting this out as asserts are slow aren't they?
        if obs.shape != self.observation_shape:
            raise RuntimeError("obs shape does not match")

        self.latest_observation = obs

    def _get_info(self):
        info = {}
        self.latest_info = info

    def _get_reward_metrics(self):
        eval_dict = self.gp.evaluate_metrics(self.data.map, world_size=self.world_size)
        self.latest_top_frac_mean_error = eval_dict[TOP_FRAC_MEAN_ERROR]
        self.latest_total_mean_error = eval_dict[MEAN_ERROR_KEY]
        self.num_visited = -(self.latest_var - 1).sum()

    def get_est_reward(self, observations):
        mean = observations[
            :, 0 : self.world_sample_points_size[0] * self.world_sample_points_size[1]
        ]
        var = observations[
            :, self.world_sample_points_size[0] * self.world_sample_points_size[1] :
        ]

        est_reward = mean - 2 * var

        return est_reward.mean(axis=1)

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
