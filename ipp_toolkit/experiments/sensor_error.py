from ipp_toolkit.config import MEAN_KEY
from ipp_toolkit.data.random_2d import RandomGaussianProcess2D
import matplotlib.pyplot as plt
from ipp_toolkit.planners.samplers import HighestUpperBoundStochasticPlanner

from ipp_toolkit.sensors.sensors import GaussianNoisyPointSensor
from ipp_toolkit.world_models.gaussian_process_regression import (
    GaussianProcessRegressionWorldModel,
)
import numpy as np
import imageio


class SensorErrorExperiments:
    def __init__(self, world_size, n_points, overlap_inds, noise_sdev=0):
        self.world_size = world_size
        random_seed = np.random.randint(1000)
        print(random_seed)

        self.groundtruth_data = RandomGaussianProcess2D(
            world_size=world_size,
            n_points=n_points,
            overlap_ind=0,
            random_seed=random_seed,
        )
        self.noisy_data = RandomGaussianProcess2D(
            world_size=world_size,
            n_points=n_points,
            overlap_ind=overlap_inds,
            random_seed=random_seed,
        )
        self.groundtruth_sensor = GaussianNoisyPointSensor(
            self.groundtruth_data, noise_sdev=0
        )
        self.noisy_sensor = GaussianNoisyPointSensor(
            self.noisy_data, noise_sdev=noise_sdev
        )

        self.noisy_model = GaussianProcessRegressionWorldModel()
        self.error_model = GaussianProcessRegressionWorldModel()

        self.noisy_planner = HighestUpperBoundStochasticPlanner(
            (0, 0), grid_end=world_size
        )
        self.error_planner = HighestUpperBoundStochasticPlanner(
            (0, 0), grid_end=world_size
        )

        self.error = self.noisy_data.map - self.groundtruth_data.map
        if True:

            fig, axs = plt.subplots(1, 3)
            cb_0 = axs[0].imshow(self.groundtruth_data.map)
            cb_1 = axs[1].imshow(self.noisy_data.map)
            cb_2 = axs[2].imshow(self.error)
            plt.colorbar(cb_0, ax=axs[0])
            plt.colorbar(cb_1, ax=axs[1])
            plt.colorbar(cb_2, ax=axs[2])
            plt.show()

    def run(
        self, initial_point=(0, 0), video_file=None, error_video_file=None, _run=None
    ):
        # Initialize model
        for i in range(20):
            x = np.hstack(
                (
                    np.random.uniform(0, self.world_size[0]),
                    np.random.uniform(0, self.world_size[1]),
                )
            )
            y = self.noisy_sensor.sample(x)
            self.noisy_model.add_observation(x, y)

        self.noisy_model.train_model()

        # Initialize path
        noisy_plan = [initial_point]
        error_plan = [initial_point]

        if video_file is not None:
            writer = imageio.get_writer(video_file, fps=20)
        if error_video_file is not None:
            error_writer = imageio.get_writer(error_video_file, fps=20)

        # Planning loop
        for i in range(50):
            # Sample the noisy sensor
            for loc in noisy_plan:
                y = self.noisy_sensor.sample(loc)
                self.noisy_model.add_observation(loc, y)

            self.noisy_model.train_model()
            noisy_pred_map = self.noisy_model.test_model(
                world_size=self.world_size, gt_data=self.noisy_data.map, vis=VIS
            )

            noisy_plan = self.noisy_planner.plan(
                self.noisy_model, variance_scale=100000, n_steps=3
            )

            for loc in error_plan:
                pred = self.noisy_model.sample_belief(loc)[MEAN_KEY]
                real = self.groundtruth_sensor.sample(loc)
                error = pred - real
                self.error_model.add_observation(loc, error)

            self.error_model.train_model()
            error_pred_map = self.error_model.test_model(
                world_size=self.world_size, gt_data=self.error, vis=VIS
            )

            error_plan = self.noisy_planner.plan(
                self.noisy_model, variance_scale=100000
            )
            f, axs = plt.subplots(2, 3)
            total_pred = noisy_pred_map - error_pred_map

            total_pred_error = total_pred - self.groundtruth_data.map

            cb0 = axs[0, 0].imshow(noisy_pred_map)
            max_pred_error = np.max(np.abs(error_pred_map))
            cb1 = axs[0, 1].imshow(
                error_pred_map,
                vmin=-max_pred_error,
                vmax=max_pred_error,
                cmap="seismic",
            )
            max_real_error = np.max(np.abs(self.error))
            cb2 = axs[0, 2].imshow(
                self.error, vmin=-max_real_error, vmax=max_real_error, cmap="seismic"
            )

            cb3 = axs[1, 0].imshow(total_pred)
            cb4 = axs[1, 1].imshow(self.groundtruth_data.map)
            max_error = np.max(np.abs(total_pred_error))
            cb5 = axs[1, 2].imshow(
                total_pred_error, vmin=-max_error, vmax=max_error, cmap="seismic"
            )

            axs[0, 0].set_title("noisy pred")
            axs[0, 1].set_title("error pred")
            axs[0, 2].set_title("sensor error")

            axs[1, 0].set_title("total pred")
            axs[1, 1].set_title("gt")
            axs[1, 2].set_title("total pred error")

            plt.colorbar(cb0, ax=axs[0, 0])
            plt.colorbar(cb1, ax=axs[0, 1])
            plt.colorbar(cb2, ax=axs[0, 2])

            plt.colorbar(cb3, ax=axs[1, 0])
            plt.colorbar(cb4, ax=axs[1, 1])
            plt.colorbar(cb5, ax=axs[1, 2])
            plt.show()

        writer.close()
        error_writer.close()
        _run.add_artifact(video_file)
        _run.add_artifact(error_video_file)
