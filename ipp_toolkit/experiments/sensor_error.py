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
    def __init__(self, world_size, n_points, overlap_inds, noise_sdev=0, random_seed=2):
        self.world_size = world_size

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

        if True:
            error = self.groundtruth_data.map - self.noisy_data.map

            fig, axs = plt.subplots(1, 3)
            cb_0 = axs[0].imshow(self.groundtruth_data.map)
            cb_1 = axs[1].imshow(self.noisy_data.map)
            cb_2 = axs[2].imshow(error)
            plt.colorbar(cb_0, ax=axs[0])
            plt.colorbar(cb_1, ax=axs[1])
            plt.colorbar(cb_2, ax=axs[2])
            plt.show()

    def run(self, initial_point=(0, 0), video_file=None, _run=None):
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
        last_loc = initial_point
        plan = [last_loc]

        if video_file is not None:
            writer = imageio.get_writer(video_file, fps=20)

        # Planning loop
        for i in range(50):
            for loc in plan:
                y = self.noisy_sensor.sample(loc)
                self.noisy_model.add_observation(loc, y)

            last_loc = plan[-1]

            self.noisy_model.train_model()
            img = self.noisy_model.test_model(
                world_size=self.world_size, gt_data=self.noisy_data.map,
            )

            if video_file is not None:
                writer.append_data(img)

            plan = self.noisy_planner.plan(self.noisy_model)

        writer.close()
        _run.add_artifact(video_file)
