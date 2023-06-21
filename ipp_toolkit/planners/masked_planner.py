from ipp_toolkit.planners.planners import BasePlanner
import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.config import VIS
from ipp_toolkit.planners.utils import compute_gridded_samples_from_mask
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import VIS_LEVEL_2
from ipp_toolkit.planners.utils import order_locations_tsp, points_to_regions
from scipy.spatial.distance import cdist
from ipp_toolkit.visualization.utils import show_or_save_plt
import logging


class BaseGriddedPlanner(BasePlanner):
    def vis(self, sampled_points, savepath=None, title="Random plan", _run=None):
        plt.close()
        plt.clf()
        plt.imshow(self.data.image[..., :3])
        # Note that the convention is switched for plotting
        plt.plot(sampled_points[:, 1], sampled_points[:, 0])
        plt.scatter(sampled_points[:, 1], sampled_points[:, 0])
        plt.title(title)
        show_or_save_plt(savepath=savepath, _run=_run)

    @classmethod
    def get_planner_name(cls):
        return "base_gridder_planner"


class RandomSamplingMaskedPlanner(BaseGriddedPlanner):
    def __init__(self, data):
        self.data = data
        self.valid_locs = self.data.get_valid_loc_points()

    def plan(self, n_samples, vis=VIS, savepath=None, **kwargs):
        num_points = self.valid_locs.shape[0]
        random_inds = np.random.choice(num_points, n_samples)
        sampled_points = self.valid_locs[random_inds].astype(int)
        if vis:
            self.vis(
                sampled_points=sampled_points, savepath=savepath, title="Random sampler"
            )
        return sampled_points

    @classmethod
    def get_planner_name(cls):
        return "random_sampling_masked_planner"


class LawnmowerMaskedPlanner(BaseGriddedPlanner):
    def __init__(
        self, data: MaskedLabeledImage, n_total_samples, initial_loc, **kwargs
    ):
        self.data = data
        self.current_loc = initial_loc
        self.samples = compute_gridded_samples_from_mask(
            self.data.mask, n_total_samples, return_exact_number=True
        )
        # TODO deal with this
        if np.random.random() > 0.5:
            logging.warn("flipping sample order in lawnmower")
            self.samples = np.flip(self.samples, axis=0)
        self.last_sampled_index = 0

    def plan(self, n_samples, vis=VIS_LEVEL_2, savepath=None, **kwargs):
        if self.current_loc is not None:
            current_loc = np.expand_dims(self.current_loc, axis=0)
            dists = cdist(self.samples, current_loc)[:, 0]
            nearest_point = np.argmin(dists)
            self.samples = np.concatenate(
                (
                    self.samples[
                        nearest_point:
                    ],  # Start at the point and finish all the samples
                    np.flip(
                        self.samples[:nearest_point], axis=0
                    ),  # Go back to near the last unsampled point and go backward to the beginning
                ),
                axis=0,
            )
            self.start_loc_set = True
        sampled_points = self.samples[
            self.last_sampled_index : self.last_sampled_index + n_samples
        ]
        self.last_sampled_index += n_samples

        if vis:
            self.vis(
                sampled_points=sampled_points,
                savepath=savepath,
                title=self.get_planner_name(),
            )
        return sampled_points

    @classmethod
    def get_planner_name(cls):
        return "lawnmower_planner"


class IceCreamConePlanner(BaseGriddedPlanner):
    def __init__(self, data: BaseWorldModel, initial_loc: np.ndarray = None):
        super().__init__(data, initial_loc)


class CompassLinesPlanner(BaseGriddedPlanner):
    def __init__(self, data: BaseWorldModel, initial_loc: np.ndarray = None):
        super().__init__(data, initial_loc)
        self.directions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        self.direction_ind = np.random.choice(4)

    def plan(
        self,
        n_samples: int,
        pathlength=None,
        pred_dict: dict = {},
        values=None,
        **kwargs,
    ):
        dists = np.linspace(0, pathlength / 2, n_samples)
        dists = np.expand_dims(dists, axis=1)

        direction = self.directions[self.direction_ind]
        steps_from_start = dists * direction
        plan = np.expand_dims(self.current_loc, axis=0) + steps_from_start
        plan = plan.astype(int)

        self.direction_ind += 1
        self.direction_ind = self.direction_ind % 4

        if self.expand_region_pixels != 1:
            plan = points_to_regions(plan, self.expand_region_pixels)

        return plan


class TrianglesLinesPlanner(BaseGriddedPlanner):
    def __init__(
        self,
        data: BaseWorldModel,
        initial_loc: np.ndarray = None,
        expand_region_pixels=1,
    ):
        super().__init__(data, initial_loc, expand_region_pixels=expand_region_pixels)
        one_over_sqrt_half = np.sqrt(0.5)

        self.directions = {
            0: np.array([[1, 0], [one_over_sqrt_half, one_over_sqrt_half]]),
            1: np.array([[0, 1], [-one_over_sqrt_half, one_over_sqrt_half]]),
            2: np.array([[-1, 0], [-one_over_sqrt_half, -one_over_sqrt_half]]),
            3: np.array([[0, -1], [one_over_sqrt_half, -one_over_sqrt_half]]),
        }
        # Each long leg of the triangle
        self.short_side_scaling_factor = 0.7653668647301797 / (2 + 0.7653668647301797)
        self.long_side_scaling_factor = 1 / (2 + 0.7653668647301797)
        self.direction_ind = np.random.choice(4)

    def plan(
        self,
        n_samples: int,
        pathlength=None,
        pred_dict: dict = {},
        values=None,
        **kwargs,
    ):
        n_samples_long_legs = int(n_samples / 2) - 1
        dists = np.linspace(
            0, pathlength * self.long_side_scaling_factor, n_samples_long_legs
        )
        dists = np.expand_dims(dists, axis=1)

        first_direction, second_direction = self.directions[self.direction_ind]

        first_steps_from_start = dists * first_direction
        second_steps_from_start = dists * second_direction

        last_first_leg = first_steps_from_start[-1]
        first_last_leg = second_steps_from_start[-1]
        middle_components = np.stack(
            [
                2 / 3 * last_first_leg + 1 / 3 * first_last_leg,
                1 / 3 * last_first_leg + 2 / 3 * first_last_leg,
            ]
        )

        steps_from_start = np.concatenate(
            (
                first_steps_from_start,
                middle_components,
                np.flip(second_steps_from_start, axis=0),
            ),
            axis=0,
        )
        plan = np.expand_dims(self.current_loc, axis=0) + steps_from_start
        plan = plan.astype(int)

        self.direction_ind += 1
        self.direction_ind = self.direction_ind % 4

        if self.expand_region_pixels != 1:
            plan = points_to_regions(plan, self.expand_region_pixels)

        return plan


class RandomWalkMaskedPlanner(BaseGriddedPlanner):
    def __init__(self, data: MaskedLabeledImage):
        self.data = data
        self.current_location = np.array(self.data.mask.shape) / 2

    def _get_random_step(self, step_size):
        angle = np.random.rand() * 2 * np.pi
        step = np.array([np.sin(angle), np.cos(angle)]) * step_size
        return step

    def _is_within_bounds(self, loc):
        return np.all(loc >= 0) and np.all(loc < self.data.mask.shape)

    def plan(self, n_samples, step_size, vis=False, savepath=None, **kwargs):
        sampled_points = np.zeros((0, 2))
        for i in range(n_samples):
            valid_mask = False
            while not valid_mask:
                step = self._get_random_step(step_size=step_size)
                candidate_location = self.current_location + step
                int_candidate_location = candidate_location.astype(int)
                valid_mask = (
                    self._is_within_bounds(int_candidate_location)
                    and self.data.mask[
                        int_candidate_location[0], int_candidate_location[1]
                    ]
                )

            self.current_location = candidate_location
            sampled_points = np.concatenate(
                (sampled_points, np.expand_dims(candidate_location, axis=0)), axis=0
            )
        sampled_points = sampled_points.astype(int)
        sampled_points = order_locations_tsp(sampled_points, open_path=True)
        if vis:
            self.vis(
                sampled_points=sampled_points,
                savepath=savepath,
                title="Random walk plan",
            )
        return sampled_points

    @classmethod
    def get_planner_name(cls):
        return "random_masked_walk_planner"


class MostUncertainPlanner(BaseGriddedPlanner):
    def __init__(self, data: MaskedLabeledImage):
        self.data = data
        self.random_sampling_planner = RandomSamplingMaskedPlanner(data=data)
        self.valid_locs = self.data.get_valid_loc_points()

    def plan(self, n_samples, interestingness_image, **kwargs):
        """
        TODO this is untested
        """
        if interestingness_image is None:
            random_plan = self.random_sampling_planner.plan(n_samples=n_samples)
            return random_plan

        valid_interestingness = interestingness_image[self.data.mask]

        # Avoid any bias toward one region with ties
        random_inds = np.random.choice(
            valid_interestingness.shape[0], size=valid_interestingness.shape[0]
        )
        valid_interestingness = valid_interestingness[random_inds]
        valid_locs = self.valid_locs[random_inds]
        highest_inds = np.argsort(valid_interestingness)[-n_samples:]
        locs = valid_locs[highest_inds]
        return locs.astype(int)

    @classmethod
    def get_planner_name(cls):
        return "most_uncertain_planner"
