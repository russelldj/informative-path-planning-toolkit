from ipp_toolkit.planners.planners import BasePlanner
import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.config import VIS
from ipp_toolkit.planners.utils import compute_gridded_samples_from_mask
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.config import VIS_LEVEL_2


class BaseGriddedPlanner(BasePlanner):
    def vis(self, sampled_points, savepath=None, title="Random plan"):
        plt.close()
        plt.clf()
        plt.imshow(self.data.image[..., :3])
        # Note that the convention is switched for plotting
        plt.plot(sampled_points[:, 1], sampled_points[:, 0])
        plt.scatter(sampled_points[:, 1], sampled_points[:, 0])
        plt.title(title)
        if savepath is not None:
            plt.savefig(savepath)
        else:
            plt.show()
        plt.close()
        plt.clf()
        plt.cla()

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
        sampled_points = np.concatenate(
            (sampled_points, sampled_points[-1:, :]), axis=0
        )
        if vis:
            self.vis(
                sampled_points=sampled_points, savepath=savepath, title="Random sampler"
            )
        return sampled_points

    @classmethod
    def get_planner_name(cls):
        return "random_sampling_masked_planner"


class LawnmowerMaskedPlanner(BaseGriddedPlanner):
    def __init__(self, data: MaskedLabeledImage, n_total_samples):
        self.data = data
        self.samples = compute_gridded_samples_from_mask(
            self.data.mask, n_total_samples
        )
        if np.random.random() > 0.5:
            self.samples = np.flip(self.samples, axis=0)
        self.last_sampled_index = 0

    def plan(self, n_samples, vis=VIS_LEVEL_2, savepath=None, **kwargs):

        sampled_points = self.samples[
            self.last_sampled_index : self.last_sampled_index + n_samples
        ]
        self.last_sampled_index += n_samples

        if vis:
            self.vis(
                sampled_points=sampled_points,
                savepath=savepath,
                title="Lawnmower planner",
            )
        return sampled_points

    @classmethod
    def get_planner_name(cls):
        return "lawnmower_planner"


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
