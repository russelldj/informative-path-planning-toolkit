from ipp_toolkit.planners.masked_planner import BaseGriddedPlanner
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.visualization.utils import add_colorbar
from scipy.spatial.distance import cdist, pdist
from ipp_toolkit.planners.utils import order_locations_tsp
from ipp_toolkit.config import UNCERTAINTY_KEY
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def image_argmax(img: np.ndarray, n_samples: int):
    n_columns = img.shape[1]
    flat_img = img.flatten()
    # TODO make this robust to negative values
    flat_img = np.ma.masked_array(np.nan_to_num(flat_img))
    sorted_inds = np.argsort(flat_img)
    top_n_inds = sorted_inds[-n_samples]
    i_values = top_n_inds // n_columns
    j_values = top_n_inds % n_columns
    ij_values = np.vstack((i_values, j_values)).T
    return ij_values


class GreedyEntropyPlanner(BaseGriddedPlanner):
    def __init__(
        self,
        data: MaskedLabeledImage,
        predictor: UncertainMaskedLabeledImagePredictor,
        current_loc=None,
        budget_fraction_per_sample=0.5,
    ):
        self.data = data
        self.predictor = deepcopy(predictor)
        self.current_loc = np.atleast_2d(current_loc)
        self.budget_fraction_per_sample = budget_fraction_per_sample

    def _plan_unbounded(self, n_samples, vis):
        plan = []
        for _ in range(n_samples):
            uncertainty = self.predictor.predict_values_and_uncertainty()[
                UNCERTAINTY_KEY
            ]
            next_loc = image_argmax(uncertainty, n_samples=1)
            if vis:
                plt.scatter(next_loc[:, 1], next_loc[:, 0], c="k")
                plt.imshow(uncertainty)
                plt.colorbar()
                plt.pause(0.1)
            self.predictor.update_model(next_loc, np.zeros(next_loc.shape[0]))
            plan.append(next_loc)
        plan = np.concatenate(plan, axis=0)
        # TODO order these points
        return plan

    def _get_bounds_additional_cost(self, path: np.ndarray, candidate_locs: np.ndarray):
        """Obtain the minimum ammount which adding a new node to the path can increase it 

        Args:
            path (np.ndarray): locations of the planned path
            candidate_locs (np.ndarray): locations you might want to add
        """
        if path.shape[0] == 0:
            raise ValueError()
        elif path.shape[0] == 1:
            dist = np.linalg.norm(path - candidate_locs, axis=1)
            # You need to go there and back
            lower_bound_cost = dist * 2
            upper_bound_cost = dist * 2
        else:
            # THis is used instead of pdist because it's easier to interpret
            dists_between_current_points = cdist(path, path)
            dists_from_current_points_to_candidates = cdist(path, candidate_locs)
            lower_bound_costs = []
            upper_bound_costs = []
            for i in range(dists_between_current_points.shape[0]):
                for j in range(i):
                    baseline_cost = dists_between_current_points[i, j]
                    i_to_sample_cost = dists_from_current_points_to_candidates[i]
                    j_to_sample_cost = dists_from_current_points_to_candidates[j]
                    increased_cost = i_to_sample_cost + j_to_sample_cost - baseline_cost
                    lower_bound_costs.append(increased_cost)
                    # Path
                    if i - 1 == j:
                        upper_bound_costs.append(increased_cost)

            lower_bound_cost = np.min(lower_bound_costs, axis=0)
            upper_bound_cost = np.min(upper_bound_costs, axis=0)

        return lower_bound_cost, upper_bound_cost

    def _plan_bounded(self, n_samples, pathlength, vis=False):
        remaining_budget = pathlength

        path = self.current_loc
        valid_locs = self.data.get_valid_loc_points()
        for i in range(n_samples):
            # Get the upper and lower bounds for adding a new sample to the plan
            lower_bound_cost, upper_bound_cost = self._get_bounds_additional_cost(
                path=path, candidate_locs=valid_locs
            )
            # For now, just choose to use one
            cost = upper_bound_cost

            # TODO recompute this in a better way
            # This is how much we can add to the total cost
            additional_budget = remaining_budget * self.budget_fraction_per_sample
            # This is how much we have for the path, including the new sample
            total_budget = pathlength - remaining_budget * (
                1 - self.budget_fraction_per_sample
            )

            # Find which ones are within budget given the cost metric
            valid_locs_within_budget = valid_locs[cost < additional_budget]

            # Generate the entropy map
            uncertainty = self.predictor.predict_values_and_uncertainty()[
                UNCERTAINTY_KEY
            ]
            # Compute images
            img = self.data.get_image_for_flat_values(cost)
            invalid_mask = img > additional_budget
            img[invalid_mask] = np.nan
            uncertainty[invalid_mask] = np.nan

            # Sample a new loc
            valid_new_loc = False
            while not valid_new_loc:
                candidate_new_loc = image_argmax(uncertainty, n_samples=1)
                print(f"Trying {candidate_new_loc}")
                candidate_path = np.concatenate((path, candidate_new_loc), axis=0)
                if candidate_path.shape[0] > 2:
                    ordered_candidate_path, cost = order_locations_tsp(
                        candidate_path, return_cost=True
                    )
                else:
                    # It's ordered by default
                    ordered_candidate_path = candidate_path
                    # Go out and back
                    cost = (
                        np.linalg.norm(
                            ordered_candidate_path[0] - ordered_candidate_path[1]
                        )
                        * 2
                    )
                print(f"cost: {cost}, total_budget: {total_budget}")
                if cost < total_budget:
                    path = ordered_candidate_path
                    remaining_budget = pathlength - cost
                    valid_new_loc = True
            if vis:
                f, axs = plt.subplots(1, 3)

                axs[0].plot(path[:, 1], path[:, 0])
                axs[1].plot(path[:, 1], path[:, 0])
                axs[2].plot(path[:, 1], path[:, 0])

                axs[0].scatter(candidate_new_loc[:, 1], candidate_new_loc[:, 0], c="k")
                axs[1].scatter(candidate_new_loc[:, 1], candidate_new_loc[:, 0], c="k")
                axs[2].scatter(candidate_new_loc[:, 1], candidate_new_loc[:, 0], c="k")

                axs[2].imshow(self.data.image)

                add_colorbar(axs[0].imshow(img))
                add_colorbar(axs[1].imshow(uncertainty))
                plt.show()
            # Update the uncertainty with the new loc
            self.predictor.update_model(
                candidate_new_loc, np.zeros(candidate_new_loc.shape[0])
            )
        return path.astype(int)

    def plan(self, n_samples: int, pathlength=None, vis=False, vis_dist=False):
        """
        Generate samples by taking the highest entropy sample
        after fitting the model on all previous samples

        Args:
            n_samples (int): How many to sample 
            current_loc (_type_, optional): _description_. Defaults to None.
            vis (bool, optional): Should you visualize entropies. Defaults to False.

        Returns:
            _type_: plan
        """
        if pathlength is None:
            path = self._plan_unbounded(n_samples=n_samples, vis=vis_dist)
        else:
            path = self._plan_bounded(
                n_samples=n_samples, pathlength=pathlength, vis=vis,
            )
        self.current_loc = path[-1:]
        if vis:
            self.vis(path)
        return path
