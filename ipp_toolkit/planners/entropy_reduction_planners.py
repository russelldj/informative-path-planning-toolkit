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
from ipp_toolkit.visualization.utils import show_or_save_plt
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from ipp_toolkit.config import DATA_FOLDER
from pathlib import Path
from tqdm import tqdm


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


def probability_weighted_samples(img: np.ndarray, n_samples: int, power=2):
    valid_samples = np.where(np.isfinite(img))
    valid_samples = np.vstack(valid_samples).T
    probs = img[valid_samples[:, 0], valid_samples[:, 1]]
    probs = np.power(probs, power)
    probs = probs / np.sum(probs)
    inds = np.random.choice(probs.shape[0], size=(n_samples), p=probs)
    ij_values = valid_samples[inds]
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

    def _plan_bounded(
        self,
        n_samples,
        pathlength,
        vis=False,
        max_GP_fits=20,
        uncertainty_weighting_power=4,
        use_upper_bound=True,
    ):
        """_summary_

        Args:
            n_samples (_type_): _description_
            pathlength (_type_): _description_
            vis (bool, optional): _description_. Defaults to False.
            n_GP_fits (int, optional): _description_. Defaults to 20.
            uncertainty_weighting_power (int, optional): The uncertainty is raised to this power
                                                         for both candidate sampling and recording the best values.
                                                         Defaults to 4.

        Returns:
            _type_: _description_
        """
        # Initialize budget to pathlength and path to the current location
        remaining_budget = pathlength
        path = self.current_loc
        # Determine which data points are valid
        valid_locs = self.data.get_valid_loc_points()

        frac_valid_per_sample = []
        map_uncertainties_per_sample = []

        for i in tqdm(range(n_samples)):
            # Get the upper and lower bounds for adding a new sample to the plan
            lower_bound_cost, upper_bound_cost = self._get_bounds_additional_cost(
                path=path, candidate_locs=valid_locs
            )
            # For now, just choose to use one
            cost = upper_bound_cost if use_upper_bound else lower_bound_cost

            # TODO recompute this in a better way
            # This is how much we can add to the total cost
            additional_budget = remaining_budget * self.budget_fraction_per_sample
            # This is how much we have for the path, including the new sample
            total_budget = pathlength - remaining_budget * (
                1 - self.budget_fraction_per_sample
            )

            # Generate the entropy map
            uncertainty = self.predictor.predict_values_and_uncertainty()[
                UNCERTAINTY_KEY
            ]
            # Compute distances
            distance_img = self.data.get_image_for_flat_values(cost)
            invalid_mask = distance_img > additional_budget
            distance_img[invalid_mask] = np.nan
            # Mask uncertainties outside of valid distances
            valid_uncertainty = uncertainty.copy()
            valid_uncertainty[invalid_mask] = np.nan

            # Bookkeeping
            lowest_map_uncertainty = np.inf
            n_GP_fits = 0
            total_candidate_locs = 0

            # Sample until you get enough GP fits
            while n_GP_fits < max_GP_fits:
                total_candidate_locs += 1
                # Get a new location by probability weighted sampling
                candidate_new_loc = probability_weighted_samples(
                    valid_uncertainty, n_samples=1, power=uncertainty_weighting_power
                )
                # Add this location to the path
                candidate_path = np.concatenate((path, candidate_new_loc), axis=0)
                # Order the path
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

                # Check validity if it's valid, check if it's the best
                if cost < total_budget:
                    n_GP_fits += 1
                    # Add the sample to a copy of the predictor
                    temporary_predictor = deepcopy(self.predictor)
                    temporary_predictor.update_model(
                        candidate_new_loc, np.zeros(candidate_new_loc.shape[0])
                    )
                    # Compute the uncertainty after adding that sample
                    candidate_map_uncertainty = temporary_predictor.predict_all()[
                        UNCERTAINTY_KEY
                    ]
                    normed_map_uncertainty = np.linalg.norm(
                        candidate_map_uncertainty[self.data.mask],
                        ord=uncertainty_weighting_power,
                    )
                    # We're looking for the lowest map uncerainty
                    if normed_map_uncertainty < lowest_map_uncertainty:
                        selected_ordered_path = ordered_candidate_path
                        remaining_budget = pathlength - cost
                        selected_new_loc = candidate_new_loc
                        lowest_map_uncertainty = normed_map_uncertainty
                        updated_uncertainty = candidate_map_uncertainty

            # Bookkeeping
            path = selected_ordered_path
            map_uncertainties_per_sample.append(lowest_map_uncertainty)
            frac_valid_per_sample.append(max_GP_fits / n_GP_fits)

            if vis:
                f, axs = plt.subplots(2, 2)
                [ax.plot(path[:, 1], path[:, 0]) for ax in axs.flatten()]

                [
                    ax.scatter(selected_new_loc[:, 1], selected_new_loc[:, 0], c="k")
                    for ax in axs.flatten()
                ]

                axs[0, 0].imshow(self.data.image[..., :3])

                add_colorbar(axs[0, 1].imshow(distance_img))
                add_colorbar(axs[1, 0].imshow(uncertainty))
                add_colorbar(axs[1, 1].imshow(updated_uncertainty))
                axs[0, 0].set_title("Features")
                axs[0, 1].set_title("Distance field")
                axs[1, 0].set_title("Initial uncertainty")
                axs[1, 1].set_title("Updated uncertainty")

                show_or_save_plt(
                    savepath=Path(DATA_FOLDER, "entropy_reduction", f"plan_{i:03d}.png")
                )
            # Update the uncertainty with the new loc
            self.predictor.update_model(
                candidate_new_loc, np.zeros(candidate_new_loc.shape[0])
            )
        if vis:
            plt.clf()
            f, axs = plt.subplots(1, 2)
            axs[0].plot(map_uncertainties_per_sample)
            axs[1].plot(frac_valid_per_sample)
            axs[0].set_title("map uncertainty")
            axs[1].set_title("frac valid TSP")
            show_or_save_plt(
                savepath=Path(DATA_FOLDER, "entropy_reduction", "summary.png")
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
