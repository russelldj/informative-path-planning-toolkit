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
from ipp_toolkit.planners.utils import order_locations_tsp, points_to_regions
from ipp_toolkit.config import UNCERTAINTY_KEY
from ipp_toolkit.visualization.utils import show_or_save_plt
import matplotlib.pyplot as plt
import numpy as np
import warnings
from copy import deepcopy
from ipp_toolkit.config import VIS_FOLDER
from pathlib import Path
from tqdm import tqdm
from imageio import imwrite
import sacred


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


def probability_weighted_samples(
    img: np.ndarray, n_samples: int, invalid_mask: np.ndarray = None, power=2
):
    # Select which regions are valid
    valid_samples = np.where(np.logical_not(invalid_mask))
    valid_samples = np.vstack(valid_samples).T

    probs = img[valid_samples[:, 0], valid_samples[:, 1]]
    probs = np.power(probs, power)
    probs = probs / np.sum(probs)
    inds = np.random.choice(probs.shape[0], size=(n_samples), p=probs)
    ij_values = valid_samples[inds]
    return ij_values


class RAPTORSPlanner(BaseGriddedPlanner):
    def __init__(
        self,
        data: MaskedLabeledImage,
        predictor: UncertainMaskedLabeledImagePredictor,
        budget_fraction_per_sample=1.0,
        initial_loc=None,
        expand_region_pixels=1,
        gp_fits_per_iteration=20,
        samples_per_region=100,
        n_test_locs=int(1e6),
        n_candidate_locs=500,
        per_sample_weighting_power=0,
        _run: sacred.Experiment = None,
    ):
        self.data = data
        self.current_loc = np.expand_dims(initial_loc, axis=0)
        self.expand_region_pixels = expand_region_pixels
        self.budget_fraction_per_sample = budget_fraction_per_sample
        self.gp_fits_per_iteration = gp_fits_per_iteration
        self.samples_per_region = samples_per_region
        self.n_test_locs = n_test_locs
        self.n_candidate_locs = n_candidate_locs
        self.per_sample_weighting_power = per_sample_weighting_power

        self.test_locs = None
        self.n_plans = 0

        self._run = _run
        # Set the predictor to reflect that the first sample is added
        self.predictor = deepcopy(predictor)
        current_patch = points_to_regions(
            self.current_loc,
            self.expand_region_pixels,
            samples_per_region=self.samples_per_region,
        )
        dummy_value = np.zeros(current_patch.shape[0])
        self.predictor.update_model(current_patch, dummy_value)

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

    def _order_path(self, current_path, new_loc):
        ## Add the new location to the path
        candidate_path = np.concatenate((current_path, new_loc), axis=0)
        if candidate_path.shape[0] > 2:
            ordered_candidate_path, candidate_pathlength = order_locations_tsp(
                candidate_path,
                return_cost=True,
            )
        else:
            # It's ordered by default
            ordered_candidate_path = candidate_path
            # Go out and back
            candidate_pathlength = (
                np.linalg.norm(ordered_candidate_path[0] - ordered_candidate_path[1])
                * 2
            )
        return ordered_candidate_path, candidate_pathlength

    def _get_additional_distance(
        self, path: np.ndarray, candidate_locs: np.ndarray, use_upper_bound: bool
    ):
        """Obtain the minimum ammount which adding a new node to the path can increase it

        Args:
            path (np.ndarray): locations of the planned path
            candidate_locs (np.ndarray): locations you might want to add
            use_upper_bound: return the upper bound cost rather than lower
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

        cost = upper_bound_cost if use_upper_bound else lower_bound_cost
        return cost

    def _select_bounded_sample(
        self,
        pathlength_budget,
        max_GP_fits,
        uncertainty_weighting_power,
        prior_uncertainty,
        invalid_mask,
        distance_img,
        current_path,
        iteration,
        vis=False,
    ):
        # Bookkeeping
        lowest_map_uncertainty = np.inf
        n_GP_fits = 0
        total_candidate_locs = 0

        # Sample until you get enough GP fits
        while n_GP_fits < max_GP_fits:
            total_candidate_locs += 1
            # Get a new location by probability weighted sampling
            candidate_new_loc = probability_weighted_samples(
                prior_uncertainty,
                n_samples=1,
                invalid_mask=invalid_mask,
                power=uncertainty_weighting_power,
            )
            # Add this location to the path
            candidate_path = np.concatenate((current_path, candidate_new_loc), axis=0)
            # Order the path
            if candidate_path.shape[0] > 2:
                ordered_candidate_path, candidate_pathlength = order_locations_tsp(
                    candidate_path,
                    return_cost=True,
                )
                # Remove the duplicate return-to-home
                # ordered_candidate_path = ordered_candidate_path[:-1]
            else:
                # It's ordered by default
                ordered_candidate_path = candidate_path
                # Go out and back
                candidate_pathlength = (
                    np.linalg.norm(
                        ordered_candidate_path[0] - ordered_candidate_path[1]
                    )
                    * 2
                )

            # Check validity if it's valid, check if it's the best
            if candidate_pathlength < pathlength_budget:
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
                    selected_new_loc = candidate_new_loc
                    selected_pathlength = candidate_pathlength
                    lowest_map_uncertainty = normed_map_uncertainty
                    updated_uncertainty = candidate_map_uncertainty
        if vis:
            _, axs = plt.subplots(2, 3)
            [
                ax.plot(selected_ordered_path[:, 1], selected_ordered_path[:, 0], c="r")
                for ax in axs.flatten()
            ]

            [
                ax.scatter(selected_new_loc[:, 1], selected_new_loc[:, 0], c="k")
                for ax in axs.flatten()
            ]
            # vis_image = self.data.get_vis_image()[..., :3]
            feature_image = self.data.image[..., :3]
            vis_image = self.data.get_vis_image()[..., :3]

            warnings.filterwarnings("ignore", module="matplotlib\..*")
            distance_img[invalid_mask] = np.nan
            masked_prior_uncertainty = prior_uncertainty.copy()
            masked_prior_uncertainty[invalid_mask] = np.nan

            axs[0, 0].imshow((vis_image))
            axs[1, 0].imshow((feature_image))

            add_colorbar(axs[0, 1].imshow(distance_img))
            add_colorbar(axs[1, 1].imshow(masked_prior_uncertainty))

            add_colorbar(axs[0, 2].imshow(prior_uncertainty))
            add_colorbar(axs[1, 2].imshow(updated_uncertainty))

            axs[0, 0].set_title("Features")
            axs[1, 0].set_title("Raw image")

            axs[0, 1].set_title("Distance field")
            axs[1, 1].set_title("Masked initial uncertainty")

            axs[0, 2].set_title("Initial uncertainty")
            axs[1, 2].set_title("Updated uncertainty")

            show_or_save_plt(
                savepath=Path(
                    VIS_FOLDER,
                    "entropy_reduction",
                    f"plan_{iteration:03d}.png",
                    _run=self._run,
                ),
                _run=self._run,
            )
        return (
            selected_ordered_path,
            selected_new_loc,
            selected_pathlength,
            lowest_map_uncertainty,
            n_GP_fits,
        )

    def select_next_sample_randomized(
        self,
        candidate_locs,
        candidate_patch_locs,
        test_locs,
        mean_prior_uncertainty,
        additional_distance,
        remaining_budget,
        pathlength_budget,
        max_GP_fits,
        current_path,
        uncertainty_weighting_power=2,
        per_sample_weighting=None,
        vis=False,
        tag=None,
    ):
        # Bookkeeping
        largest_uncertainty_reduction = -np.inf  # The best reduction
        n_GP_fits = 0  # The number of time the GP has been fit
        total_candidate_locs = 0  # The number of candidate locs that have been tried

        # Precompute the probabilities for each sample
        remaining_budget_per_sample = np.clip(
            remaining_budget - additional_distance, a_min=0, a_max=None
        )
        uncertainty_to_power = np.power(
            mean_prior_uncertainty, uncertainty_weighting_power
        )
        if per_sample_weighting is not None:
            candidate_sample_weighting = per_sample_weighting[
                candidate_locs[:, 0], candidate_locs[:, 1]
            ]
            test_sample_weighting = per_sample_weighting[
                test_locs[:, 0], test_locs[:, 1]
            ]
            candidate_sample_weighting = np.power(
                candidate_sample_weighting, self.per_sample_weighting_power
            )
            test_sample_weighting = np.power(
                test_sample_weighting, self.per_sample_weighting_power
            )
        else:
            candidate_sample_weighting = np.ones(candidate_locs.shape[0])
            test_sample_weighting = np.ones(test_locs.shape[0])
        probs = (
            uncertainty_to_power
            * remaining_budget_per_sample
            * candidate_sample_weighting
        )
        # Normalize probabilities
        probs = probs / np.sum(probs)

        attempted_locs = []
        scores = []

        # initial uncertainty
        test_locs_uncertainty = self.predictor.predict_subset_locs(test_locs)[
            UNCERTAINTY_KEY
        ]
        # TODO figure out a better way to score the uncertainty
        initial_map_uncertainty = np.linalg.norm(
            test_locs_uncertainty * test_sample_weighting,
            ord=uncertainty_weighting_power,
        )

        # Sample until you get enough GP fits
        while n_GP_fits < max_GP_fits:
            total_candidate_locs += 1

            ## Get a new location by probability weighted sampling
            candidate_ind = np.random.choice(probs.shape[0], 1, p=probs)[0]
            candidate_new_loc = candidate_locs[candidate_ind : candidate_ind + 1]

            # Add the sample and order the path
            ordered_candidate_path, candidate_pathlength = self._order_path(
                current_path=current_path, new_loc=candidate_new_loc
            )

            # Check validity. If it's valid, check if it's the best
            if candidate_pathlength < pathlength_budget:
                n_GP_fits += 1

                ## Add the sample to a copy of the predictor
                temporary_predictor = deepcopy(self.predictor)
                candidate_new_patch = points_to_regions(
                    candidate_new_loc,
                    expand_pixels=self.expand_region_pixels,
                    samples_per_region=self.samples_per_region,
                )
                # Simulate adding a new measurement
                temporary_predictor.update_model(
                    candidate_new_patch, np.zeros(candidate_new_patch.shape[0])
                )

                # Compute the uncertainty after adding that sample
                test_locs_uncertainty = temporary_predictor.predict_subset_locs(
                    test_locs
                )[UNCERTAINTY_KEY]
                # TODO figure out a better way to score the uncertainty
                uncertainty_reduction = initial_map_uncertainty - np.linalg.norm(
                    test_locs_uncertainty * test_sample_weighting,
                    ord=uncertainty_weighting_power,
                )
                remaining_budget_fraction = (
                    pathlength_budget - candidate_pathlength
                ) / pathlength_budget
                # uncertainty_reduction *= remaining_budget_fraction
                # We're looking for the lowest map uncerainty
                if uncertainty_reduction > largest_uncertainty_reduction:
                    selected_ordered_path = ordered_candidate_path
                    # Select the patch corresponding to the candidate loc
                    selected_new_patch = candidate_patch_locs[
                        candidate_ind
                        * self.samples_per_region : (candidate_ind + 1)
                        * self.samples_per_region
                    ]
                    selected_pathlength = candidate_pathlength
                    largest_uncertainty_reduction = uncertainty_reduction
                if vis:
                    scores.append(largest_uncertainty_reduction)
            elif vis:
                scores.append(0)
            if vis:
                attempted_locs.append(candidate_new_loc)
        if vis:
            plt.close()
            _, axs = plt.subplots(2, 3)

            [ax.imshow(self.data.vis_image[..., :3]) for ax in axs.flatten()]
            axs[1, 2].imshow(np.clip(self.data.image[..., :3] / 6 + 0.5, 0, 1))

            add_colorbar(
                axs[0, 0].scatter(
                    candidate_locs[:, 1], candidate_locs[:, 0], c=uncertainty_to_power
                )
            )
            add_colorbar(
                axs[0, 1].scatter(
                    candidate_locs[:, 1],
                    candidate_locs[:, 0],
                    c=remaining_budget_per_sample,
                )
            )
            add_colorbar(
                axs[1, 0].scatter(
                    candidate_locs[:, 1],
                    candidate_locs[:, 0],
                    c=probs,
                )
            )
            attempted_locs = np.concatenate(attempted_locs, axis=0)
            add_colorbar(
                axs[1, 1].scatter(
                    attempted_locs[:, 1],
                    attempted_locs[:, 0],
                    c=scores,
                )
            )
            [
                ax.plot(
                    current_path[:, 1], current_path[:, 0], c="r", label="Current path"
                )
                for ax in axs.flatten()
            ]
            [
                ax.scatter(current_path[:, 1], current_path[:, 0], c="r")
                for ax in axs.flatten()
            ]
            [ax.axis("off") for ax in axs.flatten()]

            plt.legend()
            axs[0, 0].set_title("Candidate point entropy")
            axs[0, 1].set_title("Candidate point additional distance")
            axs[0, 2].set_title("Image")

            axs[1, 0].set_title("Candidate sampling probability")
            axs[1, 1].set_title("Candidate entropy reduction")
            axs[1, 2].set_title("Features")

            show_or_save_plt(savepath=f"vis/RAPTORS/sampling_{tag}.png")
        return (
            selected_new_patch,
            selected_ordered_path,
            selected_pathlength,
            initial_map_uncertainty - largest_uncertainty_reduction,
            n_GP_fits,
        )

    def _plan_bounded_randomized(
        self,
        n_samples,
        pathlength_budget,
        max_GP_fits,
        vis=True,
        use_upper_bound=True,
        uncertainties=[],
        per_sample_weighting=None,
    ):
        """_summary_

        Args:
            n_samples (_type_): _description_
            pathlength_budget (_type_): _description_
            vis (bool, optional): _description_. Defaults to False.
            n_GP_fits (int, optional): _description_. Defaults to 20.
            uncertainty_weighting_power (int, optional): The uncertainty is raised to this power
                                                         for both candidate sampling and recording the best values.
                                                         Defaults to 4.

        Returns:
            _type_: _description_
        """

        # Initialize budget to pathlength and path to the current location
        if vis and self.n_plans == 0:
            imwrite("vis/RAPTORS/image.png", self.data.vis_image[..., :3])
            imwrite(
                "vis/RAPTORS/features.png",
                np.clip(self.data.image[..., :3] / 6 + 0.5, 0, 1),
            )

        path = self.current_loc
        current_pathlength = 0

        if self.test_locs is None:
            # Determine which data points are valid
            self.test_locs = self.data.get_random_valid_loc_points(
                n_points=self.n_test_locs, with_replacement=False
            ).astype(int)
        # These should be within the bound of the region we can get to
        candidate_locs = self.data.get_random_valid_loc_points(
            n_points=self.n_candidate_locs
        ).astype(int)

        # TODO only sample feasible region
        valid_candidate_locs = np.logical_and.reduce(
            (
                candidate_locs[:, 0] > self.expand_region_pixels,
                candidate_locs[:, 1] > self.expand_region_pixels,
                candidate_locs[:, 0]
                < self.data.image.shape[0] - self.expand_region_pixels - 1,
                candidate_locs[:, 1]
                < self.data.image.shape[1] - self.expand_region_pixels - 1,
            )
        )
        candidate_locs = candidate_locs[valid_candidate_locs]

        # Sample the patches
        candidate_patch_locs = points_to_regions(
            candidate_locs,
            self.expand_region_pixels,
            samples_per_region=self.samples_per_region,
        )

        # The number of sample points
        for i in tqdm(range(n_samples)):
            tag = f"plan_{self.n_plans:03d}_sample_{i:03d}"
            # Generate the current uncertainty map
            prior_candidate_patch_uncertainty = self.predictor.predict_subset_locs(
                candidate_patch_locs
            )[UNCERTAINTY_KEY]

            if vis:
                _, axs = plt.subplots(2, 2)
                axs[0, 0].set_title("Input image")
                axs[0, 1].set_title("Candidate patch uncertainty")
                axs[1, 0].set_title("Features")
                axs[1, 1].set_title("Test points uncertainty")

                prior_uncertainty_vis_image = np.full(
                    self.data.image.shape[:2],
                    fill_value=np.nan,
                )
                test_uncertainty_vis_image = np.full(
                    self.data.image.shape[:2],
                    fill_value=np.nan,
                )

                prior_uncertainty_vis_image[
                    candidate_patch_locs[:, 0], candidate_patch_locs[:, 1]
                ] = prior_candidate_patch_uncertainty

                test_uncertainty_vis_image[
                    self.test_locs[:, 0], self.test_locs[:, 1]
                ] = self.predictor.predict_subset_locs(self.test_locs)[UNCERTAINTY_KEY]

                [ax.plot(path[:, 1], path[:, 0], c="r") for ax in axs.flatten()]
                [ax.scatter(path[:, 1], path[:, 0], c="r") for ax in axs.flatten()]

                axs[0, 0].imshow(self.data.vis_image[..., :3])
                axs[1, 0].imshow(np.clip(self.data.image[..., :3] / 6.0 + 0.5, 0, 1))
                add_colorbar(
                    axs[0, 1].imshow(prior_uncertainty_vis_image, vmin=0, vmax=1)
                )
                add_colorbar(
                    axs[1, 1].imshow(test_uncertainty_vis_image, vmin=0, vmax=1)
                )

                show_or_save_plt(savepath=f"vis/RAPTORS/path_{tag}.png")

            # Expand each point to a patch
            per_patch_uncertainty = np.reshape(
                prior_candidate_patch_uncertainty, (candidate_locs.shape[0], -1)
            )
            per_patch_mean_uncertainty = np.mean(per_patch_uncertainty, axis=1)

            # Get the estimated additional distance for adding a new sample
            distance_per_sample = self._get_additional_distance(
                path=path,
                candidate_locs=candidate_locs,
                use_upper_bound=use_upper_bound,
            )

            # Select the next sample
            (
                selected_new_patch,
                path,
                current_pathlength,
                lowest_map_uncertainty,
                n_GP_fits,
            ) = self.select_next_sample_randomized(
                candidate_locs=candidate_locs,
                candidate_patch_locs=candidate_patch_locs,
                test_locs=self.test_locs,
                mean_prior_uncertainty=per_patch_mean_uncertainty,
                additional_distance=distance_per_sample,
                remaining_budget=pathlength_budget - current_pathlength,
                pathlength_budget=pathlength_budget,
                current_path=path,
                max_GP_fits=max_GP_fits,
                tag=tag,
                per_sample_weighting=per_sample_weighting,
                vis=False,
            )
            # Update the uncertainty with the selected locs
            self.predictor.update_model(
                selected_new_patch, np.zeros(selected_new_patch.shape[0])
            )
            uncertainties.append(lowest_map_uncertainty)
            if self._run is not None:
                self._run.log_scalar("pathlength", current_pathlength)
                self._run.log_scalar("lowest uncertainty", lowest_map_uncertainty)
                self._run.log_scalar("frac valid TSP", max_GP_fits / n_GP_fits)

        plt.close()

        plt.plot(uncertainties)
        plt.xlabel("Number of samples added")
        plt.ylabel(r"L_2 norm of test sample entropy")
        plt.title("Map entropy versus samples")
        show_or_save_plt(savepath=f"vis/RAPTORS/uncertainties_{self.n_plans:03d}.png")
        return path.astype(int)

    def _plan_bounded(
        self,
        n_samples,
        pathlength_budget,
        max_GP_fits,
        vis=False,
        uncertainty_weighting_power=4,
        use_upper_bound=True,
    ):
        """_summary_

        Args:
            n_samples (_type_): _description_
            pathlength_budget (_type_): _description_
            vis (bool, optional): _description_. Defaults to False.
            n_GP_fits (int, optional): _description_. Defaults to 20.
            uncertainty_weighting_power (int, optional): The uncertainty is raised to this power
                                                         for both candidate sampling and recording the best values.
                                                         Defaults to 4.

        Returns:
            _type_: _description_
        """
        # Initialize budget to pathlength and path to the current location
        path = self.current_loc
        current_pathlength = 0
        # Determine which data points are valid
        valid_locs = self.data.get_valid_loc_points()

        for i in range(n_samples):
            # Generate the current uncertainty map
            prior_uncertainty = self.predictor.predict_values_and_uncertainty()[
                UNCERTAINTY_KEY
            ]

            # Get the estimated additional distance for adding a new sample
            distance_per_sample = self._get_additional_distance(
                path=path, candidate_locs=valid_locs, use_upper_bound=use_upper_bound
            )

            # Massage the distances back into an image and set invalid ones to nan
            distance_img = self.data.get_image_for_flat_values(distance_per_sample)
            invalid_mask = (
                distance_img
                > (pathlength_budget - current_pathlength)
                * self.budget_fraction_per_sample
            )

            (
                path,
                selected_new_loc,
                current_pathlength,
                lowest_map_uncertainty,
                n_GP_fits,
            ) = self._select_bounded_sample(
                max_GP_fits=max_GP_fits,
                uncertainty_weighting_power=uncertainty_weighting_power,
                prior_uncertainty=prior_uncertainty,
                invalid_mask=invalid_mask,
                current_path=path,
                distance_img=distance_img,
                pathlength_budget=(
                    (
                        current_pathlength  # The current length
                        + (
                            (pathlength_budget - current_pathlength)
                            * self.budget_fraction_per_sample
                            # Plus a fraction of the remaining budget
                        )
                    )
                ),
                iteration=i,
                vis=vis,
            )

            # Update the uncertainty with the selected loc
            self.predictor.update_model(
                selected_new_loc, np.zeros(selected_new_loc.shape[0])
            )
            if self._run is not None:
                self._run.log_scalar("pathlength", current_pathlength)
                self._run.log_scalar("lowest uncertainty", lowest_map_uncertainty)
                self._run.log_scalar("frac valid TSP", max_GP_fits / n_GP_fits)
        print("Finished path")
        return path.astype(int)

    def plan(
        self,
        n_samples: int,
        pathlength=None,
        pred_dict={},
        observation_dict={},
        vis=False,
        vis_dist=False,
        savepath=None,
    ):
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
            if "mean" in pred_dict:
                predicted_classes = pred_dict["mean"]
                unique_values, counts = np.unique(predicted_classes, return_counts=True)
                inverse_counts = np.sum(counts) / counts
                per_sample_weighting = np.zeros_like(predicted_classes, dtype=float)
                for i, unique_value in enumerate(unique_values):
                    per_sample_weighting[
                        unique_value == predicted_classes
                    ] = inverse_counts[i]
            else:
                per_sample_weighting = None
            path = self._plan_bounded_randomized(
                n_samples=n_samples,
                pathlength_budget=pathlength,
                max_GP_fits=self.gp_fits_per_iteration,
                vis=vis,
                per_sample_weighting=per_sample_weighting,
            )
        # TODO handle this better
        if vis:
            self.vis(
                path,
                savepath=Path(VIS_FOLDER, "entropy_reduction", "path.png"),
                _run=self._run,
            )

        if self.expand_region_pixels != 1:
            path = points_to_regions(path, self.expand_region_pixels)

        self.n_plans += 1

        return path
