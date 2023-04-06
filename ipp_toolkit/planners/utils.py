import matplotlib.pyplot as plt
import numpy as np
from ipp_toolkit.config import PAUSE_DURATION
from scipy.spatial.distance import cdist
from numpy import meshgrid
from ipp_toolkit.visualization.utils import show_or_save_plt, remove_ticks
from ipp_toolkit.config import VIS_LEVEL_2
from python_tsp.distances.euclidean_distance import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
import time


def visualize_plan(
    image_data,
    interestingness_image,
    centers,
    plan,
    labels,
    savepath,
    cmap="tab20",
    pause_duration=PAUSE_DURATION,
    vis_subset=VIS_LEVEL_2,
    vis_path=VIS_LEVEL_2,
    vis_fit=VIS_LEVEL_2,
):
    if vis_subset:
        plt.imshow(image_data.image)
        plt.scatter(
            plan[:, 1], plan[:, 0], c="r", label="Selected locations", edgecolors="k"
        )
        plt.scatter(
            centers[:, 1], centers[:, 0], label="Un-selected locations", edgecolors="k"
        )
        plt.legend()
        remove_ticks()
        show_or_save_plt("vis/subset_selection.png")

    if vis_path:
        plt.imshow(image_data.image)
        plt.scatter(
            plan[:, 1], plan[:, 0], c="r", label="Selected locations", edgecolors="k"
        )
        plt.plot(plan[:, 1], plan[:, 0], c="r", label="Path")
        plt.legend()
        remove_ticks()
        show_or_save_plt("vis/plan.png")

    if vis_fit:
        plt.imshow(image_data.label)
        plt.scatter(
            plan[:, 1], plan[:, 0], c="r", label="Selected locations", edgecolors="k"
        )
        plt.colorbar()
        plt.title("Target quantity")
        plt.legend()
        remove_ticks()
        show_or_save_plt("vis/sampling_selected_locations.png")

    clusters = np.ones(image_data.mask.shape) * np.nan
    clusters[image_data.mask] = labels
    f, axs = plt.subplots(1, 2 if interestingness_image is None else 3)
    axs[0].imshow(image_data.image[..., :3])
    axs[1].imshow(clusters, cmap=cmap)

    axs[0].set_title("First three imagery channels")
    axs[1].set_title("Cluster inds")

    if interestingness_image is not None:
        cb = axs[2].imshow(interestingness_image)
        axs[2].set_title("Interestingness score")
        plt.colorbar(cb, ax=axs[2])

    [add_candidates_and_plan(ax, centers, plan, cmap=cmap, vis_plan=True) for ax in axs]

    if savepath is not None:

        plt.savefig(savepath)
        plt.pause(pause_duration)
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()


def add_candidates_and_plan(ax, centers, plan, cmap="tab20", vis_plan=True):
    """
    Plotting convenience for adding candidate locations and final trajectory
    """
    n_locations = centers.shape[0]

    ax.scatter(
        centers[:, 1],
        centers[:, 0],
        c=np.arange(n_locations),
        cmap=cmap,
        edgecolors="k",
        label="",
    )
    if vis_plan:
        ax.plot(plan[:, 1], plan[:, 0], c="k")


def compute_mask(input_mask, visit_n_locations):
    """
    Compute the mask. This is trivial if the mask is binary. 
    for floats it's the top visit_n_locations values
    """
    if visit_n_locations is None:
        mask = np.squeeze(input_mask)
    else:
        ordered_locs = np.argsort(input_mask)
        mask = np.zeros_like(input_mask, dtype=bool)
        top_locs = ordered_locs[-visit_n_locations:]
        mask[top_locs] = True
    return mask


def compute_n_sampled(mask, visit_n_locations):
    """
    Return an objective this variable is free to change, otherwise nothing. This coresponds to a 
    0- or 1-length tuple
    """
    if visit_n_locations is None:
        return (np.sum(mask),)
    else:
        return ()


def compute_interestingness_objective(interestingness_scores, mask):
    """
    Compute interestingness of a masked set of points
    """
    if interestingness_scores is None:
        intrestesting_return = ()
    else:
        sampled_interestingness = interestingness_scores[mask]
        sum_interestingness = np.sum(sampled_interestingness)
        intrestesting_return = (-sum_interestingness,)
    return intrestesting_return


def compute_average_min_dist(
    candidate_location_features, mask, previous_location_features
):
    """
    Compute the average distance from each unsampled point to the nearest sampled one. Returns a 1-length tuple for compatability
    """

    # If nothing or everything is sampled is sampled, the value is that of the max dist between candidates
    if np.all(mask) or np.all(np.logical_not(mask)):
        empty_value = np.max(
            cdist(candidate_location_features, candidate_location_features)
        )
        return (empty_value,)

    # Compute the features for sampled and not sampled points
    not_sampled = candidate_location_features[np.logical_not(mask)]
    sampled = candidate_location_features[mask]
    if previous_location_features is not None:
        sampled = np.concatenate((sampled, previous_location_features))

    # Compute the distance for each sampled point to each un-sampled point
    dists = cdist(sampled, not_sampled)
    # Take the min distance from each unsampled point to a sampled point. This relates to how well described it is
    min_dists = np.min(dists, axis=0)
    # Average this distance cost over all unsampled points
    average_min_dist = np.mean(min_dists)

    return (average_min_dist,)


def estimate_path_length(candidate_locations, mask, current_location=None):
    """
    Compute the traveling salesman least cost path for the selected locations.
    Returns a 1-tuple for compatability
    """
    sampled_locations = candidate_locations[mask]
    if current_location is not None:
        sampled_locations = np.concatenate(
            (np.array([current_location]), sampled_locations), axis=0
        )

    distance_matrix = euclidean_distance_matrix(sampled_locations)
    # Make it free to return to the first point
    distance_matrix[:, 0] = 0
    _, cost = solve_tsp_local_search(distance_matrix)
    return (cost,)


def order_locations_tsp(
    locations,
    current_location=None,
    solver=solve_tsp_simulated_annealing,
    open_path=False,
    return_cost=False,
):
    # Optionally add the start location
    if current_location is not None:
        locations = np.concatenate((np.array([current_location]), locations), axis=0)

    # Compute the distances between points
    distance_matrix = euclidean_distance_matrix(locations)

    # The path doesn't need to come back to the start
    if open_path:
        # Make it free to return to the first location
        distance_matrix[:, 0] = 0

    # Solve TSP and order the path
    permutation, cost = solver(distance_matrix)

    if not open_path:
        # Add a return to the start in the plan
        permutation = permutation + [0]

    ordered_locations = locations[permutation]

    if current_location is not None:
        # Ensure the current location is the first element of the plan
        assert ordered_locations[0] == current_location
        # Remove the current location from the plan
        ordered_locations = ordered_locations[1:]

    if return_cost:
        return ordered_locations, cost

    return ordered_locations


def get_gridded_points(image_shape, resolution):
    half_remainders = (
        np.array([dim_size % resolution for dim_size in image_shape]) / 2.0
    )
    i_inds, j_inds = [
        np.arange(half_remainder + resolution / 2.0, size, resolution,)
        for half_remainder, size in zip(half_remainders, image_shape)
    ]
    i_samples, j_samples = meshgrid(i_inds, j_inds, indexing="ij")
    i_samples, j_samples = [
        samples.flatten().astype(int) for samples in (i_samples, j_samples)
    ]
    sample_points = np.vstack((i_samples, j_samples)).T
    return sample_points


def compute_gridded_samples_from_mask(
    mask,
    n_samples,
    n_bisections=100,
    return_exact_number=False,
    flip_alternate_rows=True,
):
    n_points = mask.size
    upper_bound = np.ceil(np.sqrt(n_points / n_samples)).astype(int)
    lower_bound = 1
    oversampled_points = None
    left_to_right_points = None

    for _ in range(n_bisections):
        resolution = np.sqrt(upper_bound * lower_bound)
        points = get_gridded_points(mask.shape, resolution)
        valid_points = mask[points[:, 0], points[:, 1]]
        n_valid_points = np.sum(valid_points)
        if n_valid_points == n_samples:
            left_to_right_points = points[valid_points]
            break
        elif n_valid_points < n_samples:
            upper_bound = resolution
        else:
            lower_bound = resolution
            oversampled_points = points[valid_points]

    # Especially for a rectangular grid, there may be no resolution that
    # gets you exactly what you want
    if return_exact_number and left_to_right_points is None:
        random_inds = np.random.choice(
            oversampled_points.shape[0], n_samples, replace=False
        )
        # This is important or else they won't be ordered left to right
        sorted_random_inds = sorted(random_inds)
        left_to_right_points = oversampled_points[sorted_random_inds, :]
    else:
        left_to_right_points = oversampled_points

    if flip_alternate_rows:
        _, unique_i_inds = np.unique(left_to_right_points[:, 0], return_inverse=True)
        for i in range(np.max(unique_i_inds) + 1):
            if i % 2 == 0:
                continue
            matching_inds = unique_i_inds == i
            left_to_right_points[matching_inds, :] = np.flip(
                left_to_right_points[matching_inds, :], axis=0
            )
    return left_to_right_points

