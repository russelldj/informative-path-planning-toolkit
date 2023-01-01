import matplotlib.pyplot as plt
import numpy as np
from ipp_toolkit.config import PAUSE_DURATION
from scipy.spatial.distance import cdist


def visualize_plan(
    image_data,
    interestingness_image,
    centers,
    plan,
    labels,
    savepath,
    cmap="tab20",
    pause_duration=PAUSE_DURATION,
):
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
