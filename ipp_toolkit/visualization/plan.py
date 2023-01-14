from ipp_toolkit.config import PAUSE_DURATION
import numpy as np
import matplotlib.pyplot as plt


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
