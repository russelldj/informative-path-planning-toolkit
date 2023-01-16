from ipp_toolkit.config import PAUSE_DURATION
import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.visualization.utils import remove_ticks, show_or_save_plt
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.config import VIS_LEVEL_1, VIS_LEVEL_2, VIS_LEVEL


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
    image_data: MaskedLabeledImage,
    interestingness_image,
    centers,
    plan,
    labels,
    savepath,
    cmap="tab20",
    pause_duration=PAUSE_DURATION,
    vis_plan=VIS_LEVEL_1,
    vis_subset=VIS_LEVEL_2,
    vis_path=VIS_LEVEL_2,
    vis_fit=VIS_LEVEL_2,
):
    if vis_subset:
        plt.imshow(image_data.image[..., :3])
        plt.title("Satellite image")
        plt.scatter(
            centers[:, 1], centers[:, 0], label="Un-selected locations", edgecolors="k"
        )
        plt.scatter(
            plan[:, 1], plan[:, 0], c="r", label="Selected locations", edgecolors="k"
        )
        plt.legend()
        remove_ticks()
        show_or_save_plt("vis/subset_selection.png")

    if vis_path:
        plt.imshow(image_data.image[..., :3])
        plt.title("Satellite image")
        plt.scatter(
            plan[:, 1], plan[:, 0], c="r", label="Selected locations", edgecolors="k"
        )
        plt.plot(plan[:, 1], plan[:, 0], c="r", label="Path")
        plt.legend()
        remove_ticks()
        show_or_save_plt("vis/plan.png")

    if vis_fit:
        plt.imshow(
            image_data.label,
            vmin=image_data.vis_vmin,
            vmax=image_data.vis_vmax,
            cmap=image_data.cmap,
        )
        plt.colorbar()
        plt.scatter(
            plan[:, 1],
            plan[:, 0],
            facecolors="none",
            label="Selected locations",
            edgecolors="tab:orange",
            linewidths=3,
        )
        plt.title("Target quantity")
        plt.legend()
        remove_ticks()
        show_or_save_plt("vis/sampling_selected_locations.png")

    if vis_plan:
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

        [
            add_candidates_and_plan(ax, centers, plan, cmap=cmap, vis_plan=True)
            for ax in axs
        ]

        show_or_save_plt(savepath, pause_duration=pause_duration, fig_size=(20, 13))
