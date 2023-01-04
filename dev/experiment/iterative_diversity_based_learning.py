import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ipp_toolkit.config import DATA_FOLDER, VIS
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import (
    ImageNPMaskedLabeledImage,
    torchgeoMaskedDataManger,
)
from ipp_toolkit.experiments.comparing_ipp_approaches import (
    run_repeated_exp,
    compare_random_vs_diversity,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=200)
    parser.add_argument("--visit-n-locations", type=int, default=20)
    args = parser.parse_args()
    return args


coral_folder = Path(DATA_FOLDER, "maps/coral")
forest_folder = Path(DATA_FOLDER, "maps/safeforest")
safeforest_gmaps_folder = Path(DATA_FOLDER, "maps/safeforest_gmaps")
aiira_folder = Path(DATA_FOLDER, "maps/aiira")
yellowcat_folder = Path(DATA_FOLDER, "maps/yellowcat")


def compute_greenness(data_manager, vis=VIS):
    img = data_manager.image
    magnitude = np.linalg.norm(img[..., 0::2], axis=2)
    green = img[..., 1]
    greenness = green.astype(float) / (magnitude.astype(float) + 0.00001)
    greenness = np.clip(greenness, 0, 4) / 4
    greenness[np.logical_not(data_manager.mask)] = np.nan
    if vis:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        plt.colorbar(axs[1].imshow(greenness), ax=axs[1])
        axs[0].set_title("Original image")
        axs[1].set_title("Psuedo-label")
        plt.show()
    return greenness


def run_torchgeo(n_clusters, visit_n_locations, vis=VIS):
    data_manager = torchgeoMaskedDataManger(vis_all_chips=False)
    compare_random_vs_diversity(
        data_manager,
        n_clusters=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=vis,
    )


def run_forest_ortho(data_folder, n_clusters=12, visit_n_locations=8, vis=VIS):
    dem, ortho, mask_filename = [
        Path(data_folder, x + ".tif")
        for x in (
            "left_camera_dem",
            "left_camera_32x_downsample",
            "left_camera_mask_32x_downsample",
        )
    ]

    data_manager = ImageNPMaskedLabeledImage(ortho, mask_filename)
    data_manager.label = compute_greenness(data_manager, vis=VIS)
    run_repeated_exp(
        data_manager=data_manager,
        n_clusters=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=vis,
    )


def run_forest_gmap(n_clusters=12, visit_n_locations=8, vis=VIS):
    ortho = "/home/frc-ag-1/dev/research/informative-path-planning-toolkit/data/maps/safeforest_gmaps/google_earth_site.png"

    data_manager = ImageNPMaskedLabeledImage(ortho, blur_sigma=4)
    data_manager.label = compute_greenness(data_manager, vis=VIS)
    run_repeated_exp(
        data_manager=data_manager,
        n_clusters=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=vis,
    )


if __name__ == "__main__":
    args = parse_args()
    # run_forest_ortho(
    #    forest_folder,
    #    n_clusters=args.n_clusters,
    #    visit_n_locations=args.visit_n_locations,
    # )
    # run_forest_gmap(
    #    n_clusters=args.n_clusters, visit_n_locations=args.visit_n_locations,
    # )
    run_torchgeo(
        n_clusters=args.n_clusters, visit_n_locations=args.visit_n_locations, vis=True
    )

