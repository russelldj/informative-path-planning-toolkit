import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ipp_toolkit.utils.sampling import get_flat_samples
import ubelt as ub

from ipp_toolkit.config import DATA_FOLDER
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.planners.diversity_planner import DiversityPlanner
from imageio import imread, imwrite


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--visit-n-locations", type=int, default=5)
    args = parser.parse_args()
    return args


coral_folder = Path(DATA_FOLDER, "maps/coral")
forest_folder = Path(DATA_FOLDER, "maps/safeforest")
safeforest_gmaps_folder = Path(DATA_FOLDER, "maps/safeforest_gmaps")
aiira_folder = Path(DATA_FOLDER, "maps/aiira")
yellowcat_folder = Path(DATA_FOLDER, "maps/yellowcat")


def run(data_folder, n_clusters=12, visit_n_locations=8):
    filenames = [Path(data_folder, x + ".npy") for x in ("X_wv", "valid_wv", "Y")]
    data_manager = MaskedLabeledImage(*filenames)
    planner = DiversityPlanner()
    plan = planner.plan(
        data_manager,
        n_locations=n_clusters,
        visit_n_locations=visit_n_locations,
        savepath=f"vis/coral_diversity_ipp_{n_clusters}.png",
        blur_scale=20,
    )


def run_forest(data_folder, n_clusters=12, visit_n_locations=8):
    dem, ortho, mask_filename = [
        Path(data_folder, x + ".tif")
        for x in ("left_camera_dem", "left_camera", "left_camera_mask")
    ]

    data_manager = MaskedLabeledImage(
        ortho, mask_name=mask_filename, downsample=8, blur_sigma=2
    )
    planner = DiversityPlanner()
    plan = planner.plan(
        data_manager,
        n_locations=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=True,
        savepath=f"vis/safeforest_diversity_ipp_{n_clusters}.png",
    )


def run_yellowcat(data_folder, n_clusters, visit_n_locations):
    yellowcat_file = Path(data_folder, "20221028_M7_orthophoto.tif")
    data_manager = MaskedLabeledImage(
        yellowcat_file, use_last_channel_mask=True, downsample=8, blur_sigma=2,
    )
    plan = DiversityPlanner().plan(
        data_manager,
        n_locations=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=True,
        savepath=f"vis/yellow_cat_diversity_ipp_{n_clusters}.png",
    )


def run_safeforest_gmaps(data_folder, n_clusters, visit_n_locations):
    file = Path(data_folder, "safeforest_test.png")
    data_manager = MaskedLabeledImage(file, use_last_channel_mask=False, downsample=4)
    plan = DiversityPlanner().plan(
        data_manager,
        n_locations=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=True,
        savepath=f"vis/safeforest_gmaps_diversity_ipp_{n_clusters}.png",
    )


def run_aiira(data_folder, n_clusters, visit_n_locations):
    file = Path(data_folder, "random_field.png")
    data_manager = MaskedLabeledImage(file, use_last_channel_mask=False, downsample=4)
    plan = DiversityPlanner().plan(
        data_manager,
        n_locations=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=True,
        savepath=f"vis/aiira_diversity_ipp_{n_clusters}.png",
    )


if __name__ == "__main__":
    args = parse_args()
    # run(
    #    coral_folder,
    #    n_clusters=args.n_clusters,
    #    visit_n_locations=args.visit_n_locations,
    # )
    # run_yellowcat(
    #    yellowcat_folder,
    #    n_clusters=args.n_clusters,
    #    visit_n_locations=args.visit_n_locations,
    # )
    # run_forest(
    #    forest_folder,
    #    n_clusters=args.n_clusters,
    #    visit_n_locations=args.visit_n_locations,
    # )
    # run_safeforest_gmaps(
    #    safeforest_gmaps_folder,
    #    n_clusters=args.n_clusters,
    #    visit_n_locations=args.visit_n_locations,
    # )
    run_aiira(
        aiira_folder,
        n_clusters=args.n_clusters,
        visit_n_locations=args.visit_n_locations,
    )
