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
    args = parser.parse_args()
    return args


coral_folder = Path(DATA_FOLDER, "maps/coral")
forest_folder = Path(DATA_FOLDER, "maps/safeforest")


def run(data_folder, n_clusters=12):
    filenames = [Path(data_folder, x + ".npy") for x in ("X_wv", "valid_wv", "Y")]
    data_manager = MaskedLabeledImage(*filenames)
    planner = DiversityPlanner()
    plan = planner.plan(data_manager, n_locations=n_clusters)


def run_forest(data_folder, n_clusters=12):
    dem, ortho, mask_filename = [
        Path(data_folder, x + ".tif")
        for x in ("left_camera_dem", "left_camera", "left_camera_mask")
    ]

    # dem_data = imread(dem)
    # mask = (dem_data > 0).astype(np.uint8)
    # imwrite(mask_filename, mask)

    data_manager = MaskedLabeledImage(
        ortho, mask_name=mask_filename, downsample=8, blur_sigma=2
    )
    # plt.imshow(data_manager.image)
    # plt.show()
    planner = DiversityPlanner()
    plan = planner.plan(data_manager, n_locations=n_clusters, vis=True)


if __name__ == "__main__":
    args = parse_args()
    run(coral_folder, n_clusters=args.n_clusters)
    run_forest(forest_folder, n_clusters=args.n_clusters)
