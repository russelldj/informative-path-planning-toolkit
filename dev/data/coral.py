import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ipp_toolkit.utils.sampling import get_flat_samples
import ubelt as ub

from ipp_toolkit.config import DATA_FOLDER
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.planners.diversity_planner import DiversityPlanner


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()
    return args


coral_folder = Path(DATA_FOLDER, "maps/coral")


def run(data_folder, n_clusters=12):
    filenames = [Path(data_folder, x + ".npy") for x in ("X_wv", "valid_wv", "Y")]
    data_manager = MaskedLabeledImage(*filenames)
    planner = DiversityPlanner()
    plan = planner.plan(data_manager, n_locations=n_clusters)


if __name__ == "__main__":
    args = parse_args()
    run(coral_folder, n_clusters=args.n_clusters)
