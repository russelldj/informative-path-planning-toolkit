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

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import r2_score


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


def run(data_folder, n_clusters=12, visit_n_locations=8, train_frac=0.1):
    filenames = [Path(data_folder, x + ".npy") for x in ("X_wv", "valid_wv", "Y")]
    data_manager = MaskedLabeledImage(*filenames)
    # plt.imshow(data_manager.label)
    # plt.show()
    X = data_manager.get_valid_images_points()
    soft_labels = data_manager.get_valid_label_points()
    y = soft_labels[:, 0]

    invalid = y < 0
    y[invalid] = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_frac)
    )
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X)

    pred_vis = np.zeros_like(data_manager.image[..., 0])
    real_vis = np.zeros_like(data_manager.image[..., 0])
    real_vis[data_manager.mask] = y
    pred_vis[data_manager.mask] = y_pred

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(real_vis, vmin=0, vmax=1)
    ax2.imshow(pred_vis, vmin=0, vmax=1)
    r2 = r2_score(y_test, lr.predict(X_test))
    plt.title(f"R^2 = {r2}")
    plt.savefig(f"vis/regression_{train_frac:03f}.png")
    plt.pause(5)

    planner = DiversityPlanner()
    plan = planner.plan(
        data_manager,
        interestingness_map=pred_vis,
        n_locations=n_clusters,
        visit_n_locations=visit_n_locations,
        savepath=f"vis/coral_diversity_ipp_{n_clusters}.png",
        blur_scale=20,
    )


if __name__ == "__main__":
    args = parse_args()
    for train_frac in [0.1, 0.01, 0.001, 0.0001]:
        run(
            coral_folder,
            n_clusters=args.n_clusters,
            visit_n_locations=args.visit_n_locations,
            train_frac=train_frac,
        )
