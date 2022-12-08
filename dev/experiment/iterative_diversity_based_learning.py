import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ipp_toolkit.config import DATA_FOLDER
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.planners.diversity_planner import DiversityPlanner
from imageio import imread
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


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


def run_forest(data_folder, n_clusters=12, visit_n_locations=8, vis=False):
    dem, ortho, mask_filename = [
        Path(data_folder, x + ".tif")
        for x in ("left_camera_dem", "left_camera", "left_camera_mask")
    ]

    input_image = imread(ortho)
    label_image = np.linalg.norm(input_image[..., :3], axis=2)
    label_image[input_image[..., 3] == 0] = np.nan

    data_manager = MaskedLabeledImage(
        input_image[..., :3],
        mask=mask_filename,
        label=label_image,
        downsample=8,
        blur_sigma=2,
    )
    if vis:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(data_manager.image)
        cb = axs[1].imshow(data_manager.label)
        plt.colorbar(cb, ax=axs[1])
        plt.show()

    standard_scalar = StandardScaler()
    all_valid_features = data_manager.get_valid_images_points()
    # Fit on the whole dataset and transform
    all_valid_features = standard_scalar.fit_transform(all_valid_features)

    sampled_normalized_X = np.empty((0, all_valid_features.shape[1]))
    sampled_y = np.empty(0)

    planner = DiversityPlanner()
    model = LinearRegression()

    # For the first iteration we have no guess of inter
    interestingess_image = None

    for i in range(10):
        plan = planner.plan(
            data_manager,
            interestingness_image=interestingess_image,
            n_locations=n_clusters,
            visit_n_locations=visit_n_locations,
            vis=True,
            savepath=f"vis/iterative_exp/plan_iter_{i}.png",
        )
        # sample based on the plan
        # Don't double count the first/last sample
        plan = plan[:-1]
        new_y_samples = data_manager.sample_batch(plan)
        new_X_samples = data_manager.sample_batch_features(plan)
        new_normalized_X_samples = standard_scalar.transform(new_X_samples)

        valid = np.isfinite(new_y_samples)
        new_normalized_X_samples = new_normalized_X_samples[valid]
        new_y_samples = new_y_samples[valid]

        sampled_normalized_X = np.concatenate(
            (sampled_normalized_X, new_normalized_X_samples), axis=0
        )
        sampled_y = np.concatenate((sampled_y, new_y_samples), axis=0)
        # TODO Fit a model based on the samples
        model.fit(sampled_normalized_X, sampled_y)
        # TODO predict an interestingess based on the samples

        pred_y = model.predict(all_valid_features)
        interestingess_image = data_manager.get_image_for_flat_values(pred_y)
        error = interestingess_image - data_manager.label

        # Visualization
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(data_manager.image)
        plt.colorbar(axs[0, 1].imshow(data_manager.label), ax=axs[0, 1])
        plt.colorbar(axs[1, 0].imshow(interestingess_image), ax=axs[1, 0])
        plt.colorbar(axs[1, 1].imshow(error), ax=axs[1, 1])
        axs[0, 0].set_title("Image")
        axs[0, 1].set_title("Label")
        axs[1, 0].set_title("Pred label")
        axs[1, 1].set_title("Pred error")
        plt.savefig(f"vis/iterative_exp/pred_iter_{i}.png")
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    run_forest(
        forest_folder,
        n_clusters=args.n_clusters,
        visit_n_locations=args.visit_n_locations,
    )
