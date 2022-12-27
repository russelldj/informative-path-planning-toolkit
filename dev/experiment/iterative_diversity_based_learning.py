import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ipp_toolkit.config import DATA_FOLDER
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.planners.diversity_planner import (
    DiversityPlanner,
    BatchDiversityPlanner,
)
from imageio import imread, imwrite
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
        for x in (
            "left_camera_dem",
            "left_camera_32x_downsample",
            "left_camera_mask_32x_downsample",
        )
    ]

    data_manager = MaskedLabeledImage(ortho, mask_filename)
    label = np.linalg.norm(data_manager.image, axis=2)
    data_manager.label = label
    imwrite(Path(data_folder, "left_camera_32x_downsample.tif"), data_manager.image)
    imwrite(
        Path(data_folder, "left_camera_mask_32x_downsample.tif"),
        data_manager.mask.astype(np.uint8),
    )
    valid_labels = data_manager.get_valid_label_points()
    print(f"All points are valid {np.all(np.isfinite(valid_labels))}")
    if vis:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(data_manager.image)
        cb = axs[1].imshow(data_manager.label)
        plt.colorbar(cb, ax=axs[1])
        plt.show()
        data_manager.vis()

    standard_scalar = StandardScaler()
    all_valid_features = data_manager.get_valid_image_points()
    # Fit on the whole dataset and transform
    all_valid_features = standard_scalar.fit_transform(all_valid_features)

    sampled_normalized_X = np.empty((0, all_valid_features.shape[1]))
    sampled_y = np.empty(0)

    planner = DiversityPlanner()
    model = LinearRegression()

    batch_planner = BatchDiversityPlanner(
        prediction_model=model,
        world_data=data_manager,
        n_candidate_locations=n_clusters,
    )
    for i in range(10):
        savepath = f"vis/iterative_exp/no_revisit_plan_iter_{i}.png"
        plan = batch_planner.plan(
            visit_n_locations=visit_n_locations, vis=True, savepath=savepath,
        )
        print(f"Saving to {savepath}")
        # Remove duplicate entry
        plan = plan[:-1]
        # Account for the fact that the plan is in (x, y) and the query needs to be i, j
        # plan = np.flip(plan, axis=1)
        values = data_manager.sample_batch(plan, assert_valid=True, vis=True)
        if not np.all(np.isfinite(values)):
            breakpoint()

        batch_planner.update_model(plan, values)
        interestingess_image = batch_planner.predict_values()
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
