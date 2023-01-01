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
from ipp_toolkit.predictors.masked_image_predictor import MaskedLabeledImagePredictor
from ipp_toolkit.planners.masked_planner import GridMaskedPlanner, RandomMaskedPlanner
from imageio import imread, imwrite
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from skimage.color import rgb2hsv


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


def run_repeated_exp(n_trials=3, **kwargs):
    all_l2_errors = []
    for _ in range(n_trials):
        all_l2_errors.append(run_exp(**kwargs))
    all_l2_errors = np.vstack(all_l2_errors)
    mean = np.mean(all_l2_errors, axis=0)
    std = np.std(all_l2_errors, axis=0)
    x = np.arange(len(mean))
    plt.close()
    plt.cla()
    plt.clf()
    plt.plot(x, mean)
    plt.fill_between(x, mean - std, mean + std)
    plt.xlabel("Number of sampling iterations")
    plt.ylabel("Test error")
    plt.show()
    breakpoint()


def run_exp(
    data_manager, n_clusters=12, visit_n_locations=8, vis=False, use_random_planner=True
):
    if vis:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(data_manager.image)
        cb = axs[1].imshow(data_manager.label)
        plt.colorbar(cb, ax=axs[1])
        plt.show()
        data_manager.vis()

    model = MLPRegressor()

    if use_random_planner:
        planner = RandomMaskedPlanner(data_manager)
    else:
        planner = BatchDiversityPlanner(data_manager, n_candidate_locations=n_clusters)
    predictor = MaskedLabeledImagePredictor(data_manager, model)
    l2_errors = []
    # No prior interestingess
    interestingess_image = None

    for i in range(5):
        savepath = f"vis/iterative_exp/no_revisit_plan_iter_{i}.png"
        plan = planner.plan(
            interestingness_image=interestingess_image,
            visit_n_locations=visit_n_locations,
            vis=True,
            savepath=savepath,
        )
        print(f"Saving to {savepath}")
        # Remove duplicate entry
        plan = plan[:-1]
        values = data_manager.sample_batch(plan, assert_valid=True, vis=True)
        predictor.update_model(plan, values)
        if not np.all(np.isfinite(values)):
            breakpoint()

        interestingess_image = predictor.predict_values()
        error = interestingess_image - data_manager.label
        l2_errors.append(np.linalg.norm(error[data_manager.mask]))
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
    return l2_errors
    plt.close()
    plt.cla()
    plt.plot(l2_errors)
    plt.xlabel("Number of iterations")
    plt.ylabel("L2 error of predictions")
    plt.savefig("vis/iterative_exp/errors_vs_iter.png")
    plt.show()


def compute_greenness(data_manager, vis=False):
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


def run_forest_ortho(data_folder, n_clusters=12, visit_n_locations=8, vis=False):
    dem, ortho, mask_filename = [
        Path(data_folder, x + ".tif")
        for x in (
            "left_camera_dem",
            "left_camera_32x_downsample",
            "left_camera_mask_32x_downsample",
        )
    ]

    data_manager = MaskedLabeledImage(ortho, mask_filename)
    data_manager.label = compute_greenness(data_manager, vis=False)
    run_repeated_exp(
        data_manager=data_manager,
        n_clusters=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=vis,
    )


def run_forest_gmap(n_clusters=12, visit_n_locations=8, vis=False):
    ortho = "/home/frc-ag-1/dev/research/informative-path-planning-toolkit/data/maps/safeforest_gmaps/google_earth_site.png"

    data_manager = MaskedLabeledImage(ortho, blur_sigma=4)
    data_manager.label = compute_greenness(data_manager, vis=False)
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
    run_forest_gmap(
        n_clusters=args.n_clusters, visit_n_locations=args.visit_n_locations,
    )
