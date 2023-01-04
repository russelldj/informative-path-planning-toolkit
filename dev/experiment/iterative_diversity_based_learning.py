import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ipp_toolkit.config import DATA_FOLDER, ERROR_IMAGE, VIS
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import (
    ImageNPMaskedLabeledImage,
    torchgeoMaskedDataManger,
)
from ipp_toolkit.planners.diversity_planner import (
    DiversityPlanner,
    BatchDiversityPlanner,
)
from ipp_toolkit.predictors.masked_image_predictor import (
    MaskedLabeledImagePredictor,
    EnsembledMaskedLabeledImagePredictor,
)
from ipp_toolkit.planners.masked_planner import GridMaskedPlanner, RandomMaskedPlanner
from ipp_toolkit.config import (
    TOP_FRAC_MEAN_ERROR,
    UNCERTAINTY_KEY,
    MEAN_KEY,
    MEAN_ERROR_KEY,
)
from imageio import imread, imwrite
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
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


def plot_errors(all_l2_errors, run_tag):
    all_l2_errors = np.vstack(all_l2_errors)
    np.save(f"vis/iterative_exp/{run_tag}_errors.npy", all_l2_errors)
    mean = np.mean(all_l2_errors, axis=0)
    std = np.std(all_l2_errors, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=f"{run_tag} mean")
    plt.fill_between(
        x, mean - std, mean + std, label=f"{run_tag} one std bound", alpha=0.3
    )


def run_repeated_exp(n_trials=10, **kwargs):
    diversity_planner_result = [
        run_exp(use_random_planner=False, **kwargs) for _ in range(n_trials)
    ]
    random_planner_result = [
        run_exp(use_random_planner=True, **kwargs) for _ in range(n_trials)
    ]

    plt.close()
    plt.cla()
    plt.clf()
    plot_errors(random_planner_result, "Random planner")
    plot_errors(diversity_planner_result, "Diversity planner")
    plt.legend()
    plt.xlabel("Number of sampling iterations")
    plt.ylabel("Test error")
    plt.savefig("vis/iterative_exp/final_values.png")
    plt.show()


def run_exp(
    data_manager,
    n_clusters=12,
    visit_n_locations=8,
    vis=VIS,
    use_random_planner=False,
    regression_task=False,
    n_flights=5,
    vmin=0,
    vmax=9,
    cmap="tab10",
    error_key=MEAN_ERROR_KEY,
):
    # Chose which model to use
    if regression_task:
        model = MLPRegressor()
    else:
        model = MLPClassifier()

    # Chose which planner to use
    if use_random_planner:
        planner = RandomMaskedPlanner(data_manager)
    else:
        planner = BatchDiversityPlanner(data_manager, n_candidate_locations=n_clusters)
    # Create the predictor
    predictor = EnsembledMaskedLabeledImagePredictor(
        data_manager, model, classification_task=True, n_ensemble_models=7
    )

    errors = []
    # No prior interestingess
    interestingness_image = None

    for i in range(n_flights):
        savepath = f"vis/iterative_exp/no_revisit_plan_iter_{i}.png"
        plan = planner.plan(
            interestingness_image=interestingness_image,
            visit_n_locations=visit_n_locations,
            vis=VIS,
            savepath=savepath,
        )
        print(f"Saving to {savepath}")
        # Remove duplicate entry
        plan = plan[:-1]
        values = data_manager.sample_batch(plan, assert_valid=True, vis=VIS)
        predictor.update_model(plan, values)

        pred = predictor.predict_values_and_uncertainty()
        interestingness_image = pred[UNCERTAINTY_KEY]
        pred_values = pred[MEAN_KEY]
        error_dict = predictor.get_errors()
        errors.append(error_dict[error_key])
        # Visualization
        if vis:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(data_manager.image)
            plt.colorbar(
                axs[0, 1].imshow(data_manager.label, vmin=vmin, vmax=vmax, cmap=cmap),
                ax=axs[0, 1],
            )
            plt.colorbar(
                axs[1, 0].imshow(pred_values, vmin=vmin, vmax=vmax, cmap=cmap),
                ax=axs[1, 0],
            )
            plt.colorbar(axs[1, 1].imshow(error_dict[ERROR_IMAGE]), ax=axs[1, 1])
            axs[0, 0].set_title("Image")
            axs[0, 1].set_title("Label")
            axs[1, 0].set_title("Pred label")
            axs[1, 1].set_title("Pred error")
            plt.savefig(f"vis/iterative_exp/pred_iter_{i}.png")
            plt.close()
    return errors


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
    run_repeated_exp(
        data_manager=data_manager,
        n_clusters=n_clusters,
        visit_n_locations=visit_n_locations,
        vis=vis,
        n_trials=3,
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
    run_torchgeo(n_clusters=args.n_clusters, visit_n_locations=args.visit_n_locations)
