import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.config import (
    ERROR_IMAGE,
    MEAN_KEY,
    UNCERTAINTY_KEY,
    VIS,
    MEAN_ERROR_KEY,
    N_FLIGHTS,
    N_TRIALS,
    VISIT_N_LOCATIONS,
    VIS,
)
from copy import deepcopy
from collections import defaultdict

from sklearn.neural_network import MLPClassifier, MLPRegressor
from ipp_toolkit.planners.diversity_planner import BatchDiversityPlanner
from ipp_toolkit.predictors.masked_image_predictor import (
    EnsembledMaskedLabeledImagePredictor,
)
from ipp_toolkit.planners.masked_planner import RandomMaskedPlanner
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage


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


def run_repeated_exp(n_trials=N_TRIALS, **kwargs):
    diversity_planner_result = [
        run_exp_default(use_random_planner=False, **kwargs) for _ in range(n_trials)
    ]
    random_planner_result = [
        run_exp_default(use_random_planner=True, **kwargs) for _ in range(n_trials)
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


def run_exp_custom(
    planner,
    data_manager: MaskedLabeledImage,
    predictor,
    n_flights=N_FLIGHTS,
    visit_n_locations=VISIT_N_LOCATIONS,
    planner_kwargs={},
    error_key=MEAN_ERROR_KEY,
    vis=VIS,
    interestingness_image=None,
):
    """
        Compute the error for one planner over multiple trials 
    """
    errors = []
    for i in range(n_flights):
        savepath = f"vis/iterative_exp/no_revisit_plan_iter_{i}.png"
        plan = planner.plan(
            visit_n_locations=visit_n_locations,
            savepath=savepath,
            interestingness_image=interestingness_image,
            **planner_kwargs,
        )
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
            vmin = data_manager.vis_vmin
            vmax = data_manager.vis_vmax
            cmap = data_manager.cmap
            _, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(data_manager.image[..., :3])
            display_label = data_manager.label.astype(float)
            display_label[np.logical_not(data_manager.mask)] = np.nan
            display_pred = pred_values.astype(float)
            display_pred[np.logical_not(data_manager.mask)] = np.nan
            display_error = error_dict[ERROR_IMAGE].astype(float)
            display_error[np.logical_not(data_manager.mask)] = np.nan

            plt.colorbar(
                axs[0, 1].imshow(display_label, vmin=vmin, vmax=vmax, cmap=cmap),
                ax=axs[0, 1],
            )
            plt.colorbar(
                axs[1, 0].imshow(display_pred, vmin=vmin, vmax=vmax, cmap=cmap),
                ax=axs[1, 0],
            )
            plt.colorbar(axs[1, 1].imshow(display_error), ax=axs[1, 1])
            axs[0, 0].set_title("Image")
            axs[0, 1].set_title("Label")
            axs[1, 0].set_title("Pred label")
            axs[1, 1].set_title("Pred error")
            plt.savefig(f"vis/iterative_exp/pred_iter_{i}.png")
            plt.close()
    return errors


def run_exp_default(
    data_manager: MaskedLabeledImage,
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
        data_manager,
        model,
        classification_task=data_manager.is_classification_dataset(),
        n_ensemble_models=7,
    )
    run_exp_custom(**locals())


def compare_planners(
    data_manager,
    predictor,
    planners,
    planner_names,
    each_planners_kwargs,
    n_trials=N_TRIALS,
    n_flights=N_FLIGHTS,
    visit_n_locations=VISIT_N_LOCATIONS,
    savefile="vis/iterative_exp/final_values.png",
    vis=VIS,
):
    """
    Compare planner performance across iterations and multiple random trials
    """
    results = {}
    for planner, planner_name, planner_kwargs in zip(
        planners, planner_names, each_planners_kwargs
    ):
        results[planner_name] = [
            run_exp_custom(
                planner=deepcopy(planner),
                predictor=predictor,
                data_manager=data_manager,
                visit_n_locations=visit_n_locations,
                n_flights=n_flights,
                vis=vis,
                planner_kwargs=planner_kwargs,
            )
            for i in range(n_trials)
        ]

    plt.close()
    plt.cla()
    plt.clf()
    for planner_name, error_values in results.items():
        plot_errors(error_values, planner_name)
    plt.legend()
    plt.xlabel("Number of sampling iterations")
    plt.ylabel("Test error")
    plt.savefig(savefile)
    plt.show()


def compare_random_vs_diversity(
    data_manager: MaskedLabeledImage,
    n_candidate_locations_diversity=200,
    n_trials=N_TRIALS,
    vis_plan=True,
    **kwargs,
):
    planners = [
        BatchDiversityPlanner(
            data_manager, n_candidate_locations=n_candidate_locations_diversity
        ),
        RandomMaskedPlanner(data_manager),
    ]
    planner_names = ["Diversity planner", "Random planner"]
    planner_kwargs = [{"vis": vis_plan}, {"vis": vis_plan}]
    if data_manager.is_classification_dataset():
        model = MLPClassifier()
    else:
        model = MLPRegressor()
    predictor = EnsembledMaskedLabeledImagePredictor(
        data_manager,
        model,
        classification_task=data_manager.is_classification_dataset(),
    )
    compare_planners(
        planners=planners,
        predictor=predictor,
        each_planners_kwargs=planner_kwargs,
        planner_names=planner_names,
        data_manager=data_manager,
        n_trials=n_trials,
        **kwargs,
    )
