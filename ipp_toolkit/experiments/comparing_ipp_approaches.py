import numpy as np
import matplotlib.pyplot as plt
import sacred
import itertools
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
    VIS_FOLDER,
)
import typing
import logging
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from ipp_toolkit.predictors.intrestingness_computers import (
    UncertaintyInterestingessComputer,
    BaseInterestingessComputer,
)

from sklearn.neural_network import MLPClassifier, MLPRegressor
from ipp_toolkit.planners.diversity_planner import BatchDiversityPlanner
from ipp_toolkit.experiments.missions import multi_flight_mission
from ipp_toolkit.predictors.masked_image_predictor import (
    EnsambledMaskedLabeledImagePredictor,
)
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from warnings import warn
from ipp_toolkit.visualization.utils import show_or_save_plt
from pathlib import Path
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcess


def plot_errors(all_l2_errors, run_tag):
    all_l2_errors = np.vstack(all_l2_errors)
    mean = np.mean(all_l2_errors, axis=0)
    std = np.std(all_l2_errors, axis=0)
    x = np.arange(len(mean))
    plt.plot(x, mean, label=f"{run_tag} mean and one std bound")
    plt.fill_between(x, mean - std, mean + std, alpha=0.3)


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
    Compute the error for one planner over multiple flights
    """

    warn(
        "This is deprecated, use multi_flight_mission", DeprecationWarning, stacklevel=2
    )
    errors = []
    for i in range(n_flights):
        savepath = Path(VIS_FOLDER, f"iterative_exp/no_revisit_plan_iter_{i}.png")
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
        error_dict = data_manager.eval_prediction(pred)
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
            axs[0, 0].set_title("Satellite")
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
        planner = RandomSamplingMaskedPlanner(data_manager)
    else:
        planner = BatchDiversityPlanner(data_manager, n_candidate_locations=n_clusters)
    # Create the predictor
    predictor = EnsambledMaskedLabeledImagePredictor(
        data_manager,
        model,
        classification_task=data_manager.is_classification_dataset(),
        n_ensamble_models=7,
    )
    run_exp_custom(**locals())


def compare_planners(
    data_manager,
    predictor,
    planners,
    each_planners_kwargs,
    interestingness_computer: BaseInterestingessComputer = UncertaintyInterestingessComputer(),
    n_trials=N_TRIALS,
    n_flights=N_FLIGHTS,
    visit_n_locations=VISIT_N_LOCATIONS,
    savepath_stem=None,
    verbose=True,
    vis_prediction_freq: bool = 10,
    _run: sacred.Experiment = None,
):
    """
    Compare planner performance across iterations and multiple random trials
    """
    results = {}
    # Get random start locs
    start_locs = data_manager.get_random_valid_loc_points(n_points=n_trials).astype(int)
    for planner, planner_kwargs in zip(planners, each_planners_kwargs):
        planner_name = planner.get_planner_name()
        if verbose:
            print(f"Running planner {planner_name}")

        results[planner_name] = [
            # TODO Migrate to multi_flight_mission
            multi_flight_mission(
                planner=deepcopy(planner),
                data_manager=deepcopy(data_manager),
                predictor=deepcopy(predictor),
                interestingness_computer=interestingness_computer,
                locations_per_flight=visit_n_locations,
                start_loc=start_locs[i],
                n_flights=n_flights,
                planner_kwargs=planner_kwargs,
                vis_predictions=i % vis_prediction_freq == 0,
                prediction_savepath_template=str(
                    Path(
                        savepath_stem,
                        f"planner_{planner_name}:trial_{i:06d}"
                        + "_pred_iter_{:06d}.png",
                    )
                ),
                _run=_run,
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
    error_savefile = savepath_stem + ".png"
    show_or_save_plt(savepath=error_savefile)
    if _run is not None:
        _run.add_artifact(error_savefile)
    return results


def compare_across_datasets_and_models(
    datasets_dict,
    predictors_dict,
    planners_dict,
    n_missions,
    n_samples_per_mission,
    path_budget_per_mission,
    n_random_trials,
    _run: sacred.Experiment = None,
):
    """_summary_

    Args:
        datasets_names (_type_): _description_
        predictors_dict (_type_): _description_
        planners_dict (_type_): _description_
        n_missions (_type_): _description_
        n_samples_per_mission (_type_): _description_
        path_budget_per_mission (_type_): _description_
        _run (sacred.Experiment, optional): _description_. Defaults to None.
    """
    # Make the cross product of all configs
    config_tuples = list(
        itertools.product(
            datasets_dict.keys(), predictors_dict.keys(), planners_dict.keys()
        )
    )
    # Repeat each option num_random_trials times
    config_tuples = list(
        itertools.chain.from_iterable(
            (itertools.repeat(config_tuples, n_random_trials))
        )
    )
    np.random.shuffle(config_tuples)

    results_dict = defaultdict(list)
    progress_bar = tqdm(config_tuples)
    from ipp_toolkit.data.domain_data import (
        ChesapeakeBayNaipLandcover7ClassificationData,
    )

    for config_tuple in progress_bar:
        progress_bar.set_description(str(config_tuple))
        # Get the instaniation funcs
        dataset_func = datasets_dict[config_tuple[0]]
        predictor_func = predictors_dict[config_tuple[1]]
        planner_func = planners_dict[config_tuple[2]]

        data = dataset_func()
        predictor = predictor_func(data)
        planner = planner_func(data, predictor)
        data.vis()
        results_dict[config_tuple].append("result")
    breakpoint()
    # full_output_dict = {}
    # for data_manager in data_managers:
    #    data_savepath = str(
    #        Path(VIS_FOLDER, "datasets", data_manager.get_dataset_name() + ".png")
    #    )
    #    data_manager.vis(savepath=data_savepath)
    #    if _run is not None:
    #        _run.add_artifact(data_savepath)

    #    data_manager_dict = {}
    #    planners = [
    #        planner_func(data_manager) for planner_func in planner_instantiation_funcs
    #    ]
    #    planner_kwargs = [
    #        planner_kwarg_func(data_manager)
    #        for planner_kwarg_func in planner_kwarg_funcs
    #    ]
    #    for predictor_instantiation_func in predictor_instantiation_funcs:
    #        predictor = predictor_instantiation_func(data_manager)

    #        # TODO make this more general
    #        if (
    #            isinstance(predictor.prediction_model, GaussianProcess)
    #            and data_manager.is_classification_dataset()
    #        ):
    #            continue
    #        savepath_stem = str(
    #            Path(
    #                VIS_FOLDER,
    #                "figures",
    #                f"{predictor.get_name()}:dataset_{data_manager.get_dataset_name()}",
    #            )
    #        )

    #        results = compare_planners(
    #            data_manager=data_manager,
    #            predictor=predictor,
    #            planners=planners,
    #            each_planners_kwargs=planner_kwargs,
    #            savepath_stem=savepath_stem,
    #            _run=_run,
    #            **kwargs,
    #        )
    #        # Compute some sort of ID which is the name of the predictor


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
        RandomSamplingMaskedPlanner(data_manager),
    ]
    planner_names = ["Diversity planner", "Random planner"]
    planner_kwargs = [{"vis": vis_plan}, {"vis": vis_plan}]
    if data_manager.is_classification_dataset():
        model = MLPClassifier()
    else:
        model = MLPRegressor()
    predictor = EnsambledMaskedLabeledImagePredictor(
        data_manager,
        model,
        classification_task=data_manager.is_classification_dataset(),
    )
    compare_planners(
        planners=planners,
        predictor=predictor,
        each_planners_kwargs=planner_kwargs,
        data_manager=data_manager,
        n_trials=n_trials,
        **kwargs,
    )
