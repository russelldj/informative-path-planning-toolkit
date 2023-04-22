import numpy as np
import matplotlib.pyplot as plt
import sacred
import itertools
from ipp_toolkit.planners.planners import BasePlanner
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
    UniformInterestingessComputer,
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
    data_manager: MaskedLabeledImage,
    predictor,
    planners_dict: typing.Dict[str, BasePlanner],
    planner_kwargs={"vis": False},
    n_flights=N_FLIGHTS,
    n_samples_per_flight=VISIT_N_LOCATIONS,
    pathlength_per_flight=None,
    savepath_stem=None,
    verbose=True,
    n_trials=10,
    vis_prediction_freq=1,
    _run: sacred.Experiment = None,
):
    """_summary_

    Args:
        data_manager (MaskedLabeledImage): _description_
        predictor (_type_): _description_
        planners_dict (typing.Dict[str, BasePlanner]): _description_
        planner_kwargs (dict, optional): _description_. Defaults to {"vis": False}.
        n_flights (_type_, optional): _description_. Defaults to N_FLIGHTS.
        n_samples_per_flight (_type_, optional): _description_. Defaults to VISIT_N_LOCATIONS.
        pathlength_per_flight (_type_, optional): _description_. Defaults to None.
        savepath_stem (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to True.
        n_trials (int, optional): _description_. Defaults to 10.
        vis_prediction_freq (int, optional): _description_. Defaults to 1.
        _run (sacred.Experiment, optional): _description_. Defaults to None.

    Returns:
        typing.Dict[str, list]: _description_
    """

    results_dict = defaultdict(list)

    for planner_name, planner in planners_dict.items():
        if verbose:
            print(f"Running planner {planner_name}")
        for i in range(n_trials):

            prediction_savepath_template = str(
                Path(
                    savepath_stem,
                    f"planner_{planner_name}:trial_{i:06d}" + "_pred_iter_{:06d}.png",
                )
            )
            vis_predictions = i % vis_prediction_freq == 0
            mission_summary = multi_flight_mission(
                planner=deepcopy(planner),
                data_manager=deepcopy(data_manager),
                predictor=deepcopy(predictor),
                samples_per_flight=n_samples_per_flight,
                planner_kwargs=planner_kwargs,
                n_flights=n_flights,
                pathlength_per_flight=pathlength_per_flight,
                vis_predictions=vis_predictions,
                prediction_savepath_template=prediction_savepath_template,
                _run=_run,
            )
            results_dict[planner_name].append(mission_summary)
    return results_dict


def compare_across_datasets_and_models(
    datasets_dict,
    predictors_dict,
    planners_instantiation_dict,
    n_flights_func,
    n_samples_per_flight_func,
    pathlength_per_flight_func,
    initial_loc_func,
    n_datasets,
    n_trials_per_dataset,
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
    Returns:
        results: typing.Dict[tuple, list] 
            Each key is a config and there is a list of summaries for each dataset.
            Each dataset summary is a dict with planners as the keys and a list of summaries as the values
    """
    # Make the cross product of all configs
    config_tuples = list(
        itertools.product(datasets_dict.keys(), predictors_dict.keys())
    )
    # Repeat each option num_random_trials times
    config_tuples = list(
        itertools.chain.from_iterable((itertools.repeat(config_tuples, n_datasets)))
    )
    np.random.shuffle(config_tuples)

    results_dict = defaultdict(list)
    progress_bar = tqdm(config_tuples)

    for config_tuple in progress_bar:
        progress_bar.set_description(str(config_tuple))
        # Get the instaniation funcs
        dataset_name, predictor_name = config_tuple
        dataset_func = datasets_dict[dataset_name]
        predictor_func = predictors_dict[predictor_name]

        data = dataset_func()
        predictor = predictor_func(data)
        initial_loc = initial_loc_func(data)
        planners_dict = {
            name: planner_cls(data, predictor, initial_loc)
            for name, planner_cls in planners_instantiation_dict.items()
        }
        n_flights = n_flights_func(data)
        n_samples_per_flight = n_samples_per_flight_func(data)
        pathlength_per_flight = pathlength_per_flight_func(data)

        savepath_stem = str(
            Path(VIS_FOLDER, "figures", f"{predictor_name}:dataset_{dataset_name}",)
        )

        dataset_summary = compare_planners(
            data_manager=data,
            predictor=predictor,
            planners_dict=planners_dict,
            n_flights=n_flights,
            n_trials=n_trials_per_dataset,
            n_samples_per_flight=n_samples_per_flight,
            savepath_stem=savepath_stem,
            pathlength_per_flight=pathlength_per_flight,
            _run=_run,
        )
        results_dict[config_tuple].append(dataset_summary)
    return results_dict


def vis_one_metrics(all_metrics_by_planner, metric):
    for planner_name, all_planner_metrics in all_metrics_by_planner.items():
        # All planner metrics is all the runs and each sublist
        metric_values = [
            [single_run_stats[metric] for single_run_stats in runs_stats]
            for runs_stats in all_planner_metrics
        ]
        # Warning, this will only work for scalar metric values
        metric_means = np.mean(metric_values, axis=0)
        metric_stds = np.std(metric_values, axis=0)
        iters = np.arange(len(metric_means))
        plt.plot(iters, metric_means, label=planner_name)
        plt.fill_between(
            iters, metric_means - metric_stds, metric_means + metric_stds, alpha=0.3
        )
    plt.legend()
    plt.show()


def visualize_across_datasets_and_models(
    results_dict: typing.Dict[
        tuple, typing.List[typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]]]
    ],
    metrics: typing.Iterable[str],
):
    """Compare planners across the random trials

    Args:
        results_dict (_type_): _description_
        metric (_type_): a list of metrics to visualize
    """
    # For now, aggregate everything together across all other config choices
    all_datasets_summaries = list(itertools.chain(*list(results_dict.values())))
    # All datasets by planners
    # For each key this should be a list of lists of dicts
    # The outer list is over configs
    # The inner one is over random trials
    # Then the dict is a dict of different statistics
    all_datasets_by_planner = {
        k: [x[k] for x in all_datasets_summaries]
        for k in all_datasets_summaries[0].keys()
    }
    # Flatten the multiple runs per dataset
    all_runs_by_planner = {
        k: list(itertools.chain(*v)) for k, v in all_datasets_by_planner.items()
    }
    # Get just the metrics, ignoring the path and observed values
    all_metrics_by_planner = {
        k: [x["metrics"] for x in v] for k, v in all_runs_by_planner.items()
    }
    # List of dicts, where each key is the planner
    for metric in metrics:
        vis_one_metrics(all_metrics_by_planner=all_metrics_by_planner, metric=metric)


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
