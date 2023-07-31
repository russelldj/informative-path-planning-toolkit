import numpy as np
from ipp_toolkit.config import (
    MEAN_KEY,
    UNCERTAINTY_KEY,
    ERROR_IMAGE,
    MEAN_ERROR_KEY,
    BIG_FIG_SIZE,
    VIS_FOLDER,
)
import typing
from pathlib import Path
from ipp_toolkit.visualization.utils import show_or_save_plt
import matplotlib.pyplot as plt
import itertools
import numpy as np
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.visualization.utils import show_or_save_plt, add_colorbar
import logging


def vis_one_metrics(all_metrics_by_planner, metric, _run=None):
    for planner_name, all_planner_metrics in all_metrics_by_planner.items():
        # All planner metrics is all the runs and each sublist
        metric_values = [
            [single_run_stats[metric] for single_run_stats in runs_stats]
            for runs_stats in all_planner_metrics
        ]
        # Warning, this will only work for scalar metric values
        metric_means = np.mean(metric_values, axis=0)
        metric_stds = np.std(metric_values, axis=0)
        iters = np.arange(len(metric_means)) + 1
        plt.plot(iters, metric_means, label=planner_name)
        plt.fill_between(
            iters, metric_means - metric_stds, metric_means + metric_stds, alpha=0.3
        )
        plt.xticks(iters)

    plt.ylabel(metric)
    plt.xlabel("Flight number")
    plt.title(f"Plot for {metric} against iterations")
    plt.legend()
    savepath = Path(VIS_FOLDER, f"{metric}.png")
    show_or_save_plt(savepath=savepath, _run=_run)
    savepath = Path(VIS_FOLDER, f"{metric}.npy")
    np.save(savepath, metric_values)
    if _run is not None:
        _run.add_artifact(savepath)


def visualize_across_datasets_and_models(
    results_dict: typing.Dict[
        tuple, typing.List[typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]]]
    ],
    metrics: typing.Iterable[str],
    _run=None,
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
        vis_one_metrics(
            all_metrics_by_planner=all_metrics_by_planner, metric=metric, _run=_run
        )


def visualize_prediction(
    data: MaskedLabeledImage,
    prediction: dict,
    savepath=None,
    executed_plan: np.ndarray = None,
    new_plan: np.ndarray = None,
    verbose=False,
    fig_size=BIG_FIG_SIZE,
):
    """
    Takes a dataset and the prediction and visualizes several quantities

    Args:
        data: the dataset
        prediction: The prediction dictionary containing at least the
                    MEAN_KEY and UNCERTAINTY_KEY. TODO update to accept only
                    the mean key.
        executed_plan: Previous plan
        new_plan: New plan
        savepath: where to save, or show if None

    Returns:
        error_dict containing all the computed metrics
    """
    image = data.image[..., :3].copy()
    label_pred = prediction[MEAN_KEY].copy().astype(float)

    if UNCERTAINTY_KEY in prediction:
        uncertainty_pred = prediction[UNCERTAINTY_KEY].copy().astype(float)
    else:
        uncertainty_pred = np.full_like(label_pred, fill_value=np.nan)
    error_dict = data.eval_prediction(prediction)

    if verbose:
        print(f"Error is {error_dict[MEAN_ERROR_KEY]}")

    error_image = error_dict[ERROR_IMAGE].astype(float)

    label = data.label.copy().astype(float)

    for x in (label_pred, uncertainty_pred, error_image, label):
        x[np.logical_not(data.mask)] = np.nan

    if image.dtype is float:
        image[np.logical_not(data.mask)] = np.nan
    else:
        image[np.logical_not(data.mask)] = 0

    plt.close()
    f, axs = plt.subplots(1, 5)
    if data.vis_image is not None:
        axs[0].imshow(data.vis_image[..., :3])
    axs[1].imshow(np.clip(image / 6 + 0.5, 0, 1))

    # add_colorbar(axs[0, 1].imshow(uncertainty_pred))
    if data.vis_vmin is None and data.vis_vmax is None:
        valid_label_values = label[data.mask]
        valid_label_pred_values = label_pred[data.mask]
        valid_label_and_pred_values = (valid_label_values, valid_label_pred_values)
        vmin = np.min(valid_label_and_pred_values)
        vmax = np.max(valid_label_and_pred_values)
    else:
        vmin = data.vis_vmin
        vmax = data.vis_vmax

    axs[2].imshow(label, vmin=vmin, vmax=vmax, cmap=data.cmap, interpolation="nearest")
    axs[3].imshow(
        label_pred, vmin=vmin, vmax=vmax, cmap=data.cmap, interpolation="nearest"
    )
    if data.is_classification_dataset():
        axs[4].imshow(error_image)
    else:
        max_error = np.nanmax(np.abs(error_image))
        axs[4].imshow(error_image, vmin=-max_error, vmax=max_error, cmap="seismic")

    for ax in axs.flatten():
        if executed_plan is not None:
            len_path = new_plan.shape[0]
            for i in range(int(executed_plan.shape[0] / len_path)):
                ax.scatter(
                    executed_plan[i * len_path : (i + 1) * len_path, 1],
                    executed_plan[i * len_path : (i + 1) * len_path, 0],
                )
                ax.plot(
                    executed_plan[i * len_path : (i + 1) * len_path, 1],
                    executed_plan[i * len_path : (i + 1) * len_path, 0],
                )
        if new_plan is not None:
            ax.scatter(new_plan[:, 1], new_plan[:, 0])
            ax.plot(new_plan[:, 1], new_plan[:, 0])
        ax.axis("off")

    axs[0].set_title("Image", size=18)
    axs[1].set_title("Features (first three channels)", size=18)
    axs[2].set_title("Label", size=18)
    axs[3].set_title("Predicted label", size=18)
    axs[4].set_title("Error", size=18)
    show_or_save_plt(savepath=savepath, fig_size=fig_size)
    return error_dict
