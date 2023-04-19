import numpy as np
from ipp_toolkit.config import MEAN_KEY, UNCERTAINTY_KEY, ERROR_IMAGE, MEAN_ERROR_KEY
import matplotlib.pyplot as plt
import numpy as np
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.visualization.utils import show_or_save_plt, add_colorbar
import logging


def visualize_prediction(
    data: MaskedLabeledImage,
    prediction: dict,
    savepath=None,
    executed_plan: np.ndarray = None,
    new_plan: np.ndarray = None,
    verbose=False,
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
    f, axs = plt.subplots(2, 3)
    f.delaxes(axs[1, 2])
    axs[0, 0].imshow(image)
    add_colorbar(axs[0, 1].imshow(uncertainty_pred))
    if data.is_classification_dataset():
        add_colorbar(axs[0, 2].imshow(error_image))
    else:
        max_error = np.nanmax(np.abs(error_image))
        add_colorbar(
            axs[0, 2].imshow(
                error_image, vmin=-max_error, vmax=max_error, cmap="seismic"
            )
        )
    if data.vis_vmin is None and data.vis_vmax is None:
        valid_label_values = label[data.mask]
        valid_label_pred_values = label_pred[data.mask]
        valid_label_and_pred_values = (valid_label_values, valid_label_pred_values)
        vmin = np.min(valid_label_and_pred_values)
        vmax = np.max(valid_label_and_pred_values)
    else:
        vmin = data.vis_vmin
        vmax = data.vis_vmax

    add_colorbar(axs[1, 0].imshow(label, vmin=vmin, vmax=vmax, cmap=data.cmap))
    add_colorbar(axs[1, 1].imshow(label_pred, vmin=vmin, vmax=vmax, cmap=data.cmap))
    for ax in axs.flatten():
        if executed_plan is not None:
            ax.scatter(executed_plan[:, 1], executed_plan[:, 0], c="b")
            ax.plot(executed_plan[:, 1], executed_plan[:, 0], c="b")
        if new_plan is not None:
            ax.scatter(new_plan[:, 1], new_plan[:, 0], c="g")
            ax.plot(new_plan[:, 1], new_plan[:, 0], c="g")
    axs[0, 0].set_title("Image (first three channels)")
    axs[0, 1].set_title("Uncertainty pred")
    axs[0, 2].set_title("Error")
    axs[1, 0].set_title("Label")
    axs[1, 1].set_title("Predicted label")
    show_or_save_plt(savepath=savepath)
    return error_dict
