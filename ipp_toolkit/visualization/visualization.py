import numpy as np
from ipp_toolkit.config import MEAN_KEY, UNCERTAINTY_KEY, ERROR_IMAGE
import matplotlib.pyplot as plt
import numpy as np
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage


def visualize_prediction(data: MaskedLabeledImage, prediction, predictor):
    image = data.image[..., :3].copy()
    label_pred = prediction[MEAN_KEY].copy().astype(float)
    uncertainty_pred = prediction[UNCERTAINTY_KEY].copy().astype(float)
    error_image = predictor.get_errors()[ERROR_IMAGE].copy().astype(float)
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
    plt.colorbar(axs[0, 1].imshow(uncertainty_pred), ax=axs[0, 1])
    if data.is_classification_dataset():
        plt.colorbar(axs[0, 2].imshow(error_image), ax=axs[0, 2])
    else:
        max_error = np.max(np.abs(error_image))
        plt.colorbar(
            axs[0, 2].imshow(
                error_image, vmin=-max_error, vmax=max_error, cmap="seismic"
            ),
            ax=axs[0, 2],
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

    plt.colorbar(
        axs[1, 0].imshow(label, vmin=vmin, vmax=vmax, cmap=data.cmap), ax=axs[1, 0],
    )
    plt.colorbar(
        axs[1, 1].imshow(label_pred, vmin=vmin, vmax=vmax, cmap=data.cmap),
        ax=axs[1, 1],
    )
    axs[0, 0].set_title("Image (first three channels)")
    axs[0, 1].set_title("Uncertainty pred")
    axs[0, 2].set_title("Error")
    axs[1, 0].set_title("Label")
    axs[1, 1].set_title("Predicted label")
    plt.show()
