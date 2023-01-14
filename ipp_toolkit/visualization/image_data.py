import matplotlib.pyplot as plt
from ipp_toolkit.visualization.utils import show_or_save_plt
import numpy as np


def vis_uncertainty_image(uncertainty: np.ndarray, savepath):
    plt.imshow(uncertainty)
    plt.colorbar()
    plt.title("Uncertainty map")
    show_or_save_plt(savepath=savepath)
