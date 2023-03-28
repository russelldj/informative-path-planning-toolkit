from ipp_toolkit.planners.masked_planner import BaseGriddedPlanner
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.config import UNCERTAINTY_KEY
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def image_argmax(img: np.ndarray, n_samples: int):
    n_columns = img.shape[1]
    flat_img = img.flatten()
    sorted_inds = np.argsort(flat_img)
    top_n_inds = sorted_inds[-n_samples]
    i_values = top_n_inds // n_columns
    j_values = top_n_inds % n_columns
    ij_values = np.vstack((i_values, j_values)).T
    return ij_values


class GreedyEntropyPlanner(BaseGriddedPlanner):
    def __init__(
        self, data: MaskedLabeledImage, predictor: UncertainMaskedLabeledImagePredictor
    ):
        self.data = data
        self.predictor = deepcopy(predictor)

    def plan(self, n_samples: int, current_loc=None, vis=False):
        """
        Generate samples by taking the highest entropy sample
        after fitting the model on all previous samples

        Args:
            n_samples (int): How many to sample 
            current_loc (_type_, optional): _description_. Defaults to None.
            vis (bool, optional): Should you visualize entropies. Defaults to False.

        Returns:
            _type_: plan
        """
        plan = []
        for _ in range(n_samples):
            uncertainty = self.predictor.predict_values_and_uncertainty()[
                UNCERTAINTY_KEY
            ]
            next_loc = image_argmax(uncertainty, n_samples=1)
            if vis:
                plt.scatter(next_loc[:, 1], next_loc[:, 0], c="k")
                plt.imshow(uncertainty)
                plt.colorbar()
                plt.show()
            self.predictor.update_model(next_loc, np.zeros(next_loc.shape[0]))
            plan.append(next_loc)
        plan = np.concatenate(plan, axis=0)
        return plan

