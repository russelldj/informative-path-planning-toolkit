from ipp_toolkit.predictors.masked_image_predictor import (
    MaskedLabeledImagePredictor,
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.config import TORCH_DEVICE
from ipp_toolkit.predictors.pytorch_predictor import PytorchPredictor
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcess
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np


class KNNClassifierMaskedImagePredictor(MaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image,
        use_locs_for_prediction=False,
        n_neighbors=1,
        knn_kwargs={},
    ):
        prediction_model = KNeighborsClassifier(n_neighbors=n_neighbors, **knn_kwargs)
        super().__init__(
            masked_labeled_image,
            prediction_model=prediction_model,
            use_locs_for_prediction=use_locs_for_prediction,
            classification_task=True,
        )


class PytorchKNN:
    def __init__(self, n_neighbors, device=TORCH_DEVICE, **kwargs):
        self.n_neighbors = n_neighbors
        self.train_X = None
        self.train_y = None
        self.device = device

    def fit(self, X, y):
        self.train_X = torch.Tensor(X).to(self.device)
        self.train_y = torch.Tensor(y).to(int).to(self.device)

    def predict(self, X):
        X = torch.Tensor(X).to(self.device)
        dist = torch.cdist(torch.unsqueeze(self.train_X, 0), torch.unsqueeze(X, 0))
        dist = dist[0]
        knn = dist.topk(self.n_neighbors, dim=0, largest=False)
        inds = knn.indices
        labels = self.train_y[inds]
        most_common_labels = torch.mode(labels, dim=0).values
        np_labels = most_common_labels.detach().cpu().numpy()
        if np.any(np.isnan(np_labels)):
            breakpoint()
        return np_labels


class PytorchKNNClassifierMaskedImagePredictor(MaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image,
        use_locs_for_prediction=False,
        n_neighbors=1,
        knn_kwargs={},
    ):
        prediction_model = PytorchKNN(n_neighbors=n_neighbors, **knn_kwargs)
        super().__init__(
            masked_labeled_image,
            prediction_model=prediction_model,
            use_locs_for_prediction=use_locs_for_prediction,
            classification_task=True,
            pred_batch_size=5e3,
        )


class GaussianProcessMaskedImagePredictor(UncertainMaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image,
        use_locs_for_prediction=False,
        classification_task: bool = True,
        gp_kwargs={},
    ):
        uncertain_prediction_model = GaussianProcess(
            is_classification_task=classification_task, **gp_kwargs
        )
        super().__init__(
            masked_labeled_image,
            uncertain_prediction_model=uncertain_prediction_model,
            use_locs_for_prediction=use_locs_for_prediction,
            classification_task=classification_task,
        )
