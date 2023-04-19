from ipp_toolkit.predictors.masked_image_predictor import (
    MaskedLabeledImagePredictor,
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.predictors.pytorch_predictor import PytorchPredictor
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcess
from sklearn.neighbors import KNeighborsClassifier


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

