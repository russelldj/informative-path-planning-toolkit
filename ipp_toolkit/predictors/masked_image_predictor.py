from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.config import (
    TOP_FRAC,
    TOP_FRAC_MEAN_ERROR,
    MEAN_ERROR_KEY,
    ERROR_IMAGE,
)
from ipp_toolkit.config import MEAN_KEY, UNCERTAINTY_KEY, ERROR_IMAGE
from ipp_toolkit.predictors.uncertain_predictors import (
    EnsamblePredictor,
    UncertainPredictor,
)
from tqdm import tqdm


class MaskedLabeledImagePredictor:
    def __init__(
        self,
        masked_labeled_image: MaskedLabeledImage,
        prediction_model,
        use_locs_for_prediction=False,
        classification_task: bool = True,
        pred_batch_size=1e8,
    ):
        """
        Arguments
            classification_tasks: Is this a classification (not regression) task
        """
        self.masked_labeled_image = masked_labeled_image
        self.prediction_model = prediction_model
        self.use_locs_for_prediction = use_locs_for_prediction
        self.classification_task = classification_task
        self.pred_batch_size = int(pred_batch_size)
        self._setup()

    def get_name(self):
        """Return the human-readable name"""
        return "base_predictor"

    def _setup(self):
        self.prediction_scaler = StandardScaler()
        self.all_prediction_features = None

        self.previous_sampled_locs = np.empty((0, 2))

        self.labeled_prediction_features = None
        self.labeled_prediction_values = None
        self.loc_to_index_map = None

    def _preprocess_features(self):
        # Preprocessing is done
        if not self.all_prediction_features is None:
            return

        if self.use_locs_for_prediction:
            self.all_prediction_features = (
                self.masked_labeled_image.get_valid_loc_images_points()
            )
        else:
            self.all_prediction_features = (
                self.masked_labeled_image.get_valid_image_points()
            )

        self.all_prediction_features = self.prediction_scaler.fit_transform(
            self.all_prediction_features
        )
        feature_inds = np.arange(self.all_prediction_features.shape[0]).astype(int)
        self.loc_to_index_map = self.masked_labeled_image.get_image_for_flat_values(
            feature_inds
        )

    def _get_feature_inds_from_locs(self, locs: np.ndarray):
        """Get the indices into the flat array of features for locs

        Args:
            locs (_type_): Locations (i,j) by n rows to get the index of

        Returns:
            _type_: _description_
        """
        locs = locs.astype(int)
        inds = self.loc_to_index_map[locs[:, 0], locs[:, 1]]
        return inds

    def _get_candidate_location_features(
        self,
        centers: np.ndarray,
        use_locs_for_clustering: bool,
        scaler=None,
    ):
        # TODO this is weird and should be renamed/revisited
        """
        Obtain a feature representation of each location

        Args:
            image_data: image features
            centers: locations to sample at (i, j) ints. Size (n, 2)
            use_locs_for_clustering: include location information in features

        Returns:
            candidate_location_features: (n, m) features for each point
        """
        if centers is None or centers.shape[0] == 0:
            return None

        centers = centers.astype(int)
        features = self.masked_labeled_image.image[centers[:, 0], centers[:, 1]]
        if use_locs_for_clustering:
            features = np.hstack((centers, features))

        if scaler is None:
            scaler = StandardScaler()
            candidate_location_features = scaler.fit_transform(features)
            # TODO raise warning here
        else:
            candidate_location_features = scaler.transform(features)

        return candidate_location_features

    def update_model(self, locs: np.ndarray, values: np.ndarray):
        """
        This is called after the new data has been sampled

        locs: (n,2)
        values: (n,)
        """
        self._preprocess_features()

        self.previous_sampled_locs = np.concatenate(
            (self.previous_sampled_locs, locs), axis=0
        )
        sampled_location_features = self._get_candidate_location_features(
            locs,
            self.use_locs_for_prediction,
            self.prediction_scaler,
        )

        # Update features, dealing with the possibility of the array being empty
        if self.labeled_prediction_features is None:
            self.labeled_prediction_features = sampled_location_features
            self.labeled_prediction_values = values
        else:
            self.labeled_prediction_features = np.concatenate(
                (self.labeled_prediction_features, sampled_location_features), axis=0
            )
            self.labeled_prediction_values = np.concatenate(
                (self.labeled_prediction_values, values), axis=0
            )
        self.prediction_model.fit(
            self.labeled_prediction_features,
            self.labeled_prediction_values,
        )

    def predict_values(self):
        """
        Use prediction model to predict the values for the whole world
        """
        pred_ys = []
        for i in range(0, self.all_prediction_features.shape[0], self.pred_batch_size):
            pred_y = self.prediction_model.predict(
                self.all_prediction_features[i : i + self.pred_batch_size]
            )
            pred_ys.append(pred_y)
        pred_y = np.concatenate(pred_ys, axis=0)
        pred_image_y = self.masked_labeled_image.get_image_for_flat_values(pred_y)
        return pred_image_y

    def predict_all(self):
        """
        This is a convenience function which is shared between all the classes.
        It simply predicts all quantities this predictor can and returns them as a dict
        """
        pred_image_y = self.predict_values()
        return {MEAN_KEY: pred_image_y}


class UncertainMaskedLabeledImagePredictor(MaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image: MaskedLabeledImage,
        uncertain_prediction_model: UncertainPredictor,
        use_locs_for_prediction=False,
        classification_task=True,
    ):
        """
        frac_per_model: what fraction of the data to train each data on
        n_ensamble_models: how many models to use
        """
        self.all_prediction_features = None
        self.use_locs_for_prediction = use_locs_for_prediction
        self.masked_labeled_image = masked_labeled_image
        self.classification_task = classification_task

        self.prediction_model = uncertain_prediction_model

        self._setup()

    def get_name(self):
        """Return the human-readable name"""
        return f"uncertain_predictor_with_{self.prediction_model.get_name()}_model"

    def predict_values(self):
        # TODO try to minimize rewriting
        """
        Use prediction model to predict the values for the whole world
        """
        predicted_mean = self.predict_values_and_uncertainty()[MEAN_KEY]
        return predicted_mean

    def predict_values_and_uncertainty(self):
        self._preprocess_features()
        predictions = self.prediction_model.predict_uncertain(
            self.all_prediction_features
        )
        mean_image = self.masked_labeled_image.get_image_for_flat_values(
            predictions[MEAN_KEY]
        )
        uncertainty_image = self.masked_labeled_image.get_image_for_flat_values(
            predictions[UNCERTAINTY_KEY]
        )
        return {MEAN_KEY: mean_image, UNCERTAINTY_KEY: uncertainty_image}

    def predict_all(self):
        return self.predict_values_and_uncertainty()

    def predict_subset_locs(self, locs):
        """Predict for a subset of locs

        Args:
            locs (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._preprocess_features()
        inds = self._get_feature_inds_from_locs(locs).astype(int)
        subset_features = self.all_prediction_features[inds]
        subset_predictions = self.prediction_model.predict_uncertain(subset_features)
        return subset_predictions


# TODO determine whether we want to keep this around
class EnsambledMaskedLabeledImagePredictor(UncertainMaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image: MaskedLabeledImage,
        prediction_model: UncertainPredictor,
        use_locs_for_prediction=False,
        n_ensamble_models=3,
        frac_per_model: float = 0.5,
        classification_task=True,
    ):
        """
        frac_per_model: what fraction of the data to train each data on
        n_ensamble_models: how many models to use
        """

        uncertain_prediction_model = EnsamblePredictor(
            prediction_model=prediction_model,
            n_ensamble_models=n_ensamble_models,
            frac_per_model=frac_per_model,
            classification_task=classification_task,
        )

        super().__init__(
            masked_labeled_image=masked_labeled_image,
            uncertain_prediction_model=uncertain_prediction_model,
            use_locs_for_prediction=use_locs_for_prediction,
            classification_task=classification_task,
        )
        self._setup()

    def get_name(self):
        """Return the human-readable name"""
        return f"ensambled_predictor_with_{self.prediction_model.get_name()}_model"
