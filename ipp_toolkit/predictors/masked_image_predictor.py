from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from ipp_toolkit.config import (
    TOP_FRAC,
    TOP_FRAC_MEAN_ERROR,
    MEAN_ERROR_KEY,
    ERROR_IMAGE,
)
from sklearn.base import clone
from ipp_toolkit.config import MEAN_KEY, UNCERTAINTY_KEY, ERROR_IMAGE
from sklearn.metrics import accuracy_score


class MaskedLabeledImagePredictor:
    def __init__(
        self,
        masked_labeled_image,
        prediction_model,
        use_locs_for_prediction=False,
        classification_task: bool = True,
    ):
        """
        Arguments
            classification_tasks: Is this a classification (not regression) task
        """
        self.masked_labeled_image = masked_labeled_image
        self.prediction_model = prediction_model
        self.use_locs_for_prediction = use_locs_for_prediction
        self.classification_task = classification_task

        self.prediction_scaler = StandardScaler()
        self.all_prediction_features = None

        self.previous_sampled_locs = np.empty((0, 2))

        self.labeled_prediction_features = None
        self.labeled_prediction_values = None

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

    def _get_candidate_location_features(
        self, centers: np.ndarray, use_locs_for_clustering: bool, scaler=None,
    ):
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
            locs, self.use_locs_for_prediction, self.prediction_scaler,
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
            self.labeled_prediction_features, self.labeled_prediction_values
        )

    def predict_values(self):
        """
        Use prediction model to predict the values for the whole world
        """
        pred_y = self.prediction_model.predict(self.all_prediction_features)
        pred_image_y = self.masked_labeled_image.get_image_for_flat_values(pred_y)
        return pred_image_y

    def get_errors(self, ord=2):
        """
        Arguments:
            ord: The order of the error norm
        """
        pred = self.predict_values()
        flat_label = self.masked_labeled_image.get_valid_label_points()
        flat_pred = pred[self.masked_labeled_image.mask]
        if self.classification_task:
            accuracy = accuracy_score(flat_label, flat_pred)
            error_image = pred != self.masked_labeled_image.label
            return_dict = {MEAN_ERROR_KEY: 1 - accuracy, ERROR_IMAGE: error_image}
        else:
            flat_error = flat_pred - flat_label
            sorted_inds = np.argsort(flat_label)
            # Find the indices for the top fraction of ground truth points
            top_frac_inds = sorted_inds[-int(TOP_FRAC * len(sorted_inds)) :]
            top_frac_errors = flat_error[top_frac_inds]
            error_image = pred - self.masked_labeled_image.label
            error_image[np.logical_not(self.masked_labeled_image.mask)] = np.nan
            return_dict = {
                TOP_FRAC_MEAN_ERROR: np.linalg.norm(top_frac_errors, ord=ord),
                MEAN_ERROR_KEY: np.linalg.norm(flat_error, ord=ord),
                ERROR_IMAGE: error_image,
            }
        return return_dict


class EnsembledMaskedLabeledImagePredictor(MaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image,
        prediction_model,
        use_locs_for_prediction=False,
        n_ensemble_models=3,
        frac_per_model: float = 0.5,
        classification_task: bool = True,
    ):
        """
        frac_per_model: what fraction of the data to train each data on
        n_ensemble_models: how many models to use
        """
        self.masked_labeled_image = masked_labeled_image
        self.n_ensemble_models = n_ensemble_models
        self.frac_per_model = frac_per_model
        self.classification_task = classification_task

        # Create a collection of independent predictors. Each one will be fit on a subset of data
        self.estimators = [
            MaskedLabeledImagePredictor(
                self.masked_labeled_image,
                prediction_model=clone(prediction_model),
                use_locs_for_prediction=use_locs_for_prediction,
            )
            for _ in range(n_ensemble_models)
        ]

    def update_model(self, locs: np.ndarray, values: np.ndarray):
        """
        This is called after the new data has been sampled

        locs: (n,2)
        values: (n,)
        """
        for i in range(self.n_ensemble_models):
            n_points = locs.shape[0]
            chosen_loc_inds = np.random.choice(
                n_points, size=int(n_points * self.frac_per_model), replace=False
            )
            chosen_locs = locs[chosen_loc_inds]
            chosen_values = values[chosen_loc_inds]
            self.estimators[i].update_model(chosen_locs, chosen_values)

    def predict_values(self):
        """
        Use prediction model to predict the values for the whole world
        """
        predicted_mean = self.predict_values_and_uncertainty()[MEAN_KEY]
        return predicted_mean

    def predict_values_and_uncertainty(self):
        # Generate a prediction with each model
        predictions = [e.predict_values() for e in self.estimators]
        # Average over all the models
        if not self.classification_task:
            # The mean and unceratinty are just the mean and variance across all models
            mean_prediction = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
        else:
            # Get the max valid pixel
            max_class = (
                np.max([p[self.masked_labeled_image.mask] for p in predictions]).astype(
                    int
                )
                + 1
            )
            # Encode predicts as a count of one-hot predictions for subsequent proceessing
            one_hot_predictions = np.zeros(
                (predictions[0].shape[0], predictions[0].shape[1], max_class)
            )
            # Initialize uncertainty
            uncertainty = np.zeros((predictions[0].shape[0], predictions[0].shape[1]))
            for pred in predictions:
                # TODO determine if some type of advanced indexing can be used here
                for i in range(max_class):
                    one_hot_predictions[(pred == i), i] += 1
            mean_prediction = np.argmax(one_hot_predictions, axis=2)
            for i in range(max_class):
                # Find the pixels where class i was not the mode-predicted class
                does_not_match_mode_pred = mean_prediction != i
                # Figure out how many predictors predicted this class to be correct, if it wasn't voted on by the mode
                num_to_update = one_hot_predictions[..., i][does_not_match_mode_pred]
                # Update the uncertainty by this number
                uncertainty[does_not_match_mode_pred] += num_to_update
            # Normalize it so the uncertainty is never more than 1
            uncertainty /= self.n_ensemble_models

        return_dict = {MEAN_KEY: mean_prediction, UNCERTAINTY_KEY: uncertainty}
        return return_dict

