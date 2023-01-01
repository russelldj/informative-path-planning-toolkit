from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from sklearn.preprocessing import StandardScaler
import numpy as np


class MaskedLabeledImagePredictor:
    def __init__(
        self, masked_labeled_image, prediction_model, use_locs_for_prediction=False
    ):
        """
        Arguments
        """
        self.masked_labeled_image = masked_labeled_image
        self.prediction_model = prediction_model
        self.use_locs_for_prediction = use_locs_for_prediction

        self.prediction_scaler = StandardScaler()
        self.all_prediction_features = None

        self.previous_sampled_locs = np.empty((0, 2))

        self.labeled_prediction_features = None
        self.labeled_prediction_values = np.empty((0,))

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
        else:
            self.labeled_prediction_features = np.concatenate(
                (self.labeled_prediction_features, sampled_location_features), axis=0
            )
        # Update values, since we can pre-allocate an empty array
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
        self.interestingness_image = self.masked_labeled_image.get_image_for_flat_values(
            pred_y
        )
        return self.interestingness_image
