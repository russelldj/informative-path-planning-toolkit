from ipp_toolkit.planners.masked_planner import BaseGriddedPlanner
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.planners.candidate_location_selector import (
    ClusteringCandidateLocationSelector,
    GridCandidateLocationSelector,
)
import matplotlib.pyplot as plt
from ipp_toolkit.planners.utils import order_locations_tsp
import numpy as np


def index_with_cartesian_product(array, inds):
    # TODO profile
    sub_array = array[inds, :]
    sub_sub_array = sub_array[:, inds]
    return sub_sub_array


class MutualInformationPlanner(BaseGriddedPlanner):
    def __init__(self, data: MaskedLabeledImage):
        self.data_manager = data

    def plan(
        self,
        n_samples: int,
        GP_predictor: UncertainMaskedLabeledImagePredictor,
        n_candidates: int = 400,
        vis=False,
    ):
        node_locations = self.get_node_locations(
            GP_predictor=GP_predictor, n_candidates=n_candidates
        )
        # Get the features from the nodes
        scaled_features = GP_predictor._get_candidate_location_features(
            node_locations,
            use_locs_for_clustering=True,
            scaler=GP_predictor.prediction_scaler,
        )

        # Obtain the covariance of the features
        covariance = GP_predictor.prediction_model.predict_covariance(scaled_features)

        selected_inds = self.mutual_info_selection(Sigma=covariance, k=n_samples, V=())
        selected_locs = node_locations[selected_inds]

        # Now it's time to order the features
        ordered_locs = order_locations_tsp(selected_locs)

        if vis:
            self.vis(ordered_locs)
        return ordered_locs

    def get_node_locations(self, GP_predictor, n_candidates, using_clustering=False):
        if using_clustering:
            # Find the clusters in the environment
            clusterer = ClusteringCandidateLocationSelector(
                self.data_manager.image.shape[:2],
                use_dense_spatial_region=False,
                scaler=GP_predictor.prediction_scaler,
            )
            features = self.data_manager.get_valid_loc_images_points()
            locs = self.data_manager.get_valid_loc_points()

            node_locations = clusterer.select_locations(
                features=features,
                mask=self.data_manager.mask,
                loc_samples=locs,
                n_clusters=n_candidates,
            )[0]
        else:
            cluster = GridCandidateLocationSelector(self.data_manager.image.shape[:2])
            node_locations = cluster.select_locations(
                loc_samples=self.data_manager.get_valid_loc_points(),
                n_clusters=n_candidates,
            )[0]
        return node_locations

    def mutual_info_selection(self, Sigma: np.ndarray, k: int, V=(), vis_covar=True):
        """
        Algorithm 1 from
        Near-optimal sensor placements in Gaussian processes:
        Theory, efficient algorithms and empirical studies
        Arguments:
            Sigma: Covariance matrix
            k: the number of samples to take
            V: the indices of samples which cannot be selected

        Returns
            The indices into the set that generated Sigma of the selected locations
        """
        # Can you add additional samples which you want to model well?
        # Cast for higher precision
        Sigma = Sigma.astype(np.float64)
        if vis_covar:
            plt.imshow(Sigma)
            plt.colorbar()
            plt.show()
        A = np.array([], dtype=int)
        n_locs = Sigma.shape[0]
        S = [i for i in range(n_locs) if i not in V]

        # Note j is unused, it is simply a counter
        for j in range(k):
            # Compute the set of not-added points
            A_bar = np.array([i for i in range(n_locs) if i not in A])
            # Extract the covariance for those points
            Sigma_AA = index_with_cartesian_product(Sigma, A)
            Sigma_A_bar_A_bar = index_with_cartesian_product(Sigma, A_bar)
            # Invert the covariance
            Sigma_AA_inv = np.linalg.inv(Sigma_AA)
            Sigma_A_bar_A_bar_inv = np.linalg.inv(Sigma_A_bar_A_bar)

            # Skip ones which we cannot add or those which are already added
            S_minus_A = [i for i in S if i not in A]
            gamma_ys = []
            # Compute the gamma_y values for each element which has not been selected but can be
            for y in S_minus_A:
                sigma_y = Sigma[y, y]
                Sigma_y_A = Sigma[y, A]
                Sigma_y_A_bar = Sigma[y, A_bar]
                Sigma_A_y = Sigma[y, A]
                Sigma_A_bar_y = Sigma[y, A_bar]
                A_prod = np.matmul(Sigma_y_A, np.matmul(Sigma_AA_inv, Sigma_A_y))
                A_bar_prod = np.matmul(
                    Sigma_y_A_bar, np.matmul(Sigma_A_bar_A_bar_inv, Sigma_A_bar_y)
                )
                gamma_y = np.divide(sigma_y - A_prod, sigma_y - A_bar_prod)
                gamma_ys.append(gamma_y)
            # Because of infinities, there may be multiple highest values
            # We want to randomize which one is selected to avoid a bias
            # Select all elements where the highest value occurs
            highest_inds = np.where(np.array(gamma_ys) == np.max(gamma_ys))[0]
            # Chose one of these elements
            highest_ind = np.random.choice(highest_inds)
            # Note that this is an index into
            y_star = S_minus_A[highest_ind]
            A = np.concatenate((A, [y_star]))

        return A
