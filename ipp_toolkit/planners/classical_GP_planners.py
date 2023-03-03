from ipp_toolkit.planners.masked_planner import BaseGriddedPlanner
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.planners.candidate_location_selector import (
    ClusteringCandidateLocationSelector,
    GridCandidateLocationSelector,
)
from python_tsp.heuristics import solve_tsp_simulated_annealing, solve_tsp_local_search
import matplotlib.pyplot as plt
from ipp_toolkit.planners.utils import order_locations_tsp
import numpy as np
from tqdm import tqdm
from python_tsp.distances.euclidean_distance import euclidean_distance_matrix
import logging


def index_with_cartesian_product(array, i_inds, j_inds=None):
    if j_inds is None:
        j_inds = i_inds
    # TODO profile
    sub_array = array[i_inds, :]
    sub_sub_array = sub_array[:, j_inds]
    return sub_sub_array


def get_node_locations(
    data_manager, GP_predictor, n_candidates, using_clustering=False
):
    if using_clustering:
        # Find the clusters in the environment
        clusterer = ClusteringCandidateLocationSelector(
            data_manager.image.shape[:2],
            use_dense_spatial_region=False,
            scaler=GP_predictor.prediction_scaler,
        )
        features = data_manager.get_valid_loc_images_points()
        locs = data_manager.get_valid_loc_points()

        node_locations = clusterer.select_locations(
            features=features,
            mask=data_manager.mask,
            loc_samples=locs,
            n_clusters=n_candidates,
        )[0]
    else:
        cluster = GridCandidateLocationSelector(data_manager.image.shape[:2])
        node_locations = cluster.select_locations(
            loc_samples=data_manager.get_valid_loc_points(), n_clusters=n_candidates,
        )[0]
    return node_locations


def get_locs_and_covar(
    data_manager, GP_predictor, n_candidates, start_location=None, end_location=None
):
    node_locations = get_node_locations(
        data_manager=data_manager, GP_predictor=GP_predictor, n_candidates=n_candidates,
    )
    # Prepend the start
    if start_location is not None:
        node_locations = np.concatenate(([start_location], node_locations), axis=0)
    # Append the end
    if end_location is not None:
        node_locations = np.concatenate((node_locations, [end_location]), axis=0)
    # Get the features from the nodes
    scaled_features = GP_predictor._get_candidate_location_features(
        node_locations,
        use_locs_for_clustering=True,
        scaler=GP_predictor.prediction_scaler,
    )

    # Obtain the covariance of the features
    covariance = GP_predictor.prediction_model.predict_covariance(scaled_features)
    return node_locations, covariance


def mutual_info_selection(Sigma: np.ndarray, k: int, V=(), vis_covar=False):
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
    for j in tqdm(range(k)):
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
        if np.any(np.logical_not(np.isfinite(gamma_ys))):
            logging.warn("Infinite result detected in gamma_ys")
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


def open_path_tsp_cost(full_distance_matrix, indices, solver):
    sub_distance_matrix = index_with_cartesian_product(full_distance_matrix, indices)
    # Make the path open
    sub_distance_matrix[:, 0] = 0
    permutations, cost = solver(sub_distance_matrix)
    assert permutations[0] == 0
    return cost


class MutualInformationPlanner(BaseGriddedPlanner):
    def __init__(self, data: MaskedLabeledImage):
        self.data_manager = data

    @classmethod
    def get_planner_name(cls):
        return "diversity_planner"

    def plan(
        self,
        n_samples: int,
        GP_predictor: UncertainMaskedLabeledImagePredictor,
        n_candidates: int = 1000,
        vis=False,
    ):
        node_locations, covariance = get_locs_and_covar(
            data_manager=self.data_manager,
            GP_predictor=GP_predictor,
            n_candidates=n_candidates,
        )
        # Select samples based on greedy mutual information
        selected_inds = mutual_info_selection(Sigma=covariance, k=n_samples, V=())
        selected_locs = node_locations[selected_inds]

        # Now it's time to order the features
        ordered_locs = order_locations_tsp(selected_locs)

        if vis:
            self.vis(ordered_locs)
        return ordered_locs


def compute_entropy_of_covariance_matrix(covar):
    # Referencing this post
    # https://math.stackexchange.com/questions/4377621/how-can-mutual-information-between-random-variables-decrease-with-increasing-cor
    entropy = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * covar))
    return entropy


def mutual_info_value(sampled_inds, covariance_matrix):
    """
    Taken from equation 2 in Efficient Informative Sensing using Multiple Robots

    Referencing this post
    https://math.stackexchange.com/questions/4377621/how-can-mutual-information-between-random-variables-decrease-with-increasing-cor

    Assumes a square covariance matrix
    """
    n_samples = covariance_matrix.shape[0]
    # TODO look at speeding this up with sets
    unsampled_inds = np.array([i for i in range(n_samples) if i not in sampled_inds])
    sampled_inds = np.array(sampled_inds)
    Sigma_sampled = index_with_cartesian_product(covariance_matrix, sampled_inds)
    Sigma_unsampled = index_with_cartesian_product(covariance_matrix, unsampled_inds)

    conditions = [
        np.linalg.cond(x) for x in (Sigma_sampled, Sigma_unsampled, covariance_matrix)
    ]

    sampled_det = np.linalg.det(Sigma_sampled)
    unsampled_det = np.linalg.det(Sigma_unsampled)
    full_det = np.linalg.det(covariance_matrix)
    mutual_information = 0.5 * np.log(sampled_det * unsampled_det / full_det)
    print(mutual_information)
    return mutual_information


class RecursiveGreedyPlanner(BaseGriddedPlanner):
    def __init__(self, data: MaskedLabeledImage):
        """
        Implements A Recursive Greedy Algorithm for Walks in Directed Graphs by Chekuri and Pal
        
        """
        self.data_manager = data

    def recursive_greedy(self, s: int, t: int, B: float, X: list, i: int):
        """
        Algorithm 1 

        Args:
            s: index into the covariance and distance matrix of the start
            t: index into the covariance and distance matrix for the end
            B: The path length budget 
            X: The samples which have already been added
            i: Recursion iteration
        """

        """
        1. If (l(s, t) > B) return Infeasible 
        2. P ← s, t 
        3. Base case: i = 0. return P 
        4. m← fX(P)
        5. For each v ∈ V do
            (a) For 1 ≤ B1 ≤ B do 
                i. P1 ← RG(s, v, B1,X, i - 1)
                ii. P2 ← RG(v, t, B -B1,X U V(P1), i - 1) 
                iii. If (fX(P1 · P2) > m) P ← P1 · P2 m← fX(P)
        6. return P
        """
        # Step 1 is handled as a precondition
        # Step 2
        P = [s, t]
        # Step 3
        P = np.random.choice(
            self.sample_covariance.shape[0], int(self.sample_covariance.shape[0] / 2)
        )
        m = mutual_info_value(P, self.sample_covariance)

    def plan(
        self,
        n_samples: int,
        GP_predictor: UncertainMaskedLabeledImagePredictor,
        start_location,
        end_location=None,
        n_candidates: int = 30,
        budget=1000,
        vis=False,
        tsp_solver=solve_tsp_local_search,
        recursion_depth=10,
    ):
        # Record which solver we'll be using
        self.tsp_solver = tsp_solver

        assert (
            self.data_manager.mask[start_location[0], start_location[1]]
            and self.data_manager.mask[end_location[0], end_location[1]]
        ), "start and end are not both in valid regions"

        # Preliminaries
        # Compute the sampling locations, with the start location prepended and the end
        # location appended. The covariance is predicted by the GP and recorded for later use
        node_locations, self.sample_covariance = get_locs_and_covar(
            data_manager=self.data_manager,
            GP_predictor=GP_predictor,
            n_candidates=n_candidates,
            start_location=start_location,
            end_location=end_location,
        )

        # Compute the euclidean distance matrix for all locations
        self.full_distance_matrix = euclidean_distance_matrix(node_locations)
        s = 0
        t = self.full_distance_matrix.shape[0] - 1

        # Scale the correlation matrix for better stability
        self.sample_covariance /= np.max(self.sample_covariance)
        print(np.linalg.det(self.sample_covariance))

        # Check the start-goal path feasibility outside of the recursion
        if self.full_distance_matrix[s, t] > budget:
            return None
        plt.imshow(self.sample_covariance)
        plt.colorbar()
        plt.show()

        sol = self.recursive_greedy(s=0, t=0, B=budget, X=[], i=recursion_depth)
        breakpoint()

