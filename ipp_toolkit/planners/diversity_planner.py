from ipp_toolkit.config import PLANNING_RESOLUTION
from ipp_toolkit.planners.planners import GridWorldPlanner
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
from sklearn.cluster import KMeans
from python_tsp.distances.euclidean_distance import euclidean_distance_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage


def solve_tsp(points):
    distance_matrix = euclidean_distance_matrix(points)
    print(distance_matrix.shape)
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    permutation = permutation + [permutation[0]]
    path = points[permutation]
    return path


def compute_centers(features, n_clusters, max_fit_points=None):
    standard_scalar = StandardScaler()
    features = standard_scalar.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters)

    if max_fit_points is None:
        kmeans.fit(features)
    else:
        sample_inds = np.random.choice(features.shape[0], size=(max_fit_points))
        kmeans.fit(features[sample_inds, :])
    dists = kmeans.transform(features)
    cluster_inds = kmeans.predict(features)
    # Find the most representative sample for each cluster
    inds = np.argmin(dists, axis=0)
    centers = features[inds]

    centers = standard_scalar.inverse_transform(centers)
    return centers, cluster_inds


class DiversityPlanner:
    def plan(
        self,
        image_data: MaskedLabeledImage,
        n_locations=8,
        current_location=None,
        n_spectral_bands=5,
        vis=True,
    ):
        """
        Arguments:
            world_model: the belief of the world
            current_location: The location (n,)
            n_steps: How many planning steps to take

        Returns:
            A plan specifying the list of locations
        """
        image_samples = image_data.get_valid_images_points()[:, :n_spectral_bands]
        loc_samples = image_data.get_valid_loc_points()
        features = np.hstack((loc_samples, image_samples))
        centers, labels = compute_centers(features, n_clusters=n_locations)
        # Take the i, j coordinate
        locs = centers[:, :2]
        if current_location is not None:
            locs = np.concatenate((np.atleast_2d(current_location), locs), axis=0)
        plan = solve_tsp(locs)
        if current_location is not None:
            print("path is not sorted")
        if vis:
            clusters = np.ones(image_data.mask.shape) * np.nan
            clusters[image_data.mask] = labels
            plt.imshow(clusters, cmap="tab20")
            plt.plot(plan[:, 1], plan[:, 0], c="k")
            plt.scatter(
                centers[:, 1],
                centers[:, 0],
                c=np.arange(n_locations),
                cmap="tab20",
                edgecolors="k",
                label="",
            )
            plt.show()
        return plan
