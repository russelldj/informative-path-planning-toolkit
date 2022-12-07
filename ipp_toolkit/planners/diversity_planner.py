from ipp_toolkit.config import PLANNING_RESOLUTION
from ipp_toolkit.planners.planners import GridWorldPlanner
import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from python_tsp.distances.euclidean_distance import euclidean_distance_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from skimage.filters import gaussian

from platypus import NSGAII, Problem, Real, Binary, nondominated


def solve_tsp(points):
    distance_matrix = euclidean_distance_matrix(points)
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    permutation = permutation + [permutation[0]]
    path = points[permutation]
    return path


def compute_candidate_locations(
    features: np.ndarray,
    n_clusters: int,
    loc_samples,
    img_size,
    max_fit_points=None,
    gaussian_sigma: int = 5,
    use_dense_spatial_region: bool = True,
):
    """
    Clusters the image and then finds a large spatial region of similar appearances

    Args:
        features:
        n_clusters:
        loc_samples: 
        img_size:
        max_fit_points: how many points to use for kmeans fitting
        gaussian_sigma: spread of the blur kernel used to sample the region of interest
        use_dense_spatial_region: chose the max location by taking a the center of a region of many similar points
    
    Returns:
        centers: i,j locations of the centers
        cluster_inds: the predicted labels for each point
    """
    standard_scalar = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters)
    # Normalize features elementwise
    features = standard_scalar.fit_transform(features)

    if max_fit_points is None:
        # Fit on all the points
        kmeans.fit(features)
    else:
        # Fit on a subset of points
        sample_inds = np.random.choice(
            features.shape[0], size=(max_fit_points), replace=False
        )
        feature_subset = features[sample_inds, :]
        kmeans.fit(feature_subset)
    # Predict the cluster membership for each data point
    cluster_inds = kmeans.predict(features)

    # Two approaches, one is a dense spatial cluster
    # the other is the point nearest the k-means centroid
    if use_dense_spatial_region:
        # Build an array where each channel is a binary map for one class
        per_class_layers = np.zeros((img_size[0], img_size[1], n_clusters), dtype=bool)
        # Populate the binary map
        per_class_layers[
            loc_samples[:, 0].astype(int), loc_samples[:, 1].astype(int), cluster_inds
        ] = True
        # Smooth the binary map layer-wise
        smoothed_layers = [
            gaussian(per_class_layers[..., i], gaussian_sigma)
            for i in range(n_clusters)
        ]
        # Find the argmax location for each smoothed binary map
        # This is just fancy indexing for that
        centers = [
            np.unravel_index(smoothed_image.flatten().argmax(), smoothed_image.shape)
            for smoothed_image in smoothed_layers
        ]
        centers = np.vstack(centers)
    else:
        # Obtain the distance to each class centroid
        center_dists = kmeans.transform(features)
        closest_points = np.argmin(center_dists, axis=0)
        centers = loc_samples[closest_points].astype(int)
    return centers, cluster_inds


class DiversityPlanner:
    def plan(
        self,
        image_data: MaskedLabeledImage,
        n_locations=8,
        current_location=None,
        n_spectral_bands=5,
        use_locs_for_clustering=True,
        vis=True,
        visit_n_locations=5,
        savepath=None,
        blur_scale=5,
        use_dense_spatial_region_candidates: bool = True,
    ):
        """
        Arguments:
            world_model: the belief of the world
            current_location: The location (n,)
            n_steps: How many planning steps to take
            use_dense_spatial_region_candidates: Select the candidate locations using high spatial density of similar type

        Returns:
            A plan specifying the list of locations
        """
        # Get the spectral data
        image_samples = image_data.get_valid_images_points()[:, :n_spectral_bands]
        # Get the spatial data
        loc_samples = image_data.get_valid_loc_points()

        # Determine whether you want to cluster using spatial data or not
        if use_locs_for_clustering:
            features = np.hstack((loc_samples, image_samples))
        else:
            features = image_samples

        centers, labels = compute_candidate_locations(
            features,
            n_clusters=n_locations,
            loc_samples=loc_samples,
            img_size=image_data.image.shape[:2],
            gaussian_sigma=blur_scale,
            use_dense_spatial_region=use_dense_spatial_region_candidates,
        )

        features = image_data.image[centers[:, 0], centers[:, 1]]
        features_and_centers = np.hstack((centers, features))
        standard_scalar = StandardScaler()
        features_and_centers_normalized = standard_scalar.fit_transform(
            features_and_centers
        )
        # Optimization
        def objective(mask):
            empty_value = np.max(
                cdist(features_and_centers_normalized, features_and_centers_normalized)
            )
            mask = np.array(mask[0])
            num_sampled = np.sum(mask)
            if np.all(mask) or np.all(np.logical_not(mask)):
                return (empty_value, num_sampled)
            sampled = features_and_centers_normalized[mask]
            not_sampled = features_and_centers_normalized[np.logical_not(mask)]
            dists = cdist(sampled, not_sampled)
            min_dists = np.min(dists, axis=0)
            average_min_dist = np.mean(min_dists)
            assert len(min_dists) == (len(mask) - num_sampled)
            return (average_min_dist, num_sampled)

        problem = Problem(1, 2)
        problem.types[:] = Binary(n_locations)
        problem.function = objective
        print("Begining optimization")
        algorithm = NSGAII(problem)
        algorithm.run(1000)
        results = nondominated(algorithm.result)

        results_dict = {int(np.sum(r.variables)): r.variables for r in results}
        # Ensure that the number of samples you want is present
        possible_n_visit_locations = np.array(list(results_dict.keys()))
        diffs = np.abs(possible_n_visit_locations - visit_n_locations)
        visit_n_locations = possible_n_visit_locations[np.argmin(diffs)]
        final_mask = np.squeeze(np.array(results_dict[visit_n_locations]))

        # Take the i, j coordinate
        locs = centers[final_mask]
        if current_location is not None:
            locs = np.concatenate((np.atleast_2d(current_location), locs), axis=0)
        plan = solve_tsp(locs)
        if current_location is not None:
            print("path is not sorted")
        if vis:
            self._vis(results, image_data, centers, plan, n_locations, labels, savepath)
        return plan

    def _vis(self, results, image_data, centers, plan, n_locations, labels, savepath):
        plt.scatter(
            [s.objectives[1] for s in results], [s.objectives[0] for s in results]
        )
        plt.xlabel("Number of sampled locations")
        plt.ylabel("Average distance of unsampled locations")
        plt.pause(5)
        clusters = np.ones(image_data.mask.shape) * np.nan
        clusters[image_data.mask] = labels
        f, axs = plt.subplots(1, 2)
        axs[0].imshow(clusters, cmap="tab20")
        axs[0].plot(plan[:, 1], plan[:, 0], c="k")
        axs[0].scatter(
            centers[:, 1],
            centers[:, 0],
            c=np.arange(n_locations),
            cmap="tab20",
            edgecolors="k",
            label="",
        )
        axs[1].imshow(image_data.image[..., :3])
        axs[1].scatter(
            centers[:, 1],
            centers[:, 0],
            c=np.arange(n_locations),
            cmap="tab20",
            edgecolors="k",
            label="",
        )
        axs[1].plot(plan[:, 1], plan[:, 0], c="k")
        if savepath is not None:
            plt.savefig(savepath, dpi=800)
            plt.pause(5)
            plt.clf()
            plt.cla()
        else:
            plt.show()
