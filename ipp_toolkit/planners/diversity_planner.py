import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from python_tsp.distances.euclidean_distance import euclidean_distance_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
from ipp_toolkit.utils.optimization.optimization import topsis
from skimage.filters import gaussian
import time


from platypus import NSGAII, Problem, Binary, nondominated

from ipp_toolkit.config import (
    CLUSTERING_ELAPSED_TIME,
    TSP_ELAPSED_TIME,
    OPTIMIZATION_ELAPSED_TIME,
)


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
        n_optimization_iters=1000,
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
        self.log_dict = {}
        # Get the spectral data
        image_samples = image_data.get_valid_images_points()[:, :n_spectral_bands]
        # Get the spatial data
        loc_samples = image_data.get_valid_loc_points()

        # Determine whether you want to cluster using spatial data or not
        if use_locs_for_clustering:
            features = np.hstack((loc_samples, image_samples))
        else:
            features = image_samples

        # Get the candiate regions
        centers, labels = self._compute_candidate_locations(
            features,
            n_clusters=n_locations,
            loc_samples=loc_samples,
            img_size=image_data.image.shape[:2],
            gaussian_sigma=blur_scale,
            use_dense_spatial_region=use_dense_spatial_region_candidates,
        )

        # Get the features to use for optimization
        candidate_location_features = self._get_candidate_location_features(
            image_data, centers, use_locs_for_clustering
        )
        # Compute the pareto front of possible plans
        pareto_results = self._compute_optimal_subset(
            candidate_location_features=candidate_location_features,
            n_optimization_iters=n_optimization_iters,
        )
        selected_locations_mask = self._get_solution_from_pareto(
            pareto_results=pareto_results, visit_n_locations=visit_n_locations
        )

        # Take the i, j coordinate
        selected_locs = centers[selected_locations_mask]
        if current_location is not None:
            selected_locs = np.concatenate(
                (np.atleast_2d(current_location), selected_locs), axis=0
            )
            # TODO make this a warning
            # TODO actually fix this so it returns a sorted path
            print(
                "Warning: path is not sorted and may not start from the current location"
            )

        # Execute the shortest path solver on the set of selected locations
        plan = self._solve_tsp(selected_locs)

        if vis:
            self._visualize_plan(
                pareto_results, image_data, centers, plan, n_locations, labels, savepath
            )
        return plan

    def _solve_tsp(self, points):
        start_time = time.time()
        distance_matrix = euclidean_distance_matrix(points)
        permutation, _ = solve_tsp_simulated_annealing(distance_matrix)
        permutation = permutation + [permutation[0]]
        path = points[permutation]
        self.log_dict[TSP_ELAPSED_TIME] = time.time() - start_time
        return path

    def _compute_candidate_locations(
        self,
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
        start_time = time.time()

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
            per_class_layers = np.zeros(
                (img_size[0], img_size[1], n_clusters), dtype=bool
            )
            # Populate the binary map
            per_class_layers[
                loc_samples[:, 0].astype(int),
                loc_samples[:, 1].astype(int),
                cluster_inds,
            ] = True
            # Smooth the binary map layer-wise
            smoothed_layers = [
                gaussian(per_class_layers[..., i], gaussian_sigma)
                for i in range(n_clusters)
            ]
            # Find the argmax location for each smoothed binary map
            # This is just fancy indexing for that
            centers = [
                np.unravel_index(
                    smoothed_image.flatten().argmax(), smoothed_image.shape
                )
                for smoothed_image in smoothed_layers
            ]
            centers = np.vstack(centers)
        else:
            # Obtain the distance to each class centroid
            center_dists = kmeans.transform(features)
            closest_points = np.argmin(center_dists, axis=0)
            centers = loc_samples[closest_points].astype(int)

        self.log_dict[CLUSTERING_ELAPSED_TIME] = time.time() - start_time
        return centers, cluster_inds

    def _get_candidate_location_features(
        self, image_data: np.ndarray, centers: np.ndarray, use_locs_for_clustering: bool
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
        features = image_data.image[centers[:, 0], centers[:, 1]]
        if use_locs_for_clustering:
            features = np.hstack((centers, features))
        standard_scalar = StandardScaler()
        candidate_location_features = standard_scalar.fit_transform(features)
        return candidate_location_features

    def _compute_optimal_subset(
        self, candidate_location_features: np.ndarray, n_optimization_iters: int
    ):
        """ 
        Compute the best subset of locations to visit from a set of candidates

        Args:
            candidate_location_features
            n_optimization_iters: number of iterations of NSGA-II to perform

        Returns:
            pareto_results: the set of pareto-optimal results
        """
        start_time = time.time()

        # Optimization objective
        def objective(mask):
            empty_value = np.max(
                cdist(candidate_location_features, candidate_location_features)
            )
            mask = np.array(mask[0])
            num_sampled = np.sum(mask)
            if np.all(mask) or np.all(np.logical_not(mask)):
                return (empty_value, num_sampled)
            sampled = candidate_location_features[mask]
            not_sampled = candidate_location_features[np.logical_not(mask)]
            dists = cdist(sampled, not_sampled)
            min_dists = np.min(dists, axis=0)
            average_min_dist = np.mean(min_dists)
            assert len(min_dists) == (len(mask) - num_sampled)
            return (average_min_dist, num_sampled)

        n_locations = candidate_location_features.shape[0]

        problem = Problem(1, 2)
        problem.types[:] = Binary(n_locations)
        problem.function = objective
        algorithm = NSGAII(problem)
        algorithm.run(n_optimization_iters)
        pareto_results = nondominated(algorithm.result)

        self.log_dict[OPTIMIZATION_ELAPSED_TIME] = time.time() - start_time
        return pareto_results

    def _get_solution_from_pareto(
        self,
        pareto_results: list,
        visit_n_locations: int,
        min_visit_locations: int,
        use_topsis=True,
    ):
        """
        Select a solution from the pareto front. Currently, we just select one
        that has a given number of visited locations

        Args:
            pareto_results: list of pareto solutions
            visit_n_locations: how many points to visit
        """
        if use_topsis:
            pareto_values = np.array([s.objectives for s in pareto_results])
            _, topsis_index = topsis(parateo_values=pareto_values)
            final_mask = np.squeeze(pareto_results[topsis_index].variables)
        else:
            results_dict = {
                int(np.sum(r.variables)): r.variables for r in pareto_results
            }
            # Ensure that the number of samples you want is present
            possible_n_visit_locations = np.array(list(results_dict.keys()))
            diffs = np.abs(possible_n_visit_locations - visit_n_locations)
            visit_n_locations = possible_n_visit_locations[np.argmin(diffs)]
            final_mask = np.squeeze(np.array(results_dict[visit_n_locations]))

        return final_mask

    def _visualize_plan(
        self, results, image_data, centers, plan, n_locations, labels, savepath
    ):
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
