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


def compute_centers(features, n_clusters, loc_samples, max_fit_points=None):
    standard_scalar = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters)
    if max_fit_points is None:
        features = standard_scalar.fit_transform(features)
        kmeans.fit(features)
        dists = kmeans.transform(features)
    else:
        sample_inds = np.random.choice(features.shape[0], size=(max_fit_points))
        feature_subset = features[sample_inds, :]
        transformed_feature_subset = standard_scalar.fit_transform(feature_subset)
        kmeans.fit(transformed_feature_subset)
        transformed_features = standard_scalar.transform(features)
        dists = kmeans.transform(transformed_features)
    cluster_inds = kmeans.predict(features)
    # Find the most representative sample for each cluster
    inds = np.argmin(dists, axis=0)
    centers = loc_samples[inds]

    return centers, cluster_inds


def compute_centers_density(
    features, n_clusters, loc_samples, img_size, max_fit_points=None, gaussian_sigma=5
):
    standard_scalar = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters)
    features = standard_scalar.fit_transform(features)
    if max_fit_points is None:
        kmeans.fit(features)
    else:
        sample_inds = np.random.choice(features.shape[0], size=(max_fit_points))
        feature_subset = features[sample_inds, :]
        kmeans.fit(feature_subset)
    cluster_inds = kmeans.predict(features)
    per_class_layers = np.zeros((img_size[0], img_size[1], n_clusters), dtype=bool)
    per_class_layers[
        loc_samples[:, 0].astype(int), loc_samples[:, 1].astype(int), cluster_inds
    ] = True

    smoothed_layers = [
        gaussian(per_class_layers[..., i], gaussian_sigma) for i in range(n_clusters)
    ]
    # Find the most representative sample for each cluster
    centers = [
        np.unravel_index(smoothed_image.flatten().argmax(), smoothed_image.shape)
        for smoothed_image in smoothed_layers
    ]
    centers = np.vstack(centers)
    return centers, cluster_inds


def topsis(parateo_values):
    neg_ideal = np.min(parateo_values, keepdims=True, axis=0)
    pos_ideal = np.max(parateo_values, keepdims=True, axis=0)

    pos_dist = np.linalg.norm(parateo_values - pos_ideal)
    neg_dist = np.linalg.norm(parateo_values - neg_ideal)
    ratio = neg_dist / (neg_dist + pos_dist)
    best_index = np.argmax(ratio)
    selected_pareto_value = parateo_values[best_index]
    return selected_pareto_value, best_index


class DiversityPlanner:
    def plan(
        self,
        image_data: MaskedLabeledImage,
        interestingness_map: np.ndarray = None,
        n_locations: int = 8,
        n_spectral_bands=5,
        use_locs_for_clustering=True,
        vis=True,
        visit_n_locations=5,
        savepath=None,
        blur_scale=5,
        current_location=None,
    ):
        """
        Arguments:
            image_data: The input gridded data
            n_locations: How many candidate locations to generate
            interestingness_score: Gridded data the same size as the image explaining how useful each sample is

        Returns:
            A plan specifying the list of locations
        """
        image_samples = image_data.get_valid_images_points()[:, :n_spectral_bands]
        loc_samples = image_data.get_valid_loc_points()
        if use_locs_for_clustering:
            features = np.hstack((loc_samples, image_samples))
        else:
            features = image_samples

        centers, labels = compute_centers_density(
            features,
            n_clusters=n_locations,
            loc_samples=loc_samples,
            img_size=image_data.image.shape[:2],
            gaussian_sigma=blur_scale,
        )

        features = image_data.image[centers[:, 0], centers[:, 1]]
        features_and_centers = np.hstack((centers, features))
        standard_scalar = StandardScaler()
        features_and_centers_normalized = standard_scalar.fit_transform(
            features_and_centers
        )

        if interestingness_map is not None:
            interestingness_scores = interestingness_map[centers[:, 0], centers[:, 1]]
        else:
            interestingness_scores = None
        # Optimization
        def objective(mask):
            """
            Return negative interestingness
            """
            mask = np.array(mask[0])
            if interestingness_scores is None:
                intrestesting_return = ()
            else:
                sampled_interestingness = interestingness_scores[mask]
                sum_interestingness = np.sum(sampled_interestingness)
                intrestesting_return = (-sum_interestingness,)

            empty_value = np.max(
                cdist(features_and_centers_normalized, features_and_centers_normalized)
            )
            num_sampled = np.sum(mask)
            if np.all(mask) or np.all(np.logical_not(mask)):
                return (empty_value, num_sampled) + intrestesting_return
            sampled = features_and_centers_normalized[mask]
            not_sampled = features_and_centers_normalized[np.logical_not(mask)]
            dists = cdist(sampled, not_sampled)
            min_dists = np.min(dists, axis=0)
            average_min_dist = np.mean(min_dists)
            assert len(min_dists) == (len(mask) - num_sampled)

            return (average_min_dist, num_sampled) + intrestesting_return

        if interestingness_scores is None:
            problem = Problem(1, 2)
        else:
            problem = Problem(1, 3)
        problem.types[:] = Binary(n_locations)
        problem.function = objective
        print("Begining optimization")
        algorithm = NSGAII(problem)
        algorithm.run(1000)
        results = nondominated(algorithm.result)

        # results_dict = {int(np.sum(r.variables)): r.variables for r in results}
        ## Ensure that the number of samples you want is present
        # possible_n_visit_locations = np.array(list(results_dict.keys()))
        # diffs = np.abs(possible_n_visit_locations - visit_n_locations)
        # visit_n_locations = possible_n_visit_locations[np.argmin(diffs)]
        # final_mask = np.squeeze(np.array(results_dict[visit_n_locations]))
        pareto_values = np.array(
            [
                [s.objectives[0] for s in results],
                [s.objectives[1] for s in results],
                [s.objectives[2] for s in results],
            ]
        ).T

        valid_pareto_inds = np.where(pareto_values[:, 1] == visit_n_locations)[0]
        valid_pareto_values = pareto_values[valid_pareto_inds]

        solution, solution_ind = topsis(valid_pareto_values)
        solution_ind_in_original = valid_pareto_inds[solution_ind]
        final_mask = np.squeeze(results[solution_ind_in_original].variables)

        # Take the i, j coordinate
        locs = centers[final_mask]
        if current_location is not None:
            locs = np.concatenate((np.atleast_2d(current_location), locs), axis=0)
        plan = solve_tsp(locs)
        if current_location is not None:
            print("path is not sorted")
        if vis:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            n = 100

            ax.scatter(pareto_values[:, 0], pareto_values[:, 1], pareto_values[:, 2])
            ax.scatter(solution[0], solution[1], solution[2], c="r")

            ax.set_xlabel("Diversity cost")
            ax.set_ylabel("Num sampled")
            ax.set_zlabel("Interestingness")

            plt.show()

            # plt.xlabel("Number of sampled locations")
            # plt.ylabel("Average distance of unsampled locations")
            # plt.pause(5)

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
        return plan
