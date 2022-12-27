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

plt.rcParams["figure.figsize"] = (20, 13)

from platypus import NSGAII, Problem, Binary, Real, nondominated

from ipp_toolkit.config import (
    CLUSTERING_ELAPSED_TIME,
    TSP_ELAPSED_TIME,
    OPTIMIZATION_ELAPSED_TIME,
    OPTIMIZATION_ITERS,
    PAUSE_DURATION,
)


def add_candidates_and_plan(ax, centers, plan, cmap="tab20"):
    """
    Plotting convenience for adding candidate locations and final trajectory
    """
    n_locations = centers.shape[0]

    ax.scatter(
        centers[:, 1],
        centers[:, 0],
        c=np.arange(n_locations),
        cmap=cmap,
        edgecolors="k",
        label="",
    )
    ax.plot(plan[:, 1], plan[:, 0], c="k")


def compute_mask(input_mask, visit_n_locations):
    """
    Compute the mask. This is trivial if the mask is binary. 
    for floats it's the top visit_n_locations values
    """
    if visit_n_locations is None:
        mask = np.squeeze(input_mask)
    else:
        ordered_locs = np.argsort(input_mask)
        mask = np.zeros_like(input_mask, dtype=bool)
        top_locs = ordered_locs[-visit_n_locations:]
        mask[top_locs] = True
    return mask


def compute_n_sampled(mask, visit_n_locations):
    """
    Return an objective this variable is free to change, otherwise nothing. This coresponds to a 
    0- or 1-length tuple
    """
    if visit_n_locations is None:
        return (np.sum(mask),)
    else:
        return ()


def compute_interestingness_objective(interestingness_scores, mask):
    """
    Compute interestingness of a masked set of points
    """
    if interestingness_scores is None:
        intrestesting_return = ()
    else:
        sampled_interestingness = interestingness_scores[mask]
        sum_interestingness = np.sum(sampled_interestingness)
        intrestesting_return = (-sum_interestingness,)
    return intrestesting_return


def compute_average_min_dist(
    candidate_location_features, mask, previous_location_features
):
    """
    Compute the average distance from each unsampled point to the nearest sampled one. Returns a 1-length tuple for compatability
    """

    # If nothing or everything is sampled is sampled, the value is that of the max dist between candidates
    if np.all(mask) or np.all(np.logical_not(mask)):
        empty_value = np.max(
            cdist(candidate_location_features, candidate_location_features)
        )
        return (empty_value,)

    # Compute the features for sampled and not sampled points
    not_sampled = candidate_location_features[np.logical_not(mask)]
    sampled = candidate_location_features[mask]
    if previous_location_features is not None:
        sampled = np.concatenate((sampled, previous_location_features))

    # Compute the distance for each sampled point to each un-sampled point
    dists = cdist(sampled, not_sampled)
    # Take the min distance from each unsampled point to a sampled point. This relates to how well described it is
    min_dists = np.min(dists, axis=0)
    # Average this distance cost over all unsampled points
    average_min_dist = np.mean(min_dists)

    return (average_min_dist,)


class DiversityPlanner:
    def plan(
        self,
        image_data: MaskedLabeledImage,
        interestingness_image: np.ndarray = None,
        mask: np.ndarray = None,
        previous_sampled_points: np.ndarray = None,
        candidate_locations: np.ndarray = None,
        labels: np.ndarray = None,
        scaler: StandardScaler = None,
        n_locations=8,
        current_location=None,
        n_spectral_bands=5,
        use_locs_for_clustering=True,
        vis=True,
        visit_n_locations=5,
        savepath=None,
        blur_scale=5,
        use_dense_spatial_region_candidates: bool = True,
        constrain_n_samples_in_optim: bool = True,
        n_optimization_iters=OPTIMIZATION_ITERS,
    ):
        """
        Arguments:
            world_model: the belief of the world
            mask: Legal locations
            current_location: The location (n,)
            n_steps: How many planning steps to take
            use_dense_spatial_region_candidates: Select the candidate locations using high spatial density of similar type
            constrain_n_samples_in_optim: whether to constrain the number of samples in the optimzation or allow it to vary

        Returns:
            A plan specifying the list of locations
        """
        self.log_dict = {}
        # Get the spectral data
        image_samples = image_data.get_valid_image_points()[:, :n_spectral_bands]
        # Get the spatial data
        loc_samples = image_data.get_valid_loc_points()

        # Determine whether you want to cluster using spatial data or not
        if use_locs_for_clustering:
            features = np.hstack((loc_samples, image_samples))
        else:
            features = image_samples

        if candidate_locations is None or labels is None or scaler is None:
            # Get the candiate regions
            candidate_locations, labels, scaler = self._compute_candidate_locations(
                features,
                mask,
                n_clusters=n_locations,
                loc_samples=loc_samples,
                img_size=image_data.image.shape[:2],
                gaussian_sigma=blur_scale,
                use_dense_spatial_region=use_dense_spatial_region_candidates,
            )

        # Get the features to use for optimization
        candidate_location_features = self._get_candidate_location_features(
            image_data, candidate_locations, use_locs_for_clustering, scaler
        )
        previous_location_features = self._get_candidate_location_features(
            image_data, previous_sampled_points, use_locs_for_clustering, scaler
        )

        # Get the per-sample interestingness
        candidate_location_interestingness = self._get_candidate_location_interestingness(
            interestingness_image, candidate_locations
        )

        # Compute the pareto front of possible plans
        pareto_results = self._compute_optimal_subset(
            candidate_location_features=candidate_location_features,
            n_optimization_iters=n_optimization_iters,
            interestingness_scores=candidate_location_interestingness,
            visit_n_locations=(
                visit_n_locations if constrain_n_samples_in_optim else None
            ),
            previous_location_features=previous_location_features,
        )

        # Solve for the pareto-optimal set of values
        selected_locations_mask, selected_objectives = self._get_solution_from_pareto(
            pareto_results=pareto_results, visit_n_locations=visit_n_locations
        )

        # Take the i, j coordinate
        selected_locs = candidate_locations[selected_locations_mask]
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
                image_data,
                interestingness_image,
                candidate_locations,
                plan,
                labels,
                savepath,
            )
            self._visualize_pareto_front(
                pareto_results,
                selected_objectives,
                remove_n_sampled_locations_obj=constrain_n_samples_in_optim,
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
        mask: np.ndarray,
        n_clusters: int,
        loc_samples,
        img_size,
        max_fit_points=None,
        gaussian_sigma: int = 5,
        use_dense_spatial_region: bool = True,
        scaler: StandardScaler = None,
    ):
        """
        Clusters the image and then finds a large spatial region of similar appearances

        Args:
            features:
            mask: binary mask representing valid area to sample
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
        kmeans = KMeans(n_clusters=n_clusters)
        if scaler is None:
            scaler = StandardScaler()
            # Normalize features elementwise
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)

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
            masked_smoothed_layers = []
            for l in smoothed_layers:
                l[np.logical_not(mask)] = 0
                masked_smoothed_layers.append(l)
            # Find the argmax location for each smoothed binary map
            # This is just fancy indexing for that
            centers = [
                np.unravel_index(msi.flatten().argmax(), msi.shape)
                for msi in masked_smoothed_layers
            ]
            centers = np.vstack(centers)
        else:
            # Obtain the distance to each class centroid
            center_dists = kmeans.transform(features)
            closest_points = np.argmin(center_dists, axis=0)
            centers = loc_samples[closest_points].astype(int)

        self.log_dict[CLUSTERING_ELAPSED_TIME] = time.time() - start_time
        return centers, cluster_inds, scaler

    def _get_candidate_location_features(
        self,
        image_data: np.ndarray,
        centers: np.ndarray,
        use_locs_for_clustering: bool,
        scaler=None,
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
        features = image_data.image[centers[:, 0], centers[:, 1]]
        if use_locs_for_clustering:
            features = np.hstack((centers, features))

        if scaler is None:
            scaler = StandardScaler()
            candidate_location_features = scaler.fit_transform(features)
        else:
            candidate_location_features = scaler.transform(features)

        return candidate_location_features

    def _get_candidate_location_interestingness(
        self, interestingness_image: np.ndarray, centers: np.ndarray
    ):
        """
        Takes an interestingness_image and samples it based on centers to obtain a per-sample interestingness

        Args
            interestingness_image: (n,m) image representing per-location interestingness
            centers: (k,2) places to sample

        Returns:
            np.ndarray (k,) of per-center interestingness, organized the same way
            or 
            None
        """
        if interestingness_image is None:
            return None

        interestingness_scores = interestingness_image[centers[:, 0], centers[:, 1]]
        return interestingness_scores

    def _compute_optimal_subset(
        self,
        candidate_location_features: np.ndarray,
        interestingness_scores: np.ndarray = None,
        n_optimization_iters: int = OPTIMIZATION_ITERS,
        visit_n_locations=None,
        previous_location_features=None,
    ):
        """ 
        Compute the best subset of locations to visit from a set of candidates

        Args:
            candidate_location_features
            interestingness_scores: per_point interestingness for visiting. Higher is better
            n_optimization_iters: number of iterations of NSGA-II to perform
            visit_n_locations: int | None
                if set force the optimization to use continous-valued decision variables
                The solution will be chosen as the top sample_n_location variables
        previous_location_features: features from previously visited locations

        Returns:
            pareto_results: the set of pareto-optimal results
        """
        start_time = time.time()

        # Optimization objective. Uses the local variables interestingness_scores and candiate_location_features
        def objective(mask):
            """
            Return negative interestingness
            """
            mask = compute_mask(mask, visit_n_locations)

            # Compute num sampled objective
            num_sampled = compute_n_sampled(mask, visit_n_locations)

            # Def compute interestingness objective
            interestingness_return = compute_interestingness_objective(
                interestingness_scores, mask
            )
            average_min_dist = compute_average_min_dist(
                candidate_location_features, mask, previous_location_features
            )
            # Concatinates three tuples. Num sampled and interestingess_return may be
            # 0-length. Average min distance is guaranteed to be meaninful
            return num_sampled + average_min_dist + interestingness_return

        n_locations = candidate_location_features.shape[0]
        # Count up the number of valid objectives
        num_objectives = (
            1 + int(interestingness_scores is not None) + int(visit_n_locations is None)
        )

        if visit_n_locations is None:
            problem = Problem(1, num_objectives)
            # Each variable represents whether that location is used
            problem.types[:] = Binary(n_locations)
        else:
            problem = Problem(n_locations, num_objectives)
            problem_types = [Real(0, 1)] * n_locations
            problem.types[:] = problem_types
        problem.function = objective
        algorithm = NSGAII(problem)
        algorithm.run(n_optimization_iters)
        pareto_results = nondominated(algorithm.result)

        # Ensure that masks are binary for downstream use
        for p in pareto_results:
            p.variables = compute_mask(p.variables, visit_n_locations)

        self.log_dict[OPTIMIZATION_ELAPSED_TIME] = time.time() - start_time
        return pareto_results

    def _get_solution_from_pareto(
        self,
        pareto_results: list,
        visit_n_locations: int,
        use_topsis=False,
        min_visit_locations: int = 2,
    ):
        """
        Select a solution from the pareto front. Currently, we just select one
        that has a given number of visited locations

        Args:
            pareto_results: list of pareto solutions
            visit_n_locations: how many points to visit

        Returns:
            The mask for teh selected solution
            The objectives for the selected solution
        """
        valid_pareto_results = [
            r for r in pareto_results if np.sum(r.variables) >= min_visit_locations
        ]

        if use_topsis:
            pareto_values = np.array([s.objectives for s in valid_pareto_results])
            _, topsis_index = topsis(parateo_values=pareto_values)
            final_mask = np.squeeze(pareto_results[topsis_index].variables)
            final_objectives = np.squeeze(pareto_results[topsis_index].objectivse)
        else:
            results_dict = {int(np.sum(r.variables)): r for r in valid_pareto_results}
            # Ensure that the number of samples you want is present
            possible_n_visit_locations = np.array(list(results_dict.keys()))
            diffs = np.abs(possible_n_visit_locations - visit_n_locations)
            visit_n_locations = possible_n_visit_locations[np.argmin(diffs)]
            selected_solution = results_dict[visit_n_locations]
            final_mask = np.squeeze(np.array(selected_solution.variables))
            final_objectives = selected_solution.objectives

        return final_mask, final_objectives

    def _visualize_pareto_front(
        self,
        pareto_solutions,
        selected_objectives,
        pause_duration=PAUSE_DURATION,
        remove_n_sampled_locations_obj: bool = False,
        labels=(
            "Number of sampled locations",
            "Average distance of unsampled locations",
            "Sum interestingness score",
        ),
    ):
        """
        Args:
            show_n_sampled_locations: should you show the sampled locations objective

        """
        # close existing figures
        plt.close()

        pareto_objectives = np.array([s.objectives for s in pareto_solutions])

        # If the number of sample locations is constrained, this objective is not present
        if remove_n_sampled_locations_obj:
            # pareto_objectives = pareto_objectives[:, 1:]
            labels = labels[1:]

        dimensionality = pareto_objectives.shape[1]
        if dimensionality == 1:
            print(f"Objective value for one-dim problem {pareto_objectives[0, 0]}")
            return
        if dimensionality == 2:
            # Normal 2d plot
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            ax = plt

        elif dimensionality == 3:
            # Set up 3d plot
            ax = plt.figure().add_subplot(projection="3d")
            # Plot z axis which would otherwise be missing
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

        else:
            raise ValueError(
                f"Cannot show problem with {dimensionality} dimensions, only 3 or fewer"
            )
        # Show all candidate objectives
        ax.scatter(*pareto_objectives.T, label="Candidate solutions")
        # Show the selected location
        ax.scatter(*selected_objectives, c="r", s=50, label="Chosen solution")

        plt.legend()
        plt.pause(pause_duration)

    def _visualize_plan(
        self,
        image_data,
        interestingess_image,
        centers,
        plan,
        labels,
        savepath,
        cmap="tab20",
        pause_duration=PAUSE_DURATION,
    ):
        clusters = np.ones(image_data.mask.shape) * np.nan
        clusters[image_data.mask] = labels
        f, axs = plt.subplots(1, 2 if interestingess_image is None else 3)
        axs[0].imshow(image_data.image[..., :3])
        axs[1].imshow(clusters, cmap=cmap)

        axs[0].set_title("First three imagery channels")
        axs[1].set_title("Cluster inds")

        if interestingess_image is not None:
            cb = axs[2].imshow(interestingess_image)
            axs[2].set_title("Interestingess score")
            plt.colorbar(cb, ax=axs[2])

        [add_candidates_and_plan(ax, centers, plan, cmap=cmap) for ax in axs]

        if savepath is not None:

            plt.savefig(savepath)
            plt.pause(pause_duration)
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.show()


class BatchDiversityPlanner(DiversityPlanner):
    def __init__(
        self,
        prediction_model,
        world_data: MaskedLabeledImage,
        n_candidate_locations: int = 8,
        n_spectral_bands=5,
        use_dense_spatial_region_candidates: bool = True,
        blur_scale=5,
        use_locs_for_clustering=True,
        use_locs_for_prediction=True,
    ):
        """
        Args:
            prediction_model: maps from samples to labels
            world_data: the masked features. Note we don't use the labels here
            n_spectral_bands: how many spectral features to use
            use_dense_spatial_region_candidates: Select the candidate locations using high spatial density of similar type
            use_locs_for_clustering: use location in clustering decisions
            blur_scale: scale of gaussian kernel for spatial candidates
            n_candidate_locations: number of candidate regions to sample
        """
        self.prediction_model = prediction_model
        self.world_data = world_data
        self.n_candidate_locations = n_candidate_locations

        self.n_spectral_bands = n_spectral_bands
        self.use_dense_spatial_region_candidates = use_dense_spatial_region_candidates
        self.use_locs_for_clustering = use_locs_for_clustering
        self.use_locs_for_prediction = use_locs_for_prediction
        self.blur_scale = blur_scale

        self.log_dict = {}

        self.planner = DiversityPlanner()

        self.clustering_scaler = StandardScaler()
        self.prediction_scaler = StandardScaler()

        self.previous_sampled_locs = np.empty((0, 2))
        self.interestingness_image = None  # Prior believe on interestingess

        self.all_prediction_features = None
        self.clustering_features = None
        self.loc_samples = None

        # Supervised learning features
        self.labeled_prediction_features = None
        self.labeled_prediction_values = np.empty(0)

        self.candidate_locations = None
        self.cluster_labels = None

    def _preprocess_features(self):
        # Preprocessing is done
        if not (
            self.clustering_features is None
            or self.all_prediction_features is None
            or self.loc_samples is None
        ):
            return

        image_features = self.world_data.get_valid_image_points()
        if self.use_locs_for_prediction or self.use_locs_for_clustering:
            self.loc_samples = self.world_data.get_valid_loc_points()

        if self.use_locs_for_clustering:
            self.clustering_features = np.concatenate(
                (image_features, self.loc_samples), axis=1
            )
        else:
            self.clustering_features = image_features

        if self.use_locs_for_prediction:
            self.all_prediction_features = np.concatenate(
                (image_features, self.loc_samples), axis=1
            )
        else:
            self.all_prediction_features = image_features

        self.all_prediction_features = self.prediction_scaler.fit_transform(
            self.all_prediction_features
        )
        self.clustering_scaler.fit(self.clustering_features)

    def plan(
        self,
        interestingness_image: np.ndarray = None,
        current_location=None,
        vis=True,
        visit_n_locations=5,
        savepath=None,
        constrain_n_samples_in_optim: bool = True,
        n_optimization_iters=OPTIMIZATION_ITERS,
    ):
        """
        Arguments:
            world_model: the belief of the world
            mask: Where it is legal to sample
            current_location: The location (n,)
            n_steps: How many planning steps to take
            constrain_n_samples_in_optim: whether to constrain the number of samples in the optimzation or allow it to vary

        Returns:
            A plan specifying the list of locations
        """
        self.log_dict = {}
        # Preprocess features if this hasn't been done yet
        self._preprocess_features()

        # Overwrite the default interestingess image if provide
        if self.interestingness_image is None and interestingness_image is not None:
            self.interestingness_image = interestingness_image

        # Get the candidate locations
        if self.cluster_labels is None or self.candidate_locations is None:
            print("computing new cluster centers")
            (
                self.candidate_locations,
                self.cluster_labels,
                _,
            ) = self._compute_candidate_locations(
                self.clustering_features,
                self.world_data.mask,
                n_clusters=self.n_candidate_locations,
                loc_samples=self.loc_samples,
                img_size=self.world_data.image.shape[:2],
                gaussian_sigma=self.blur_scale,
                use_dense_spatial_region=self.use_dense_spatial_region_candidates,
                scaler=self.clustering_scaler,
            )

        # Generate the plan
        plan = self.planner.plan(
            image_data=self.world_data,
            interestingness_image=self.interestingness_image,
            previous_sampled_points=self.previous_sampled_locs,
            candidate_locations=self.candidate_locations,
            labels=self.cluster_labels,
            n_locations=self.n_candidate_locations,
            current_location=current_location,
            visit_n_locations=visit_n_locations,
            vis=vis,
            constrain_n_samples_in_optim=constrain_n_samples_in_optim,
            savepath=savepath,
            n_optimization_iters=n_optimization_iters,
            scaler=self.clustering_scaler,
        )

        return plan

    def update_model(self, locs: np.ndarray, values: np.ndarray):
        """
        This is called after the new data has been sampled

        locs: (n,2)
        values: (n,)
        """
        self.previous_sampled_locs = np.concatenate(
            (self.previous_sampled_locs, locs), axis=0
        )
        sampled_location_features = self._get_candidate_location_features(
            self.world_data, locs, self.use_locs_for_prediction, self.prediction_scaler
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
        print(f"{self.labeled_prediction_features} {self.labeled_prediction_values}")
        self.prediction_model.fit(
            self.labeled_prediction_features, self.labeled_prediction_values
        )

    def predict_values(self):
        """
        Use prediction model to predict the values for the whole world
        """
        pred_y = self.prediction_model.predict(self.all_prediction_features)
        self.interestingness_image = self.world_data.get_image_for_flat_values(pred_y)
        return self.interestingness_image
