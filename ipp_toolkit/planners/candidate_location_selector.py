import time

import numpy as np
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
import matplotlib.pyplot as plt
from ipp_toolkit.config import PAUSE_DURATION
from ipp_toolkit.planners.utils import compute_gridded_samples_from_mask
from ipp_toolkit.visualization.utils import show_or_save_plt, remove_ticks


class ClusteringCandidateLocationSelector:
    def __init__(
        self,
        img_size,
        max_fit_points=10000,
        gaussian_sigma: int = 5,
        use_dense_spatial_region: bool = True,
        scaler: StandardScaler = None,
        **kwargs,
    ):
        """
        """
        self.img_size = img_size
        self.max_fit_points = max_fit_points
        self.gaussian_sigma = gaussian_sigma
        self.use_dense_spatial_region = use_dense_spatial_region
        self.scaler = scaler
        self.kmeans = None

        self.cluster_inds = None
        self.centers = None

    def select_locations(
        self, features: np.ndarray, mask: np.ndarray, n_clusters: int, loc_samples,
    ):
        """
        Clusters the image and then finds a large spatial region of similar appearances

        Args:
            features:
            mask: binary mask representing valid area to sample
            n_clusters:
            loc_samples: Locations of the features

        Returns:
            centers: i,j locations of the centers
            cluster_inds: the predicted labels for each point
        """
        start_time = time.time()
        cluster_inds = self.cluster(features=features, n_clusters=n_clusters)

        # Two approaches, one is a dense spatial cluster
        # the other is the point nearest the k-means centroid
        if self.use_dense_spatial_region:
            # Build an array where each channel is a binary map for one class
            per_class_layers = np.zeros(
                (self.img_size[0], self.img_size[1], n_clusters), dtype=bool
            )
            # Populate the binary map
            per_class_layers[
                loc_samples[:, 0].astype(int),
                loc_samples[:, 1].astype(int),
                cluster_inds,
            ] = True
            # Smooth the binary map layer-wise
            smoothed_layers = [
                gaussian(per_class_layers[..., i], self.gaussian_sigma)
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
            center_dists = self.kmeans.transform(features)
            closest_points = np.argmin(center_dists, axis=0)
            centers = loc_samples[closest_points].astype(int)

        elapsed_time = time.time() - start_time
        self.centers = centers
        return centers, cluster_inds, self.scaler, elapsed_time

    def cluster(self, features, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters)
        if self.scaler is None:
            self.scaler = StandardScaler()
            # Normalize features elementwise
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        if self.max_fit_points is None:
            # Fit on all the points
            self.kmeans.fit(features)
        else:
            # Fit on a subset of points
            sample_inds = np.random.choice(
                features.shape[0], size=(self.max_fit_points), replace=False
            )
            feature_subset = features[sample_inds, :]
            self.kmeans.fit(feature_subset)
        # Predict the cluster membership for each data point
        self.cluster_inds = self.kmeans.predict(features)
        return self.cluster_inds

    def vis(
        self,
        data_manager: MaskedLabeledImage,
        savepath="vis/candidate_locations.png",
        show_as_scaler=False,
        show_as_id=False,
    ):
        if self.cluster_inds is None:
            raise ValueError("Must compute cluster inds before visualization")

        vis_image = data_manager.get_image_for_flat_values(self.cluster_inds)
        plt.close()

        colors = np.arange(self.centers.shape[0])
        n_plots = 1 + np.sum([show_as_scaler, show_as_id])
        f, axs = plt.subplots(1, n_plots)

        axs0 = axs if n_plots == 1 else axs[0]
        axs0.imshow(data_manager.image[..., :3])
        axs0.scatter(
            self.centers[:, 1],
            self.centers[:, 0],
            label="Candidate locations",
            edgecolors="k",
        )
        axs0.set_title("Satellite image")

        if show_as_id:
            axs[1].imshow(vis_image, cmap="tab20")
            axs[1].scatter(
                self.centers[:, 1],
                self.centers[:, 0],
                c=colors,
                cmap="tab20",
                edgecolors="k",
                label="Candidate locations",
            )
            axs[1].set_title("Class index visualized as a repeating colormap (tab20)")

        if show_as_scaler:
            axs[2].imshow(vis_image)
            axs[2].scatter(
                self.centers[:, 1],
                self.centers[:, 0],
                c=colors,
                edgecolors="k",
                label="Candidate locations",
            )
            axs[2].set_title("Class index visualized as a scaler")
        if n_plots > 1:
            [ax.legend() for ax in axs]
        else:
            plt.legend()

        remove_ticks()
        show_or_save_plt(savepath=savepath)

        plt.close()


class GridCandidateLocationSelector:
    def __init__(self, img_size, **kwargs):
        self.img_size = img_size
        self.centers = None

    def select_locations(self, loc_samples, n_clusters, **kwargs):
        start_time = time.time()
        valid_locations_image = np.zeros(self.img_size, dtype=bool)
        loc_samples = loc_samples.astype(int)
        valid_locations_image[loc_samples[:, 0], loc_samples[:, 1]] = True
        self.centers = compute_gridded_samples_from_mask(
            valid_locations_image, n_samples=n_clusters
        )
        elapsed_time = time.time() - start_time
        return self.centers, None, None, elapsed_time

    def vis(self, data_manager: MaskedLabeledImage, savepath=None, **kwargs):

        plt.imshow(data_manager.image[..., :3])
        plt.scatter(
            self.centers[:, 1],
            self.centers[:, 0],
            label="Candidate locations",
            edgecolors="k",
        )
        plt.title("Satellite image")
        plt.legend()
        show_or_save_plt(savepath=savepath)
