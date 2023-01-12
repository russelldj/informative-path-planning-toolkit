import time

import numpy as np
from skimage.filters import gaussian
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage
import matplotlib.pyplot as plt
from ipp_toolkit.config import PAUSE_DURATION


class CandidateLocationSelector:
    def __init__(
        self,
        img_size,
        max_fit_points=10000,
        gaussian_sigma: int = 5,
        use_dense_spatial_region: bool = True,
        scaler: StandardScaler = None,
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
        self, data_manager: MaskedLabeledImage, savepath=None, show_as_scaler=False
    ):
        if self.cluster_inds is None:
            raise ValueError("Must compute cluster inds before visualization")

        vis_image = data_manager.get_image_for_flat_values(self.cluster_inds)
        plt.close()

        colors = np.arange(self.centers.shape[0])

        f, axs = plt.subplots(1, 3 if show_as_scaler else 2)

        axs[0].imshow(data_manager.image)
        axs[0].scatter(
            self.centers[:, 1],
            self.centers[:, 0],
            label="Sampled locations",
            edgecolors="k",
        )
        axs[0].set_title("Image")

        axs[1].imshow(vis_image, cmap="tab20")
        axs[1].scatter(
            self.centers[:, 1],
            self.centers[:, 0],
            c=colors,
            cmap="tab20",
            edgecolors="k",
            label="Sampled locations",
        )
        axs[1].set_title("Class index visualized as a repeating colormap (tab20)")

        if show_as_scaler:
            axs[2].imshow(vis_image)
            axs[2].scatter(
                self.centers[:, 1],
                self.centers[:, 0],
                c=colors,
                edgecolors="k",
                label="Sampled locations",
            )
            axs[2].set_title("Class index visualized as a scaler")

        [ax.legend() for ax in axs]

        if savepath is None:
            plt.show()
            # plt.pause(PAUSE_DURATION)
            breakpoint()
        else:
            plt.savefig(savepath)

        plt.close()
