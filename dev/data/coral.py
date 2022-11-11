import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ipp_toolkit.utils.sampling import get_flat_samples
from sklearn.cluster import KMeans
import ubelt as ub

from sklearn.preprocessing import StandardScaler
from ipp_toolkit.config import DATA_FOLDER
from python_tsp.heuristics import solve_tsp_simulated_annealing
from python_tsp.distances.euclidean_distance import euclidean_distance_matrix
from argparse import ArgumentParser
from ipp_toolkit.data.MaskedLabeledImage import MaskedLabeledImage


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()
    return args


coral_folder = Path(DATA_FOLDER, "maps/coral")


def solve_tsp(points):
    distance_matrix = euclidean_distance_matrix(points)
    print(distance_matrix.shape)
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    permutation = permutation + [permutation[0]]
    path = points[permutation]
    return path


def compute_centers(
    i_locs, j_locs, first_spectral_images, mask, n_clusters, max_fit_points=None
):
    standard_scalar = StandardScaler()
    features = [feature[mask] for feature in [i_locs, j_locs] + first_spectral_images]
    features = np.vstack(features).T
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


def run(data_folder, n_clusters=12):
    filenames = [Path(data_folder, x + ".npy") for x in ("X_wv", "valid_wv", "Y")]
    spectral_image, mask, label = [np.load(file) for file in filenames]
    mask = mask[..., 0].astype(bool)

    data_manager = MaskedLabeledImage(*filenames)
    image_samples = data_manager.get_valid_images_points()
    loc_samples = data_manager.get_valid_loc_points()

    fig, axs = plt.subplots(3, 4)
    locs = data_manager.get_locs()
    i_locs = locs[..., 0]
    j_locs = locs[..., 1]

    for i in range(2):
        for j in range(4):
            channel = spectral_image[..., i * 4 + j]
            # print(f"min: {np.min(channel)}, max: {np.max(channel)}")
            axs[i, j].imshow(channel, vmin=0, vmax=0.35)
            # axs[i, j].hist(channel.flatten())
    i_locs[np.logical_not(mask)] = np.nan
    j_locs[np.logical_not(mask)] = np.nan
    first_spectral_images = [spectral_image[..., i] for i in range(5)]

    centers, cluster_inds = compute_centers(
        i_locs, j_locs, first_spectral_images, mask, n_clusters
    )
    path = solve_tsp(centers)

    ub.ensuredir("vis/coral")
    clusters = np.ones(mask.shape) * np.nan
    clusters[mask] = cluster_inds
    axs[2, 0].imshow(clusters, cmap="tab10")
    # Swap axes
    # markerfacecolor='w',
    #         markeredgewidth=1.5, markeredgecolor=(r, g, b, 1)

    axs[2, 0].scatter(
        centers[:, 1],
        centers[:, 0],
        c=np.arange(n_clusters),
        cmap="tab10",
        edgecolors="k",
    )
    axs[2, 1].imshow(label)
    axs[2, 2].imshow(i_locs)
    axs[2, 3].imshow(j_locs)
    plt.show()
    plt.savefig(f"vis/coral/montoge_{n_clusters}.png")
    plt.clf()

    plt.imshow(clusters, cmap="tab10")
    plt.plot(path[:, 1], path[:, 0], c="k")
    plt.scatter(
        centers[:, 1],
        centers[:, 0],
        c=np.arange(n_clusters),
        cmap="tab10",
        edgecolors="k",
        label="",
    )
    plt.legend()
    plt.savefig(f"vis/coral/{n_clusters}.png")
    plt.show()
    plt.clf()


if __name__ == "__main__":
    args = parse_args()
    run(coral_folder, n_clusters=args.n_clusters)
