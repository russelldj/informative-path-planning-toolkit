import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ipp_toolkit.utils.sampling import get_flat_samples
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler


DATA_FOLDER = (
    "/home/david/dev/research/informative-path-planning-toolkit/data/maps/coral"
)


def run(data_folder, n_clusters=12):
    spectral_image, mask, label = [
        np.load(Path(data_folder, x + ".npy")) for x in ("X_wv", "valid_wv", "Y")
    ]
    mask = mask[..., 0].astype(bool)

    fig, axs = plt.subplots(3, 4)

    samples, initial_shape = get_flat_samples(np.array(mask.shape[:2]) - 1, 1)
    i_locs, j_locs = [np.reshape(samples[:, i], initial_shape) for i in range(2)]

    for i in range(2):
        for j in range(4):
            channel = spectral_image[..., i * 4 + j]
            # print(f"min: {np.min(channel)}, max: {np.max(channel)}")
            axs[i, j].imshow(channel, vmin=0, vmax=0.35)
            # axs[i, j].hist(channel.flatten())
    i_locs[np.logical_not(mask)] = np.nan
    j_locs[np.logical_not(mask)] = np.nan
    first_spectral_images = [spectral_image[..., i] for i in range(5)]

    standard_scalar = StandardScaler()
    features = [feature[mask] for feature in [i_locs, j_locs] + first_spectral_images]
    features = np.vstack(features).T
    features = standard_scalar.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)

    cluster_inds = kmeans.labels_
    dists = kmeans.transform(features)
    # Find the most representative sample for each cluster
    inds = np.argmin(dists, axis=0)
    centers = features[inds]

    centers = standard_scalar.inverse_transform(centers)

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

    plt.imshow(clusters, cmap="tab10")
    plt.scatter(
        centers[:, 1],
        centers[:, 0],
        c=np.arange(n_clusters),
        cmap="tab10",
        edgecolors="k",
    )
    plt.show()


if __name__ == "__main__":
    run(DATA_FOLDER, n_clusters=8)
