from ipp_toolkit.data.MaskedLabeledImage import (
    ImageNPMaskedLabeledImage,
    torchgeoMaskedDataManger,
)
from pathlib import Path
from ipp_toolkit.config import DATA_FOLDER, VIS
import numpy as np
from torchgeo.datasets import Chesapeake7
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ipp_toolkit.utils.data.dvc import pull_dvc_data
import copy


def compute_greenness(data_manager, vis=VIS):
    img = data_manager.image
    magnitude = np.linalg.norm(img[..., 0::2], axis=2)
    green = img[..., 1]
    greenness = green.astype(float) / (magnitude.astype(float) + 0.00001)
    greenness = np.clip(greenness, 0, 4) / 4
    greenness[np.logical_not(data_manager.mask)] = np.nan
    if vis:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        plt.colorbar(axs[1].imshow(greenness), ax=axs[1])
        axs[0].set_title("Original image")
        axs[1].set_title("Psuedo-label")
        plt.show()
    return greenness


def take_top_k_classes(label_map: np.ndarray, k: int):
    """
    The label map is a class label per pixel. This returns a new map where each of the top k-1 classes
    are preserved. All other classes are combined into one "background" class.
    """
    unique, counts = np.unique(label_map, return_counts=True)
    sorted_inds = np.argsort(counts)
    sorted_unique = np.flip(unique[sorted_inds])

    # set up everything with the background class
    output_label_map = np.ones_like(label_map) * (k - 1)
    for i, class_index in enumerate(sorted_unique[: (k - 1)]):
        output_label_map[label_map == class_index] = i
    return output_label_map


class CoralLandsatClassificationData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/coral/X_wv.npy"),
        mask=Path(DATA_FOLDER, "maps/coral/valid_wv.npy"),
        label=Path(DATA_FOLDER, "maps/coral/Y.npy"),
        **kwargs,
    ):
        super().__init__(
            image=image,
            mask=mask,
            label=label,
            cmap="tab10",
            n_classes=3,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            **kwargs,
        )
        self.label = np.argmax(self.label, axis=2)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/coral"))


class YellowcatDroneClassificationData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/yellowcat/20221028_M7_orthophoto.tif"),
        n_pseudo_classes=8,
        fit_on_n_points=10000,
        **kwargs,
    ):
        super().__init__(
            image=image,
            downsample=16,
            use_last_channel_mask=True,
            n_classes=n_pseudo_classes,
            cmap="tab10",
            vis_vmin=-0.5,
            vis_vmax=9.5,
            **kwargs,
        )
        kmeans = KMeans(n_clusters=n_pseudo_classes)
        valid_features = self.get_valid_image_points()
        subset_inds = np.random.choice(
            valid_features.shape[0], min(fit_on_n_points, valid_features.shape[0])
        )
        kmeans.fit_predict(valid_features[subset_inds])
        labels = kmeans.predict(valid_features)
        self.label = self.get_image_for_flat_values(labels)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/yellowcat"))


class ChesapeakeBayNaipLandcover7ClassificationData(torchgeoMaskedDataManger):
    def __init__(
        self,
        naip_tiles=(
            "m_3807511_ne_18_060_20181104.tif",
            "m_3807511_se_18_060_20181104.tif",
            "m_3807512_nw_18_060_20180815.tif",
            "m_3807512_sw_18_060_20180815.tif",
        ),
        chesapeake_dataset=Chesapeake7,
        **kwargs,
    ):
        super().__init__(
            naip_tiles=naip_tiles,
            chesapeake_dataset=chesapeake_dataset,
            n_classes=7,
            cmap="tab10",
            vis_vmin=-0.5,
            vis_vmax=9.5,
            **kwargs,
        )


class SafeForestOrthoGreennessRegressionData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/safeforest/left_camera.tif"),
        mask=Path(DATA_FOLDER, "maps/safeforest/left_camera_mask.tif"),
        downsample=8,
        blur_sigma=2,
        **kwargs,
    ):
        super().__init__(
            image=image,
            mask=mask,
            downsample=downsample,
            blur_sigma=blur_sigma,
            vis_vmin=None,
            vis_vmax=None,
            **kwargs,
        )
        self.label = compute_greenness(self)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/safeforest"))


class SafeForestGMapGreennessRegressionData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/safeforest_gmaps/safeforest_test.png"),
        downsample=4,
        **kwargs,
    ):
        super().__init__(
            image=image, downsample=downsample, vis_vmin=None, vis_vmax=None, **kwargs,
        )
        self.label = compute_greenness(self)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/safeforest_gmaps"))


class AIIRAGreennessRegresssionData(ImageNPMaskedLabeledImage):
    def __init__(
        self, image=Path(DATA_FOLDER, "maps/aiira/random_field.png"), **kwargs
    ):
        super().__init__(image=image, download=4, **kwargs)
        self.label = compute_greenness(self)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/aiira"))


class CupriteASTERMineralClassificationData(ImageNPMaskedLabeledImage):
    """
    Data from the Cuprite, NV mining area. This location is well-studied in geology so there is 
    extensive remote sensing and field work done about the area. This data was most extensively
    used by Alberto Candela, now at NASA JPL. 

    This data has features from the ASTER satellite measurements. I believe these observations
    are upsampled from 15m/px to 3.5m/px to match that of the AVIRIS data product.

    The label maps are obtained from the Tetracorder software to the best of my knowledge. The 
    label maps are reduced from 215 classes to the top 10, preserving those 9 most prevalent classes
    and the last class is an aggregation of everything else. 
    """

    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/cuprite/aster/aster_cube_norm.npy"),
        label=Path(DATA_FOLDER, "maps/cuprite/labels/mineral.npy"),
    ):
        # TODO update plotting options
        super().__init__(
            image=image,
            label=label,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            cmap="tab10",
            n_classes=10,
        )
        # Condense the channels
        self.label = take_top_k_classes(self.label, 10)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/cuprite"))


class CupriteAVIRISASTERMineralClassificationData(ImageNPMaskedLabeledImage):
    """
    See CupriteASTERMineralClassificationData for a general description.

    This data was obtained from the AVIRIS aerial collection and spectrally downsampled
    to match that of ASTER
    """

    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/cuprite/aster/aviris_aster_cube_norm.npy"),
        label=Path(DATA_FOLDER, "maps/cuprite/labels/mineral.npy"),
    ):
        # TODO update plotting options
        super().__init__(
            image=image,
            label=label,
            use_zero_allchannels_mask=True,
            drop_last_image_channel=False,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            cmap="tab10",
            n_classes=10,
        )
        self.label = take_top_k_classes(self.label, 10)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/cuprite"))


class CupriteAVIRISMineralClassificationData(ImageNPMaskedLabeledImage):
    """
    See CupriteASTERMineralClassificationData for a general description.

    This data was obtained from the AVIRIS aerial collection.
    """

    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/cuprite/aviris/aviris_cube_2um_norm.npy"),
        label=Path(DATA_FOLDER, "maps/cuprite/labels/mineral.npy"),
    ):
        # TODO update plotting options
        super().__init__(
            image=image,
            label=label,
            use_zero_allchannels_mask=True,
            drop_last_image_channel=False,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            cmap="tab10",
            n_classes=10,
        )
        self.label = take_top_k_classes(self.label, 10)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/cuprite"))


ALL_LABELED_DOMAIN_DATASETS = {
    "cuprite_aviris": CupriteAVIRISMineralClassificationData,
    "cuprite_aster": CupriteASTERMineralClassificationData,
    "cuprite_aster_aviris": CupriteAVIRISASTERMineralClassificationData,
    "aiira": AIIRAGreennessRegresssionData,
    "safeforest_gmap": SafeForestGMapGreennessRegressionData,
    "safeforest_ortho": SafeForestOrthoGreennessRegressionData,
    "yellowcat": YellowcatDroneClassificationData,
    "chesapeake": ChesapeakeBayNaipLandcover7ClassificationData,
    "coral": CoralLandsatClassificationData,
}

ALL_DOMAIN_DATASETS = {
    **ALL_LABELED_DOMAIN_DATASETS,
}
