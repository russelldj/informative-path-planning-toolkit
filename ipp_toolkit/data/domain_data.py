from ipp_toolkit.data.masked_labeled_image import (
    ImageNPMaskedLabeledImage,
    torchgeoMaskedDataManger,
)
from torchgeo.datasets import ReforesTree
from pathlib import Path
from ipp_toolkit.config import DATA_FOLDER, VIS
import numpy as np
from torchgeo.datasets import Chesapeake7, Chesapeake13
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from ipp_toolkit.utils.data.dvc import pull_dvc_data
import copy
from matplotlib import cm
from sklearn.manifold import TSNE
from skimage.io import imread
import pandas as pd


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

    @classmethod
    def get_dataset_name(cls):
        return "coral_landsat_classification"


class CoralLandsatRegressionData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/coral/X_wv.npy"),
        mask=Path(DATA_FOLDER, "maps/coral/valid_wv.npy"),
        label=Path(DATA_FOLDER, "maps/coral/Y.npy"),
        class_ID=0,
        **kwargs,
    ):
        """
        Args:
            class_ID: Which class to use
        """
        super().__init__(
            image=image, mask=mask, label=label, vis_vmin=0, vis_vmax=1, **kwargs,
        )
        self.label = self.label[..., class_ID]
        self.label[self.label < 0] = 0

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/coral"))

    @classmethod
    def get_dataset_name(cls):
        return "coral_landsat_regression"


class GascolaNAIPUnlabeledData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/gascola/naip.tiff"),
        bounds=(7000, 9000, 3500, 4500),
        **kwargs,
    ):
        """
        Args:
            bounds: i_min, i_max, j_min, j_max
        """
        super().__init__(
            image=image, **kwargs,
        )
        i_min, i_max, j_min, j_max = bounds
        self.image = self.image[i_min:i_max, j_min:j_max, :]
        self.mask = self.mask[i_min:i_max, j_min:j_max]

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/gascola"))

    @classmethod
    def get_dataset_name(cls):
        return "gascola_naip_unlabeled"


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

    @classmethod
    def get_dataset_name(cls):
        return "yellowcat"


class ChesapeakeBayNaipLandcover7ClassificationData(torchgeoMaskedDataManger):
    def __init__(
        self,
        naip_urls=("https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807511_ne_18_060_20181104.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/vt/2018/vt_060cm_2018/42072/m_4207220_ne_18_060_20181123.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/ma/2018/ma_060cm_2018/42072/m_4207258_nw_18_060_20181123.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/41076/m_4107640_nw_18_060_20191005.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/va/2018/va_060cm_2018/39078/m_3907854_se_17_060_20181219.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/40078/m_4007840_se_17_060_20191010.tif"),
        chesapeake_dataset=Chesapeake7,
        download=False,
        chip_size=400,
        n_classes=7,
        cmap="tab10",
        vis_vmin=-0.5,
        vis_vmax=9.5,
        **kwargs,
    ):
        super().__init__(
            naip_urls=naip_urls,
            label_dataset_cls=chesapeake_dataset,
            n_classes=n_classes,
            cmap=cmap,
            vis_vmin=vis_vmin,
            vis_vmax=vis_vmax,
            download=download,
            chip_size=chip_size,
            **kwargs,
        )
        self.image = self.image.astype(np.uint8)

    @classmethod
    def get_dataset_name(cls):
        return "chesapeake7"

class ChesapeakeBayNaipLandcover13ClassificationData(torchgeoMaskedDataManger):
    def __init__(
        self,
        naip_urls=("https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807511_ne_18_060_20181104.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/vt/2018/vt_060cm_2018/42072/m_4207220_ne_18_060_20181123.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/ma/2018/ma_060cm_2018/42072/m_4207258_nw_18_060_20181123.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/41076/m_4107640_nw_18_060_20191005.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/va/2018/va_060cm_2018/39078/m_3907854_se_17_060_20181219.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/40078/m_4007840_se_17_060_20191010.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/40078/m_4007845_sw_17_060_20190922.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/39077/m_3907709_sw_18_060_20190925.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/md/2018/md_060cm_2018/39077/m_3907744_nw_18_060_20181211.tif",
                    "https://naipeuwest.blob.core.windows.net/naip/v002/va/2018/va_060cm_2018/39077/m_3907752_se_18_060_20181111.tif"),
        chesapeake_dataset=Chesapeake13,
        download=False,
        chip_size=400,
        n_classes=13,
        cmap="tab20",
        vis_vmin=-0.5,
        vis_vmax=19.5,
        **kwargs,
    ):
        super().__init__(
            naip_urls=naip_urls,
            label_dataset_cls=chesapeake_dataset,
            n_classes=n_classes,
            cmap=cmap,
            vis_vmin=vis_vmin,
            vis_vmax=vis_vmax,
            download=download,
            chip_size=chip_size,
            **kwargs,
        )
        self.image = self.image.astype(np.uint8)

    @classmethod
    def get_dataset_name(cls):
        return "chesapeake13"


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

    @classmethod
    def get_dataset_name(cls):
        return "safeforest_ortho"


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

    @classmethod
    def get_dataset_name(cls):
        return "safeforest_gmaps"


class AIIRAGreennessRegresssionData(ImageNPMaskedLabeledImage):
    def __init__(
        self, image=Path(DATA_FOLDER, "maps/aiira/random_field.png"), **kwargs
    ):
        super().__init__(image=image, download=4, **kwargs)
        self.label = compute_greenness(self)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/aiira"))

    @classmethod
    def get_dataset_name(cls):
        return "aiira"


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
        site=None,
    ):
        """_summary_

        Args:
            image (_type_, optional): _description_. Defaults to Path(DATA_FOLDER, "maps/cuprite/aster/aster_cube_norm.npy").
            label (_type_, optional): _description_. Defaults to Path(DATA_FOLDER, "maps/cuprite/labels/mineral.npy").
            site (_type_, optional): If a letter, take a crop to match Alberto's prior work. Defaults to None.
        """
        # TODO update plotting options
        super().__init__(
            image=image,
            label=label,
            downsample=1,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            cmap="tab10",
            n_classes=10,
        )
        if site is not None:
            if site == "A":
                i_lim, j_lim = [1760, 1840], [1600, 1680]
            elif site == "B":
                i_lim, j_lim = [1070, 1270], [1550, 1750]
            elif site == "C":
                i_lim, j_lim = [1050, 1250], [1700, 1900]
            elif site == "D":
                i_lim, j_lim = [1550, 1850], [1900, 2200]
            crop = self.get_crop(i_lim=i_lim, j_lim=j_lim)
            self.image = crop.image
            self.mask = crop.mask
            self.label = crop.label
            self.locs = crop.locs

        # Condense the channels
        self.label = take_top_k_classes(self.label, 10)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/cuprite"))

    @classmethod
    def get_dataset_name(cls):
        return "cuprite_aster"

    def vis(self, vis_TSNE=False):
        if vis_TSNE:
            valid_features = self.get_valid_image_points()
            valid_labels = self.get_valid_label_points()
            n_points = valid_features.shape[0]
            random_inds = np.random.choice(n_points, n_points) < 10000

            valid_features = valid_features[random_inds]
            valid_labels = valid_labels[random_inds]

            X_embedded = TSNE(
                n_components=2, learning_rate="auto", init="random", perplexity=3
            ).fit_transform(valid_features)
            plt.scatter(
                X_embedded[:, 0], X_embedded[:, 1], c=valid_labels, cmap="tab10"
            )
            plt.show()
        super().vis()


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
            downsample=1,
            use_value_allchannels_mask=0,
            drop_last_image_channel=False,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            cmap="tab10",
            n_classes=10,
        )
        self.label = take_top_k_classes(self.label, 10)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/cuprite"))

    @classmethod
    def get_dataset_name(cls):
        return "cuprite_aviris_aster"


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
            downsample=1,
            use_value_allchannels_mask=0,
            drop_last_image_channel=False,
            vis_vmin=-0.5,
            vis_vmax=9.5,
            cmap="tab10",
            n_classes=10,
        )
        self.label = take_top_k_classes(self.label, 10)

    def download(self):
        pull_dvc_data(Path(DATA_FOLDER, "maps/cuprite"))

    @classmethod
    def get_dataset_name(cls):
        return "cuprite"


class ReforesTreeClassificationData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        item_id: int = 0,
        use_classes_as_targets: bool = True,
        download=False,
        **kwargs,
    ):
        """
        item_id: which image to use, ordered by the internal index
        use_classes_as_targets: predict the classification, not the biomass regression
        """
        dataset = ReforesTree(
            root=Path(DATA_FOLDER, "torchgeo", "reforestree"), download=download
        )
        item = dataset[item_id]
        image = np.transpose(item["image"].cpu().numpy(), (1, 2, 0))
        boxes = item["boxes"].cpu().numpy()
        label = item["label"].cpu().numpy()
        agb = item["agb"].cpu().numpy()

        areas = self.get_areas(boxes)
        abg_density = agb / areas

        class_label_map = self.fill_boxes(boxes, label, image.shape[:2])
        abg_map = self.fill_boxes(
            boxes, abg_density, image_shape=image.shape[:2], dtype=float
        )
        # Shift these by one to account for the background being zero
        self.class2idx = {k: v + 1 for k, v in dataset.class2idx.items()}

        # TODO masking isn't handled correctly because the invalid regions are white
        if use_classes_as_targets:
            label = class_label_map
            cmap = "tab10"
            vis_vmin = -0.5
            vis_vmax = 9.5
        else:
            label = abg_map
            cmap = None
            vis_vmin = None
            vis_vmax = None

        super().__init__(
            image,
            mask=None,
            label=label,
            n_classes=len(self.class2idx),
            use_value_allchannels_mask=255,
            cmap=cmap,
            vis_vmax=vis_vmax,
            vis_vmin=vis_vmin,
            **kwargs,
        )

    @classmethod
    def get_dataset_name(cls):
        return "reforestree"

    def get_areas(self, boxes):
        if len(boxes.shape) == 0:
            return np.zeros((0, 1))
        elif len(boxes.shape) == 1:
            boxes = np.expand_dims(boxes, axis=0)
            breakpoint()
        i_dif = boxes[:, 2] - boxes[:, 0]
        j_dif = boxes[:, 3] - boxes[:, 1]
        areas = i_dif * j_dif
        return areas

    def fill_boxes(self, boxes, values, image_shape, dtype=np.uint8):
        canvas = np.zeros(image_shape, dtype=dtype)
        for (i_low, j_low, i_high, j_high), value in zip(boxes, values):
            j_low, i_low, j_high, i_high = [
                int(x) for x in (i_low, j_low, i_high, j_high)
            ]
            canvas[i_low:i_high, j_low:j_high] = value
        return canvas


ALL_LABELED_DOMAIN_DATASETS = {
    x.get_dataset_name(): x
    for x in [
        CupriteAVIRISMineralClassificationData,
        CupriteASTERMineralClassificationData,
        CupriteAVIRISASTERMineralClassificationData,
        AIIRAGreennessRegresssionData,
        SafeForestGMapGreennessRegressionData,
        SafeForestOrthoGreennessRegressionData,
        YellowcatDroneClassificationData,
        ChesapeakeBayNaipLandcover7ClassificationData,
        ReforesTreeClassificationData,
        CoralLandsatClassificationData,
        CoralLandsatRegressionData,
    ]
}

ALL_DOMAIN_DATASETS = {
    **ALL_LABELED_DOMAIN_DATASETS,
}
