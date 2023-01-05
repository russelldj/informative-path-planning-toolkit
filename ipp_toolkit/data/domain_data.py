from ipp_toolkit.data.MaskedLabeledImage import (
    ImageNPMaskedLabeledImage,
    torchgeoMaskedDataManger,
)
from pathlib import Path
from ipp_toolkit.config import DATA_FOLDER
import numpy as np
from torchgeo.datasets import Chesapeake7

from sklearn.cluster import KMeans


class CoralLandsatClassificationData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/coral/X_wv.npy"),
        mask=Path(DATA_FOLDER, "maps/coral/valid_wv.npy"),
        label=Path(DATA_FOLDER, "maps/coral/Y.npy"),
        **kwargs
    ):
        super().__init__(
            image=image,
            mask=mask,
            label=label,
            cmap="tab10",
            vis_vmin=-0.5,
            vis_vmax=9.5,
            **kwargs
        )
        self.label = np.argmax(self.label, axis=2)


class YellowcatDroneClassificationData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/yellowcat/20221028_M7_orthophoto.tif"),
        n_pseudo_classes=8,
        fit_on_n_points=10000,
        **kwargs
    ):
        super().__init__(
            image=image,
            downsample=16,
            use_last_channel_mask=True,
            n_classes=n_pseudo_classes,
            cmap="tab10",
            vis_vmin=-0.5,
            vis_vmax=9.5,
            **kwargs
        )
        kmeans = KMeans(n_clusters=n_pseudo_classes)
        valid_features = self.get_valid_image_points()
        subset_inds = np.random.choice(
            valid_features.shape[0], min(fit_on_n_points, valid_features.shape[0])
        )
        kmeans.fit_predict(valid_features[subset_inds])
        labels = kmeans.predict(valid_features)
        self.label = self.get_image_for_flat_values(labels)


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
        **kwargs
    ):
        super().__init__(
            naip_tiles=naip_tiles,
            chesapeake_dataset=chesapeake_dataset,
            cmap="tab10",
            vis_vmin=-0.5,
            vis_vmax=9.5,
            **kwargs
        )

