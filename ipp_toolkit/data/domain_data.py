from ipp_toolkit.data.MaskedLabeledImage import ImageNPMaskedLabeledImage
from pathlib import Path
from ipp_toolkit.config import DATA_FOLDER
import numpy as np


class CoralLandsatClassificationData(ImageNPMaskedLabeledImage):
    def __init__(
        self,
        image=Path(DATA_FOLDER, "maps/coral/X_wv.npy"),
        mask=Path(DATA_FOLDER, "maps/coral/valid_wv.npy"),
        label=Path(DATA_FOLDER, "maps/coral/Y.npy"),
        **kwargs
    ):
        super().__init__(image=image, mask=mask, label=label, **kwargs)
        self.label = np.argmax(self.label, axis=2)
