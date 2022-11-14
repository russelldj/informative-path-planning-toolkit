import numpy as np
from ipp_toolkit.data.data import GridData2D
from ipp_toolkit.utils.sampling import get_flat_samples
from imageio import imread
from pathlib import Path
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt


def load_image_npy(filename):
    if Path(filename).suffix == ".npy":
        data = np.load(filename)
    else:
        data = imread(filename)
    return data


class MaskedLabeledImage(GridData2D):
    def __init__(
        self, image_name, mask_name=None, label_name=None, downsample=1, blur_sigma=None
    ):
        """
        image_name:
        mask_name:
        image_name:
        """
        self.image = load_image_npy(image_name)

        if mask_name is not None:
            self.mask = np.squeeze(load_image_npy(mask_name)).astype(bool)
        else:
            self.mask = np.ones(self.image.shape[:2], dtype=bool)

        if label_name is not None:
            self.label = load_image_npy(label_name)
        else:
            self.label = None

        world_size = self.image.shape[:2]
        assert len(self.mask.shape) == 2
        assert len(self.image.shape) == 3
        assert self.label is None or len(self.label.shape) in (2, 3)

        assert self.mask.shape[:2] == world_size
        assert self.label is None or self.label.shape[:2] == world_size

        if downsample != 1:
            output_size = (np.array(self.image.shape[:2]) / downsample).astype(int)
            # Resize image by channel for memory reasons
            resized_channels = []
            for i in range(self.image.shape[2]):
                # Warning: this converts to a float image in the range (0,1)
                resized_channel = resize(self.image[..., i], output_size)
                resized_channels.append(resized_channel)
            self.image = np.stack(resized_channels, axis=2)
            if self.mask is not None:
                self.mask = resize(self.mask, output_size, anti_aliasing=False)
            if self.label is not None:
                self.label = resize(self.label, output_size, anti_aliasing=True)

        if blur_sigma is not None:
            self.image = gaussian(self.image, sigma=blur_sigma)
        samples, initial_shape = get_flat_samples(np.array(self.image.shape[:2]) - 1, 1)
        i_locs, j_locs = [np.reshape(samples[:, i], initial_shape) for i in range(2)]
        self.locs = np.stack([i_locs, j_locs], axis=2)

        super().__init__(world_size)

    def get_image_channel(self, channel: int):
        return self.image[..., channel]

    def get_label_channel(self, channel: int):
        return self.label[..., channel]

    def get_mask(self):
        return self.mask

    def get_locs(self):
        return self.locs

    def get_valid_images_points(self):
        return self.image[self.mask]

    def get_valid_label_points(self):
        return self.label[self.mask]

    def get_valid_loc_points(self):
        return self.locs[self.mask]

