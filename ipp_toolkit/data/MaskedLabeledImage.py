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


def load_image_npy_passthrough(filename_or_data):
    if isinstance(filename_or_data, np.ndarray):
        return filename_or_data
    return load_image_npy(filename_or_data)


class MaskedLabeledImage(GridData2D):
    def __init__(
        self,
        image,
        mask=None,
        label=None,
        downsample=1,
        blur_sigma=None,
        use_last_channel_mask: bool = False,
    ):
        """
        image: str | np.array
        mask: str | np.array | None
        image: str | np.array | None
        """

        self.image = load_image_npy_passthrough(image)

        if use_last_channel_mask and mask is not None:
            raise ValueError("Do not specify use_last_channel and provide a mask name")
        elif mask is not None:
            self.mask = np.squeeze(load_image_npy_passthrough(mask)).astype(bool)
        elif use_last_channel_mask:
            self.mask = self.image[..., -1] > 0
            self.image = self.image[..., :-1]
        else:
            self.mask = np.ones(self.image.shape[:2], dtype=bool)

        if label is not None:
            self.label = load_image_npy_passthrough(label)
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

    def get_valid_image_points(self):
        return self.image[self.mask]

    def get_valid_label_points(self):
        return self.label[self.mask]

    def get_valid_loc_points(self):
        return self.locs[self.mask]

    def get_valid_loc_images_points(self):
        locs = self.get_valid_loc_points()
        features = self.get_valid_image_points()
        return np.concatenate((locs, features), axis=1)

    def sample_batch(self, locs, assert_valid=False, vis=False):
        """
        locs: (n, 2), in i,j format
        """
        if vis:
            f, axs = plt.subplots(1, 2)
            axs[0].imshow(self.label)
            axs[1].imshow(self.mask)
            # This is i,j convention
            axs[0].scatter(locs[:, 1], locs[:, 0])
            axs[1].scatter(locs[:, 1], locs[:, 0])

        invalid_points = np.logical_not(self.mask[locs[:, 0], locs[:, 1]])
        if assert_valid and np.any(invalid_points):
            invalid_locs = locs[invalid_points]
            f, axs = plt.subplots(1, 2)
            axs[0].imshow(self.label)
            axs[1].imshow(self.mask)
            # This is i,j convention
            axs[0].scatter(invalid_locs[:, 1], invalid_locs[:, 0])
            axs[1].scatter(invalid_locs[:, 1], invalid_locs[:, 0])
            breakpoint()
            raise ValueError("Sampled invalid points")
        sample_values = self.label[locs[:, 0], locs[:, 1]]
        sample_values[invalid_points] = np.nan
        return sample_values

    def sample_batch_locs(self, locs):
        invalid_points = np.logical_not(self.mask[locs[:, 0], locs[:, 1]])
        # This should be a no-op for now, but might change in the future
        sample_locs = self.locs[locs[:, 0], locs[:, 1]]
        sample_locs[invalid_points] = np.nan
        return sample_locs

    def sample_batch_features(self, locs):
        invalid_points = np.logical_not(self.mask[locs[:, 0], locs[:, 1]])
        sample_features = self.image[locs[:, 0], locs[:, 1]]
        sample_features[invalid_points] = np.nan
        return sample_features

    def sample_batch_locs_and_features(self, locs):
        """
        Args:
            locs: np.ndarray, (n, 2)

        Returns:
            (n, 2 + n_features) 
        """
        sample_features = self.sample_batch_features(locs)
        sample_locs = self.sample_batch_locs()

        return np.concatenate((sample_locs, sample_features), axis=1)

    def get_image_for_flat_values(self, flat_values):
        """
        The number of flat values must match the number of valid entires in the mask

        The remainder of the image will be black
        """
        image = np.full_like(self.mask, fill_value=np.nan, dtype=float)
        image[self.mask] = flat_values

        return image

    def vis(self):
        f, axs = plt.subplots(1, 3)
        axs[0].imshow(self.image)
        if self.mask is not None:
            axs[1].imshow(self.mask)
        if self.label is not None:
            plt.colorbar(axs[2].imshow(self.label), ax=axs[2])
        plt.show()
