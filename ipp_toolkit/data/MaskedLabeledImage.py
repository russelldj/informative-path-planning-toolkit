import numpy as np
from ipp_toolkit.data.data import GridData2D
from ipp_toolkit.utils.sampling import get_flat_samples


def load_image_npy(filename):
    data = np.load(filename)
    return data


class MaskedLabeledImage(GridData2D):
    def __init__(
        self, image_name, mask_name, label_name,
    ):
        self.image = load_image_npy(image_name)
        self.mask = np.squeeze(load_image_npy(mask_name)).astype(bool)
        self.label = load_image_npy(label_name)

        world_size = self.image.shape[:2]
        assert len(self.mask.shape) == 2
        assert len(self.image.shape) == 3
        assert len(self.label.shape) == 3

        assert self.mask.shape[:2] == world_size
        assert self.label.shape[:2] == world_size

        samples, initial_shape = get_flat_samples(np.array(self.mask.shape[:2]) - 1, 1)
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

