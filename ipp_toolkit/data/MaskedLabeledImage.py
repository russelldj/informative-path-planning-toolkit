import logging
import os
import tempfile
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac
import rioxarray
from imageio import imread
from skimage.filters import gaussian
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, Chesapeake7, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

from ipp_toolkit.config import VIS, DATA_FOLDER
from ipp_toolkit.data.data import GridData2D
from ipp_toolkit.utils.sampling import get_flat_samples


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


def multichannel_gaussian(image, blur_sigma):
    if image.dtype is float:
        image = np.stack(
            [gaussian(image[..., i], sigma=blur_sigma) for i in range(image.shape[2])],
            axis=2,
        )
    else:
        float_image = image.astype(float)
        float_image = np.stack(
            [
                gaussian(float_image[..., i], sigma=blur_sigma)
                for i in range(float_image.shape[2])
            ],
            axis=2,
        )
        image = (float_image * 255.0).astype(np.uint8)
    return image


class MaskedLabeledImage(GridData2D):
    def __init__(
        self,
        downsample: Union[int, float] = 1,
        blur_sigma: Union[int, float] = None,
        cmap="viridis",
        n_classes=0,
        vis_vmin=None,
        vis_vmax=None,
    ):
        """
        Arguments:
            downsample: how much to downsample the image
            blur_sigma: how much to blur the downsampled image
            classification_dataset: Is this a classification (not regression) dataset
        """
        self.cmap = cmap
        self.n_classes = n_classes
        self.vis_vmin = vis_vmin
        self.vis_vmax = vis_vmax

        world_size = self.image.shape[:2]
        assert len(self.mask.shape) == 2
        assert len(self.image.shape) == 3
        assert self.label is None or len(self.label.shape) in (2, 3)

        assert self.mask.shape[:2] == world_size
        assert self.label is None or self.label.shape[:2] == world_size
        self._downsample_and_blur(
            downsample=downsample, blur_sigma=blur_sigma, world_size=world_size
        )

    def _downsample_and_blur(self, downsample, blur_sigma, world_size):
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
            self.image = multichannel_gaussian(self.image, blur_sigma)
        samples, initial_shape = get_flat_samples(np.array(self.image.shape[:2]) - 1, 1)
        i_locs, j_locs = [np.reshape(samples[:, i], initial_shape) for i in range(2)]
        self.locs = np.stack([i_locs, j_locs], axis=2)

        super().__init__(world_size)

    def vis(self, vmin=None, vmax=None, cmap=None):
        if cmap is None:
            cmap = self.cmap
        if vmin is None:
            vmin = self.vis_vmin
        if vmax is None:
            vmax = self.vis_vmax
        n_valid = np.sum([x is not None for x in (self.image, self.mask, self.label)])
        _, axs = plt.subplots(1, n_valid)
        n_plotted = 1
        axs[0].imshow(self.image[..., :3])
        axs[0].set_title(f"Satellite image\n 3 / {self.image.shape[2]} channels")
        if self.mask is not None:
            plt.colorbar(axs[1].imshow(self.mask, vmin=False, vmax=True), ax=axs[1])
            axs[1].set_title("Mask")
            n_plotted += 1
        if self.label is not None:
            display_label = self.label.copy().astype(float)
            display_label[np.logical_not(self.mask)] = np.nan
            plt.colorbar(
                axs[n_plotted].imshow(display_label, vmin=vmin, vmax=vmax, cmap=cmap),
                ax=axs[n_plotted],
            )
            axs[n_plotted].set_title("Label")

        plt.show()

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

    def sample_batch(self, locs, assert_valid=False, vis=VIS):
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
        # This cannot be done for integers
        if sample_values.dtype is float:
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

    def is_classification_dataset(self):
        """
            Are the labels class IDs or regression
        """
        return self.n_classes > 0


class ImageNPMaskedLabeledImage(MaskedLabeledImage):
    def __init__(
        self,
        image,
        mask=None,
        label=None,
        use_last_channel_mask: bool = False,
        use_zero_allchannels_mask: bool = False,
        drop_last_image_channel: bool = None,
        downsample=1,
        blur_sigma=None,
        **kwargs,
    ):
        """
        image: str | np.array
        mask: str | np.array | None
        image: str | np.array | None
        use_zero_allchannels_mask: the mask is valid for the locations which are not zero on all channels
        drop_last_image_channel: if None, defaults to use_last_channel_mask. Drop the last image channel
            # TODO this should be updated to simply a range of channels to include
        """
        # TODO these should be moved to the real baseclass

        self.image = load_image_npy_passthrough(image)

        if mask is not None:
            if use_last_channel_mask or use_zero_allchannels_mask:
                logging.warning(
                    "Ignoring use_last_channel_mask or use_zero_allchannels_mask due to a mask being directly provided"
                )
            self.mask = np.squeeze(load_image_npy_passthrough(mask)).astype(bool)
        elif use_last_channel_mask:
            self.mask = self.image[..., -1] > 0
        elif use_zero_allchannels_mask:
            self.mask = np.sum(self.image, axis=-1) > 0
        else:
            self.mask = np.ones(self.image.shape[:2], dtype=bool)

        # Setting it to the default if unset
        if drop_last_image_channel is None:
            drop_last_image_channel = use_last_channel_mask
        if drop_last_image_channel:
            self.image = self.image[..., :-1]

        if label is not None:
            self.label = load_image_npy_passthrough(label)
        else:
            self.label = None
        super().__init__(downsample=downsample, blur_sigma=blur_sigma, **kwargs)


class STACMaskedLabeledImage(MaskedLabeledImage):
    def __init__(
        self,
        image_item_url,
        label_item_url=None,
        image_asset="visual",
        label_asset="visual",
        downsample=1,
        blur_sigma=None,
        **kwargs,
    ):
        self.image, self.mask = self._get_data_from_stac(image_item_url, image_asset)
        self.label, _ = self._get_data_from_stac(label_item_url, label_asset)

        super().__init__(downsample=downsample, blur_sigma=blur_sigma, **kwargs)

    def _get_data_from_stac(self, url, asset):
        if url is None:
            return None, None
        item = pystac.Item.from_file(url)
        signed_item = planetary_computer.sign(item)

        # Open one of the data assets (other asset keys to use: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP', 'visual')
        asset_href = signed_item.assets[asset].href
        logging.info("Begining to read data")
        ds = rioxarray.open_rasterio(asset_href)
        masked_array = ds.to_masked_array()
        logging.info("Done converting data into masked array")
        masked_array = np.transpose(masked_array, (1, 2, 0))
        image = masked_array.data
        mask = np.logical_not(masked_array.mask)
        return image, mask


class torchgeoMaskedDataManger(MaskedLabeledImage):
    """
    Currently this takes a sample from the 
    """

    def __init__(
        self,
        data_root=Path(DATA_FOLDER, "torchgeo"),
        vis_all_chips=False,
        naip_url="https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/",
        naip_tiles=(
            "m_3807511_ne_18_060_20181104.tif",
            "m_3807511_se_18_060_20181104.tif",
            "m_3807512_nw_18_060_20180815.tif",
            "m_3807512_sw_18_060_20180815.tif",
        ),
        chesapeake_dataset=Chesapeake7,
        downsample=1,
        blur_sigma=None,
        **kwargs,
    ):
        """
        Arguments:
            data_root: pathlike
                Where to store the data
            vis_all_chips: Show all the chips from the dataloader
            naip_url: Where to download the naip data from
            naip_tiles: image names to download
            chesapeake_dataset: Which chesapeake dataset to use
        """
        # Initialize everything
        naip_root = os.path.join(data_root, "naip")
        chesapeake_root = os.path.join(data_root, "chesapeake")
        # Download naip tiles
        for tile in naip_tiles:
            download_url(naip_url + tile, naip_root)
        # Create naip and
        self.naip = NAIP(naip_root)
        self.chesapeake = chesapeake_dataset(
            chesapeake_root, crs=self.naip.crs, res=self.naip.res, download=True
        )
        # Take the interesection of these datasets
        self.dataset = self.naip & self.chesapeake
        # Create a sampler and dataloader
        sampler = RandomGeoSampler(self.dataset, size=1000, length=10)
        dataloader = DataLoader(self.dataset, sampler=sampler, collate_fn=stack_samples)
        if vis_all_chips:
            for sample in dataloader:
                image = np.transpose(sample["image"].numpy()[0], (1, 2, 0))
                target = sample["mask"].numpy()[0, 0]
                f, axs = plt.subplots(1, 2)
                axs[0].imshow(image[..., :3])
                plt.colorbar(
                    axs[1].imshow(target, cmap="tab10", vmin=0, vmax=9), ax=axs[1]
                )
                plt.show()
        else:
            # TODO figure out why next doesn't work
            for sample in dataloader:
                # Take the first random chip
                break
        self.image = np.transpose(sample["image"].numpy()[0], (1, 2, 0))
        self.label = sample["mask"].numpy()[0, 0]
        self.mask = np.ones(self.image.shape[:2], dtype=bool)
        super().__init__(downsample=downsample, blur_sigma=blur_sigma, **kwargs)

