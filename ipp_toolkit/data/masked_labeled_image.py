import logging
import os
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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, Chesapeake7, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler
from sklearn.metrics import accuracy_score

from ipp_toolkit.config import (
    DATA_FOLDER,
    MEAN_ERROR_KEY,
    MEAN_KEY,
    VIS,
    ERROR_IMAGE,
    N_TOP_FRAC,
    TOP_FRAC,
    TOP_FRAC_MEAN_ERROR,
)
from ipp_toolkit.data.data import GridData2D
from ipp_toolkit.utils.sampling import get_flat_samples
from ipp_toolkit.visualization.utils import show_or_save_plt, add_colorbar


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
        image,
        mask,
        label,
        vis_image=None,
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
        self.image = image
        self.mask = mask
        self.label = label
        self.vis_image = vis_image
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
            if self.vis_image is not None:
                resized_channels = []
                for i in range(self.vis_image.shape[2]):
                    # Warning: this converts to a float image in the range (0,1)
                    resized_channel = resize(self.vis_image[..., i], output_size)
                    resized_channels.append(resized_channel)
                self.vis_image = np.stack(resized_channels, axis=2)
            if self.mask is not None:
                self.mask = resize(self.mask, output_size, anti_aliasing=False)
            if self.label is not None:
                self.label = resize(
                    self.label,
                    output_size,
                    anti_aliasing=True,
                    order=0 if self.is_classification_dataset else 1,
                )

        if blur_sigma is not None:
            self.image = multichannel_gaussian(self.image, blur_sigma)

        self._set_locs()
        super().__init__(world_size)

    def _set_locs(self):
        samples, initial_shape = get_flat_samples(np.array(self.image.shape[:2]) - 1, 1)
        i_locs, j_locs = [np.reshape(samples[:, i], initial_shape) for i in range(2)]
        self.locs = np.stack([i_locs, j_locs], axis=2)

    def vis(self, vmin=None, vmax=None, cmap=None, savepath=None):
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
            add_colorbar(axs[1].imshow(self.mask, vmin=False, vmax=True))
            axs[1].set_title("Mask")
            n_plotted += 1
        if self.label is not None:
            display_label = self.label.copy().astype(float)
            display_label[np.logical_not(self.mask)] = np.nan
            add_colorbar(
                axs[n_plotted].imshow(display_label, vmin=vmin, vmax=vmax, cmap=cmap)
            )
            axs[n_plotted].set_title("Label")

        show_or_save_plt(savepath=savepath)

    def get_vis_image(self):
        if self.vis_image is not None:
            return self.vis_image
        return self.image

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

    def get_random_valid_loc_points(self, n_points, with_replacement: bool = True):
        """
        Chose a random set of n_points valid locations. with_replacement determines whether duplicates can be obtained
        """
        all_valid_loc_points = self.get_valid_loc_points()
        chosen_inds = np.random.choice(
            all_valid_loc_points.shape[0], n_points, replace=with_replacement
        )
        chosen_valid_loc_points = all_valid_loc_points[chosen_inds]
        return chosen_valid_loc_points

    def get_valid_loc_images_points(self):
        locs = self.get_valid_loc_points()
        features = self.get_valid_image_points()
        return np.concatenate((locs, features), axis=1)

    def get_crop(self, i_lim, j_lim):
        new_dataset_image = self.image[i_lim[0] : i_lim[1], j_lim[0] : j_lim[1]].copy()
        new_dataset_mask = self.mask[i_lim[0] : i_lim[1], j_lim[0] : j_lim[1]].copy()
        new_dataset_label = self.label[i_lim[0] : i_lim[1], j_lim[0] : j_lim[1]].copy()
        new_dataset = MaskedLabeledImage(
            image=new_dataset_image,
            mask=new_dataset_mask,
            label=new_dataset_label,
            n_classes=self.n_classes,
            cmap=self.cmap,
            vis_vmin=self.vis_vmin,
            vis_vmax=self.vis_vmax,
        )
        return new_dataset

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
            plt.show()
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
        sample_locs = self.sample_batch_locs(locs)

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

    def eval_prediction(
        self, prediction: dict, norm_ord: int = 1, top_frac: float = TOP_FRAC
    ):
        """
        Args:
            prediction: Some prediction for the target quantity. Should at least contain
            the MEAN_KEY
            norm_ord: The order of the norm to evaluate the error
            top_frac: fraction of top-valued points to evaluate
        
        Returns:
            A dict with at least the MEAN_ERROR_KEY and optionally other metrics and visualizations
        """
        # Currently, only the mean is considered in the prediction accuracy
        mean_pred = prediction[MEAN_KEY]
        # Flatten the prediction for the valid region
        flat_pred = mean_pred[self.mask]

        # Get flat label values for the valid region
        flat_label = self.get_valid_label_points()

        # Metrics are different for classification and regression tasks
        if self.is_classification_dataset():
            # Compute the accuracy and error
            accuracy = accuracy_score(flat_label, flat_pred)
            error = 1 - accuracy
            # Record for output
            return_dict = {MEAN_ERROR_KEY: error}

            error_image = mean_pred != self.label

        else:
            flat_error = flat_pred - flat_label
            sorted_inds = np.argsort(flat_label)
            # Find the indices for the top fraction of ground truth points
            n_top_frac = int(top_frac * len(sorted_inds))
            top_frac_inds = sorted_inds[-n_top_frac:]
            top_frac_errors = flat_error[top_frac_inds]
            n_points = len(flat_error)
            return_dict = {
                TOP_FRAC_MEAN_ERROR: np.linalg.norm(top_frac_errors, ord=norm_ord)
                / n_top_frac,
                MEAN_ERROR_KEY: np.linalg.norm(flat_error, ord=norm_ord) / n_points,
                N_TOP_FRAC: n_top_frac,
            }

            error_image = mean_pred - self.label
        # Mask out invalid regions so they don't mess up visualization
        error_image[np.logical_not(self.mask)] = np.nan
        return_dict[ERROR_IMAGE] = error_image
        return return_dict

    @classmethod
    def get_dataset_name(cls):
        return "base_dataset"


class ImageNPMaskedLabeledImage(MaskedLabeledImage):
    def __init__(
        self,
        image,
        mask=None,
        label=None,
        use_last_channel_mask: bool = False,
        use_value_allchannels_mask: bool = False,
        drop_last_image_channel: bool = None,
        downsample=1,
        blur_sigma=None,
        download: bool = False,
        **kwargs,
    ):
        """
        image: str | np.array
        mask: str | np.array | None
        image: str | np.array | None
        use_value_allchannels_mask: set to a value if that value in all channels indicates it's invalid 
        drop_last_image_channel: if None, defaults to use_last_channel_mask. Drop the last image channel
            # TODO this should be updated to simply a range of channels to include
        download: try to download data, may be a no-op
        """
        # TODO these should be moved to the real baseclass

        if download:
            self.download()

        self.image = load_image_npy_passthrough(image)

        if mask is not None:
            if use_last_channel_mask or use_value_allchannels_mask:
                logging.warning(
                    "Ignoring use_last_channel_mask or use_zero_allchannels_mask due to a mask being directly provided"
                )
            self.mask = np.squeeze(load_image_npy_passthrough(mask)).astype(bool)
        elif use_last_channel_mask:
            self.mask = self.image[..., -1] > 0
        elif use_value_allchannels_mask:
            self.mask = np.sum(self.image != use_value_allchannels_mask, axis=-1) > 0
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

        super().__init__(
            self.image,
            self.mask,
            self.label,
            downsample=downsample,
            blur_sigma=blur_sigma,
            **kwargs,
        )

    def download(self):
        """Attempt to download data"""
        pass


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
        naip_url="https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/",
        naip_tiles=("m_3807511_ne_18_060_20181104.tif",),
        features_dataset_cls=NAIP,
        label_dataset_cls=Chesapeake7,
        downsample=1,
        blur_sigma=None,
        download: bool = False,
        vis=False,
        chip_size=1000,
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
            download: whether to download
        """
        # Initialize everything
        features_root = os.path.join(data_root, "naip")
        labels_root = os.path.join(data_root, "chesapeake")
        # TODO make this more general
        if download:
            # Download naip tiles
            for tile in naip_tiles:
                download_url(naip_url + tile, features_root)
        # Create naip and chesapeake
        features = features_dataset_cls(features_root)
        label = label_dataset_cls(
            labels_root, crs=features.crs, res=features.res, download=download
        )
        # Take the interesection of these datasets
        dataset = features & label
        # Create a sampler and dataloader
        sampler = RandomGeoSampler(dataset, size=chip_size, length=1)
        # Num workers must be 1 or there will be a memory leak that will be
        # very hard to debug
        dataloader = DataLoader(
            dataset, sampler=sampler, collate_fn=stack_samples, num_workers=1
        )
        data_dict = next(iter(dataloader))
        image = data_dict["image"]
        label = data_dict["mask"]
        image, label = [
            np.transpose(x[0].detach().cpu().numpy(), axes=(1, 2, 0))
            for x in (image, label)
        ]
        label = label[..., -1]
        if vis:
            f, axs = plt.subplots(1, 2)
            axs[0].imshow(image[..., :3] / 255)
            axs[1].imshow(label, cmap="tab10", vmin=0, vmax=9)
            plt.show()
        mask = np.ones_like(label, dtype=bool)
        super().__init__(
            image=image,
            mask=mask,
            label=label,
            downsample=downsample,
            blur_sigma=blur_sigma,
            **kwargs,
        )

