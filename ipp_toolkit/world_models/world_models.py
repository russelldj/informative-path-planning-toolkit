from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ubelt as ub
from ipp_toolkit.config import (
    GRID_RESOLUTION,
    MEAN_ERROR_KEY,
    MEAN_KEY,
    MEAN_UNCERTAINTY_KEY,
    TOP_FRAC_MEAN_ERROR,
    TOP_FRAC_MEAN_VARIANCE,
    UNCERTAINTY_KEY,
)
from ipp_toolkit.utils.sampling import get_flat_samples


class BaseWorldModel:
    def __init__(self, world_extent=None):
        self.world_extent = world_extent

    def add_observation(self, location, value):
        """Add a new observation from the sensor

        Arguments:
            location: ArrayLike
                Where the sample was taken
            value: ArrayLike
                What the sample was
        """
        raise NotImplementedError()

    def sample_belief(self, location):
        """Samples a single belief from the model

        Arguments:
            location: where to sample the belief

        Returns:
            A dict containing (1,m) samples, with 1 point corresponding to the sample
            location and m features. The keys index the different quantites
        """
        raise NotImplementedError()

    def sample_belief_array(self, locations):
        """Samples n beliefs from different locations from the model

        Arguments:
            locations: Where to sample the beliefs. Each row should represent
            a location

        Returns:
            A dict containing (n,m) samples, with n points corresponding to the sample
            locations and m features. The keys index the different quantites
        """
        raise NotImplementedError()

    def sample_belief_grid(
        self, world_size=(10, 10), resolution=GRID_RESOLUTION, world_start=(0, 0),
    ):
        """Samples n beliefs from different locations from the model

        Arguments:
            world_size: The size of the world to sample
            resolution: The resolution of the sampling grid
            world_start: where the top left corner of your world is

        Returns:
            a dict containing the quantaties, for example the predicted mean
            and variance
        """
        samples, initial_shape = get_flat_samples(
            world_size, resolution, world_start=world_start
        )
        values_dict = self.sample_belief_array(samples)
        values_dict = {k: np.reshape(v, initial_shape) for k, v in values_dict.items()}
        return values_dict

    def evaluate_metrics(
        self,
        ground_truth,
        world_size=(10, 10),
        resolution=GRID_RESOLUTION,
        world_start=(0, 0),
        top_fraction=0.40,
    ):
        """Evaluates the error across a grid

        Arguments:
            ground_truth: the real values
            world_size: the size
            resolution: the size between samples
            world_start: the top left corner of the world
            top_fraction: Take the metrics on the top fraction of the world

        Returns:
            dict containing the metric values
        """
        values_dict = self.sample_belief_grid(world_size, resolution, world_start)
        mean = values_dict[MEAN_KEY]
        error_map = mean - ground_truth
        # mean_error = np.mean(np.abs(error_map))
        mean_error = np.linalg.norm(error_map)
        return_dict = {MEAN_ERROR_KEY: mean_error}

        sorted_inds = np.argsort(ground_truth.flatten())
        sorted_gt = ground_truth.flatten()[sorted_inds]
        sorted_preds = mean.flatten()[sorted_inds]
        sorted_error = sorted_preds - sorted_gt

        top_k = int(len(sorted_gt) * top_fraction)
        if top_k == 0:
            raise ValueError("No samples")

        # return_dict[TOP_FRAC_MEAN_ERROR] = np.mean(np.abs(sorted_error[-top_k:]))
        return_dict[TOP_FRAC_MEAN_ERROR] = np.linalg.norm(sorted_error[-top_k:])
        if UNCERTAINTY_KEY in values_dict:
            return_dict[MEAN_UNCERTAINTY_KEY] = np.mean(values_dict[UNCERTAINTY_KEY])
            flat_variance = values_dict[UNCERTAINTY_KEY].flatten()
            return_dict[TOP_FRAC_MEAN_VARIANCE] = np.mean(flat_variance[-top_k:])

        return return_dict

    def test_model(
        self,
        world_size=(10, 10),
        resolution=GRID_RESOLUTION,
        world_start=(0, 0),
        gt_data=None,
        vis: bool = True,
        savefile=None,
        plot=False,
        **kwargs,
    ):
        """
        Various testing options

        Arguments:
            world_size: the size
            resolution: the size between samples
            world_start: the top left corner of the world
            gt_data: the real values
            vis: whether to visualize
            savefile: where to save the image
            **kwargs: visualizaition keyword args
        """
        values_dict = self.sample_belief_grid(world_size, resolution, world_start)
        mean = values_dict[MEAN_KEY]
        var = values_dict[UNCERTAINTY_KEY]
        if vis:
            return self.visualize(
                mean=mean,
                variance=var,
                world_size=world_size,
                world_start=world_start,
                gt_data=gt_data,
                savefile=savefile,
                plot=plot,
                **kwargs,
            )
        else:
            return mean

    def visualize(
        self,
        mean,
        variance,
        world_size=(10, 10),
        world_start=(0, 0),
        gt_data=None,
        savefile=None,
        plot=False,
        ticks=True,
        error_min=None,
        error_max=None,
        gt_min=None,
        gt_max=None,
    ):
        """Visualize the predictions

        Arguments:
            mean: predicted value
            variance: predicted variance
            world_size: the size
            world_start: the top left corner of the world
            gt_data: the real values
            savefile: where to save the image
            ticks: whether to have ticks on the image plots
        """
        if gt_data is None:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
            all_axs = (ax1, ax2)
        else:
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3))
            all_axs = (ax1, ax2, ax3, ax4)

        extent = (
            world_start[1],
            world_start[1] + world_size[1],
            world_start[0],
            world_start[0] + world_size[0],
        )  # left, right, bottom, top

        cb0 = ax1.imshow(mean, extent=extent, vmin=gt_min, vmax=gt_max)
        ax1.set_title("predicted")
        cb1 = ax2.imshow(variance, extent=extent)
        ax2.set_title("model variance")

        # breakpoint()
        # [
        #    x.scatter(
        #        self.X.detach().cpu().numpy()[:, 1],
        #        world_size[0] - self.X.detach().cpu().numpy()[:, 0],
        #        c="w",
        #        marker="+",
        #    )
        #    for x in all_axs
        # ]

        plt.colorbar(cb0, ax=ax1, orientation="vertical")
        plt.colorbar(cb1, ax=ax2, orientation="vertical")

        if gt_data is not None:

            cb2 = ax3.imshow(gt_data, extent=extent, vmin=gt_min, vmax=gt_max)
            plt.colorbar(cb2, ax=ax3, orientation="vertical")
            ax3.set_title("ground truth")

            error = mean - gt_data
            cb3 = ax4.imshow(error, cmap="seismic", vmin=-1, vmax=1, extent=extent)
            plt.colorbar(cb3, ax=ax4, orientation="vertical")
            ax4.set_title("error")
        if not ticks:
            [x.set_xticks([]) for x in all_axs]
            [x.set_yticks([]) for x in all_axs]

        if savefile is not None:
            savefile = Path(savefile)
            ub.ensuredir(savefile.parent)
            plt.savefig(savefile)
            plt.close()
        else:
            img = mplfig_to_npimage(f)
            if plot:
                plt.pause(3)
            plt.close()
            return img
