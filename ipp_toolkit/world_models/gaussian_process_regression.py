import math
from statistics import variance
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.utils.sampling import get_flat_samples
from ipp_toolkit.config import GRID_RESOLUTION, MEAN_KEY, VARIANCE_KEY
from pathlib import Path
import ubelt as ub

from moviepy.video.io.bindings import mplfig_to_npimage


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegressionWorldModel(BaseWorldModel):
    def __init__(self, training_iters=50, device="cuda:0"):
        self.training_iters = training_iters

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.model = None
        self.X = None
        self.y = None

        self.device = device

    def add_observation(self, location, value):
        # Find optimal model hyperparameters
        location = torch.unsqueeze(torch.Tensor(location).cuda(), dim=0)
        value = torch.Tensor(np.atleast_1d(value)).cuda()
        if self.X is None:
            self.X = location
            self.y = value
        else:
            self.X = torch.vstack((self.X, location))
            self.y = torch.hstack((self.y, value))

    def train_model(self, verbose=False):
        # Setup
        self.model = ExactGPModel(self.X, self.y, self.likelihood).cuda()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters

        self.model.train()
        self.likelihood.train()

        for i in range(self.training_iters):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.y)
            loss.backward()
            if verbose:
                print(
                    "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                    % (
                        i + 1,
                        self.training_iters,
                        loss.item(),
                        self.model.covar_module.base_kernel.lengthscale.item(),
                        self.model.likelihood.noise.item(),
                    )
                )
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    def sample_belief(self, location):
        location = np.atleast_2d(location)
        return self.sample_belief_array(location)

    def sample_belief_array(self, locations):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            locations = torch.Tensor(locations).to(self.device)
            observed_pred = self.likelihood(self.model(locations))
            variance = observed_pred.variance
            mean = observed_pred.mean

        return {
            MEAN_KEY: mean.detach().cpu().numpy(),
            VARIANCE_KEY: variance.detach().cpu().numpy(),
        }

    def predict_grid(
        self, world_size=(10, 10), resolution=GRID_RESOLUTION, world_start=(0, 0),
    ):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            samples, initial_shape = get_flat_samples(
                world_size, resolution, world_start=world_start
            )
            samples = torch.Tensor(samples).to(self.device)
            observed_pred = self.likelihood(self.model(samples))
            variance = observed_pred.variance
            mean = observed_pred.mean
            mean, variance = [torch.reshape(x, initial_shape) for x in (mean, variance)]
            mean, variance = [x.detach().cpu().numpy() for x in (mean, variance)]
        return mean, variance

    def test_model(
        self,
        world_size=(10, 10),
        resolution=GRID_RESOLUTION,
        world_start=(0, 0),
        gt_data=None,
        vis: bool = True,
        savefile=None,
    ):
        mean, variance = self.predict_grid(world_size, resolution, world_start)

        if vis:
            if gt_data is None:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
                all_axs = (ax1, ax2)
            else:
                f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(4, 3))
                all_axs = (ax1, ax2, ax3, ax4)

            extent = (
                world_start[0],
                world_start[0] + world_size[0],
                world_start[1],
                world_start[1] + world_size[1],
            )  # left, right, bottom, top

            cb0 = ax1.imshow(mean, extent=extent, vmin=0, vmax=1)
            ax1.set_title("predicted")
            cb1 = ax2.imshow(variance, extent=extent)
            ax2.set_title("model variance")

            [
                x.scatter(
                    self.X.detach().cpu().numpy()[:, 1],
                    world_size[1] - self.X.detach().cpu().numpy()[:, 0],
                    c="w",
                    marker="+",
                )
                for x in all_axs
            ]

            plt.colorbar(cb0, ax=ax1, orientation="vertical")
            plt.colorbar(cb1, ax=ax2, orientation="vertical")

            if gt_data is not None:

                cb2 = ax3.imshow(gt_data, extent=extent, vmin=0, vmax=1)
                plt.colorbar(cb2, ax=ax3, orientation="vertical")
                ax3.set_title("ground truth")

                error = mean - gt_data
                cb3 = ax4.imshow(error, cmap="seismic", vmin=-1, vmax=1, extent=extent)
                plt.colorbar(cb3, ax=ax4, orientation="vertical")
                ax4.set_title("error")
            [x.set_xticks([]) for x in (ax1, ax2, ax3, ax4)]
            [x.set_yticks([]) for x in (ax1, ax2, ax3, ax4)]

            if savefile is not None:
                savefile = Path(savefile)
                ub.ensuredir(savefile.parent)
                plt.savefig(savefile)
                plt.close()
            else:
                img = mplfig_to_npimage(f)
                plt.close()
                return img
