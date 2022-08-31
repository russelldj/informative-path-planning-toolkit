import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.utils.sampling import get_flat_samples


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
        value = torch.Tensor(value).cuda()
        if self.X is None:
            self.X = location
            self.y = value
        else:
            self.X = torch.vstack((self.X, location))
            self.y = torch.hstack((self.y, value))

    def train_model(self):
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

    def test_model(
        self,
        world_size=(10, 10),
        resolution=0.101,
        world_start=(-10, -10),
        vis: bool = True,
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
        if vis:
            f, axs = plt.subplots(1, 3, figsize=(4, 3))
            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            cb0 = axs[0].imshow(mean)
            cb1 = axs[1].imshow(variance)
            plt.colorbar(cb0, ax=axs[0], orientation="vertical")
            plt.colorbar(cb1, ax=axs[1], orientation="vertical")
            plt.show()
