import math
from statistics import variance
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import GRID_RESOLUTION, MEAN_KEY, UNCERTAINTY_KEY


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, train_x, train_y, likelihood, lengthscale=None, lengthscale_std=None
    ):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if lengthscale is None or lengthscale_std is None:
            lengthscale_prior = None
        else:
            lengthscale_prior = gpytorch.priors.NormalPrior(
                lengthscale, lengthscale_std
            )
        rbf = gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
        rbf.lengthscale = lengthscale_prior
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegressionWorldModel(BaseWorldModel):
    def __init__(
        self, training_iters=1, lengthscale=None, lengthscale_std=None, device="cuda:0"
    ):
        self.training_iters = training_iters

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.model = None
        self.X = None
        self.y = None

        self.lengthscale = lengthscale
        self.lengthscale_std = lengthscale_std
        self.device = device

    def add_observation(self, location, value, unsqueeze=True):
        # Find optimal model hyperparameters
        if unsqueeze:
            location = torch.unsqueeze(torch.Tensor(location).cuda(), dim=0)
        else:
            location = torch.Tensor(location).cuda()
        value = torch.Tensor(np.atleast_1d(value)).cuda()
        if self.X is None:
            self.X = location
            self.y = value
        else:
            self.X = torch.vstack((self.X, location))
            self.y = torch.hstack((self.y, value))

    def setup_model(self):
        self.model = ExactGPModel(
            self.X,
            self.y,
            self.likelihood,
            lengthscale=self.lengthscale,
            lengthscale_std=self.lengthscale_std,
        ).cuda()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def train_model(self, verbose=False):
        # Setup
        self.setup_model()

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
        if self.model is None:
            self.setup_model()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            locations = torch.Tensor(locations).to(self.device)
            observed_pred = self.likelihood(self.model(locations))
            variance = observed_pred.variance
            mean = observed_pred.mean

        return {
            MEAN_KEY: mean.detach().cpu().numpy(),
            UNCERTAINTY_KEY: variance.detach().cpu().numpy(),
        }
