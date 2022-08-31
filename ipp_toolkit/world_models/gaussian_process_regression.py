import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel


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
    def __init__(self, training_iters=50):
        self.training_iters = training_iters

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        # "Loss" for GPs - the marginal log likelihood
        self.model = None  # ExactGPModel(train_x, train_y, likelihood).cuda()
        self.X = None
        self.y = None

    def add_observation(self, location, value):
        # Find optimal model hyperparameters
        location = np.expand_dims(location, axis=0)
        value = np.expand_dims(value, axis=0)
        if self.X is None:
            self.X = location
            self.y = value
        else:
            self.X = np.vstack(self.X, location)
            self.X = np.vstack(self.y, value)

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

    def test_model(self):
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            axis_samples_test = axis_samples_train
            test_x = torch.meshgrid(axis_samples_test, axis_samples_test).cuda()
            test_x = [x.flatten() for x in test_x]
            test_x = torch.vstack(test_x).T
            observed_pred = likelihood(model(test_x))

        with torch.no_grad():
            # Initialize plot
            f, axs = plt.subplots(1, 3, figsize=(4, 3))
            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            lower, upper = [torch.reshape(x, initial_shape) for x in (lower, upper)]
            cb0 = axs[0].imshow(lower, vmin=-2, vmax=16)
            cb1 = axs[1].imshow(upper, vmin=-2, vmax=16)
            plt.colorbar(cb0, ax=axs[0], orientation="vertical")
            plt.colorbar(cb1, ax=axs[1], orientation="vertical")
            plt.show()
        plt.show()
