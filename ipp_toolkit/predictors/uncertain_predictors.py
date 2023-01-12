import math
from statistics import variance
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import GRID_RESOLUTION, MEAN_KEY, UNCERTAINTY_KEY


class UncertainPredictor:
    def fit(self, X, y):
        """
        Fit the underlying model
        
        Arguments:
            X: features
            y: labels 
        """
        # TODO figure out the right way to work with ABCs
        raise NotImplementedError("Abstract base class")

    def predict_uncertain(self, X):
        """
        Predict the label and the uncertainty uncertainty of the label

        Arguments:
            X: features

        Returns:
            a dict containing the predicted label and the uncertainty about that label
        """

        raise NotImplementedError("Abstract base class")

    def uncertainty(self, X):
        """
        Predict the uncertainty of the label

        Arguments:
            X: features

        Returns:
            The predicted uncertainty of the label
        """
        return self.predict_uncertain(X)[UNCERTAINTY_KEY]

    def predict(self, X):
        """
        Predict the label

        Arguments:
            X: features
        
        Returns:
            the predicted label
        """
        return self.predict_uncertain(X)[MEAN_KEY]


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        rbf = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)

    def forward(self, x):

        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessRegression(UncertainPredictor):
    def __init__(self, training_iters=1, device="cuda:0"):
        self.training_iters = training_iters

        # initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.model = None
        self.device = device

    def _setup_model(self, X, y):
        self.model = ExactGPModel(X, y, self.likelihood).cuda()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def fit(self, X, y, verbose=False):

        # Transform x and y to the right device
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)

        # Setup
        self._setup_model(X, y)

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
            output = self.model(X)
            # Calc loss and backprop gradients
            loss = -self.mll(output, y)
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

    def predict_uncertain(self, X):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X = torch.Tensor(X).to(self.device)
            # TODO see if it's faster to only predict the mean or variance
            pred = self.likelihood(self.model(X))

        return {
            MEAN_KEY: pred.mean.detach().cpu().numpy(),
            UNCERTAINTY_KEY: pred.variance.detach().cpu().numpy(),
        }

