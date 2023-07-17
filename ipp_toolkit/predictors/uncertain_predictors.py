import math
from statistics import variance
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

from ipp_toolkit.world_models.world_models import BaseWorldModel
from ipp_toolkit.config import GRID_RESOLUTION, MEAN_KEY, UNCERTAINTY_KEY, TORCH_DEVICE
import logging


class UncertainPredictor:
    def get_name(self):
        return "uncertain_predictor"

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


# We will use the simplest form of GP model, exact inference
class DirichletGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size((num_classes,))
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            train_x.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        ard_num_dims=1,
        noise=None,
        rbf_lengthscale=None,
        output_scale=None,
        device=TORCH_DEVICE,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        rbf = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(rbf)

        if noise is not None:
            self.likelihood.noise = torch.Tensor(np.atleast_1d(noise)).to(device=device)
        if rbf_lengthscale is not None:
            rbf.lengthscale = torch.Tensor(rbf_lengthscale).to(device=device)
        if output_scale is not None:
            self.covar_module._set_outputscale(output_scale)

    def forward(self, x):
        mean_x = self.mean_module(x)

        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess(UncertainPredictor):
    def __init__(
        self,
        training_iters=50,
        device=TORCH_DEVICE,
        kernel_kwargs={},
        is_classification_task=False,
        num_classes: int = None,
    ):
        self.training_iters = training_iters
        self.kernel_kwargs = kernel_kwargs
        # initialize likelihood and model
        self.likelihood = None
        self.model = None
        self.device = device
        self.is_classification_task = is_classification_task
        self.num_classes = num_classes

        self.unique_label_values = None  # Used only for classification

    def get_name(self):
        return "gaussian_process"

    def _setup_model(self, X, y, ard_num_dims=1):
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)
        if self.is_classification_task:
            y = y.to(torch.int64)

            num_classes = int(y.max() + 1)
            breakpoint()
            self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
                num_features=num_classes, num_classes=num_classes
            ).to(self.device)
            self.model = GPClassificationModel(X).to(self.device)
            self.mll = gpytorch.mlls.VariationalELBO(
                self.likelihood, self.model, y.numel()
            ).to(self.device)
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.model = ExactGPModel(
                X, y, self.likelihood, ard_num_dims=ard_num_dims, **self.kernel_kwargs
            ).to(self.device)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.model
            )

    def fit(self, X, y, verbose=True):
        # Transform x and y to the right device
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)

        if self.is_classification_task:
            self.unique_label_values, y = torch.unique(y, return_inverse=True)
            y = y.to(int)

        # Setup
        self._setup_model(X, y, ard_num_dims=X.shape[1])

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
            breakpoint()
            loss = -self.mll(output, y)
            breakpoint()
            loss.backward()
            if verbose and i % 50 == 49:
                print(
                    "Iter %d/%d - Loss: %.3f"
                    % (
                        i + 1,
                        self.training_iters,
                        loss.item(),
                    )
                )
                print(f"noise: {self.model.likelihood.noise}")
                print(f"lengthscale: {self.model.covar_module.base_kernel.lengthscale}")
                print(f"outputscale: {self.model.covar_module.outputscale}")
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    def predict_uncertain(self, X):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X = torch.Tensor(X).to(self.device)
            # TODO see if it's faster to only predict the mean or variance
            pred = self.likelihood(self.model(X))

        if self.is_classification_task:
            # TODO validate this further
            pred_value = pred.loc.max(0)[1]
            # Remap in case all classes weren't represented
            pred_value = self.unique_label_values[pred_value]
            pred_value = pred_value.detach().cpu().numpy()
            pred_uncertainty = pred.variance.sum(0).detach().cpu().numpy()
        else:
            pred_value = pred.mean.detach().cpu().numpy()
            pred_uncertainty = pred.variance.detach().cpu().numpy()

        return {MEAN_KEY: pred_value, UNCERTAINTY_KEY: pred_uncertainty}

    def predict_covariance(self, X):
        # TODO I'm not quite sure if this should be done like this
        self._setup_model(X, y=np.zeros_like(X[:, 0]), ard_num_dims=X.shape[1])

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X = torch.Tensor(X).to(self.device)
            posterior = self.model(X)
            covariance = posterior.covariance_matrix
            # This invalidates the prior training
            self.model = None
            return covariance.detach().cpu().numpy()


class EnsamblePredictor(UncertainPredictor):
    def __init__(
        self,
        prediction_model,
        n_ensamble_models=3,
        frac_per_model: float = 0.5,
        classification_task=True,
    ):
        self.n_ensamble_models = n_ensamble_models
        self.frac_per_model = frac_per_model
        self.classification_task = classification_task

        # Create a collection of independent predictors. Each one will be fit on a subset of data
        self.estimators = [deepcopy(prediction_model) for _ in range(n_ensamble_models)]

    def fit(self, X, y):
        for i in range(self.n_ensamble_models):
            n_points = X.shape[0]
            chosen_inds = np.random.choice(
                n_points, size=int(n_points * self.frac_per_model), replace=False
            )
            chosen_X = X[chosen_inds]
            chosen_y = y[chosen_inds]
            self.estimators[i].fit(chosen_X, chosen_y)

    def predict_uncertain(self, X):
        # Generate a prediction with each model
        predictions = [e.predict(X) for e in self.estimators]
        # Average over all the models
        if not self.classification_task:
            # The mean and unceratinty are just the mean and variance across all models
            mean_prediction = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
        else:
            # Get the max valid pixel
            max_class = np.max(predictions).astype(int) + 1
            # Encode predicts as a count of one-hot predictions for subsequent proceessing
            one_hot_predictions = np.zeros((X.shape[0], max_class))
            # Initialize uncertainty
            for pred in predictions:
                # TODO determine if some type of advanced indexing can be used here
                one_hot_predictions[
                    np.arange(X.shape[0]).astype(int), pred.astype(int)
                ] += 1
            mean_prediction = np.argmax(one_hot_predictions, axis=1)
            n_matching_predictions = one_hot_predictions[
                np.arange(X.shape[0]).astype(int), mean_prediction
            ]
            n_not_matching = self.n_ensamble_models - n_matching_predictions

            # Normalize it so the uncertainty is never more than 1
            uncertainty = n_not_matching / self.n_ensamble_models

        return_dict = {MEAN_KEY: mean_prediction, UNCERTAINTY_KEY: uncertainty}
        return return_dict
