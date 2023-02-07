import torch
import numpy as np
from ipp_toolkit.config import NN_TRAINING_EPOCHS

from itertools import chain
from torch import nn


class MLPNetwork(nn.Module):
    def __init__(self, input_dim=9, output_dim=10, hidden_dims=(100, 100)):
        super(MLPNetwork, self).__init__()
        hidden_layers = [
            (nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU())
            for i in range(len(hidden_dims) - 1)
        ]
        hidden_layers = list(chain.from_iterable(hidden_layers))
        all_layers = (
            [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
            + hidden_layers
            + [nn.Linear(hidden_dims[-1], output_dim)]
        )
        self.linear_relu_stack = nn.Sequential(*all_layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class PytorchPredictor:
    """
    Wraps a pytorch model to handle conversions to and from numpy
    """

    def __init__(
        self,
        model: torch.nn.Module = MLPNetwork(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_trained: bool = False,
        criterion=torch.nn.MSELoss(),
        optimizer_instantiation_function=lambda net_params: torch.optim.SGD(
            net_params, lr=0.001, momentum=0.9
        ),
        training_epochs: int = NN_TRAINING_EPOCHS,
        classification_task: bool = False,
    ):
        """
        model: The underlying model to be used
        device: the device to be used for training and test
        is_trained: can this model be used for predictions without being fit
        criterion: loss function, must take in (pred, target) and return the loss
        optimizer_instantiation_function: function: net params -> optimizer
            Takes in the model parameters and creates a torch.optim optimizer
        training_epochs: how many full passes through the dataset to run
        classification_task: are labels integer class values
        """

        self.model = model
        self.device = device
        self.is_trained = is_trained
        self.criterion = criterion
        self.optimizer_instantiation_function = optimizer_instantiation_function
        self.training_epochs = training_epochs
        self.classification_task = classification_task
        self.model.to(device=self.device)

    def _to_device(self, X):
        X_torch = torch.Tensor(X).to(self.device)
        return X_torch

    def _from_device(self, X: torch.Tensor):
        return X.detach().cpu().numpy()

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: int = False,
    ):
        """
        Note, this resumes training from wherever the model was 

        X: features (TODO choose dim)
        y: targets (TODO choose dim)
        verbose: whether to print training statistics
        """
        # Create the optimizer, wrapping the model params
        optimizer = self.optimizer_instantiation_function(self.model.parameters())

        X = self._to_device(X)
        y = self._to_device(y)
        if self.classification_task:
            y = y.to(dtype=int)
        else:
            # TODO figure out how to deal with squeezing and unsqueezing
            y = torch.unsqueeze(y, 1)

        for epoch in range(self.training_epochs):
            # TODO think about wrapping in a data loader
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            if verbose:
                # print statistics
                print(f"[{epoch + 1}] loss: {loss.item():.3f}")

    def predict(self, X):
        X = self._to_device(X)
        y_pred = self.model(X)
        if self.classification_task:
            y_pred = torch.argmax(y_pred, dim=1)
        y_pred = self._from_device(y_pred)
        if not self.classification_task:
            # TODO figure out how to deal with squeezing and unsqueezing
            y_pred = y_pred.squeeze()
        return y_pred
