import torch
import numpy as np
from ipp_toolkit.config import NN_TRAINING_EPOCHS


class PytorchPredictor:
    """
    Wraps a pytorch model to handle conversions to and from numpy
    """

    def __init__(self, model: torch.nn.Module, device: str, is_trained: bool = False):
        """
        model: The underlying model to be used
        device: the device to be used for training and test
        is_trained: can this model be used for predictions without being fit
        """

        self.model = model
        self.device = device
        self.is_trained = is_trained
        self.model.to(device=self.device)

    def _to_device(self, X):
        X_torch = torch.Tensor(X, device=self.device)
        return X_torch

    def _from_device(self, X: torch.Tensor):
        return X.detach().cpu().numpy()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_instantiation_function=lambda net_params: torch.optim.SGD(
            net_params, lr=0.001, momentum=0.9
        ),
        training_epochs: int = NN_TRAINING_EPOCHS,
    ):
        """
        Note, this resumes training from wherever the model was 

        X: features (TODO choose dim)
        y: targets (TODO choose dim)
        criterion: loss function, must take in (pred, target) and return the loss
        optimizer_instantiation_function: function: net params -> optimizer
            Takes in the model parameters and creates a torch.optim optimizer
        training_epochs: how many full passes through the dataset to run
        """
        # Create the optimizer, wrapping the model params
        optimizer = optimizer_instantiation_function(self.model.parameters())

        X = self._to_device(X)
        y = self._to_device(y)

        for epoch in range(training_epochs):
            # TODO think about wrapping in a data loader
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = self.model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # print statistics
            print(f"[{epoch + 1}] loss: {loss.item():.3f}")

    def predict(self, X):
        X = self._to_device(X)
        y_pred = self.model(X)
        y_pred = self._from_device(y_pred)
        return y_pred
