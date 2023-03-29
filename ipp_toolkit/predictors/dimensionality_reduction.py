import torch
import torch.nn as nn
from torch.optim import Adam
from ipp_toolkit.config import TORCH_DEVICE
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
from tqdm import tqdm


"""
Taken from here
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
"""


class VAEEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        """Follows Table 5.1 in Bayesian models for science exploration

        Args:
            data_dim (_type_): _description_
            latent_dim (_type_): _description_
        """
        super(VAEEncoder, self).__init__()

        encoder1_dim = int(2 / 3 * (data_dim + latent_dim))
        encoder2_dim = int(1 / 3 * (data_dim + latent_dim))

        self.encoder_1 = nn.Linear(data_dim, encoder1_dim)
        self.encoder_2 = nn.Linear(encoder1_dim, encoder2_dim)
        self.mean = nn.Linear(encoder2_dim, latent_dim)
        self.log_var = nn.Linear(encoder2_dim, latent_dim)

        self.activation = nn.ReLU()
        self.training = True

    def forward(self, x):
        x = self.activation(self.encoder_1(x))
        x = self.activation(self.encoder_2(x))
        mean = self.mean(x)
        log_var = self.log_var(x)
        # encoder produces mean and log of variance
        # (i.e., parateters of simple tractable normal distribution "q"
        return mean, log_var


class VAEDecoder(nn.Module):
    def __init__(self, data_dim, latent_dim):
        super(VAEDecoder, self).__init__()
        decoder1_dim = int(1 / 3 * (data_dim + latent_dim))
        decoder2_dim = int(2 / 3 * (data_dim + latent_dim))

        self.decoder_1 = nn.Linear(latent_dim, decoder1_dim)
        self.decoder_2 = nn.Linear(decoder1_dim, decoder2_dim)
        self.output = nn.Linear(decoder2_dim, data_dim)

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.decoder_1(x))
        x = self.activation(self.decoder_2(x))
        output = self.sigmoid(self.output(x))
        return output


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(TORCH_DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


def VAE_loss(x, x_hat, mean, log_var, weighting_lambda=0.1):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    print(f"reproj: {reproduction_loss} weighted KLD: {weighting_lambda * KLD}")
    return reproduction_loss + weighting_lambda * KLD


class VAEDimensionalityReducer:
    def __init__(
        self,
        data_dim,
        latent_dim,
        batch_size=10000,
        lr=1e-3,
        epochs=200,
        chunk_size=10000,
        loss_function=VAE_loss,
        weighting_lambda=1,
        device=TORCH_DEVICE,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_function = loss_function
        self.device = device
        self.chunk_size = chunk_size
        self.weighting_lambda = weighting_lambda
        self.encoder = VAEEncoder(data_dim=data_dim, latent_dim=latent_dim)
        self.decoder = VAEDecoder(data_dim=data_dim, latent_dim=latent_dim)

        self.model = VAEModel(Encoder=self.encoder, Decoder=self.decoder).to(
            device=device
        )
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.scaler = None

    def scale_by_chunks(self, X, inverse=False):
        if inverse:
            for i in range(0, X.shape[0], self.chunk_size):
                X[i : i + self.chunk_size] = self.scaler.inverse_transform(
                    X[i : i + self.chunk_size]
                )
        else:
            for i in range(0, X.shape[0], self.chunk_size):
                X[i : i + self.chunk_size] = self.scaler.transform(
                    X[i : i + self.chunk_size]
                )

        return X

    def fit(self, X):
        logging.info("Start training VAE...")
        self.model.train()
        num_samples = X.shape[0]
        random_order = np.random.permutation(num_samples)
        self.scaler = StandardScaler()
        # Fit on at most 10000 points
        self.scaler.fit(X[random_order[: self.chunk_size]])
        # Transform
        logging.warning("About to scale data")
        # X = self.scale_by_chunks(X)
        print(f"mean {np.mean(X, axis=0)}")
        print(f"var {np.var(X, axis=0)}")
        logging.warning("Done scaling data")
        # Convert to tensor
        X = torch.Tensor(X).to(self.device)
        for epoch in tqdm(range(self.epochs)):
            overall_loss = 0
            for i in range(0, num_samples, self.batch_size):
                selected_inds = random_order[i : i + self.batch_size]
                x = X[selected_inds]

                self.optimizer.zero_grad()

                x_hat, mean, log_var = self.model(x)
                loss = self.loss_function(
                    x, x_hat, mean, log_var, weighting_lambda=self.weighting_lambda
                )

                overall_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            print(
                "\tEpoch",
                epoch + 1,
                "complete!",
                "\tAverage Loss: ",
                overall_loss / num_samples,
            )

    def transform(self, X):
        X = torch.Tensor(X).to(self.device)
        ys = []
        for i in tqdm(range(0, X.shape[0], self.batch_size)):
            x = X[i : i + self.batch_size]
            y = self.encoder(x)[0].detach().cpu().numpy()
            ys.append(y)
        ys = np.concatenate(ys, axis=0)
        print(f"mean {np.mean(ys, axis=0)}")
        print(f"var {np.var(ys, axis=0)}")
        return ys
