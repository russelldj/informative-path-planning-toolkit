import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from ipp_toolkit.config import TORCH_DEVICE
from ipp_toolkit.predictors.masked_image_predictor import MaskedLabeledImagePredictor


class MOSAIKImagePredictor(MaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image,
        prediction_model=None,  # Unused
        use_locs_for_prediction=False,  # Unused
        classification_task: bool = True,  # Unused
        n_features: int = 512,
        kernel_width: int = 7,
        n_PCA_components=6,
        bias=0,
        spatial_pooling_factor=10,
        n_features_at_once=128,
        device=TORCH_DEVICE,
    ):
        """
        Args:
            kernel_width: must be odd
            n_features: note that some will be the negation
            masked_labeled_image,
            prediction_model: Unused
            use_locs_for_prediction: Unused
            classification_task: Unused
            n_features: The number of features corresponding to the random kernels
            kernel_width: width in pixels of the random kernel
            n_PCA_components: The number of components to compress to
            bias: bias during convolution,
            spatial_pooling_factor: How much to downsample the feature map
            n_features_at_once: How many features to convolve at a time to avoid OOM
            device: Which device to use for computation
        """
        assert kernel_width % 2 == 1

        self.device = device
        self.masked_labeled_image = masked_labeled_image
        self.kernel_width = kernel_width
        self.n_PCA_components = n_PCA_components
        self.n_features = n_features
        self.spatial_pooling_factor = spatial_pooling_factor
        self.n_features_at_once = n_features_at_once

        self.biases = (torch.zeros(n_features // 2, requires_grad=False) + bias).to(
            self.device
        )
        self.compressed_spatial_features = None

    def _compute_features(self, vis=True):
        """ """
        mean = np.mean(self.masked_labeled_image.image, axis=(0, 1))
        std = np.std(self.masked_labeled_image.image, axis=(0, 1))
        mean, std = [np.expand_dims(x, (0, 1)) for x in (mean, std)]
        self.normalized_image = (self.masked_labeled_image.image - mean) / std

        self._compute_kernels()
        self.features = self._forward()
        return self._compress_features()

    def _compute_kernels(self):
        i_size, j_size = self.normalized_image.shape[:2]

        kernel_offset = self.kernel_width // 2
        sample_locs = np.stack(
            (
                np.random.randint(
                    kernel_offset, i_size - kernel_offset, self.n_features // 2
                ),
                np.random.randint(
                    kernel_offset, j_size - kernel_offset, self.n_features // 2
                ),
            ),
            axis=0,
        ).T
        weights = [
            self.normalized_image[
                i - kernel_offset : i + kernel_offset + 1,
                j - kernel_offset : j + kernel_offset + 1,
            ]
            for i, j in sample_locs
        ]

        # normalization_factor = self.kernel_width ** 2 * self.normalized_image.shape[-1]
        normalization_factor = 1

        weights = [np.transpose(w, (2, 0, 1)) for w in weights]
        try:
            weights = np.stack(weights, axis=0) / normalization_factor
        except ValueError:
            shapes = np.array([w.shape for w in weights])
            print(np.unique(shapes, axis=0, return_inverse=True))
            breakpoint()
        self.weights = torch.Tensor(weights).to(self.device)

    def _forward(self):
        # Preprocess
        x = self.normalized_image
        x = torch.Tensor(x / 255.0).to(self.device)
        x = torch.permute(x, (2, 0, 1))

        output_spatial_res = tuple(
            np.array(self.normalized_image.shape[:2]) // self.spatial_pooling_factor
        )

        xs = []
        for i in range(0, self.weights.shape[0], self.n_features_at_once):
            # Positive version
            x1a = F.relu(
                F.conv2d(
                    x,
                    self.weights[i : i + self.n_features_at_once],
                    bias=self.biases[i : i + self.n_features_at_once],
                    stride=1,
                    padding=0,
                ),
                inplace=True,
            )
            x1a = F.adaptive_avg_pool2d(x1a, output_spatial_res)

            # Negative version
            x1b = F.relu(
                -F.conv2d(
                    x,
                    self.weights[i : i + self.n_features_at_once],
                    bias=self.biases[i : i + self.n_features_at_once],
                    stride=1,
                    padding=0,
                ),
                inplace=False,
            )
            x1b = F.adaptive_avg_pool2d(x1b, output_spatial_res)

            xs.extend([x1a, x1b])
        output = torch.cat(xs, dim=0)
        output = torch.permute(output, (1, 2, 0))
        return output.detach().cpu().numpy()

    def _compress_features(self):
        flat_features = np.reshape(self.features, (-1, self.features.shape[-1]))
        pca = PCA(n_components=self.n_PCA_components)
        compressed_features = pca.fit_transform(flat_features)
        compressed_spatial_features = np.reshape(
            compressed_features, self.features.shape[:2] + (-1,)
        )
        return compressed_spatial_features

    def predict_values(self):
        if self.compressed_spatial_features is None:
            self.compressed_spatial_features = self._compute_features()

        return self.compressed_spatial_features
