import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        self.pca = None
        self.standard_scalar = StandardScaler()

    def _compute_features(self, vis=True):
        """ """
        mean = np.mean(self.masked_labeled_image.image, axis=(0, 1))
        std = np.std(self.masked_labeled_image.image, axis=(0, 1))
        mean, std = [np.expand_dims(x, (0, 1)) for x in (mean, std)]
        self.normalized_image = (self.masked_labeled_image.image - mean) / std

        self._fit()
        self.compressed_spatial_features = self._forward()
        return self.compressed_spatial_features

    def _fit(self):
        self._fit_weights()
        self._fit_compression()

    def _fit_weights(self):
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

    def _fit_compression(self, n_patches=100, patch_size=41):
        i_size, j_size = self.normalized_image.shape[:2]

        kernel_offset = patch_size // 2
        sample_locs = np.stack(
            (
                np.random.randint(kernel_offset, i_size - kernel_offset, n_patches),
                np.random.randint(kernel_offset, j_size - kernel_offset, n_patches),
            ),
            axis=0,
        ).T
        patches = [
            self.normalized_image[
                i - kernel_offset : i + kernel_offset + 1,
                j - kernel_offset : j + kernel_offset + 1,
            ]
            for i, j in sample_locs
        ]
        patch_features = []
        for patch in patches:
            patch_feature = self.inference(patch)
            patch_features.append(patch_feature)

        patch_features = np.concatenate(patch_features, axis=0)
        flat_features = np.reshape(patch_features, (-1, patch_features.shape[-1]))
        # TODO fit PCA
        self.pca = PCA(n_components=self.n_PCA_components)
        compressed_flat_features = self.pca.fit_transform(flat_features)
        # Fit standard scalar
        self.standard_scalar.fit(compressed_flat_features)

    def _forward(self, tile_size=1000):
        shape = self.normalized_image.shape
        kernel_offset = self.kernel_width // 2
        padded_normalized_image = np.pad(
            self.normalized_image,
            ((kernel_offset, kernel_offset), (kernel_offset, kernel_offset), (0, 0)),
        )
        output_image = np.zeros((shape[0], shape[1], self.n_PCA_components))

        # Compute with spatial chunks to avoid OOM
        # Properly respect boundary effects
        for i in range(0, shape[0], tile_size):
            for j in range(0, shape[1], tile_size):
                chip = padded_normalized_image[
                    i : i + tile_size + 2 * kernel_offset,
                    j : j + tile_size + 2 * kernel_offset,
                ]
                output_image[
                    i : i + tile_size, j : j + tile_size, :
                ] = self.compress_features(self.inference(chip))

        # Preprocess
        return output_image

    def inference(self, x):
        # , weights, biases, spatial_pooling_factor, n_features_at_once, device
        x = torch.Tensor(x / 255.0).to(self.device)
        x = torch.permute(x, (2, 0, 1))
        output_spatial_res = tuple(
            np.array(x.shape[-2:]) // self.spatial_pooling_factor
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
            if self.spatial_pooling_factor != 1:
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
            if self.spatial_pooling_factor != 1:
                x1b = F.adaptive_avg_pool2d(x1b, output_spatial_res)

            xs.extend([x1a.detach().cpu().numpy(), x1b.detach().cpu().numpy()])
        output = np.concatenate(xs, axis=0)
        output = np.transpose(output, (1, 2, 0))
        return output

    def compress_features(self, features):
        features_shape = features.shape
        flat_features = np.reshape(features, (-1, features.shape[-1]))
        compressed_features = self.pca.transform(flat_features)
        unitized_compressed_features = self.standard_scalar.transform(
            compressed_features
        )
        unitized_compressed_spatial_features = np.reshape(
            unitized_compressed_features, features_shape[:2] + (-1,)
        )

        return unitized_compressed_spatial_features

    def predict_values(self):
        if self.compressed_spatial_features is None:
            self.compressed_spatial_features = self._compute_features()

        return self.compressed_spatial_features
