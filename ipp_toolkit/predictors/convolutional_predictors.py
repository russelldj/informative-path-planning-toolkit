from ipp_toolkit.predictors.masked_image_predictor import MaskedLabeledImagePredictor

import torch
import torch.nn.functional as F
import numpy as np
from ipp_toolkit.config import TORCH_DEVICE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ipp_toolkit.visualization.utils import add_colorbar
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class MOSAIKImagePredictor(MaskedLabeledImagePredictor):
    def __init__(
        self,
        masked_labeled_image,
        prediction_model=None,  # Unused
        use_locs_for_prediction=False,  # Unused
        classification_task: bool = True,  # Unused
        n_features: int = 512,
        kernel_width: int = 7,
        bias=0,
        spatial_pooling_factor=10,
        device=TORCH_DEVICE,
    ):
        """
        Args:
            kernel_width: must be odd
            n_features: note that some will be the negation 
        """
        self.device = device

        self.masked_labeled_image = masked_labeled_image
        assert kernel_width % 2 == 1
        self.kernel_width = kernel_width
        self.n_features = n_features
        self.spatial_pooling_factor = spatial_pooling_factor

        self.biases = torch.zeros(n_features // 2, requires_grad=False) + bias
        self.biases = self.biases.to(self.device)

        self._compute_features()

    def _compute_features(self, vis=True):
        """
        """
        mean = np.mean(self.masked_labeled_image.image, axis=(0, 1))
        std = np.std(self.masked_labeled_image.image, axis=(0, 1))
        mean, std = [np.expand_dims(x, (0, 1)) for x in (mean, std)]
        self.normalized_image = (self.masked_labeled_image.image - mean) / std

        self._compute_kernels()
        self.features = self._forward()
        self._compress_features()

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

        # Positive version
        x1a = F.relu(
            F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0),
            inplace=True,
        )
        x1a = F.adaptive_avg_pool2d(x1a, output_spatial_res)

        # Negative version
        x1b = F.relu(
            -F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=0),
            inplace=False,
        )
        x1b = F.adaptive_avg_pool2d(x1b, output_spatial_res)
        output = torch.cat((x1a, x1b), dim=0)
        output = torch.permute(output, (1, 2, 0))
        return output.detach().cpu().numpy()

    def _compress_features(self, vis=True, num_tsne=10000):
        flat_features = np.reshape(self.features, (-1, self.features.shape[-1]))
        pca = PCA(n_components=6)
        compressed_features = pca.fit_transform(flat_features)
        compressed_spatial_features = np.reshape(
            compressed_features, self.features.shape[:2] + (-1,)
        )
        if vis:
            flat_labels = self.masked_labeled_image.label.flatten()

            X_train, X_test, y_train, y_test = train_test_split(
                compressed_features, flat_labels, test_size=0.9999, random_state=42
            )
            svc = LinearSVC()
            svc.fit(X_train, y_train)
            y_pred = svc.predict(X_test)

            conf = confusion_matrix(y_test, y_pred)
            add_colorbar(plt.imshow(conf))
            plt.show()

            print("starting tsne visualization")
            tsne = TSNE()
            sampled_points = np.random.choice(compressed_features.shape[0], num_tsne)

            tsne_embedded_compressed_features = tsne.fit_transform(
                compressed_features[sampled_points, :]
            )
            plt.scatter(
                tsne_embedded_compressed_features[:, 0],
                tsne_embedded_compressed_features[:, 1],
                c=flat_labels[sampled_points],
                vmin=-0.5,
                vmax=9.5,
                cmap="tab10",
            )
            plt.show()

            for i in range(0, compressed_spatial_features.shape[-1]):
                f, ax = plt.subplots(1, 2)
                chip = compressed_spatial_features[..., i : i + 1]
                print(np.mean(chip, (0, 1)))
                ax[0].imshow(self.masked_labeled_image.image[..., :3])
                add_colorbar(ax[1].imshow(chip))
                plt.show()
