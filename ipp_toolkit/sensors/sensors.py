from multiprocessing.sharedctypes import Value
from numpy.typing import ArrayLike
import numpy as np

from ipp_toolkit.data.data import BaseData
from typing import Union

from ipp_toolkit.data.structured_2d import Uniform2D


class BaseSensor:
    def __init__(self, data):
        self.data = data

    def sample(self, location):
        """
        Args:
            location: Any
                All the parameters describing the location of the sensor
        Returns:
            observations: ArrayLike
                A vector of observations
        """
        raise NotImplementedError


class PointSensor(BaseSensor):
    def __init__(self, data):
        super().__init__(data)

    def sample(self, location):
        value = self.data.sample(location)
        return value


class GaussianNoisyPointSensor(PointSensor):
    """
    Samples have Gaussian noise added on top of the true value
    """

    def __init__(
        self,
        data: BaseData,
        noise_sdev: Union[BaseData, float, None] = None,
        noise_bias: Union[BaseData, float] = 0,
    ):
        """
        data:
            Scalar 2D field representing the data
        noise_bias
        """
        if isinstance(noise_sdev, (float, int)):
            noise_sdev = Uniform2D(value=noise_sdev)
        elif noise_sdev is None:
            noise_sdev = Uniform2D(value=1)

        if isinstance(noise_bias, (float, int)):
            noise_bias = Uniform2D(value=noise_bias)

        self.data_sampler = PointSensor(data)
        self.sdev_sampler = PointSensor(noise_sdev)
        self.bias_smapler = PointSensor(noise_bias)

    def sample(self, location, no_noise=False):
        raw_sample = self.data_sampler.sample(location)
        if no_noise:
            return raw_sample

        noise_sdev = self.sdev_sampler.sample(location)
        noise_bias = self.bias_smapler.sample(location)
        noisy_sample = raw_sample + noise_bias + np.random.normal(scale=noise_sdev)
        return noisy_sample
