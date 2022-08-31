from ipp_toolkit.data.random_2d import RandomGaussian2D
from ipp_toolkit.sensors.sensors import (
    PointSensor,
    GaussianNoisyPointSensor,
)

data = RandomGaussian2D(world_size=(20, 100))
sensor = GaussianNoisyPointSensor(data)
[print(sensor.sample([15, 10])) for _ in range(10)]
breakpoint()
