from informative_path_planning_toolkit.data.random_blob_2d import RandomGaussian2D
from informative_path_planning_toolkit.sensors.sensors import PointSensor

data = RandomGaussian2D(world_size=(20, 100))
sensor = PointSensor(data)
print(sensor.sample([15, 10]))
breakpoint()
