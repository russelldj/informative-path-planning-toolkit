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
