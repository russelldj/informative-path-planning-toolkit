class BaseSensor:
    def sample(self, location):
        """
        Args:
            location: Any
                All the parameters describing the location of the sensor
        Returns:
            observations: ArrayLike
                A vector of observations
        """
