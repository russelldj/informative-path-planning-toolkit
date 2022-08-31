class BaseWorldModel:
    def __init__(self, world_extent=None):
        self.world_extent = world_extent

    def add_observation(self, location, value):
        """Add a new observation from the sensor

        Arguments:
            location: ArrayLike
                Where the sample was taken
            value: ArrayLike
                What the sample was
        """
        raise NotImplementedError()


class GaussianProcessWorldModel(BaseWorldModel):
    def __init__():
        super().__init__()

    def add_observation(self, location, value):
        pass

