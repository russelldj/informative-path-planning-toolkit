import numpy as np


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

    def sample_belief(self, location):
        """Samples a single belief from the model

        Arguments:
            location: where to sample the belief
        """
        raise NotImplementedError()

    def sample_belief_array(self, locations):
        """Samples n beliefs from different locations from the model

        Arguments:
            locations: Where to sample the beliefs. Each row should represent 
            a location
        """
        raise NotImplementedError()
