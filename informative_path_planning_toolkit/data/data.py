class BaseData:
    def __init__(self):
        pass

    def sample(self, location):
        """
        Args:
            location: Any
        Returns:
            A vector of observations
        """
        raise NotImplementedError()

    def show(self):
        """
        Visualize the data
        """
        raise NotImplementedError()
