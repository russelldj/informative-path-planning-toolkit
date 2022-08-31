import numpy as np
from informative_path_planning_toolkit.data.data import GridData2D, BaseData


class Uniform2D(BaseData):
    def __init__(self, value: float = 0):
        self.value = value

    def sample(self, location):
        return self.value
