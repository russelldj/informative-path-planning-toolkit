import numpy as np
from pathlib import Path

MEAN_KEY = "mean"
VARIANCE_KEY = "var"
MEAN_ERROR_KEY = "mean_error"
MEAN_VARIANCE_KEY = "mean_variance"
TOP_FRAC_MEAN_ERROR = "top_frac_mean_error"
TOP_FRAC_MEAN_VARIANCE = "top_frac_mean_variance"

GRID_RESOLUTION = 0.25
PLANNING_RESOLUTION = 0.5
VIS_RESOLUTION = 0.1

FLOAT_EPS = np.finfo(float).eps

DATA_FOLDER = Path(Path(__file__).parent, "../data")