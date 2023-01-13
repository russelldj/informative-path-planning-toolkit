from pathlib import Path

import numpy as np

MEAN_KEY = "mean"
UNCERTAINTY_KEY = "var"
MEAN_ERROR_KEY = "mean_error"
MEAN_UNCERTAINTY_KEY = "mean_variance"
TOP_FRAC_MEAN_ERROR = "top_frac_mean_error"
TOP_FRAC_MEAN_VARIANCE = "top_frac_mean_variance"
ERROR_IMAGE = "error_image"

TOP_FRAC = 0.4

GRID_RESOLUTION = 0.25
PLANNING_RESOLUTION = 0.5
VIS_RESOLUTION = 0.1

FLOAT_EPS = np.finfo(float).eps

DATA_FOLDER = Path(Path(__file__).parent, "../data")

# Diversity planner
TSP_ELAPSED_TIME = "tsp_elapsed_time"
CLUSTERING_ELAPSED_TIME = "clustering_elapsed_time"
OPTIMIZATION_ELAPSED_TIME = "optimization_elapsed_time"

OPTIMIZATION_ITERS = 10000

## Visualization
VIS_LEVEL = 0
PAUSE_DURATION = 0.1
VIS = False
FIG_SIZE = (5, 3.5)

# visualization levels, higher is more likely to get visualized
VIS_LEVEL = 2
VIS_LEVEL_0 = 0
VIS_LEVEL_1 = 1
VIS_LEVEL_2 = 2
VIS_LEVEL_3 = 3

# Experiments
N_FLIGHTS = 10
VISIT_N_LOCATIONS = 20
N_TRIALS = 10
