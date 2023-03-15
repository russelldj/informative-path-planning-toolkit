from pathlib import Path

import numpy as np
import torch

MEAN_KEY = "mean"
UNCERTAINTY_KEY = "var"
MEAN_ERROR_KEY = "mean_error"
MEAN_UNCERTAINTY_KEY = "mean_variance"
TOP_FRAC_MEAN_ERROR = "top_frac_mean_error"
TOP_FRAC_MEAN_VARIANCE = "top_frac_mean_variance"
N_TOP_FRAC = "n_top_frac"
ERROR_IMAGE = "error_image"

TOP_FRAC = 0.4

GRID_RESOLUTION = 0.25
PLANNING_RESOLUTION = 0.5
VIS_RESOLUTION = 0.1

FLOAT_EPS = np.finfo(float).eps

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
VIS_FOLDER = Path(Path(__file__).parent, "..", "vis")

# Diversity planner
TSP_ELAPSED_TIME = "tsp_elapsed_time"
CLUSTERING_ELAPSED_TIME = "clustering_elapsed_time"
OPTIMIZATION_ELAPSED_TIME = "optimization_elapsed_time"

OPTIMIZATION_ITERS = 10000

## Visualization
PAUSE_DURATION = 0.1
VIS = False
SMALL_FIG_SIZE = (5, 3.5)
MED_FIG_SIZE = (10, 7)
BIG_FIG_SIZE = (20, 14)

# visualization levels, lower is more likely to get visualized
# Visualized if vis_level > (vis_level_indicator)
VIS_LEVEL = 3
VIS_LEVEL_0 = VIS_LEVEL > 0
VIS_LEVEL_1 = VIS_LEVEL > 1
VIS_LEVEL_2 = VIS_LEVEL > 2
VIS_LEVEL_3 = VIS_LEVEL > 3

# Experiments
N_FLIGHTS = 10
VISIT_N_LOCATIONS = 20
N_TRIALS = 10

# Neural networks
NN_TRAINING_EPOCHS = 1000
TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# GP params
GP_KERNEL_PARAMS = {
    "aiira": {
        "noise": 0.00010010461846832186,
        "rbf_lengthscale": [26.38989, 26.749512, 2.972929, 4.815648, 4.9283924],
        "output_scale": 0.026678644120693207,
    },
    "safeforest_gmaps": {
        "noise": 0.00010010476398747414,
        "rbf_lengthscale": [27.399124, 27.364677, 3.070873, 2.6928146, 4.1541324],
        "output_scale": 0.023601967841386795,
    },
    "safeforest_ortho": {
        "noise": 0.00010010667756432667,
        "rbf_lengthscale": [
            39.23713,
            39.630306,
            1.0986562,
            0.31501293,
            0.7527026,
            64.4776,
        ],
        "output_scale": 0.004418130032718182,
    },
    "coral_landsat_regression": {
        "noise": 0.0002531635109335184,
        "rbf_lengthscale": [
            0.11592584,
            0.12620367,
            11.426642,
            1.9107863,
            0.90792924,
            0.4770281,
            3.3797588,
            0.8065388,
            0.8833334,
            10.46065,
        ],
        "output_scale": 0.012055214494466782,
    },
}
