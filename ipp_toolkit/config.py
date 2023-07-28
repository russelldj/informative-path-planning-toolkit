from pathlib import Path

import numpy as np
import torch

MEAN_KEY = "mean"
UNCERTAINTY_KEY = "var"
MEAN_ERROR_KEY = "mean_error"
BALANCED_CLASS_ERROR_KEY = "balanced_class_error"
MEAN_UNCERTAINTY_KEY = "mean_variance"
TOP_FRAC_MEAN_ERROR = "top_frac_mean_error"
TOP_FRAC_MEAN_VARIANCE = "top_frac_mean_variance"
N_TOP_FRAC = "n_top_frac"
ERROR_IMAGE = "error_image"

PLANNING_TIME_KEY = "planning_time"

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
GP_KERNEL_PARAMS_W_LOCS = {
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

GP_KERNEL_PARAMS_WOUT_LOCS = {
    "aiira": {
        "noise": 0.00010010577534558252,
        "rbf_lengthscale": np.array(
            [[4.3891683, 9.8652315, 7.2327757]], dtype=np.float32
        ),
        "output_scale": 0.16533610224723816,
    },
    "safeforest_gmaps": {
        "noise": 0.00010010693222284317,
        "rbf_lengthscale": np.array([[5.60179, 5.1897674, 9.661916]], dtype=np.float32),
        "output_scale": 0.4135790169239044,
    },
    "safeforest_ortho": {
        "noise": 0.00010011204722104594,
        "rbf_lengthscale": np.array(
            [[1.5761911, 0.41952115, 1.1059576, 35.16553]], dtype=np.float32
        ),
        "output_scale": 0.0262079406529665,
    },
    "coral_landsat_regression": {
        "noise": 0.002736506052315235,
        "rbf_lengthscale": np.array(
            [
                [
                    1.3000101,
                    1.3159883,
                    1.0959761,
                    1.8342868,
                    0.472861,
                    0.9711884,
                    2.9737582,
                    14.546533,
                ]
            ],
            dtype=np.float32,
        ),
        "output_scale": 0.012053688988089561,
    },
}

NAIP_URLS = (
    "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/m_3807511_ne_18_060_20181104.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/vt/2018/vt_060cm_2018/42072/m_4207220_ne_18_060_20181123.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/ma/2018/ma_060cm_2018/42072/m_4207258_nw_18_060_20181123.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/41076/m_4107640_nw_18_060_20191005.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/va/2018/va_060cm_2018/39078/m_3907854_se_17_060_20181219.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/40078/m_4007840_se_17_060_20191010.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/40078/m_4007845_sw_17_060_20190922.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/pa/2019/pa_60cm_2019/39077/m_3907709_sw_18_060_20190925.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/md/2018/md_060cm_2018/39077/m_3907744_nw_18_060_20181211.tif",
    "https://naipeuwest.blob.core.windows.net/naip/v002/va/2018/va_060cm_2018/39077/m_3907752_se_18_060_20181111.tif",
)
