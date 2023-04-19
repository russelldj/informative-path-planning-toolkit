from ipp_toolkit.data.domain_data import (
    ChesapeakeBayNaipLandcover7ClassificationData,
    CupriteASTERMineralClassificationData,
)
from ipp_toolkit.planners.entropy_reduction_planners import GreedyEntropyPlanner
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcess
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.predictors.convolutional_predictors import MOSAIKImagePredictor
from ipp_toolkit.config import GP_KERNEL_PARAMS_WOUT_LOCS
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
from ipp_toolkit.data.masked_labeled_image import ImageNPMaskedLabeledImage
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver
from pathlib import Path
from ipp_toolkit.config import VIS_FOLDER

ex = Experiment("mosaik")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


@ex.config
def config():
    n_iters = 100
    n_steps = 10
    kernel_scale = 0.5
    pathlength = 600


def run_trial(n_steps, kernel_scale, pathlength, _run):
    # data = ChesapeakeBayNaipLandcover7ClassificationData(download=True)

    # predictor = MOSAIKImagePredictor(data, spatial_pooling_factor=1, n_features=512)
    # compressed_spatial_features = predictor.predict_values()
    # data = ImageNPMaskedLabeledImage(
    #    compressed_spatial_features, label=data.label, downsample=4
    # )
    data = CupriteASTERMineralClassificationData(site="B")

    current_loc = np.expand_dims(
        [
            np.random.randint(0, data.image.shape[0]),
            np.random.randint(0, data.image.shape[1]),
        ],
        axis=0,
    )
    current_value = data.sample_batch(current_loc)

    # gp_kwargs = GP_KERNEL_PARAMS_WOUT_LOCS[data.get_dataset_name()]
    gp_kwargs = {
        "noise": None,
        "rbf_lengthscale": np.concatenate(
            (np.ones(9) * 4.57488270e-01, (np.ones(2) * 2000))
        )
        * 3,
        "output_scale": None,
    }
    # gp_kwargs = GP_KERNEL_PARAMS_WOUT_LOCS[data.get_dataset_name()]

    gp = GaussianProcess(kernel_kwargs=gp_kwargs, training_iters=0)
    predictor = UncertainMaskedLabeledImagePredictor(
        data, gp, use_locs_for_prediction=True
    )
    predictor.update_model(current_loc, current_value)

    planner = GreedyEntropyPlanner(
        data,
        predictor,
        current_loc=current_loc,
        budget_fraction_per_sample=0.25,
        _run=_run,
    )
    plan = planner.plan(n_steps, vis=True, pathlength=pathlength)
    values = data.sample_batch(plan)
    predictor.update_model(plan, values)


@ex.automain
def main(n_iters, n_steps, kernel_scale, pathlength, _run):
    for _ in range(n_iters):
        run_trial(n_steps, kernel_scale=kernel_scale, pathlength=pathlength, _run=_run)
