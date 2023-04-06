from ipp_toolkit.data.domain_data import ChesapeakeBayNaipLandcover7ClassificationData
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
from ipp_toolkit.config import DATA_FOLDER

ex = Experiment("mosaik")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


@ex.config
def config():
    n_iters = 100


def run_trial(_run):
    data = ChesapeakeBayNaipLandcover7ClassificationData()

    predictor = MOSAIKImagePredictor(data, spatial_pooling_factor=1, n_features=512)
    compressed_spatial_features = predictor.predict_values()
    data = ImageNPMaskedLabeledImage(compressed_spatial_features, label=data.label, downsample=4)

    current_loc = np.expand_dims([int(x / 2) for x in data.image.shape[:2]], axis=0)
    current_value = data.sample_batch(current_loc)

    # gp_kwargs = GP_KERNEL_PARAMS_WOUT_LOCS[data.get_dataset_name()]
    gp_kwargs = {
        "noise": 0.0001,
        "rbf_lengthscale": np.array([[1, 1, 1, 1, 1, 1]], dtype=np.float32),
        "output_scale": 1,
    }
    # gp_kwargs = GP_KERNEL_PARAMS_WOUT_LOCS[data.get_dataset_name()]

    gp = GaussianProcess(kernel_kwargs=gp_kwargs, training_iters=0)
    predictor = UncertainMaskedLabeledImagePredictor(data, gp)
    predictor.update_model(current_loc, current_value)

    planner = GreedyEntropyPlanner(
        data, predictor, current_loc=current_loc, budget_fraction_per_sample=0.25,
    )
    plan = planner.plan(20, vis=False, pathlength=600)
    values = data.sample_batch(plan)
    predictor.update_model(plan, values)
    files = Path(DATA_FOLDER, "entropy_reduction").glob("*")
    [_run.add_artifact(f) for f in files]


@ex.automain
def main(n_iters, _run):
    for _ in range(n_iters):
        run_trial(_run)
