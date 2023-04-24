from ipp_toolkit.planners.masked_planner import LawnmowerMaskedPlanner
from ipp_toolkit.planners.entropy_reduction_planners import GreedyEntropyPlanner
from ipp_toolkit.data.domain_data import ChesapeakeBayNaipLandcover7ClassificationData
from ipp_toolkit.experiments.comparing_ipp_approaches import (
    compare_across_datasets_and_models,
    visualize_across_datasets_and_models,
)
from ipp_toolkit.predictors.convolutional_predictors import MOSAIKImagePredictor
from ipp_toolkit.data.masked_labeled_image import ImageNPMaskedLabeledImage
from ipp_toolkit.predictors.convenient_predictors import (
    KNNClassifierMaskedImagePredictor,
    GaussianProcessMaskedImagePredictor,
)
from ipp_toolkit.config import GP_KERNEL_PARAMS_WOUT_LOCS

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np

ex = Experiment("aifsg_experiments")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


def create_semi_greedy(data, predictor, initial_loc):
    kernel_kwargs = {
        "noise": None,
        "rbf_lengthscale": np.ones(data.image.shape[-1]) * 2,
        "output_scale": 1,
    }
    predictor = GaussianProcessMaskedImagePredictor(
        masked_labeled_image=data,
        classification_task=False,
        gp_kwargs={"kernel_kwargs": kernel_kwargs, "training_iters": 0},
    )
    planner = GreedyEntropyPlanner(
        data,
        predictor=predictor,
        initial_loc=initial_loc,
        gp_fits_per_iteration=2,
        budget_fraction_per_sample=0.5,
    )
    return planner


def create_chesapeak_mosaik():
    data = ChesapeakeBayNaipLandcover7ClassificationData()
    predictor = MOSAIKImagePredictor(data, spatial_pooling_factor=1, n_features=512)
    compressed_spatial_features = predictor.predict_values()
    mosaiks_data = ImageNPMaskedLabeledImage(
        image=compressed_spatial_features,
        label=data.label,
        mask=data.mask,
        vis_image=np.clip(
            compressed_spatial_features / 6 + 0.5, 0, 1
        ),  # Get it mostly within the range 0,1
        cmap=data.cmap,
        vis_vmin=data.vis_vmin,
        vis_vmax=data.vis_vmax,
        n_classes=data.n_classes,
    )
    return mosaiks_data


@ex.config
def config():
    datasets_dict = {"chesapeake": create_chesapeak_mosaik}
    predictors_dict = {
        "knn": (lambda data: KNNClassifierMaskedImagePredictor(data)),
    }
    planners_instantiation_dict = {
        "semi_greedy": create_semi_greedy,
        "lawnmower": lambda data, predictor, initial_loc: LawnmowerMaskedPlanner(
            data, n_total_samples=6, initial_loc=initial_loc,
        ),
    }
    n_flights_func = lambda data: 2
    n_samples_per_flight_func = lambda data: 3
    pathlength_per_flight_func = lambda data: np.sqrt(
        data.image.shape[0] * data.image.shape[1]
    )
    initial_loc_func = lambda data: (np.array(data.image.shape[:2]) / 2).astype(int)

    n_datasets = 3
    n_trials_per_dataset = 1


@ex.automain
def main(
    datasets_dict,
    planners_instantiation_dict,
    predictors_dict,
    n_flights_func,
    n_samples_per_flight_func,
    pathlength_per_flight_func,
    initial_loc_func,
    n_datasets,
    n_trials_per_dataset,
    _run,
):
    results_dict = compare_across_datasets_and_models(
        datasets_dict=datasets_dict,
        planners_instantiation_dict=planners_instantiation_dict,
        predictors_dict=predictors_dict,
        n_flights_func=n_flights_func,
        n_samples_per_flight_func=n_samples_per_flight_func,
        pathlength_per_flight_func=pathlength_per_flight_func,
        initial_loc_func=initial_loc_func,
        n_datasets=n_datasets,
        n_trials_per_dataset=n_trials_per_dataset,
        _run=_run,
    )
    visualize_across_datasets_and_models(
        results_dict=results_dict, metrics=("mean_error",)
    )
