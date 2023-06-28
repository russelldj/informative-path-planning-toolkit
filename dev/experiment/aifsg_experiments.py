from ipp_toolkit.planners.masked_planner import (
    LawnmowerMaskedPlanner,
    CompassLinesPlanner,
    TrianglesLinesPlanner,
)
from ipp_toolkit.planners.RAPTORS_planner import RAPTORSPlanner
from ipp_toolkit.data.domain_data import ChesapeakeBayNaipLandcover7ClassificationData
from ipp_toolkit.experiments.comparing_ipp_approaches import (
    compare_across_datasets_and_models,
    visualize_across_datasets_and_models,
)
from ipp_toolkit.config import (
    BALANCED_CLASS_ERROR_KEY,
    MEAN_ERROR_KEY,
    PLANNING_TIME_KEY,
)
from ipp_toolkit.predictors.convolutional_predictors import MOSAIKImagePredictor
from ipp_toolkit.data.masked_labeled_image import ImageNPMaskedLabeledImage
from ipp_toolkit.predictors.convenient_predictors import (
    KNNClassifierMaskedImagePredictor,
    GaussianProcessMaskedImagePredictor,
)
from torchgeo.datasets import Chesapeake13, Chesapeake7

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np

ex = Experiment("aifsg_experiments")
# ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


def semi_greedy_instantiation(data, predictor, initial_loc, expand_region_pixels):
    kernel_kwargs = {
        "noise": 1e-4,
        "rbf_lengthscale": np.ones(data.image.shape[-1]) * 0.5,
        "output_scale": 1,
    }
    predictor = GaussianProcessMaskedImagePredictor(
        masked_labeled_image=data,
        classification_task=False,
        gp_kwargs={"kernel_kwargs": kernel_kwargs, "training_iters": 0},
    )
    planner = RAPTORSPlanner(
        data,
        predictor=predictor,
        initial_loc=initial_loc,
        expand_region_pixels=expand_region_pixels,
        gp_fits_per_iteration=20,
        budget_fraction_per_sample=0.5,
        samples_per_region=25,
        n_test_locs=int(1.6e5),
        n_candidate_locs=2000,
    )
    return planner


def create_chesapeak_mosaik():
    data = ChesapeakeBayNaipLandcover7ClassificationData(
        chip_size=400,
        chesapeake_dataset=Chesapeake7,
        n_classes=7,
        cmap="tab10",
        vis_vmin=-0.5,
        vis_vmax=9.5,
        download=True,
    )
    predictor = MOSAIKImagePredictor(data, spatial_pooling_factor=1, n_features=512)
    compressed_spatial_features = predictor.predict_values()
    mosaiks_data = ImageNPMaskedLabeledImage(
        image=compressed_spatial_features,
        label=data.label,
        mask=data.mask,
        vis_image=data.image,
        # np.clip(
        #    compressed_spatial_features / 6 + 0.5, 0, 1
        # ),  # Get it mostly within the range 0,1
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
        "knn": (lambda data: KNNClassifierMaskedImagePredictor(data, n_neighbors=7)),
    }
    n_flights_func = lambda data: 4
    n_samples_per_flight_func = lambda data: 10
    pathlength_per_flight_func = (
        lambda data: np.sqrt(data.image.shape[0] * data.image.shape[1]) * 1
    )
    expand_region_pixels = 2
    planners_instantiation_dict = {
        # "compass_lines": lambda data, predictor, initial_loc: CompassLinesPlanner(
        #    data, initial_loc=initial_loc
        # ),
        "GSB-IPP": semi_greedy_instantiation,
        "triangles_lines": lambda data, predictor, initial_loc, expand_region_pixels: TrianglesLinesPlanner(
            data, initial_loc=initial_loc, expand_region_pixels=expand_region_pixels
        ),
        # "lawnmower": lambda data, predictor, initial_loc: LawnmowerMaskedPlanner(
        #    data, n_total_samples=40, initial_loc=initial_loc,
        # ),
    }
    initial_loc_func = lambda data: (np.array(data.image.shape[:2]) / 2).astype(int)

    n_datasets = 10
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
    expand_region_pixels,
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
        expand_region_pixels=expand_region_pixels,
        _run=_run,
    )

    visualize_across_datasets_and_models(
        results_dict=results_dict,
        metrics=(MEAN_ERROR_KEY, BALANCED_CLASS_ERROR_KEY, PLANNING_TIME_KEY),
        _run=_run,
    )
