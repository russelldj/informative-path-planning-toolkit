from ipp_toolkit.planners.masked_planner import (
    LawnmowerMaskedPlanner,
    CompassLinesPlanner,
    TrianglesLinesPlanner,
)
from ipp_toolkit.planners.RAPTORS_planner import RAPTORSPlanner
from ipp_toolkit.data.domain_data import (
    ChesapeakeBayNaipLandcover7ClassificationData,
    ChesapeakeBayNaipLandcover13ClassificationData,
)
from ipp_toolkit.experiments.comparing_ipp_approaches import (
    compare_across_datasets_and_models,
    visualize_across_datasets_and_models,
)
from ipp_toolkit.config import NAIP_URLS
from ipp_toolkit.predictors.convolutional_predictors import MOSAIKImagePredictor
from ipp_toolkit.data.masked_labeled_image import ImageNPMaskedLabeledImage
from ipp_toolkit.predictors.convenient_predictors import (
    PytorchKNNClassifierMaskedImagePredictor,
    KNNClassifierMaskedImagePredictor,
    GaussianProcessMaskedImagePredictor,
)
from torchgeo.datasets import Chesapeake13, Chesapeake7

from sacred import Experiment
from sacred.observers import MongoObserver
import numpy as np

ex = Experiment("thesis classification experiment")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


def RAPTORS_instantiation(
    data,
    predictor,
    initial_loc,
    expand_region_pixels,
    per_sample_weighting_power=0,
    lengthscale=0.5,
):
    kernel_kwargs = {
        "noise": 1e-4,
        "rbf_lengthscale": np.ones(data.image.shape[-1]) * lengthscale,
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
        samples_per_region=10,
        n_test_locs=int(4e4),
        n_candidate_locs=int(4e4),
        per_sample_weighting_power=per_sample_weighting_power,  # Whether or not to prioritize rare classes
    )
    return planner


def RAPTORS_rare_instantiation(
    data,
    predictor,
    initial_loc,
    expand_region_pixels,
):
    return RAPTORS_instantiation(
        data=data,
        predictor=predictor,
        initial_loc=initial_loc,
        expand_region_pixels=expand_region_pixels,
        per_sample_weighting_power=1,  # All this just to prioritize rare classes
    )


def create_chesapeak_mosaik(chip_size, naip_start_ind=None, naip_stop_ind=None):
    # Return the function, because that's what's expected
    def create_dataset():
        data = ChesapeakeBayNaipLandcover7ClassificationData(
            naip_urls=NAIP_URLS[naip_start_ind:naip_stop_ind],
            chip_size=chip_size,
            download=True,
        )
        predictor = MOSAIKImagePredictor(data, spatial_pooling_factor=1, n_features=512)
        compressed_spatial_features = predictor.predict_values()
        mosaiks_data = ImageNPMaskedLabeledImage(
            image=compressed_spatial_features,
            label=data.label,
            mask=data.mask,
            vis_image=data.image,
            cmap=data.cmap,
            vis_vmin=data.vis_vmin,
            vis_vmax=data.vis_vmax,
            n_classes=data.n_classes,
        )
        return mosaiks_data

    return create_dataset


@ex.config
def config():
    naip_start_ind = None
    naip_stop_ind = None
    chip_size = 2000
    datasets_dict = {
        "chesapeake": create_chesapeak_mosaik(
            chip_size=chip_size,
            naip_start_ind=naip_start_ind,
            naip_stop_ind=naip_stop_ind,
        )
    }
    predictors_dict = {
        "knn": (
            lambda data: PytorchKNNClassifierMaskedImagePredictor(data, n_neighbors=7)
        ),
    }
    n_flights_func = lambda data: 4
    n_samples_per_flight_func = lambda data: 10
    pathlength_per_flight_func = (
        lambda data: np.sqrt(data.image.shape[0] * data.image.shape[1]) * 0.5
    )
    expand_region_pixels = 15
    planners_instantiation_dict = {
        "coverage": lambda data, predictor, initial_loc, expand_region_pixels: TrianglesLinesPlanner(
            data, initial_loc=initial_loc, expand_region_pixels=expand_region_pixels
        ),
        "RAPTORS_rare": RAPTORS_rare_instantiation,
        "RAPTORS": RAPTORS_instantiation,
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
    compare_across_datasets_and_models(
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
