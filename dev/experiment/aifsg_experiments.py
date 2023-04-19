from ipp_toolkit.planners.masked_planner import LawnmowerMaskedPlanner
from ipp_toolkit.planners.entropy_reduction_planners import GreedyEntropyPlanner
from ipp_toolkit.data.domain_data import ChesapeakeBayNaipLandcover7ClassificationData
from ipp_toolkit.experiments.comparing_ipp_approaches import (
    compare_across_datasets_and_models,
)
from ipp_toolkit.predictors.convenient_predictors import (
    KNNClassifierMaskedImagePredictor,
    GaussianProcessMaskedImagePredictor,
)
from ipp_toolkit.config import GP_KERNEL_PARAMS_WOUT_LOCS

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("aifsg_experiments")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


def create_semi_greedy(data, predictor):
    kernel_kwargs = {
        "noise": None,
        "rbf_lengthscale": None,
        "output_scale": None,
    }
    predictor = GaussianProcessMaskedImagePredictor(
        masked_labeled_image=data, gp_kwargs={"kernel_kwargs": kernel_kwargs}
    )
    planner = GreedyEntropyPlanner(data, predictor=predictor)
    return planner


@ex.config
def config():
    datasets_dict = {"chesapeake": ChesapeakeBayNaipLandcover7ClassificationData}
    predictors_dict = {
        "knn": (lambda data: KNNClassifierMaskedImagePredictor(data)),
    }
    planners_dict = {
        "lawnmower": lambda data, predictor: LawnmowerMaskedPlanner(
            data, n_total_samples=100
        ),
        "semi_greedy": create_semi_greedy,
    }
    n_missions = lambda data: 4
    n_samples_per_mission = lambda data: 20
    path_budget_per_mission = (
        lambda data: (data.image.shape[0] * data.image.shape[1]) / 4
    )
    n_random_trials = 10


@ex.automain
def main(
    datasets_dict,
    planners_dict,
    predictors_dict,
    n_missions,
    n_samples_per_mission,
    path_budget_per_mission,
    n_random_trials,
    _run,
):
    compare_across_datasets_and_models(
        datasets_dict=datasets_dict,
        planners_dict=planners_dict,
        predictors_dict=predictors_dict,
        n_missions=n_missions,
        n_samples_per_mission=n_samples_per_mission,
        path_budget_per_mission=path_budget_per_mission,
        n_random_trials=n_random_trials,
        _run=_run,
    )
