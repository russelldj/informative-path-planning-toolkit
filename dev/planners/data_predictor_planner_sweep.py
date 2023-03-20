# %%
import torch

from ipp_toolkit.config import TORCH_DEVICE, GP_KERNEL_PARAMS
from ipp_toolkit.data.domain_data import ALL_LABELED_DOMAIN_DATASETS
from ipp_toolkit.experiments.comparing_ipp_approaches import (
    compare_across_datasets_and_models,
    compare_planners,
)
from ipp_toolkit.planners.diversity_planner import BatchDiversityPlanner
from ipp_toolkit.planners.masked_planner import (
    LawnmowerMaskedPlanner,
    RandomSamplingMaskedPlanner,
    RandomWalkMaskedPlanner,
)
from ipp_toolkit.planners.classical_GP_planners import MutualInformationPlanner
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.predictors.pytorch_predictor import MLPNetwork, PytorchPredictor
from ipp_toolkit.predictors.uncertain_predictors import (
    EnsamblePredictor,
    GaussianProcess,
)

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("data_predictor_planner_sweep")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="ipp"))


@ex.config
def config():
    dataset_names = ["coral"]

    n_candidate_locations_diversity = 50
    random_walk_frac = 8
    visit_n_locations = 3
    vis_plan = False
    n_trials = 20
    n_flights = 10
    vis_prediction_freq = 2

    n_lawnmower_samples = n_flights * visit_n_locations


@ex.automain
def main(
    dataset_names,
    n_candidate_locations_diversity,
    random_walk_frac,
    visit_n_locations,
    vis_plan,
    n_trials,
    n_flights,
    n_lawnmower_samples,
    vis_prediction_freq,
    _run,
):
    # Create your different planners
    planner_instantiation_funcs = [
        #    lambda data: BatchDiversityPlanner(
        #        data, n_candidate_locations=N_CANDIDATE_LOCATIONS_DIVERSITY
        #    ),
        lambda data: MutualInformationPlanner(
            data, GP_KERNEL_PARAMS[data.get_dataset_name()]
        ),
        lambda data: RandomSamplingMaskedPlanner(data),
        lambda data: RandomWalkMaskedPlanner(data),
        lambda data: LawnmowerMaskedPlanner(data, n_lawnmower_samples),
    ]
    # Add planner-specific keyword arguments
    planner_kwarg_funcs = [
        lambda data: {"vis": vis_plan},
        lambda data: {"vis": vis_plan},
        lambda data: {
            "vis": vis_plan,
            "step_size": data.image.shape[0] / random_walk_frac,
        },
        lambda data: {"vis": vis_plan},
    ]

    def create_pytorch_predictor(data, use_locs=False):
        # Create a prediction model
        input_dim = data.image.shape[-1]
        if use_locs:
            input_dim += 2
        output_dim = data.n_classes if data.is_classification_dataset() else 1
        model = PytorchPredictor(
            model=MLPNetwork(input_dim=input_dim, output_dim=output_dim),
            classification_task=data.is_classification_dataset(),
            criterion=torch.nn.CrossEntropyLoss()
            if data.is_classification_dataset()
            else torch.nn.MSELoss(),
            device=TORCH_DEVICE,
        )
        ensamble_model = EnsamblePredictor(
            model, classification_task=data.is_classification_dataset()
        )
        predictor = UncertainMaskedLabeledImagePredictor(
            data,
            uncertain_prediction_model=ensamble_model,
            classification_task=data.is_classification_dataset(),
            use_locs_for_prediction=use_locs,
        )
        return predictor

    predictor_instantiation_funcs = [
        lambda data: UncertainMaskedLabeledImagePredictor(
            data,
            GaussianProcess(
                device=TORCH_DEVICE,
                training_iters=0,
                kernel_kwargs=GP_KERNEL_PARAMS[data.get_dataset_name()],
            ),
            classification_task=data.is_classification_dataset(),
            use_locs_for_prediction=True,
        ),
        lambda data: create_pytorch_predictor(data, use_locs=True),
        create_pytorch_predictor,
    ]

    try:
        data_managers = [
            ALL_LABELED_DOMAIN_DATASETS[dataset_name]()
            for dataset_name in dataset_names
        ]
    except KeyError:
        print(f"Valid dataset names are {list(ALL_LABELED_DOMAIN_DATASETS.keys())}")
        exit()

    compare_across_datasets_and_models(
        data_managers=data_managers,
        predictor_instantiation_funcs=predictor_instantiation_funcs,
        planner_instantiation_funcs=planner_instantiation_funcs,
        planner_kwarg_funcs=planner_kwarg_funcs,
        visit_n_locations=visit_n_locations,
        n_trials=n_trials,
        n_flights=n_flights,
        vis_prediction_freq=vis_prediction_freq,
        _run=_run,
    )

