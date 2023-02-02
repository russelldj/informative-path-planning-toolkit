from ipp_toolkit.experiments.comparing_ipp_approaches import (
    sweep_planners_datasets_predictors,
)
from ipp_toolkit.planners.diversity_planner import BatchDiversityPlanner
from ipp_toolkit.data.domain_data import ALL_LABELED_DOMAIN_DATASETS

from ipp_toolkit.experiments.comparing_ipp_approaches import compare_planners
from ipp_toolkit.planners.diversity_planner import BatchDiversityPlanner
from ipp_toolkit.planners.masked_planner import (
    RandomSamplingMaskedPlanner,
    LawnmowerMaskedPlanner,
    RandomWalkMaskedPlanner,
)
from ipp_toolkit.data.domain_data import ALL_LABELED_DOMAIN_DATASETS
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcessRegression
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
    EnsambledMaskedLabeledImagePredictor,
)
from ipp_toolkit.predictors.intrestingness_computers import (
    UncertaintyInterestingessComputer,
)

from sklearn.neural_network import MLPRegressor, MLPClassifier

# Define the constants
N_CANDIDATE_LOCATIONS_DIVERSITY = 200
RANDOM_WALK_STEP = 50
VISIT_N_LOCATIONS = 20
VIS_PLAN = False
N_TRIALS = 10
N_FLIGHTS = 10

N_LAWNMOWER_SAMPLES = N_FLIGHTS * VISIT_N_LOCATIONS

# Name them for visualization later
planner_names = [
    "Random sampler",
    "Random walk planner",
    "Lawnmower planner",
    "Diversity planner",
]
# Add planner-specific keyword arguments
planner_kwargs = [
    {"vis": VIS_PLAN},
    {"vis": VIS_PLAN, "step_size": RANDOM_WALK_STEP},
    {"vis": VIS_PLAN},
    {"vis": VIS_PLAN},
]

# if use_ensemble:
#        # Create a gridded predictor
#        predictor = EnsambledMaskedLabeledImagePredictor(
#            data, model, classification_task=data.is_classification_dataset()
#        )
#    else:
#        predictor = UncertainMaskedLabeledImagePredictor(
#            data, model, classification_task=data.is_classification_dataset()
#        )

planner_instantiation_funcs = [
    lambda data: RandomSamplingMaskedPlanner(data),
    lambda data: RandomWalkMaskedPlanner(data),
    lambda data: LawnmowerMaskedPlanner(data, N_LAWNMOWER_SAMPLES),
    lambda data: BatchDiversityPlanner(
        data, n_candidate_locations=N_CANDIDATE_LOCATIONS_DIVERSITY
    ),
]
predictor_instantiation_funcs = [
    lambda data: EnsambledMaskedLabeledImagePredictor(
        data,
        MLPClassifier() if data.is_classification_dataset() else MLPRegressor(),
        classification_task=data.is_classification_dataset(),
    ),
    lambda data: UncertainMaskedLabeledImagePredictor(
        data,
        GaussianProcessRegression(),
        classification_task=data.is_classification_dataset(),
    ),
]


data_manager_classes = ALL_LABELED_DOMAIN_DATASETS.values()
interestingness_computer = UncertaintyInterestingessComputer()


def validation_function(data, predictor):
    if data.is_classification_dataset() and isinstance(
        predictor.prediction_model, GaussianProcessRegression
    ):
        return False
    return True


sweep_planners_datasets_predictors(
    data_manager_classes=data_manager_classes,
    planner_instantiation_funcs=planner_instantiation_funcs,
    predictor_instantiation_funcs=predictor_instantiation_funcs,
    planner_names=planner_names,
    each_planners_kwargs=planner_kwargs,
    interestingness_computer=interestingness_computer,
    validation_function=validation_function,
    n_trials=N_TRIALS,
    n_flights=N_FLIGHTS,
    visit_n_locations=VISIT_N_LOCATIONS,
)

