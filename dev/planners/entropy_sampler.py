from ipp_toolkit.data.domain_data import SafeForestGMapGreennessRegressionData
from ipp_toolkit.planners.entropy_reduction_planners import GreedyEntropyPlanner
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcess
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.config import GP_KERNEL_PARAMS_WOUT_LOCS
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
import numpy as np

data = SafeForestGMapGreennessRegressionData()

current_loc = np.expand_dims([int(x / 2) for x in data.image.shape[:2]], axis=0)
current_value = data.sample_batch(current_loc)

gp_kwargs = GP_KERNEL_PARAMS_WOUT_LOCS[data.get_dataset_name()]

gp = GaussianProcess(kernel_kwargs=gp_kwargs, training_iters=0)
predictor = UncertainMaskedLabeledImagePredictor(data, gp)
predictor.update_model(current_loc, current_value)

planner = GreedyEntropyPlanner(
    data, predictor, current_loc=current_loc, budget_fraction_per_sample=0.25,
)
plan = planner.plan(20, vis=True, pathlength=600)
values = data.sample_batch(plan)
predictor.update_model(plan, values)
