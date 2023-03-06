from ipp_toolkit.predictors.uncertain_predictors import GaussianProcessRegression
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.data.masked_labeled_image import ImageNPMaskedLabeledImage
from ipp_toolkit.data.domain_data import (
    CoralLandsatRegressionData,
    AIIRAGreennessRegresssionData,
)
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
from ipp_toolkit.planners.classical_GP_planners import (
    MutualInformationPlanner,
    RecursiveGreedyPlanner,
)
import matplotlib.pyplot as plt

from pathlib import Path
from ipp_toolkit.config import DATA_FOLDER
import numpy as np
from ipp_toolkit.visualization.visualization import visualize_prediction


coral_kernel_kwargs = {
    "noise": 0.002,
    "rbf_lengthscale": np.array(
        [
            0.5731,
            42.3674,
            67.7918,
            9.3892,
            3.0964,
            40.4809,
            1.9705,
            2.4214,
            76.9222,
            81.0072,
        ]
    ),
    "output_scale": 0.020,
}
aiira_kernel_kwargs = {
    "noise": 1e-3,
    "rbf_lengthscale": [124.4611, 125.0317, 3.0636, 4.4240, 6.2944],
    "output_scale": 0.019228380173444748,
}

FIT = False

data = CoralLandsatRegressionData()
# data = AIIRAGreennessRegresssionData()

if FIT:
    kernel_model = GaussianProcessRegression(training_iters=10000, verbose=True)
    kernel_predictor = UncertainMaskedLabeledImagePredictor(
        data,
        uncertain_prediction_model=kernel_model,
        use_locs_for_prediction=True,
        classification_task=False,
    )
    random_planner = RandomSamplingMaskedPlanner(data)
    plan = random_planner.plan(n_samples=1000, verbose=True)
    values = data.sample_batch(plan)
    kernel_predictor.update_model(plan, values)
    breakpoint()

model = GaussianProcessRegression(training_iters=0, kernel_kwargs=coral_kernel_kwargs)
predictor = UncertainMaskedLabeledImagePredictor(
    data,
    uncertain_prediction_model=model,
    use_locs_for_prediction=True,
    classification_task=False,
)
predictor._preprocess_features()

recursive_planner = RecursiveGreedyPlanner(data)
mutual_info_plan = recursive_planner.plan(
    n_samples=10,
    GP_predictor=predictor,
    start_location=[int(x / 2) for x in data.image.shape[:2]],
    budget=2000,
    vis=True,
)
samples = data.sample_batch(mutual_info_plan)
predictor.update_model(mutual_info_plan, samples)
pred = predictor.predict_all()
errors = data.eval_prediction(pred)

print(f"Mean errors: {errors['mean_error'] / np.sum(data.mask)}")

visualize_prediction(data, prediction=pred)
