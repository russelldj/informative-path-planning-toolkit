from ipp_toolkit.predictors.uncertain_predictors import GaussianProcessRegression
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)
from ipp_toolkit.data.MaskedLabeledImage import ImageNPMaskedLabeledImage
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
from ipp_toolkit.planners.classical_GP_planners import MutualInformationPlanner
import matplotlib.pyplot as plt

from pathlib import Path
from ipp_toolkit.config import DATA_FOLDER
import numpy as np
from ipp_toolkit.visualization.visualization import visualize_prediction


kernel_kwargs = {
    "noise": 0.002,
    "rbf_lengthscale": [
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
    ],
    "output_scale": 0.020,
}

image_file = Path(DATA_FOLDER, "maps/coral/X_wv.npy")
mask_file = Path(DATA_FOLDER, "maps/coral/valid_wv.npy")
label_file = Path(DATA_FOLDER, "maps/coral/Y.npy")

label = np.load(label_file)

data = ImageNPMaskedLabeledImage(
    image=image_file,
    mask=mask_file,
    label=label[..., 0],
    vis_vmin=0,
    vis_vmax=np.max(label[..., 0]),
)
data.label[data.label < 0] = 0

model = GaussianProcessRegression(training_iters=0, kernel_kwargs=kernel_kwargs)
predictor = UncertainMaskedLabeledImagePredictor(
    data,
    uncertain_prediction_model=model,
    use_locs_for_prediction=True,
    classification_task=False,
)
predictor._preprocess_features()

mutual_info_planner = MutualInformationPlanner(data)
mutual_info_plan = mutual_info_planner.plan(
    n_samples=200, GP_predictor=predictor, vis=True
)
samples = data.sample_batch(mutual_info_plan)
predictor.update_model(mutual_info_plan, samples)
pred = predictor.predict_all()
errors = predictor.get_errors(ord=1)

print(f"Mean errors: {errors['mean_error'] / np.sum(data.mask)}")

visualize_prediction(data, prediction=pred, predictor=predictor)
