from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcessRegression
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)


def train_GP(data: MaskedLabeledImage, n_samples=10000, training_iters=2000):
    gpr = GaussianProcessRegression(training_iters=training_iters)
    predictor = UncertainMaskedLabeledImagePredictor(
        data, uncertain_prediction_model=gpr
    )

    sampler = RandomSamplingMaskedPlanner(data)
    locs = sampler.plan(n_samples=n_samples)
    values = data.sample_batch(locs)
    predictor.update_model(locs, values)
    breakpoint()
