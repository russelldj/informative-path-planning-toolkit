from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.planners.masked_planner import RandomSamplingMaskedPlanner
from ipp_toolkit.predictors.uncertain_predictors import GaussianProcess
from ipp_toolkit.predictors.masked_image_predictor import (
    UncertainMaskedLabeledImagePredictor,
)


def train_GP(
    data: MaskedLabeledImage,
    n_samples=10000,
    training_iters=2000,
    use_locs_for_prediction=True,
):
    gpr = GuassianProcess(training_iters=training_iters)
    predictor = UncertainMaskedLabeledImagePredictor(
        data,
        uncertain_prediction_model=gpr,
        use_locs_for_prediction=use_locs_for_prediction,
    )

    sampler = RandomSamplingMaskedPlanner(data)
    locs = sampler.plan(n_samples=n_samples)
    values = data.sample_batch(locs)
    predictor.update_model(locs, values)

    noise = predictor.prediction_model.model.likelihood.noise.item()
    rbf_lengthscale = (
        predictor.prediction_model.model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .numpy()
    )
    output_scale = (
        predictor.prediction_model.model.covar_module.outputscale.detach().cpu().item()
    )
    return_dict = {
        "noise": noise,
        "rbf_lengthscale": rbf_lengthscale,
        "output_scale": output_scale,
    }
    return return_dict
