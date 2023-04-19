import numpy as np

from ipp_toolkit.config import MEAN_ERROR_KEY, VIS_LEVEL_3
from ipp_toolkit.data.masked_labeled_image import MaskedLabeledImage
from ipp_toolkit.planners.masked_planner import BaseGriddedPlanner
from ipp_toolkit.predictors.intrestingness_computers import BaseInterestingessComputer
from ipp_toolkit.predictors.masked_image_predictor import MaskedLabeledImagePredictor
from ipp_toolkit.utils.filenames import format_string_with_iter
from ipp_toolkit.visualization.image_data import show_or_save_plt
from ipp_toolkit.visualization.visualization import visualize_prediction
import sacred


def multi_flight_mission(
    planner: BaseGriddedPlanner,
    data_manager: MaskedLabeledImage,
    predictor: MaskedLabeledImagePredictor,
    interestingness_computer: BaseInterestingessComputer,
    samples_per_flight: int,
    n_flights: int,
    initial_interestingess_image: np.ndarray = None,
    planner_kwargs: dict = {},
    start_loc: np.ndarray = None,
    error_metric: str = MEAN_ERROR_KEY,
    planner_savepath_template: str = None,
    prediction_savepath_template: str = None,
    vis_predictions: bool = VIS_LEVEL_3,
    _run: sacred.Experiment = None,
):
    """
    This simulates running a multi-flight mission.

    Args:
        planner: The planner which decides which samples to select
        data_manager: The data manager which contains features and labels
        predictor: The prediction system which generates predictions of the label based on features
        interestingness_computer: Takes a prediction of the world and determines which regions are interesting
        samples_per_flight: How many locations to sample per flight
        n_flights: How many flights to perform
        initial_interestingness_image: An intial representation of what regions are interesting. Can be None
        planner_kwargs: The arguments to the planner
        start_loc: Where to start (i, j), or None
        error_metric: Which error metric to use
        planner_savepath_template:
            A template for where to save the file for the planner's visualization.
            It must be formatable with the .format(int) method, where the int
            represents the iteration. For example "vis/planner_iter_{:03d}.png"
        prediction_savepath_template: Where to save the error visualization. Same constraints as above
        vis_predictions: Should you show the model predictions each flight
        _run: Sacred run for logging

    Returns:
        A list of error values per flight
    """
    errors = []
    # Set the initial interestingness image
    interestingness_image = initial_interestingess_image

    executed_plan = np.zeros((0, 2))
    for flight_iter in range(n_flights):
        # Execute the plan
        new_plan = planner.plan(
            n_samples=samples_per_flight,
            interestingness_image=interestingness_image,
            savepath=format_string_with_iter(planner_savepath_template, flight_iter),
            current_loc=start_loc if flight_iter == 0 else None,
            **planner_kwargs,
        )
        # Sample values from the world
        values = data_manager.sample_batch(new_plan, assert_valid=True)
        # Update the model of the world based on sampled observations
        predictor.update_model(new_plan, values)
        # Generate predictions for the entire map
        pred_dict = predictor.predict_all()
        # Generate an interestingess image from the prediction
        interestingness_image = interestingness_computer.compute_interestingness(
            prediction_dict=pred_dict
        )
        # Compute the error of this prediction
        error_dict = data_manager.eval_prediction(pred_dict)
        # Append the error to the list of errors
        errors.append(error_dict[error_metric])
        # Visualization
        if vis_predictions:
            savepath = format_string_with_iter(
                prediction_savepath_template, flight_iter
            )
            visualize_prediction(
                data_manager,
                prediction=pred_dict,
                savepath=savepath,
                executed_plan=executed_plan,
                new_plan=new_plan,
            )
            if _run is not None:
                _run.add_artifact(savepath)
        executed_plan = np.concatenate((executed_plan, new_plan), axis=0)
    return errors
