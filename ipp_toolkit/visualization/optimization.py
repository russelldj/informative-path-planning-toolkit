import logging
import matplotlib.pyplot as plt
import numpy as np
from ipp_toolkit.config import PAUSE_DURATION
from ipp_toolkit.visualization.utils import show_or_save_plt


def visualize_pareto_front(
    pareto_solutions,
    selected_objectives,
    pause_duration=PAUSE_DURATION,
    remove_n_sampled_locations_obj: bool = False,
    labels=(
        "Number of sampled locations",
        "Average distance of unsampled locations",
        "Sum interestingness score",
    ),
    savepath="vis/pareto.png",
):
    """
    Args:
        show_n_sampled_locations: should you show the sampled locations objective

    """
    # close existing figures
    plt.close()
    plt.clf()

    pareto_objectives = np.array([s.objectives for s in pareto_solutions])

    # If the number of sample locations is constrained, this objective is not present
    if remove_n_sampled_locations_obj:
        # pareto_objectives = pareto_objectives[:, 1:]
        labels = labels[1:]

    dimensionality = pareto_objectives.shape[1]
    if dimensionality == 1:
        logging.info(f"Objective value for one-dim problem {pareto_objectives[0, 0]}")
        return
    if dimensionality == 2:
        # Normal 2d plot
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        ax = plt

    elif dimensionality == 3:
        # Set up 3d plot
        ax = plt.figure().add_subplot(projection="3d")
        # Plot z axis which would otherwise be missing
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

    else:
        raise ValueError(
            f"Cannot show problem with {dimensionality} dimensions, only 3 or fewer"
        )
    # Show all candidate objectives
    ax.scatter(*pareto_objectives.T, label="Candidate solutions")
    # Show the selected location
    ax.scatter(*selected_objectives, c="r", s=50, label="Chosen solution")

    plt.legend()

    show_or_save_plt(savepath=savepath, pause_duration=pause_duration)
