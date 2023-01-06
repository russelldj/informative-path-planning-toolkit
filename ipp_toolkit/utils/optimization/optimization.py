import numpy as np
import logging
from sklearn.preprocessing import StandardScaler


def topsis(pareto_values, normalize_axis=True):
    if normalize_axis:
        pareto_values = StandardScaler().fit_transform(pareto_values)
    if pareto_values.shape[1] == 1:
        logging.info(
            "Attempted to use topsis on 1-D problem, returning the first entry"
        )
        return pareto_values[0], 0
    neg_ideal = np.min(pareto_values, keepdims=True, axis=0)
    pos_ideal = np.max(pareto_values, keepdims=True, axis=0)

    if np.all(pos_ideal == pareto_values):
        logging.info(
            "Attempted to use topsis on problems with all identical values, returning the first entry"
        )
        return pareto_values[0], 0

    pos_dist = np.linalg.norm(pareto_values - pos_ideal)
    neg_dist = np.linalg.norm(pareto_values - neg_ideal)
    ratio = neg_dist / (neg_dist + pos_dist)
    if not np.all(np.isfinite(ratio)):
        breakpoint()
    best_index = np.argmax(ratio)
    selected_pareto_value = pareto_values[best_index]
    return selected_pareto_value, best_index


def quantile_solution(pareto_values, quantile=0.5):
    """
    Return
    """
    dim = pareto_values.shape[1]
    if dim == 1:
        logging.info(
            "Attempted to use quantile_solution on 1-D problem, returning the first entry"
        )
        return pareto_values[0], 0
    elif dim != 2:
        raise ValueError(
            "Cannot use quantile_solution on a problem with more than two dimensions"
        )

    sorted_inds = np.argsort(pareto_values[:, 0])
    chosen_ind = sorted_inds[int(quantile * pareto_values.shape[0])]
    selected_pareto_value = pareto_values[chosen_ind]
    return selected_pareto_value, chosen_ind

