import numpy as np


def topsis(parateo_values):
    neg_ideal = np.min(parateo_values, keepdims=True, axis=0)
    pos_ideal = np.max(parateo_values, keepdims=True, axis=0)

    pos_dist = np.linalg.norm(parateo_values - pos_ideal)
    neg_dist = np.linalg.norm(parateo_values - neg_ideal)
    ratio = neg_dist / (neg_dist + pos_dist)
    best_index = np.argmax(ratio)
    selected_pareto_value = parateo_values[best_index]
    return selected_pareto_value, best_index
