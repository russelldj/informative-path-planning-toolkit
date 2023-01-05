from ipp_toolkit.data.domain_data import (
    CoralLandsatClassificationData,
    ChesapeakeBayNaipLandcover7ClassificationData,
    YellowcatDroneClassificationData,
)
from ipp_toolkit.experiments.comparing_ipp_approaches import compare_random_vs_diversity

coral = CoralLandsatClassificationData()
chesapeake = ChesapeakeBayNaipLandcover7ClassificationData()
yellowcat = YellowcatDroneClassificationData()

compare_random_vs_diversity(
    data_manager=yellowcat, n_clusters=200, visit_n_locations=20, n_flights=5, vis=True
)

