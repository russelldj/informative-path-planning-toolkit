from ipp_toolkit.data.domain_data import (
    CoralLandsatClassificationData,
    ChesapeakeBayNaipLandcover7ClassificationData,
    YellowcatDroneClassificationData,
    SafeForestOrthoGreennessRegressionData,
    SafeForestGMapGreennessRegressionData,
    AIIRAGreennessRegresssionData,
)
from ipp_toolkit.experiments.comparing_ipp_approaches import compare_random_vs_diversity

# coral = CoralLandsatClassificationData()
# chesapeake = ChesapeakeBayNaipLandcover7ClassificationData()
# yellowcat = YellowcatDroneClassificationData()
# safeforest = SafeForestOrthoGreennessRegressionData()
safeforest_gmap = SafeForestGMapGreennessRegressionData()
aiira = AIIRAGreennessRegresssionData()

compare_random_vs_diversity(
    data_manager=aiira,
    n_candidate_locations_diversity=200,
    visit_n_locations=20,
    n_flights=10,
    n_trials=10,
    vis=True,
)

