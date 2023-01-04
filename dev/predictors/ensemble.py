from ipp_toolkit.predictors.masked_image_predictor import (
    EnsembledMaskedLabeledImagePredictor,
    MaskedLabeledImagePredictor,
)
from ipp_toolkit.planners.masked_planner import RandomMaskedPlanner
from ipp_toolkit.data.MaskedLabeledImage import torchgeoMaskedDataManger
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from ipp_toolkit.config import MEAN_KEY, UNCERTAINTY_KEY

data = torchgeoMaskedDataManger()
classifier = MLPClassifier()
planner = RandomMaskedPlanner(data)
predictor = EnsembledMaskedLabeledImagePredictor(data, classifier, n_ensemble_models=10)

plan = planner.plan(200, vis=False)
values = data.sample_batch(plan)
predictor.update_model(plan, values)
prediction = predictor.predict_values_and_uncertainty()
label_pred = prediction[MEAN_KEY]
uncertainty_pred = prediction[UNCERTAINTY_KEY]
error = label_pred != data.label

plt.close()
f, axs = plt.subplots(2, 3)
axs[0, 0].imshow(data.image)
plt.colorbar(axs[0, 1].imshow(uncertainty_pred), ax=axs[0, 1])
plt.colorbar(axs[0, 2].imshow(error), ax=axs[0, 2])
plt.colorbar(axs[1, 0].imshow(data.label, vmin=0, vmax=9, cmap="tab10"), ax=axs[1, 0])
plt.colorbar(axs[1, 1].imshow(label_pred, vmin=0, vmax=9, cmap="tab10"), ax=axs[1, 1])
plt.show()
