import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open("vis/sweep_metrics_203.pkl", "rb"))
biases = []
noises = []
planner_variances = []
final_mean_errors = []
top_frac_final_mean_errors = []


noise_biases = np.geomspace(0.001, 0.5, num=7)
planner_variance_scales = np.geomspace(0.001, 1000, num=7)
noise_sdevs = np.geomspace(0.001, 0.2, num=7)

results_volume = np.zeros((7, 7, 7))
top_frac_results_volume = np.zeros((7, 7, 7))

for exp in data:
    biases.append(exp["noise_bias"])
    noises.append(exp["noise_sdev"])
    planner_variances.append(exp["planner_variance_scale"])
    final_mean_errors.append(exp["metrics"]["mean_error"][-1])
    top_frac_final_mean_errors.append(exp["metrics"]["top_frac_mean_error"][-1])

    bias_ind = np.argwhere(exp["noise_bias"] == noise_biases)[0][0]
    noise_ind = np.argwhere(exp["noise_sdev"] == noise_sdevs)[0][0]
    planner_var_ind = np.argwhere(
        exp["planner_variance_scale"] == planner_variance_scales
    )[0][0]

    last_mean_error = exp["metrics"]["mean_error"][-1]
    last_top_frac_mean_error = exp["metrics"]["top_frac_mean_error"][-1]

    results_volume[bias_ind, noise_ind, planner_var_ind] = last_mean_error
    top_frac_results_volume[
        bias_ind, noise_ind, planner_var_ind
    ] = last_top_frac_mean_error

plt.imshow(results_volume[0])
plt.colorbar()
plt.yticks(range(7), noise_sdevs)
plt.xticks(range(7), planner_variance_scales)
plt.xlabel("Planner variance scale")
plt.ylabel("Noise sdev")
plt.show()


plt.scatter(biases, top_frac_final_mean_errors)
plt.xscale("log")
plt.show()
breakpoint()
