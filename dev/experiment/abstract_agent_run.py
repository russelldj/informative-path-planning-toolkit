# Needed for env instantiation
from sacred import Experiment

from ipp_toolkit.utils.rl.rl import test

ex = Experiment("test")


@ex.config
def config():
    agent_types = ["random"]
    num_trials = 1
    vis_dir = "vis"
    model_dir = "models"
    n_iters = 20
    safety_max = 100
    noise_sdev = 0
    noise_bias = 0
    world_size = (20, 20)
    sensor_size = (1, 1)
    sensor_resolution = 1.0
    obs_clip = 1.0
    obs_gp_mean_scale = 1.0
    obs_gp_std_scale = 1.0
    rew_top_frac_scale = 1.0
    rew_diff_num_visited_scale = 0.0
    write_video = False
    map_seed = None  # Random seed for the map
    action_space_discretization = None  # Or an int specifying how many samples per axis
    plot = False
    world_sample_resolution = 20 / (7 - 1e-6)  # only used for continous env
    # GP details
    # n_gp_fit_iters = 1

    # gp_lengthscale_prior = 4
    # gp_lengthscale_var_prior = 0.1


@ex.automain
def main(
    agent_types,
    num_trials,
    vis_dir,
    model_dir,
    n_iters,
    safety_max,
    noise_sdev,
    noise_bias,
    world_size,
    sensor_size,
    sensor_resolution,
    obs_clip,
    obs_gp_mean_scale,
    obs_gp_std_scale,
    rew_top_frac_scale,
    rew_diff_num_visited_scale,
    write_video,
    map_seed,
    action_space_discretization,
    plot,
    world_sample_resolution,
    # n_gp_fit_iters,
    # gp_lengthscale_prior,
    # gp_lengthscale_var_prior,
    _run,
):
    test(**locals())
