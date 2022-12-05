from sacred import Experiment

from ipp_toolkit.utils.rl.rl import train

ex = Experiment("test")


@ex.config
def config():
    agent_type = "SAC"
    model_dir = "models"
    log_dir = "logs"
    n_iters = 20
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
    map_seed = None
    action_space_discretization = None  # Or an int specifying how many samples per axis
    world_sample_resolution = 20 / (7 - 1e-6)
    # GP details
    # n_gp_fit_iters = 10
    # gp_lengthscale_prior = None
    # gp_lengthscale_var_prior = None

    # training details
    num_par = 1
    # learning_rate = 3e-4
    learning_rate = 1e-3
    n_steps = 2048
    total_timesteps = 300000
    verbose = 1
    save_freq = 1000


@ex.automain
def main(
    agent_type,
    model_dir,
    log_dir,
    n_iters,
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
    map_seed,
    action_space_discretization,
    world_sample_resolution,
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    # n_gp_fit_iters,
    # gp_lengthscale_prior,
    # gp_lengthscale_var_prior,
    _run,
):
    train(**locals())
