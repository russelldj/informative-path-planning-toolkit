# Needed for env instantiation
from sacred import Experiment

from ipp_toolkit.utils.rl.rl import test_agents, train_agent

ex = Experiment("rl_train_test")


@ex.config
def config():
    # agent_types = [
    #    "random",
    #    "TD3",
    #    "MB",
    # ]  # Which agents to train or test on
    agent_types = [
        "random",
        "Perfect",
        "PPO",
        "DQN",
        "MB",
        "DA",
        "BC",
    ]  # Which agents to train or test on
    policy = "MlpPolicy"  # What policy to use, can also be CNN
    num_trials = 100  # How many test runs to run
    vis_dir = "vis/submission"  # Where to save visualization
    model_dir = "models_harry"  # Where to save and/or load models
    n_iters = 20  # How many planning iters to run
    safety_max = 100  # ?
    noise_sdev = 0  # map sensor noise
    noise_bias = 0  # map sensor bias
    world_size = (20, 20)  # The size of the testing world
    sensor_size = (1, 1)  # How many samples to take on a grid
    sensor_resolution = 1.0  # Diff between sensor samples on the grid
    obs_clip = 1.0  # ?
    obs_gp_mean_scale = 1.0  # ?
    obs_gp_std_scale = 1.0  # ?
    rew_top_frac_scale = 1.0  # What fraction of rewards to look at
    rew_diff_num_visited_scale = 0.0  # ?
    write_video = False  # Save out results video
    map_seed = None  # Random seed for the map
    action_space_discretization = 7  # Or an int specifying how many samples per axis
    observation_space_discretization = 7
    map_lower_offset = 0.5  # The lowest value in the map
    use_interpolation_model = (
        False  # Represent belief based on interpolation rather than a grid
    )
    plot = False  # ?
    move_on_grid = False
    # GP details

    log_dir = "logs"

    # training details
    num_par = 1
    # learning_rate can be set on the command line
    LR_DICT = {
        "DQN": 0.0001,
        "PPO": 0.0003,
        "DDPG": 0.001,
        "SAC": 0.0003,
        "random": None,
        "MB": 0.0005,
        "UCB": None,
        "BC": None,
        "DA": None,
        "Perfect": None,
        "TD3": 0.0005,
    }
    learning_rate = LR_DICT[agent_types[0]]
    n_steps = 2048
    total_timesteps = 300000
    verbose = 1
    save_freq = 1000

    train = False


@ex.automain
def main(
    agent_types,
    policy,
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
    observation_space_discretization,
    map_lower_offset,
    use_interpolation_model,
    move_on_grid,
    plot,
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    log_dir,
    train,
    _run,
):
    if train:
        train_agent(agent_type=agent_types[0], **locals())
    else:
        test_agents(**locals())
