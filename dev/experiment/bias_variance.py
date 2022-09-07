from ipp_toolkit.experiments.point_sampler import point_sampler
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("bias_variance")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    error_file = "vis/error.png"
    n_iters = 200
    noise_sdev = 0.1
    noise_bias = 0
    world_size = (20, 20)
    planner_variance_scale = 100
    n_blobs = 40
    top_frac = 0.4


@ex.automain
def main(
    video_file,
    error_file,
    n_iters,
    noise_sdev,
    noise_bias,
    world_size,
    planner_variance_scale,
    n_blobs,
    top_frac,
    _run,
):
    point_sampler(
        video_file=video_file,
        error_file=error_file,
        n_iters=n_iters,
        noise_sdev=noise_sdev,
        noise_bias=noise_bias,
        world_size=world_size,
        planner_variance_scale=planner_variance_scale,
        n_blobs=n_blobs,
        top_frac=top_frac,
        _run=_run,
    )
