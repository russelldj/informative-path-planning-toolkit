from sacred import Experiment
from sacred.observers import MongoObserver
from ipp_toolkit.experiments.sensor_error import SensorErrorExperiments

ex = Experiment("sensor_error")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    error_video_file = "vis/error.mp4"
    n_points = 10
    world_size = (10, 10)
    overlap_inds = 2


@ex.automain
def main(
    video_file,
    error_video_file,
    n_points,
    world_size,
    overlap_inds,
    _run,
):

    exp = SensorErrorExperiments(
        world_size=world_size, n_points=n_points, overlap_inds=overlap_inds
    )
    exp.run(video_file=video_file, error_video_file=error_video_file, _run=_run)
