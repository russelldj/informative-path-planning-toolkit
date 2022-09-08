from ipp_toolkit.experiments.point_sampler import point_sampler
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid
import pickle
import numpy as np
from tqdm import tqdm

ex = Experiment("bias_variance")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    noise_biases = np.geomspace(0.001, 0.5, num=7)
    planner_variance_scales = np.geomspace(0.001, 1000, num=7)
    noise_sdevs = np.geomspace(0.001, 0.2, num=7)
    video_file = "vis/test.mp4"
    error_file = "vis/error.png"
    n_iters = 200
    world_size = (20, 20)
    n_blobs = 40
    top_frac = 0.4
    pickle_file = "vis/sweep_metrics.pkl"


@ex.automain
def main(
    noise_biases,
    planner_variance_scales,
    video_file,
    error_file,
    n_iters,
    noise_sdevs,
    world_size,
    n_blobs,
    top_frac,
    pickle_file,
    _run,
):
    params = list(
        ParameterGrid(
            {
                "noise_bias": noise_biases,
                "planner_variance_scale": planner_variance_scales,
                "noise_sdev": noise_sdevs,
            }
        )
    )
    results = []
    for param in tqdm(params):
        metrics = point_sampler(
            video_file=video_file,
            error_file=error_file,
            n_iters=n_iters,
            world_size=world_size,
            n_blobs=n_blobs,
            top_frac=top_frac,
            vis=False,
            _run=_run,
            **param,
        )
        result = param.copy()
        result["metrics"] = metrics
        results.append(result)
        pickle.dump(results, open(pickle_file, "wb"))
    _run.add_artifact(pickle_file)
