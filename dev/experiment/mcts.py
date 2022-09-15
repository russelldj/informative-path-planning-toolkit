from sacred import Experiment
from sacred.observers import MongoObserver
from ipp_toolkit.experiments.MCTS import MCTSExperiment

ex = Experiment("MCTS")
ex.observers.append(MongoObserver(url="localhost:27017", db_name="mmseg"))


@ex.config
def config():
    video_file = "vis/test.mp4"
    num_points = 100


@ex.automain
def main(
    video_file, num_points, _run,
):

    exp = MCTSExperiment(num_points=num_points)
    exp.run((0, 0), video_file, _run)
