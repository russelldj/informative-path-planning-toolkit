import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


class BaseAgent:
    def __init__(self, env):
        self.model = None
        self.rl_alg_class = None

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        raise NotImplementedError()

    def load_model(self, model_dir):
        raise NotImplementedError()

    def get_action(self, observation, env=None):
        raise NotImplementedError()

    def _create_model(self, cfg, env):
        raise NotImplementedError()
