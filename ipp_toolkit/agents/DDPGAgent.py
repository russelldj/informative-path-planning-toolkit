import os

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env


from ipp_toolkit.agents.BaseAgent import BaseAgent


class DDPGAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "DDPG"
        self.policy = "CnnPolicy"
        self.model_name = "dqn_model"
        self.action_space = action_space
        self.model = None

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        num_par = cfg["num_par"]
        learning_rate = cfg["learning_rate"]
        total_timesteps = cfg["total_timesteps"]
        verbose = cfg["verbose"]
        save_freq = cfg["save_freq"]
        model_dir = cfg["model_dir"]
        log_dir = cfg["log_dir"]

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        dummy_env = DummyVecEnv([lambda: env] * num_par)

        model = DDPG(
            self.policy, dummy_env, learning_rate=learning_rate, verbose=verbose,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=log_dir,
            name_prefix=self.model_name + "_checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps=int(total_timesteps),
            progress_bar=True,
            callback=checkpoint_callback,
        )

        model_path = os.path.join(model_dir, self.model_name)
        model.save(model_path)

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self.model_name)
        self.model = DDPG.load(model_path)

    def get_action(self, observation):
        if self.model is None:
            raise RuntimeError("Need to load model before getting action")

        return self.model.predict(observation, deterministic=True)
