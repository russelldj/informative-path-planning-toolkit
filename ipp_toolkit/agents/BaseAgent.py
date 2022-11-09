import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


class BaseAgent:
    def __init__(self, action_space):
        self.model = None
        self.rl_alg_class = None

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        model_dir = cfg["model_dir"]
        log_dir = cfg["log_dir"]
        num_par = cfg["num_par"]
        save_freq = cfg["save_freq"]
        total_timesteps = cfg["total_timesteps"]

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        dummy_env = DummyVecEnv([lambda: env] * num_par)
        self._create_model(cfg, dummy_env)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=log_dir,
            name_prefix=self.model_name + "_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        self.model.learn(
            total_timesteps=int(total_timesteps),
            progress_bar=True,
            callback=checkpoint_callback,
        )

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self.model_name)
        self.model = self.rl_alg_class.load(model_path)

    def get_action(self, observation):

        if self.model is None:
            raise RuntimeError("Need to load model before getting action")

        return self.model.predict(observation, deterministic=True)

    def _create_model(self, cfg, env):
        """This needs to be defined in the subclass"""
        raise NotImplementedError()
