from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
from ipp_toolkit.utils.rl.agents.RandomAgent import RandomAgent
from ipp_toolkit.utils.rl.agents.UCBAgent import UCBAgent
from stable_baselines3 import DDPG, PPO, DQN, SAC, HerReplayBuffer, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os


class BaseStableBaselinesAgent(BaseAgent):
    def __init__(self, env):
        pass

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

        model_path = os.path.join(model_dir, self.model_name)
        self.model.save(model_path)

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self.model_name)
        self.model = self.rl_alg_class.load(model_path)

    def get_action(self, observation, env=None):
        if self.model is None:
            raise RuntimeError("Need to load model before getting action")

        return self.model.predict(observation, deterministic=True)

    def _create_model(self, cfg, env):
        """This needs to be defined in the subclass"""
        raise NotImplementedError()


class SACAgent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "SAC"
        self.policy = None
        self.model_name = "sac_model"
        self.model = None
        self.rl_alg_class = SAC

    def _create_model(self, cfg, env, policy):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]

        self.model = self.rl_alg_class(self.policy, env, verbose=verbose)


class DDPGAgent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "DDPG"
        self.policy = policy
        self.model_name = "ddpg_model"
        self.model = None
        self.rl_alg_class = DDPG

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )

        self.model = self.rl_alg_class(
            self.policy,
            env,
            verbose=verbose,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [400, 300]},
        )


class DQNAgent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "DDPG"
        self.policy = None
        self.model_name = "dqn_model"
        self.model = None
        self.rl_alg_class = DQN

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]
        self.model = self.rl_alg_class(
            self.policy,
            env,
            learning_rate=learning_rate,
            verbose=verbose,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=128,
            gamma=0.99,
            train_freq=4,  # adjust this?,
            gradient_steps=1,  # adjust this?,
            target_update_interval=250,
            exploration_fraction=0.2,
            exploration_final_eps=0.1,
            tau=0.01,
        )


class PPOAgent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "PPO"
        self.policy = None
        self.model_name = "ppo_model"
        self.rl_alg_class = PPO
        self.model = None

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        n_steps = cfg["n_steps"]
        verbose = cfg["verbose"]

        self.model = self.rl_alg_class(
            self.policy,
            env,
            n_steps=n_steps,
            learning_rate=learning_rate,
            verbose=verbose,
            #gae_lambda=0.98,
            #gamma=0.99,
            ent_coef=0.00,
            use_sde=True,
            max_grad_norm=0.5,
            n_epochs=20,
            batch_size=256,
        )

class TD3Agent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "TD3"
        self.policy = None
        self.model_name = "td3_model"
        self.rl_alg_class = TD3
        self.model = None

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]

        self.model = self.rl_alg_class(
            self.policy,
            env,
            learning_rate=learning_rate,
            buffer_size=10000,
            learning_starts=10000, 
            batch_size=128, 
            tau=0.01, 
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            verbose=verbose,
        )

