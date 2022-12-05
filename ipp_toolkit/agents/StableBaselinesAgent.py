from ipp_toolkit.agents.BaseAgent import BaseAgent
from ipp_toolkit.agents.RandomAgent import RandomAgent
from stable_baselines3 import DDPG, PPO, DQN, SAC, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np


class SACAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "SAC"
        self.policy = "MlpPolicy"
        self.model_name = "sac_model"
        self.action_space = action_space
        self.model = None
        self.rl_alg_class = SAC

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]

        self.model = self.rl_alg_class(self.policy, env, verbose=verbose)


class DDPGAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "DDPG"
        self.policy = "MlpPolicy"
        self.model_name = "ddpg_model"
        self.action_space = action_space
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


class DQNAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "DDPG"
        self.policy = "MlpPolicy"
        self.model_name = "dqn_model"
        self.action_space = action_space
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


class PPOAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "PPO"
        self.policy = "MlpPolicy"
        self.model_name = "ppo_model"
        self.action_space = action_space
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
        )


class SACAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "SAC"
        self.policy = "MlpPolicy"
        self.model_name = "ppo_model"
        self.action_space = action_space
        self.rl_alg_class = SAC
        self.model = None

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]

        self.model = self.rl_alg_class(
            self.policy,
            env,
            learning_rate=learning_rate,
            verbose=verbose,
        )


agent_dict = {
    "random": RandomAgent,
    "PPO": PPOAgent,
    "DDPG": DDPGAgent,
    "DQN": DQNAgent,
    "SAC": SACAgent,
}
