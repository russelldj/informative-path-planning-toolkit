from ipp_toolkit.agents.BaseAgent import BaseAgent
from ipp_toolkit.agents.RandomAgent import RandomAgent
from stable_baselines3 import DDPG, PPO, DQN


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
        self.model = DDPG(
            self.policy,
            env,
            learning_rate=learning_rate,
            verbose=verbose,
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


agent_dict = {
    "random": RandomAgent,
    "PPO": PPOAgent,
    "DDPG": DDPGAgent,
    "DQN": DQNAgent,
}
