from ipp_toolkit.agents.BaseAgent import BaseAgent
from ipp_toolkit.agents.RandomAgent import RandomAgent
from stable_baselines3 import DDPG, PPO


class DDPGAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "DDPG"
        self.policy = "CnnPolicy"
        self.model_name = "dqn_model"
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


class PPOAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "PPO"
        self.policy = "CnnPolicy"
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
            learning_rate=learning_rate,
            n_steps=n_steps,
            verbose=verbose,
        )


agent_dict = {"random": RandomAgent, "PPO": PPOAgent, "DDPG": DDPGAgent}