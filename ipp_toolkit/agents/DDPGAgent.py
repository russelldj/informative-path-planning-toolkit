from stable_baselines3 import DDPG


from ipp_toolkit.agents.BaseAgent import BaseAgent


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
            self.policy, env, learning_rate=learning_rate, verbose=verbose,
        )
