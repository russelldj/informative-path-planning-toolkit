from stable_baselines3 import PPO


from ipp_toolkit.agents.BaseAgent import BaseAgent


class PPOLSTMAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "PPOLSTM"
        self.policy = "MlpLstmPolicy"
        self.model_name = "ppo_lstm_model"
        self.action_space = action_space
        self.model = None
        self.rl_alg_class = PPO

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
