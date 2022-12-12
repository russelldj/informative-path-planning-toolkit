from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env):
        self.name = "Random"
        self.action_space = env.action_space

    def train(self, env, cfg):
        print("Cannot train random agent.")

    def load_model(self, model_dir):
        pass

    def get_action(self, observation, env=None):
        return self.action_space.sample(), None
