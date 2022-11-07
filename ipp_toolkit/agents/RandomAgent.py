from ipp_toolkit.agents.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "Random"
        self.action_space = action_space

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        print("Cannot train random agent.")

    def load_model(self, model_dir):
        pass

    def get_action(self, observation):
        return self.action_space.sample(), None
