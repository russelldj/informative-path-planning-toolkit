
class BaseAgent:
    def __init__(self, action_space):
        pass

    def get_name(self):
        raise NotImplementedError()

    def train(self, env, cfg):
        raise NotImplementedError()

    def load_model(self, model_dir):
        raise NotImplementedError()

    def get_action(self, observation):
        raise NotImplementedError()