from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
from ipp_toolkit.utils.rl.agents.UCBAgent import UCBAgent
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from imitation.scripts.train_preference_comparisons import save_model
from pathlib import Path
import os


class BehaviorCloningAgent(BaseAgent):
    def __init__(self, action_space):
        self.name = "BC"
        self.action_space = action_space
        self.policy = None

    def train(self, env, cfg, rng=np.random.default_rng(0), min_episodes=50):

        expert = UCBAgent(self.action_space)

        get_action = lambda x: expert.get_action(x[0])
        print("Sampling UCB trajectories")
        rollouts = rollout.rollout(
            get_action,
            DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
            rng=rng,
        )
        print("Training on sampled trajectories")
        transitions = rollout.flatten_trajectories(rollouts)
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
        )

        model_dir = cfg["model_dir"]
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        bc_trainer.policy.save(Path(model_dir, "BC_model.zip"))

    def load_model(self, model_dir):
        torch.

    def get_action(self, observation, env=None):
        breakpoint()
        return self.action_space.sample(), None
