from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
from ipp_toolkit.utils.rl.agents.PerfectAgent import PerfectAgent
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
from imitation.algorithms.bc import reconstruct_policy
import torch
import tempfile
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.types import TransitionsMinimal
from tqdm import tqdm
import imitation
#from stable_baselines3.common.policies import ActorCriticPolicy


class DaggerAgent(BaseAgent):
    def __init__(self, env):
        self.name = "DA"
        self.policy = None
        self.model_name = "da_model"

    def train(
        self,
        env,
        cfg,
    ):
        model_dir = cfg["model_dir"]
        savefile = Path(model_dir, self.model_name + ".zip")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        rng = np.random.default_rng(0)
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
            batch_size=256,
        )

        expert = PerfectAgent(env)
        #TODO
        #not really sure if wrapper is needed but it worked so keeping it
        #venv = DummyVecEnv([lambda: env])
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=venv,
                scratch_dir=tmpdir,
                expert_policy=expert,
                rng=rng,
                bc_trainer=bc_trainer,
            )

            dagger_trainer.train(10000, 
                                 rollout_round_min_episodes=100, 
                                 bc_train_kwargs={"n_epochs": 100},
                                 #policy=ActorCriticPolicy,
                                 )

            dagger_trainer.save_policy(savefile)

    def load_model(self, model_dir):
        self.policy = reconstruct_policy(Path(model_dir, self.model_name + ".zip"))

    def get_action(self, observation, env):
        observation = torch.Tensor(observation).cuda()
        observation = torch.unsqueeze(observation, dim=0)
        action, _, _ = self.policy(observation)
        action = action.squeeze().detach().cpu().numpy()
        return action, None
