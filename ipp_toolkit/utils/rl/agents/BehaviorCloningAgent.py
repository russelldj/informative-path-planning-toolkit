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
from stable_baselines3.common.policies import ActorCriticPolicy
import pickle


class BehaviorCloningAgent(BaseAgent):
    def __init__(self, env):
        self.name = "BC"
        self.policy = None

    def get_expert_trajectories(self, env, n_trajectories):
        expert = PerfectAgent(env)
        all_obs = []
        all_act = []
        all_rewards = []

        for i in tqdm(range(n_trajectories)):
            obs = env.reset()
            done = False
            while not done:
                action, _ = expert.get_action(obs, env)
                all_obs.append(obs)
                all_act.append(action)
                obs, reward, done, _ = env.step(action)
                all_rewards.append(reward)
        n_transitions = len(all_act)
        all_obs = np.vstack(all_obs)
        all_act = np.vstack(all_act)
        all_infos = np.array([None] * n_transitions)
        transitions = TransitionsMinimal(all_obs, all_act, all_infos)
        print(f"Training trajectories had a mean reward of {np.mean(all_rewards)}")
        return transitions

    def train(
        self,
        env,
        cfg,
        rng=np.random.default_rng(0),
        min_episodes=2000,
        use_dagger=False,
    ):
        model_dir = cfg["model_dir"]
        savefile = Path(model_dir, "BC_model.zip")
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not use_dagger:
            print("Sampling UCB trajectories")
            transitions = self.get_expert_trajectories(env, n_trajectories=min_episodes)

            try:
                with open(Path(model_dir, "traj.pkl"), 'wb') as handle:
                    pickle.dump(transitions, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                pass

            #imitation.data.types.save(Path(model_dir, "traj.npy"), transitions)
            # venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
            # expert = UCBAgent(self.action_space)

            # get_action = lambda x: expert.get_action(x[0])
            # rollouts = rollout.rollout(
            #    get_action,
            #    venv,
            #    rollout.make_sample_until(
            #        min_timesteps=None, min_episodes=min_episodes
            #    ),
            #    rng=rng,
            # )
            # print("Training on sampled trajectories")
            # transitions = rollout.flatten_trajectories(rollouts)
            # transitions = TransitionsMinimal(
            #    transitions.obs, transitions.acts, transitions.infos
            # )

            self.bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                #policy=ActorCriticPolicy,
                batch_size = 256,
                rng=rng,
            )
            self.bc_trainer.train(n_epochs=100)
            self.bc_trainer.save_policy(savefile)
        else:
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                rng=rng,
            )
            with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
                print(tmpdir)
                dagger_trainer = SimpleDAggerTrainer(
                    venv=venv,
                    scratch_dir=tmpdir,
                    expert_policy=get_action,
                    bc_trainer=bc_trainer,
                    rng=rng,
                )
                dagger_trainer.train(2000)
                dagger_trainer.save_policy(savefile)

    def load_model(self, model_dir):
        self.policy = reconstruct_policy(Path(model_dir, "BC_model.zip"))

    def get_action(self, observation, env):
        observation = torch.Tensor(observation).cuda()
        observation = torch.unsqueeze(observation, dim=0)
        action, _, _ = self.policy(observation)
        action = action.squeeze().detach().cpu().numpy()
        return action, None
