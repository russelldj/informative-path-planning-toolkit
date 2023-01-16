from .base_agent import BaseAgent
from ipp_toolkit.trainers.model_based.ff_model import FFModel
from ipp_toolkit.trainers.model_based.MPC_policy import MPCPolicy
from ipp_toolkit.trainers.model_based.replay_buffer import ReplayBuffer
from ipp_toolkit.trainers.model_based.utils import *

import torch


class MBAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MBAgent, self).__init__()

        self.env = env.unwrapped
        self.agent_params = agent_params
        self.ensamble_size = self.agent_params["ensamble_size"]

        self.dyn_models = []
        for i in range(self.ensamble_size):
            model = FFModel(
                self.agent_params["ac_dim"],
                self.agent_params["ob_dim"],
                self.agent_params["n_layers"],
                self.agent_params["size"],
                self.agent_params["learning_rate"],
            )
            self.dyn_models.append(model)

        self.actor = MPCPolicy(
            self.env,
            ac_dim=self.agent_params["ac_dim"],
            dyn_models=self.dyn_models,
            horizon=self.agent_params["mpc_horizon"],
            N=self.agent_params["mpc_num_action_sequences"],
            sample_strategy=self.agent_params["mpc_action_sampling_strategy"],
            cem_iterations=self.agent_params["cem_iterations"],
            cem_num_elites=self.agent_params["cem_num_elites"],
            cem_alpha=self.agent_params["cem_alpha"],
        )

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):

        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensamble is trained on a different random batch of size batch_size
        losses = []
        num_data = ob_no.shape[0]
        num_data_per_ens = int(num_data / self.ensamble_size)

        for i in range(self.ensamble_size):

            # select which datapoints to use for this model of the ensamble
            # you might find the num_data_per_env variable defined above useful
            start_index = i * num_data_per_ens
            end_index = (i + 1) * num_data_per_ens

            observations = ob_no[start_index:end_index]  # TODO(Q1)
            actions = ac_na[start_index:end_index]  # TODO(Q1)
            next_observations = next_ob_no[start_index:end_index]  # TODO(Q1)

            # use datapoints to update one of the dyn_models
            model = self.dyn_models[i]  # TODO(Q1)
            log = model.update(
                observations, actions, next_observations, self.data_statistics
            )
            loss = log["Training Loss"]
            losses.append(loss)

        avg_loss = np.mean(losses)
        return {
            "Training Loss": avg_loss,
        }

    def add_to_replay_buffer(self, paths, add_sl_noise=False):

        # add data to replay buffer
        self.replay_buffer.add_rollouts(paths, noised=add_sl_noise)

        # get updated mean/std of the data in our replay buffer
        self.data_statistics = {
            "obs_mean": np.mean(self.replay_buffer.obs, axis=0),
            "obs_std": np.std(self.replay_buffer.obs, axis=0),
            "acs_mean": np.mean(self.replay_buffer.acs, axis=0),
            "acs_std": np.std(self.replay_buffer.acs, axis=0),
            "delta_mean": np.mean(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0
            ),
            "delta_std": np.std(
                self.replay_buffer.next_obs - self.replay_buffer.obs, axis=0
            ),
        }

        # update the actor's data_statistics too, so actor.get_action can be calculated correctly
        self.actor.data_statistics = sensamble_statistics
nsamble
    def sample(self, batch_size):
        # NOTE: sampling batch_nsambleensamble_size,
        # so each model in our ensamble can get trained on batch_size data
        return self.replay_buffer.sample_random_data(batch_size * self.ensamble_size)

    def save(self, path):
        torch.save(self.actor, path)
