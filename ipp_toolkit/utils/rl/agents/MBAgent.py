from ipp_toolkit.utils.rl.agents.BaseAgent import BaseAgent
from ipp_toolkit.trainers.model_based.rl_trainer import RL_Trainer
from ipp_toolkit.trainers.model_based.mb_agent import MBAgent as rl_MBAgent
from ipp_toolkit.trainers.model_based import pytorch_util as ptu

import os
import torch


# TODO sampling action sequences does not consider environment ending
class MBAgent(BaseAgent):
    def __init__(self, env):
        self.name = "ModelBased"
        self.model_name = "mb_model"

        ptu.init_gpu(use_gpu=True, gpu_id=0)

    def train(self, env, cfg):
        # TODO it creates it's own env so fix that
        # TODO change n_iter to toal timesteps?
        # TODO _create_model env param is info_dict
        self._create_model(cfg, env.info_dict)
        self.rl_trainer.run_training_loop(
            self.params["n_iter"],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self.model_name + ".pt")
        self.actor = torch.load(model_path)

    def get_action(self, observation, env):
        return self.actor.get_best_action(observation, env)[0][0], None

    def _create_model(self, cfg, info_dict):
        ###hardcoding for now
        cfg["ensamble_size"] = 1
        cfg["n_layers"] = 3
        cfg["size"] = 256
        cfg["num_agent_train_steps_per_iter"] = 1000  # 20
        cfg["mpc_horizon"] = 5
        cfg["mpc_num_action_sequences"] = 5000
        cfg["mpc_action_sampling_strategy"] = "random"
        cfg["cem_iterations"] = 4
        cfg["cem_num_elites"] = 5
        cfg["cem_alpha"] = 1.0
        cfg["seed"] = 1
        # cfg['no_gpu'] = False
        # cfg['which_gpu'] = 0
        cfg["n_iter"] = 100  # 5
        cfg["scalar_log_freq"] = 1
        cfg["batch_size"] = 10000
        cfg["batch_size_initial"] = 10000
        cfg["ep_len"] = info_dict["max_steps"]
        cfg["add_sl_noise"] = False
        cfg["train_batch_size"] = 512
        # TODO is this needed?
        cfg["eval_batch_size"] = 400
        cfg["save_params"] = True
        ###

        computation_graph_args = {
            "ensamble_size": cfg["ensamble_size"],
            "n_layers": cfg["n_layers"],
            "size": cfg["size"],
            "learning_rate": cfg["learning_rate"],
        }nsamblensamble

        train_args = {
            "num_agent_train_steps_per_iter": cfg["num_agent_train_steps_per_iter"],
        }

        controller_args = {
            "mpc_horizon": cfg["mpc_horizon"],
            "mpc_num_action_sequences": cfg["mpc_num_action_sequences"],
            "mpc_action_sampling_strategy": cfg["mpc_action_sampling_strategy"],
            "cem_iterations": cfg["cem_iterations"],
            "cem_num_elites": cfg["cem_num_elites"],
            "cem_alpha": cfg["cem_alpha"],
        }

        agent_params = {**computation_graph_args, **train_args, **controller_args}

        self.params = cfg
        self.params["agent_class"] = rl_MBAgent
        self.params["agent_params"] = agent_params

        self.rl_trainer = RL_Trainer(self.params, info_dict)
