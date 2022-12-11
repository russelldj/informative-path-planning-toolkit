from ipp_toolkit.utils.rl.agents.StableBaselinesAgent import (
    PPOAgent,
    DDPGAgent,
    DQNAgent,
    SACAgent,
)
from ipp_toolkit.utils.rl.agents.RandomAgent import RandomAgent
from ipp_toolkit.utils.rl.agents.UCBAgent import UCBAgent
from ipp_toolkit.utils.rl.agents.MBAgent import MBAgent
from ipp_toolkit.utils.rl.agents.BehaviorCloningAgent import BehaviorCloningAgent

agent_dict = {
    "random": RandomAgent,
    "UCB": UCBAgent,
    "PPO": PPOAgent,
    "DDPG": DDPGAgent,
    "DQN": DQNAgent,
    "SAC": SACAgent,
    "MB": MBAgent,
    "BC": BehaviorCloningAgent,
}
