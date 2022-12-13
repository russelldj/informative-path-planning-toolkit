from ipp_toolkit.utils.rl.agents.StableBaselinesAgent import (
    PPOAgent,
    DDPGAgent,
    DQNAgent,
    SACAgent,
    TD3Agent,
)
from ipp_toolkit.utils.rl.agents.RandomAgent import RandomAgent
from ipp_toolkit.utils.rl.agents.UCBAgent import UCBAgent
from ipp_toolkit.utils.rl.agents.MBAgent import MBAgent
from ipp_toolkit.utils.rl.agents.BehaviorCloningAgent import BehaviorCloningAgent
from ipp_toolkit.utils.rl.agents.DaggerAgent import DaggerAgent
from ipp_toolkit.utils.rl.agents.PerfectAgent import PerfectAgent

agent_dict = {
    "random": RandomAgent,
    "UCB": UCBAgent,
    "PPO": PPOAgent,
    "TD3": TD3Agent,
    "DDPG": DDPGAgent,
    "DQN": DQNAgent,
    "SAC": SACAgent,
    "MB": MBAgent,
    "BC": BehaviorCloningAgent,
    "DA": DaggerAgent,
    "Perfect": PerfectAgent,
}
