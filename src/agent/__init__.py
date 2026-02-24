"""
Agent Module — v2 (IQN)

Usage:
    from src.agent import IQNAgent
    from src.trading import make_env

    env   = make_env()
    agent = IQNAgent(
        state_dim   = env.get_state_space_size(),
        action_size = env.get_action_space_size(),
    )
"""

from .network import IQNNetwork
from .replay_buffer import ReplayBuffer
from .iqn_agent import IQNAgent

# Backward-compatibility aliases so any existing scripts importing DQNAgent still work
DQNAgent   = IQNAgent
DQNNetwork = IQNNetwork

__all__ = [
    "IQNNetwork",
    "IQNAgent",
    "ReplayBuffer",
    # Aliases
    "DQNAgent",
    "DQNNetwork",
]
