"""
Agent Module

DQN agent for cryptocurrency trading.
Provides neural network, replay buffer, agent, and trainer.

Usage:
    from src.agent import DQNAgent, DQNTrainer
    from src.trading import make_env

    # Create environment and agent
    env = make_env()
    agent = DQNAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
    )

    # Create trainer and train
    trainer = DQNTrainer(env, agent)
    trainer.train()
"""

from .network import QNetwork, create_q_network
from .replay_buffer import ReplayBuffer, Transition
from .dqn_agent import DQNAgent
from .trainer import DQNTrainer

__all__ = [
    "QNetwork",
    "create_q_network",
    "ReplayBuffer",
    "Transition",
    "DQNAgent",
    "DQNTrainer",
]
