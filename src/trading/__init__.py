"""
Trading Module

Gym-style trading environment for DQN cryptocurrency trading.
Provides environment, position management, reward calculation, and state building.

Usage:
    from src.trading import TradingEnvironment, make_env

    # Create environment
    env = make_env()

    # Or with custom config
    env = TradingEnvironment(
        config_path="config/trading.yaml",
        data_directory="data/volatility",
    )

    # Standard Gym interface
    state = env.reset()
    next_state, reward, done, info = env.step(action)
"""

from .environment import TradingEnvironment, make_env
from .position_manager import PositionManager, Position, PositionSide, TradeRecord
from .reward_calculator import RewardCalculator, RewardInfo
from .state_builder import StateBuilder

__all__ = [
    "TradingEnvironment",
    "make_env",
    "PositionManager",
    "Position",
    "PositionSide",
    "TradeRecord",
    "RewardCalculator",
    "RewardInfo",
    "StateBuilder",
]
