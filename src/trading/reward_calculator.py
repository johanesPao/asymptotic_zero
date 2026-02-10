"""
Reward Calculator - Direct PnL Alignment

Simple rule: reward = PnL (no scaling, no penalties)
$1 profit = +1.0 reward
$1 loss = -1.0 reward
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardInfo:
    """Reward breakdown."""
    base_reward: float = 0.0
    pnl_reward: float = 0.0
    penalty: float = 0.0
    bonus: float = 0.0
    total: float = 0.0
    reason: str = ""

    def calculate_total(self):
        """Sum components."""
        self.total = self.base_reward + self.pnl_reward + self.penalty + self.bonus
        return self.total


class RewardCalculator:
    """
    Direct PnL-aligned reward calculator.
    
    CRITICAL: reward must have SAME SIGN as PnL!
    - Profit → positive reward
    - Loss → negative reward
    """

    def __init__(self, config_path: str = "config/trading.yaml", initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.episode_pnl = 0.0
        self.trade_count = 0
        
        # Track detailed breakdown for dashboard
        self.step_rewards = []
        self.step_pnls = []

    def reset(self):
        """Reset for new episode."""
        self.episode_pnl = 0.0
        self.trade_count = 0
        self.step_rewards = []
        self.step_pnls = []

    def calculate_step_reward(
        self,
        action_type: str,
        action_valid: bool,
        pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        position_count: int = 0,
    ) -> RewardInfo:
        """
        Calculate step reward.
        
        SIMPLE RULE: reward = PnL (direct mapping, no scaling!)
        PLUS: Small shaping reward for unrealized PnL to guide learning
        """
        reward = RewardInfo(reason=action_type)

        # Direct mapping: reward = PnL when closing
        if pnl != 0:
            reward.pnl_reward = pnl  # Direct 1:1 mapping!
            self.episode_pnl += pnl
            self.trade_count += 1
            reward.reason = f"{action_type} (${pnl:.2f})"
        
        # IMPORTANT: Add small shaping reward for unrealized PnL
        # This helps agent learn DURING holding, not just at close
        # Scale down by 100x so it doesn't dominate the real PnL reward
        if unrealized_pnl != 0 and action_type not in ["close", "close_all", "close_worst", "force_close"]:
            reward.base_reward = unrealized_pnl * 0.01  # 1% of unrealized PnL as shaping
        
        # Track for dashboard
        reward.calculate_total()
        self.step_rewards.append(reward.total)
        self.step_pnls.append(pnl)
        
        return reward

    def calculate_episode_end_reward(
        self, final_pnl: float, total_trades: int, win_rate: float
    ) -> RewardInfo:
        """
        Episode end: NO additional reward/penalty.
        Let the step rewards speak for themselves.
        """
        reward = RewardInfo(reason="episode_end")
        # NO episode bonus - just return 0
        reward.calculate_total()
        return reward

    def get_statistics(self) -> Dict:
        """Get detailed statistics for dashboard."""
        return {
            "episode_pnl": self.episode_pnl,
            "trade_count": self.trade_count,
            "total_step_reward": sum(self.step_rewards),
            "positive_rewards": sum(r for r in self.step_rewards if r > 0),
            "negative_rewards": sum(r for r in self.step_rewards if r < 0),
            "positive_pnls": sum(p for p in self.step_pnls if p > 0),
            "negative_pnls": sum(p for p in self.step_pnls if p < 0),
        }
