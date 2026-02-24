"""
Reward Calculator — v2

Risk-adjusted reward that teaches the agent about:
  - Realized PnL (direct signal on close)
  - Unrealized PnL shaping (signal during hold)
  - Transaction costs (commission + slippage on every trade)
  - Funding rate (charged every 8h on open positions)
  - Behavioral penalties (invalid actions, holding losers)
  - Behavioral bonuses (patience on winners, good episode)
  - Sharpe-based risk adjustment (quality > quantity)

Key design principle:
  reward sign ALWAYS matches PnL sign for realized trades.
  All other components are small relative to realized PnL
  so they shape without dominating.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardInfo:
    """Full reward breakdown for transparency and debugging."""
    realized_pnl:       float = 0.0   # Direct PnL on close
    unrealized_shaping: float = 0.0   # Shaping from unrealized PnL
    cost_penalty:       float = 0.0   # Commission + slippage deduction
    funding_penalty:    float = 0.0   # Funding rate deduction
    invalid_penalty:    float = 0.0   # Invalid action penalty
    holding_penalty:    float = 0.0   # Holding a losing position
    patience_bonus:     float = 0.0   # Holding a winning position
    inactivity_penalty: float = 0.0   # Holding with no open positions
    session_bonus:      float = 0.0   # End-of-episode performance bonus
    drawdown_penalty:   float = 0.0   # Equity drawdown penalty
    sharpe_adjustment:  float = 0.0   # Risk-quality scaling
    total:              float = 0.0
    reason:             str   = ""

    def compute_total(self) -> float:
        self.total = (
            self.realized_pnl
            + self.unrealized_shaping
            + self.cost_penalty
            + self.funding_penalty
            + self.invalid_penalty
            + self.holding_penalty
            + self.patience_bonus
            + self.inactivity_penalty
            + self.session_bonus
            + self.drawdown_penalty
            + self.sharpe_adjustment
        )
        return self.total


class RewardCalculator:
    """
    Risk-adjusted reward calculator for IQN retraining.

    Instantiated once per environment. Call reset() at the start
    of each episode. Call calculate_step_reward() on every step
    and calculate_episode_end_reward() when done=True.
    """

    def __init__(
        self,
        config_path: str = "config/trading.yaml",
        initial_balance: float = 10000.0,
    ):
        self.initial_balance = initial_balance
        self.config = self._load_config(config_path)
        reward_cfg = self.config.get("reward", {})
        costs_cfg  = self.config.get("costs",  {})

        # ── Sharpe window ──────────────────────────────────────────────────────
        sharpe_cfg = reward_cfg.get("sharpe", {})
        self.sharpe_window         = sharpe_cfg.get("window_steps", 20)
        self.sharpe_scale          = sharpe_cfg.get("scale", 1.0)
        self.sharpe_min_trades     = sharpe_cfg.get("min_trades_for_sharpe", 5)

        # ── Unrealized shaping ─────────────────────────────────────────────────
        ur_cfg = reward_cfg.get("unrealized_shaping", {})
        self.ur_enabled = ur_cfg.get("enabled", True)
        self.ur_scale   = ur_cfg.get("scale", 0.01)

        # ── Cost penalty ───────────────────────────────────────────────────────
        cost_pen_cfg = reward_cfg.get("cost_penalty", {})
        self.cost_pen_enabled = cost_pen_cfg.get("enabled", True)
        self.cost_pen_scale   = cost_pen_cfg.get("scale", 1.0)
        self.taker_fee        = costs_cfg.get("taker_fee_pct", 0.04) / 100.0
        self.slippage         = costs_cfg.get("slippage_pct",  0.01) / 100.0
        self.total_trade_cost = (self.taker_fee + self.slippage) * 2  # Entry + exit

        # ── Funding rate ───────────────────────────────────────────────────────
        fund_cfg = reward_cfg.get("funding_penalty", {})
        self.funding_enabled       = fund_cfg.get("enabled", True)
        self.funding_scale         = fund_cfg.get("scale", 1.0)
        costs_fund                 = costs_cfg.get("funding_rate_pct", 0.01) / 100.0
        self.funding_rate_per_step = costs_fund / costs_cfg.get("funding_interval_steps", 96)

        # ── Invalid action ─────────────────────────────────────────────────────
        self.invalid_penalty = reward_cfg.get("invalid_action_penalty", -2.0)

        # ── Holding penalties / bonuses ────────────────────────────────────────
        self.holding_loss_penalty   = reward_cfg.get("holding_loss_penalty_per_step", -0.05)
        self.holding_loss_threshold = reward_cfg.get("holding_loss_threshold", -50.0)
        self.patience_bonus_step    = reward_cfg.get("patience_bonus_per_step", 0.02)
        self.patience_threshold     = reward_cfg.get("patience_threshold", 20.0)

        # ── Inactivity penalty (hold with no positions) ─────────────────────────
        inact_cfg = reward_cfg.get("inactivity_penalty", {})
        self.inactivity_enabled = inact_cfg.get("enabled", True)
        self.inactivity_cost    = inact_cfg.get("per_step", -0.001)

        # ── Drawdown penalty ───────────────────────────────────────────────────
        self.drawdown_scale = reward_cfg.get("drawdown_penalty_scale", 0.5)

        # ── PnL scaling (bring large PnL numbers in line with shaping signals) ──
        self.pnl_scale = reward_cfg.get("pnl_scale", 0.01)

        # ── Drawdown cap ──────────────────────────────────────────────────────
        self.max_drawdown_penalty = reward_cfg.get("max_drawdown_penalty", -0.5)

        # ── Reward normalization ───────────────────────────────────────────────
        norm_cfg = reward_cfg.get("normalization", {})
        self.norm_enabled   = norm_cfg.get("enabled", True)
        self.norm_ema_alpha = norm_cfg.get("ema_alpha", 0.001)
        self.norm_warmup_steps = norm_cfg.get("warmup_steps", 500)
        self.norm_clip_range   = norm_cfg.get("clip_range", 10.0)
        self._reward_ema_mean = 0.0
        self._reward_ema_var  = 1.0
        self._norm_step_count = 0
        self._warmup_rewards  = []

        # ── Episode end ────────────────────────────────────────────────────────
        end_cfg = reward_cfg.get("episode_end", {})
        self.end_enabled              = end_cfg.get("enabled", True)
        self.profitable_session_bonus = end_cfg.get("profitable_session_bonus", 5.0)
        self.win_rate_bonus_scale     = end_cfg.get("win_rate_bonus_scale", 2.0)
        self.sharpe_bonus_scale       = end_cfg.get("sharpe_bonus_scale", 1.0)

        # ── Episode state ──────────────────────────────────────────────────────
        self.reset()

    def _load_config(self, config_path: str) -> dict:
        from pathlib import Path
        p = Path(config_path)
        if not p.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}
        with open(p) as f:
            return yaml.safe_load(f)

    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset all episode-level accumulators."""
        self.episode_pnl       = 0.0
        self.trade_count       = 0
        self.winning_trades    = 0
        self.peak_balance      = self.initial_balance
        self.current_balance   = self.initial_balance

        # Rolling window for Sharpe calculation
        self._step_returns: deque = deque(maxlen=self.sharpe_window)
        self._realized_pnls: List[float] = []

        # Per-step tracking for statistics
        self.step_rewards:       List[float] = []
        self.step_pnls:          List[float] = []
        self.invalid_action_count: int       = 0

    # ──────────────────────────────────────────────────────────────────────────

    def calculate_step_reward(
        self,
        action_type:        str,
        action_valid:       bool,
        pnl:                float = 0.0,
        unrealized_pnl:     float = 0.0,
        position_count:     int   = 0,
        trade_notional:     float = 0.0,  # Position size in USDT (for cost calc)
        total_open_notional: float = 0.0, # Total notional of ALL open positions
        steps_in_position:  int   = 0,    # Steps this position has been open
    ) -> RewardInfo:
        """
        Calculate reward for one environment step.

        Args:
            action_type:       'hold', 'open', 'close', 'close_all',
                               'close_worst', 'force_close', 'invalid'
            action_valid:      Whether the action was successfully executed
            pnl:               Realized PnL this step (non-zero only on close)
            unrealized_pnl:    Total unrealized PnL across all open positions
            position_count:    Number of currently open positions
            trade_notional:    Dollar value of position opened/closed (for costs)
            steps_in_position: Steps the closed/open position was held

        Returns:
            RewardInfo with full breakdown
        """
        r = RewardInfo(reason=action_type)

        # ── 1. Invalid action — strong negative signal ─────────────────────────
        if not action_valid:
            r.invalid_penalty = self.invalid_penalty
            self.invalid_action_count += 1
            r.reason = f"invalid:{action_type}"
            r.compute_total()
            self.step_rewards.append(r.total)
            return r

        # ── 2. Realized PnL — scaled to match shaping signals ──────────────────
        if pnl != 0.0:
            r.realized_pnl = pnl * self.pnl_scale
            self.episode_pnl += pnl  # Track raw PnL for stats
            self.trade_count += 1
            if pnl > 0:
                self.winning_trades += 1
            self._realized_pnls.append(pnl)
            self._step_returns.append(pnl * self.pnl_scale)
            r.reason = f"{action_type}(${pnl:+.2f})"

        # ── 3. Transaction cost penalty on open and close ──────────────────────
        if self.cost_pen_enabled and trade_notional > 0 and action_type in (
            "open", "close", "close_all", "close_worst", "force_close"
        ):
            cost = trade_notional * self.total_trade_cost * self.cost_pen_scale
            r.cost_penalty = -cost

        # ── 4. Funding rate penalty — per step, per open position ──────────────
        if self.funding_enabled and position_count > 0 and total_open_notional > 0:
            # Charge funding on total open position notional (not just current trade)
            fund_cost = (
                total_open_notional * self.funding_rate_per_step * self.funding_scale
            )
            r.funding_penalty = -fund_cost

        # ── 5. Unrealized PnL shaping — only during hold/open, not on close ───
        if (
            self.ur_enabled
            and unrealized_pnl != 0.0
            and action_type not in ("close", "close_all", "close_worst", "force_close")
        ):
            r.unrealized_shaping = unrealized_pnl * self.ur_scale

        # ── 6. Holding penalties and bonuses ───────────────────────────────────
        if action_type == "hold" and position_count > 0:
            if unrealized_pnl < self.holding_loss_threshold:
                r.holding_penalty = self.holding_loss_penalty

            if unrealized_pnl > self.patience_threshold:
                r.patience_bonus = self.patience_bonus_step

        # ── 6b. Inactivity penalty — nudge agent to take positions ────────────
        if self.inactivity_enabled and action_type == "hold" and position_count == 0:
            r.inactivity_penalty = self.inactivity_cost

        # ── 7. Drawdown penalty ────────────────────────────────────────────────
        self.current_balance = self.initial_balance + self.episode_pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        drawdown = self.peak_balance - self.current_balance
        if drawdown > 0:
            raw_dd = -(drawdown * self.drawdown_scale * 0.001)
            r.drawdown_penalty = max(raw_dd, self.max_drawdown_penalty)

        # ── 8. Sharpe adjustment — scale reward by rolling risk quality ─────────
        if len(self._step_returns) >= self.sharpe_min_trades:
            returns = np.array(self._step_returns)
            mean_r  = np.mean(returns)
            std_r   = np.std(returns) + 1e-8
            rolling_sharpe = mean_r / std_r
            # Scale is small — just nudges agent toward consistent returns
            r.sharpe_adjustment = rolling_sharpe * self.sharpe_scale * 0.01

        return self._finalize_reward(r, pnl)

    def _finalize_reward(self, r: RewardInfo, pnl: float = 0.0) -> RewardInfo:
        """Helper to compute total, normalize, and track stats."""
        r.compute_total()

        # ── Reward normalization ───────────────────────────────────────────────
        if self.norm_enabled:
            raw = r.total
            self._norm_step_count += 1

            if self._norm_step_count <= self.norm_warmup_steps:
                # Warmup phase: collect raw rewards, skip normalization
                self._warmup_rewards.append(raw)

                if self._norm_step_count == self.norm_warmup_steps:
                    # Initialize EMA mean/var from collected batch
                    arr = np.array(self._warmup_rewards)
                    self._reward_ema_mean = float(np.mean(arr))
                    self._reward_ema_var  = float(np.var(arr)) + 1e-6
                    self._warmup_rewards  = []  # Free memory
            else:
                # Post-warmup: apply EMA normalization
                alpha = self.norm_ema_alpha
                self._reward_ema_mean = (1 - alpha) * self._reward_ema_mean + alpha * raw
                diff = raw - self._reward_ema_mean
                self._reward_ema_var = (1 - alpha) * self._reward_ema_var + alpha * (diff ** 2)
                std = max(self._reward_ema_var ** 0.5, 1e-6)
                r.total = raw / std

            # Always hard-clip output
            r.total = float(np.clip(r.total, -self.norm_clip_range, self.norm_clip_range))

        self.step_rewards.append(r.total)
        self.step_pnls.append(pnl)
        return r

    # ──────────────────────────────────────────────────────────────────────────

    def calculate_episode_end_reward(
        self,
        final_pnl:    float,
        total_trades: int,
        win_rate:     float,
    ) -> RewardInfo:
        """
        Episode-end bonus based on overall session quality.

        This fires once at the end of each trading day episode.
        It rewards sessions with positive PnL, high win rate,
        and good Sharpe — reinforcing the full-day objective.
        """
        r = RewardInfo(reason="episode_end")

        if not self.end_enabled:
            return self._finalize_reward(r)

        # Profitable session bonus
        if final_pnl > 0:
            r.session_bonus = self.profitable_session_bonus

        # Win rate bonus
        if win_rate > 0 and total_trades >= self.sharpe_min_trades:
            r.session_bonus += win_rate * self.win_rate_bonus_scale

        # Session Sharpe bonus
        if len(self._realized_pnls) >= self.sharpe_min_trades:
            pnls = np.array(self._realized_pnls)
            session_sharpe = np.mean(pnls) / (np.std(pnls) + 1e-8)
            if session_sharpe > 0:
                r.sharpe_adjustment = session_sharpe * self.sharpe_bonus_scale

        return self._finalize_reward(r)

    # ──────────────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        """Full episode statistics for dashboard and logging."""
        win_rate = (
            self.winning_trades / self.trade_count
            if self.trade_count > 0 else 0.0
        )
        invalid_rate = (
            self.invalid_action_count / max(len(self.step_rewards), 1)
        )

        # Session Sharpe
        session_sharpe = 0.0
        if len(self._realized_pnls) >= self.sharpe_min_trades:
            pnls = np.array(self._realized_pnls)
            session_sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-8))

        return {
            "episode_pnl":          self.episode_pnl,
            "trade_count":          self.trade_count,
            "win_rate":             win_rate,
            "winning_trades":       self.winning_trades,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_rate":  invalid_rate,
            "session_sharpe":       session_sharpe,
            "peak_balance":         self.peak_balance,
            "max_drawdown":         self.peak_balance - self.current_balance,
            "total_step_reward":    sum(self.step_rewards),
            "positive_rewards":     sum(r for r in self.step_rewards if r > 0),
            "negative_rewards":     sum(r for r in self.step_rewards if r < 0),
        }
