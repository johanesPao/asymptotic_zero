"""
Position Manager

Tracks all open positions, calculates PnL, and manages portfolio state.
Handles position opening, closing, and updates based on market prices.

Key Responsibilities:
- Track open positions (entry, price, size, side, timestamp)
- Calculate unrealized and realized PnL
- Enforce position limits and sizing rules
- Handle trading fees and slippage
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position direction."""

    LONG = 1
    SHORT = -1
    NONE = 0


@dataclass
class Position:
    """
    Represents a single open position.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        side: LONG or SHORT
        entry_price: Price at which position was opened
        size: Position size in quote currency (USDT)
        entry_time: Candle index when position was opened
        coin_index: Index in the top movers list (0-19)
        is_gainer: Whether this coin is a gainer (True) or loser (False)
    """

    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    entry_time: int
    coin_index: int
    is_gainer: bool

    # Tracking fields (updated each step)
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update(self, current_price: float):
        """Update position with current market price."""
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            # Long: profit when price goes up
            self.unrealized_pnl = (
                (current_price - self.entry_price) / self.entry_price * self.size
            )
            self.unrealized_pnl_pct = (
                (current_price - self.entry_price) / self.entry_price * 100
            )
        elif self.side == PositionSide.SHORT:
            # Short: profit when price goes down
            self.unrealized_pnl = (
                (self.entry_price - current_price) / self.entry_price * self.size
            )
            self.unrealized_pnl_pct = (
                (self.entry_price - current_price) / self.entry_price * 100
            )

    def get_exit_pnl(self, exit_price: float, fee_pct: float = 0.0) -> float:
        """
        Calculate realized PnL if position is closed at exit_price.

        Args:
            exit_price: Price to close position at
            fee_pct: Trading fee percentage (applied to exit)

        Returns:
            Realized PnL in quote currency
        """
        if self.side == PositionSide.LONG:
            gross_pnl = (exit_price - self.entry_price) / self.entry_price * self.size
        elif self.side == PositionSide.SHORT:
            gross_pnl = (self.entry_price - exit_price) / self.entry_price * self.size
        else:
            return 0.0

        # Deduct exit fee
        fee = self.size * (fee_pct / 100)
        net_pnl = gross_pnl - fee

        return net_pnl


@dataclass
class TradeRecord:
    """Record a completed trade."""

    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl: float
    pnl_pct: float
    fees_paid: float
    coin_index: int
    is_gainer: bool


class PositionManager:
    """
    Manages trading position and portfolio state.

    Tracks:
    - Open positions (max 3 by default)
    - Cash balance
    - Realized and unrealized PnL
    - Trade history
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        max_positions: int = 3,
        position_size_pct: float = 0.20,
        taker_fee_pct: float = 0.04,
    ):
        """
        Initialize the position manager.

        Args:
            initial_balance: Starting balance in USDT
            max_positions: Maximum concurrent positions
            position_size_pct: Each position as fraction of portfolio
            taker_fee_pct: Trading fee percentage
        """
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.taker_fee_pct = taker_fee_pct

        # State
        self.cash: float = initial_balance
        self.positions: Dict[int, Position] = {}  # coin_index -> Position
        self.trade_history: List[TradeRecord] = []
        self.total_fees_paid: float = 0.0
        self.current_step: int = 0
        self._peak_portfolio: float = initial_balance
        self._step_pnls: List[float] = []  # for rolling Sharpe / volatility

    def reset(self):
        """Reset to initial state for new episode."""
        self.cash = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.total_fees_paid = 0.0
        self.current_step: int = 0
        self._peak_portfolio = self.initial_balance
        self._step_pnls = []

    def set_step(self, step: int):
        """Set current timestep."""
        self.current_step = step

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + unrealized PnL)."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized

    def get_position_size(self) -> float:
        """Calculate position size based on current portfolio value."""
        portfolio_value = self.get_portfolio_value()
        return portfolio_value * self.position_size_pct

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < self.max_positions

    def has_position(self, coin_index: int) -> bool:
        """Check if we have a position on this coin."""
        return coin_index in self.positions

    def get_position(self, coin_index: int) -> Optional[Position]:
        """Get position for a coin, or None if no position."""
        return self.positions.get(coin_index)

    def update_positions(self, prices: Dict[int, float]):
        """
        Update all positions with current prices.

        Args:
            prices: Dictionary of coin_index -> current_price
        """
        for coin_index, position in self.positions.items():
            if coin_index in prices:
                position.update(prices[coin_index])

    def open_position(
        self,
        coin_index: int,
        symbol: str,
        side: PositionSide,
        price: float,
        is_gainer: bool,
    ) -> Tuple[bool, str]:
        """
        Open a new position.

        Args:
            coin_index: Index in top movers (0-19)
            symbol: Trading pair symbol
            side: LONG or SHORT
            price: Current market price
            is_gainer: Whether coin is a gainer

        Returns:
            (success, message) tuple
        """
        # Check if we can open
        if not self.can_open_position():
            return False, "Max positions reached"

        if self.has_position(coin_index):
            return False, f"Already have position on coin {coin_index}"

        # Calculate position size and fees
        # NOTE: price already has slippage applied by TradingEnvironment
        position_size = self.get_position_size()
        entry_fee = position_size * (self.taker_fee_pct / 100)

        # Check if we have enough cash
        if self.cash < position_size + entry_fee:
            return False, "Insufficient cash"

        # Deduct cash (position size is locked, fee is paid)
        self.cash -= position_size + entry_fee
        self.total_fees_paid += entry_fee

        # Create position (price is already slippage-adjusted by environment)
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            size=position_size,
            entry_time=self.current_step,
            coin_index=coin_index,
            is_gainer=is_gainer,
            current_price=price,
        )

        self.positions[coin_index] = position

        side_str = "LONG" if side == PositionSide.LONG else "SHORT"
        logger.debug(
            f"Opened {side_str} {symbol} @ {price:.6f}, size={position_size:.2f}"
        )

        return True, f"Opened {side_str} position"

    def close_position(self, coin_index: int, price: float) -> Tuple[bool, str, float]:
        """
        Close an existing position.

        Args:
            coin_index: Index of coin to close
            price: Current market price

        Returns:
            (success, message, realized_pnl) tuple
        """
        if not self.has_position(coin_index):
            return False, f"No position on coin {coin_index}", 0.0

        position = self.positions[coin_index]

        # NOTE: price already has slippage applied by TradingEnvironment
        # Calculate PnL
        exit_fee = position.size * (self.taker_fee_pct / 100)
        realized_pnl = position.get_exit_pnl(price, self.taker_fee_pct)

        # Calculate PnL percentage
        if position.side == PositionSide.LONG:
            pnl_pct = (
                (price - position.entry_price) / position.entry_price * 100
            )
        else:
            pnl_pct = (
                (position.entry_price - price) / position.entry_price * 100
            )

        # Return cash + PnL
        self.cash += position.size + realized_pnl
        self.total_fees_paid += exit_fee

        # Track PnL for rolling stats
        self._step_pnls.append(realized_pnl)

        # Update peak portfolio
        pv = self.get_portfolio_value()
        if pv > self._peak_portfolio:
            self._peak_portfolio = pv

        # Record trade
        trade = TradeRecord(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=self.current_step,
            pnl=realized_pnl,
            pnl_pct=pnl_pct,
            fees_paid=exit_fee,
            coin_index=coin_index,
            is_gainer=position.is_gainer,
        )
        self.trade_history.append(trade)

        # Remove position
        del self.positions[coin_index]

        side_str = "LONG" if position.side == PositionSide.LONG else "SHORT"
        logger.debug(
            f"Closed {side_str} {position.symbol} @ {price:.6f}, PnL={realized_pnl:.2f}"
        )

        return True, f"Closed position, PnL: {realized_pnl:.2f}", realized_pnl

    def close_all_positions(self, prices: Dict[int, float]) -> Tuple[int, float]:
        """
        Close all open positions.

        Args:
            prices: Dictionary of coin_index -> current_price

        Returns:
            (num_closed, total_pnl) tuple
        """
        total_pnl = 0.0
        closed = 0

        # Get list of positions to close (can't modify dict during iteration)
        coins_to_close = list(self.positions.keys())

        for coin_index in coins_to_close:
            if coin_index in prices:
                success, _, pnl = self.close_position(coin_index, prices[coin_index])
                if success:
                    total_pnl += pnl
                    closed += 1

        return closed, total_pnl

    def close_worst_position(self, prices: Dict[int, float]) -> Tuple[bool, float, int]:
        """
        Close the worst performing position.

        Args:
            prices: Dictionary of coin_index -> current_price

        Returns:
            (success, pnl, coin_index) tuple — coin_index is -1 if no positions
        """
        if not self.positions:
            return False, 0.0, -1

        # Update positions first
        self.update_positions(prices)

        # Find worst position
        worst_coin = min(
            self.positions.keys(), key=lambda idx: self.positions[idx].unrealized_pnl
        )

        success, _, pnl = self.close_position(worst_coin, prices[worst_coin])
        return success, pnl, worst_coin

    def get_position_features(self, num_coins: int = 20) -> np.ndarray:
        """
        Get position features for state representation.

        Args:
            num_coins: Total number of coins (gainers + losers)

        Returns:
            Array of shape (num_coins, 8) with position features
        """
        features = np.zeros((num_coins, 8), dtype=np.float32)
        total_portfolio = self.get_portfolio_value()

        for coin_index, position in self.positions.items():
            if coin_index < num_coins:
                # [0] has_position
                features[coin_index, 0] = 1.0
                # [1] position_side: -1 (short), 0 (none), 1 (long)
                features[coin_index, 1] = position.side.value
                # [2] position_size (normalized by initial balance)
                features[coin_index, 2] = position.size / self.initial_balance
                # [3] unrealized_pnl (normalized by position size)
                if position.size > 0:
                    features[coin_index, 3] = position.unrealized_pnl / position.size
                # [4] entry_price_distance
                if position.entry_price > 0:
                    features[coin_index, 4] = (
                        position.current_price - position.entry_price
                    ) / position.entry_price
                # [5] holding_time (normalized by episode length, ~88 tradeable candles)
                holding_time = self.current_step - position.entry_time
                features[coin_index, 5] = min(holding_time / 88.0, 1.0)
                # [6] hold_duration in raw steps (normalized to 0-1 by 288 max candles)
                features[coin_index, 6] = min(holding_time / 288.0, 1.0)
                # [7] position size as fraction of total portfolio
                if total_portfolio > 0:
                    features[coin_index, 7] = position.size / total_portfolio

        return features

    def get_portfolio_features(self) -> np.ndarray:
        """
        Get portfolio-level features for state representation.

        Returns:
            Array of shape (8,) with portfolio features
        """
        portfolio_value = self.get_portfolio_value()
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())

        # Rolling Sharpe from trade PnLs
        if len(self._step_pnls) >= 3:
            pnls = np.array(self._step_pnls[-20:])  # last 20 trades
            rolling_sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-8))
        else:
            rolling_sharpe = 0.0

        # Max drawdown pct
        drawdown = (self._peak_portfolio - portfolio_value) / self._peak_portfolio

        # Portfolio volatility (std of trade returns)
        if len(self._step_pnls) >= 3:
            vol = float(np.std(self._step_pnls[-20:]) / self.initial_balance)
        else:
            vol = 0.0

        features = np.array(
            [
                # [0] portfolio_value (normalized by initial balance)
                portfolio_value / self.initial_balance,
                # [1] cash_available (normalized by initial balance)
                self.cash / self.initial_balance,
                # [2] num_positions (normalized by max positions)
                len(self.positions) / self.max_positions,
                # [3] total_unrealized_pnl (normalized by initial balance)
                total_unrealized / self.initial_balance,
                # [4] total_fees_paid (normalized by initial balance)
                self.total_fees_paid / self.initial_balance,
                # [5] rolling Sharpe ratio (clipped)
                np.clip(rolling_sharpe, -3.0, 3.0),
                # [6] max drawdown percentage
                np.clip(drawdown, 0.0, 1.0),
                # [7] portfolio volatility
                np.clip(vol, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        return features

    def get_statistics(self) -> Dict:
        """Get trading statistics for the episode."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "total_fees": self.total_fees_paid,
                "final_portfolio_value": self.get_portfolio_value(),
                "return_pct": (self.get_portfolio_value() - self.initial_balance)
                / self.initial_balance
                * 100,
            }

        pnls = [t.pnl for t in self.trade_history]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        return {
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": (
                len(winning) / len(self.trade_history)
                if self.trade_history
                else 0.0
            ),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls) if pnls else 0.0,
            "max_win": max(pnls) if pnls else 0.0,
            "max_loss": min(pnls) if pnls else 0.0,
            "total_fees": self.total_fees_paid,
            "final_portfolio_value": self.get_portfolio_value(),
            "return_pct": (self.get_portfolio_value() - self.initial_balance)
            / self.initial_balance
            * 100,
        }
