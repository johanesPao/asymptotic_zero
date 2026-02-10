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
        slippage_pct: float = 0.01,
    ):
        """
        Initialize the position manager.

        Args:
            initial_balance: Starting balance in USDT
            max_positions: Maximum concurrent positions
            position_size_pct: Each position as fraction of portfolio
            taker_fee_pct: Trading fee percentage
            slippage_pct: Estimated slippage percentage
        """
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct

        # State
        self.cash: float = initial_balance
        self.positions: Dict[int, Position] = {}  # coin_index -> Position
        self.trade_history: List[TradeRecord] = []
        self.total_fees_paid: float = 0.0
        self.current_step: int = 0

    def reset(self):
        """Reset to initial state for new episode."""
        self.cash = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.total_fees_paid = 0.0
        self.current_step: int = 0

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
        position_size = self.get_position_size()
        entry_fee = position_size * (self.taker_fee_pct / 100)
        slippage_cost = position_size * (self.slippage_pct / 100)
        total_cost = entry_fee + slippage_cost

        # Adjust entry price for slippage
        if side == PositionSide.LONG:
            adjusted_price = price * (1 + self.slippage_pct / 100)
        else:
            adjusted_price = price * (1 - self.slippage_pct / 100)

        # Check if we have enough cash
        if self.cash < position_size + total_cost:
            return False, "Insufficient cash"

        # Deduct cash (position size is locked, fee is paid)
        self.cash -= position_size + total_cost
        self.total_fees_paid += entry_fee

        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=adjusted_price,
            size=position_size,
            entry_time=self.current_step,
            coin_index=coin_index,
            is_gainer=is_gainer,
            current_price=adjusted_price,
        )

        self.positions[coin_index] = position

        side_str = "LONG" if side == PositionSide.LONG else "SHORT"
        logger.debug(
            f"Opened {side_str} {symbol} @ {adjusted_price:.6f}, size={position_size:.2f}"
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

        # Adjust exit price for slippage
        if position.side == PositionSide.LONG:
            # Selling: slippage works against us
            adjusted_price = price * (1 - self.slippage_pct / 100)
        else:
            # Buying back: slippage works against us
            adjusted_price = price * (1 + self.slippage_pct / 100)

        # Calculated PnL
        exit_fee = position.size * (self.taker_fee_pct / 100)
        realized_pnl = position.get_exit_pnl(adjusted_price, self.taker_fee_pct)

        # Calculate PnL percentage
        if position.side == PositionSide.LONG:
            pnl_pct = (
                (adjusted_price - position.entry_price) / position.entry_price * 100
            )
        else:
            pnl_pct = (
                (position.entry_price - adjusted_price) / position.entry_price * 100
            )

        # Return cash + PnL
        self.cash += position.size + realized_pnl
        self.total_fees_paid += exit_fee

        # Return trade
        trade = TradeRecord(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=adjusted_price,
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
            f"Closed {side_str} {position.symbol} @ {adjusted_price:.6f}, PnL={realized_pnl:.2f}"
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

    def close_worst_position(self, prices: Dict[int, float]) -> Tuple[bool, float]:
        """
        Close the worst performing position.

        Args:
            prices: Dictionary of coin_index -> current_price

        Returns:
            (success, pnl) tuple
        """
        if not self.positions:
            return False, 0.0

        # Update positions first
        self.update_positions(prices)

        # Find worst position
        worst_coin = min(
            self.positions.keys(), key=lambda idx: self.positions[idx].unrealized_pnl
        )

        success, _, pnl = self.close_position(worst_coin, prices[worst_coin])
        return success, pnl

    def get_position_features(self, num_coins: int = 20) -> np.ndarray:
        """
        Get position features for state representation.

        Args:
            num_coins: Total number of coins (gainers + losers)

        Returns:
            Array of shape (num_coins, 6) with position features
        """
        features = np.zeros((num_coins, 6))

        for coin_index, position in self.positions.items():
            if coin_index < num_coins:
                # has_position
                features[coin_index, 0] = 1.0

                # position_side: -1 (short), 0 (none), 1 (long)
                features[coin_index, 1] = position.side.value

                # position_size (normalized by initial balance)
                features[coin_index, 2] = position.size / self.initial_balance

                # unrealized_pnl (normalized by position size)
                if position.size > 0:
                    features[coin_index, 3] = position.unrealized_pnl / position.size

                # entry_price_distance
                if position.entry_price > 0:
                    features[coin_index, 4] = (
                        position.current_price - position.entry_price
                    ) / position.entry_price

                # holding_time (normalized by episode length, ~88 tradeable candles)
                holding_time = self.current_step - position.entry_time
                features[coin_index, 5] = min(holding_time / 88.0, 1.0)

        return features

    def get_portfolio_features(self) -> np.ndarray:
        """
        Get portfolio-level features for state representation.

        Returns:
            Array of shape (5,) with portfolio features
        """
        portfolio_value = self.get_portfolio_value()
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())

        features = np.array(
            [
                # portfolio_value (normalized by initial balance)
                portfolio_value / self.initial_balance,
                # cash_available (normalized by initial balance)
                self.cash / self.initial_balance,
                # num_positions (normalized by max positions)
                len(self.positions) / self.max_positions,
                # total_unrealized_pnl (normalized by initial balance)
                total_unrealized / self.initial_balance,
                # capacity_used (how much of max positions is used)
                len(self.positions) / self.max_positions,
            ]
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
                len(winning) / len(self.trade_history) * 100
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
