"""
Trade Executor
Handles order placement and position management.
"""

from binance.client import Client
from typing import Dict, Optional, List
import time


class TradeExecutor:
    """Executes trades on Binance Futures."""

    def __init__(
        self,
        client: Client,
        leverage: int = 10,
        position_size_pct: float = 0.20,
        max_positions: int = 3,
    ):
        """
        Initialize trade executor.

        Args:
            client: Binance client
            leverage: Leverage multiplier
            position_size_pct: Position size as % of portfolio
            max_positions: Maximum concurrent positions
        """
        self.client = client
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

        # Track positions: {symbol: {'side': 'LONG'/'SHORT', 'size': float, 'entry_price': float}}
        self.positions: Dict[str, Dict] = {}

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        ticker = self.client.futures_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    def get_balance(self) -> float:
        """Get available USDT balance."""
        account = self.client.futures_account()
        return float(account["totalWalletBalance"])

    def get_position_info(self) -> Dict:
        """Get info about current positions."""
        return {
            "count": len(self.positions),
            "symbols": list(self.positions.keys()),
            "positions": self.positions.copy(),
        }

    def _calculate_pnl_pct(self, position: Dict, current_price: float) -> float:
        """Calculate PnL percentage for a position."""
        entry = position["entry_price"]

        if position["side"] == "LONG":
            return ((current_price - entry) / entry) * 100 * self.leverage
        else:  # SHORT
            return ((entry - current_price) / entry) * 100 * self.leverage

    def set_leverage(self, symbol: str):
        """Set leverage for symbol."""
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage)
        except Exception as e:
            print(f"⚠️ Leverage already set for {symbol}: {e}")

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to valid precision for symbol."""
        # Get symbol info
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        step_size = float(f["stepSize"])
                        precision = len(str(step_size).rstrip("0").split(".")[-1])
                        return round(quantity, precision)

        return round(quantity, 3)  # Default precision

    def _get_fill_price(self, order: Dict) -> float:
        """
        Extract actual average fill price from a Binance futures order response.
        Falls back to pre-fetched price only if avgPrice is missing or zero.
        """
        avg_price = float(order.get("avgPrice", 0))
        if avg_price > 0:
            return avg_price

        # Fallback: derive from cumQuote / executedQty
        cum_quote = float(order.get("cumQuote", 0))
        executed_qty = float(order.get("executedQty", 0))
        if cum_quote > 0 and executed_qty > 0:
            return cum_quote / executed_qty

        # Last resort: live price (should rarely reach here)
        return self.get_current_price(order["symbol"])

    def open_long(self, symbol: str, balance: float) -> Optional[Dict]:
        """
        Open a LONG position.

        Args:
            symbol: Trading pair
            balance: Current portfolio balance

        Returns:
            Order info dict or None if failed
        """
        if len(self.positions) >= self.max_positions:
            return None

        if symbol in self.positions:
            return None

        try:
            self.set_leverage(symbol)

            # Use live price only to calculate quantity — NOT stored as entry price
            position_value = balance * self.position_size_pct
            price_estimate = self.get_current_price(symbol)
            quantity = (position_value * self.leverage) / price_estimate
            quantity = self._round_quantity(symbol, quantity)

            order = self.client.futures_create_order(
                symbol=symbol, side="BUY", type="MARKET", quantity=quantity
            )

            # Use actual fill price from Binance response
            actual_entry = self._get_fill_price(order)
            actual_qty = float(order.get("executedQty", quantity))

            self.positions[symbol] = {
                "side": "LONG",
                "size": actual_qty,
                "entry_price": actual_entry,
                "order_id": order["orderId"],
            }

            return order
        except Exception as e:
            print(f"❌ Failed to open LONG {symbol}: {e}")
            return None

    def open_short(self, symbol: str, balance: float) -> Optional[Dict]:
        """Open a SHORT position."""
        if len(self.positions) >= self.max_positions:
            return None

        if symbol in self.positions:
            return None

        try:
            self.set_leverage(symbol)

            position_value = balance * self.position_size_pct
            price_estimate = self.get_current_price(symbol)
            quantity = (position_value * self.leverage) / price_estimate
            quantity = self._round_quantity(symbol, quantity)

            order = self.client.futures_create_order(
                symbol=symbol, side="SELL", type="MARKET", quantity=quantity
            )

            actual_entry = self._get_fill_price(order)
            actual_qty = float(order.get("executedQty", quantity))

            self.positions[symbol] = {
                "side": "SHORT",
                "size": actual_qty,
                "entry_price": actual_entry,
                "order_id": order["orderId"],
            }

            return order
        except Exception as e:
            print(f"❌ Failed to open SHORT {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close an existing position.
        
        Returns:
            Dict with 'order', 'pnl', 'pnl_pct' or None if failed.
        """
        if symbol not in self.positions:
            return None

        try:
            position = self.positions[symbol]
            side = "SELL" if position["side"] == "LONG" else "BUY"

            entry_price = position["entry_price"]
            size = position["size"]
            size_usdt = size * entry_price

            # Place closing order first
            order = self.client.futures_create_order(
                symbol=symbol, side=side, type="MARKET", quantity=size
            )

            # Use actual fill price from order response
            exit_price = self._get_fill_price(order)

            # Calculate exact realized PnL using actual fill prices
            if position["side"] == "LONG":
                pnl = (exit_price - entry_price) / entry_price * size_usdt
            else:
                pnl = (entry_price - exit_price) / entry_price * size_usdt

            pnl_pct = self._calculate_pnl_pct(position, exit_price)

            del self.positions[symbol]

            return {
                "order": order,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": position["side"],
            }
        except Exception as e:
            print(f"❌ Failed to close {symbol}: {e}")
            return None

    def close_all_positions(self) -> int:
        """Close all open positions."""
        symbols = list(self.positions.keys())
        closed = 0

        for symbol in symbols:
            if self.close_position(symbol):
                closed += 1

        return closed
