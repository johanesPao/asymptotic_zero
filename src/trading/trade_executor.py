"""
Trade Executor
Handles order placement and position management.
"""

import logging
import math
from binance.client import Client
from typing import Dict, Optional, List
import time

_log = logging.getLogger(__name__)


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

        # Cache of symbol lot-size step per symbol {symbol: step_size float}
        # Populated lazily on first order for each symbol.
        self._lot_step_cache: Dict[str, float] = {}

        # Cache of max notional per symbol at self.leverage {symbol: max_notional float}
        # Used to cap order size and avoid -2027 errors.
        self._max_notional_cache: Dict[str, float] = {}

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol."""
        ticker = self.client.futures_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    def get_balance(self) -> float:
        """Get wallet balance (cash only, no unrealized PnL).
        Used to set initial_balance at session start so that session PnL = equity_now - cash_at_start,
        which naturally equals unrealized_PnL + realized_PnL_from_closed_trades."""
        account = self.client.futures_account()
        return float(account["totalWalletBalance"])

    def get_equity(self) -> float:
        """Get total equity = wallet + unrealized PnL (totalMarginBalance)."""
        account = self.client.futures_account()
        return float(account.get("totalMarginBalance") or account["totalWalletBalance"])

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
            _log.debug(f"set_leverage {symbol}: {e}")

    def _get_lot_step(self, symbol: str) -> float:
        """Return the LOT_SIZE step for symbol, cached after first fetch."""
        if symbol not in self._lot_step_cache:
            try:
                info = self.client.futures_exchange_info()
                for s in info["symbols"]:
                    step = 1.0
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            break
                    self._lot_step_cache[s["symbol"]] = step
            except Exception as e:
                _log.warning(f"Could not fetch exchange info for {symbol}: {e}")
                return 1.0  # safe fallback — integer quantity
        return self._lot_step_cache.get(symbol, 1.0)

    def _get_max_notional(self, symbol: str) -> float:
        """Return the maximum notional allowed at self.leverage for symbol.

        Queries futures_leverage_bracket once per symbol and caches.  Falls back
        to a conservative $5 000 cap if the bracket cannot be fetched so that we
        never hit -2027 ("Exceeded maximum allowable position at current leverage").
        """
        if symbol not in self._max_notional_cache:
            try:
                brackets = self.client.futures_leverage_bracket(symbol=symbol)
                # Response: [{"symbol": "...", "brackets": [{...}, ...]}, ...]
                sym_data = brackets[0] if brackets else {}
                max_notional = 5_000.0  # conservative fallback
                for b in sym_data.get("brackets", []):
                    if b.get("initialLeverage", 0) >= self.leverage:
                        max_notional = float(b["notionalCap"])
                self._max_notional_cache[symbol] = max_notional
            except Exception as e:
                _log.warning(f"Could not fetch leverage bracket for {symbol}: {e} — using $5000 cap")
                self._max_notional_cache[symbol] = 5_000.0
        return self._max_notional_cache[symbol]

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Floor quantity to the nearest valid LOT_SIZE step.

        Uses floor (not round) so we never accidentally exceed our
        intended position size or trip a max-notional filter.
        """
        step = self._get_lot_step(symbol)
        # floor to step, then remove floating-point noise
        floored = math.floor(quantity / step) * step
        # determine decimal places from step to format cleanly
        if step >= 1.0:
            return float(int(floored))
        precision = max(0, round(-math.log10(step)))
        return round(floored, precision)

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
            raw_notional = position_value * self.leverage
            # Cap to max notional allowed at current leverage (avoids -2027)
            max_notional = self._get_max_notional(symbol)
            notional = min(raw_notional, max_notional * 0.95)  # 5% safety margin
            quantity = notional / price_estimate
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
            _log.error(f"Failed to open LONG {symbol}: {e}")
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
            raw_notional = position_value * self.leverage
            max_notional = self._get_max_notional(symbol)
            notional = min(raw_notional, max_notional * 0.95)  # 5% safety margin
            quantity = notional / price_estimate
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
            _log.error(f"Failed to open SHORT {symbol}: {e}")
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
            _log.error(f"Failed to close {symbol}: {e}")
            return None

    def sync_positions_from_binance(self) -> int:
        """Load actual open positions from Binance into the in-memory cache.

        Should be called after a bot restart so the cache reflects any
        positions that were opened during a previous session.
        Returns the number of positions loaded.
        """
        try:
            live = self.client.futures_position_information()
            count = 0
            for pos in live:
                symbol = pos["symbol"]
                amt = float(pos["positionAmt"])
                if amt == 0:
                    continue
                entry_price = float(pos["entryPrice"])
                side = "LONG" if amt > 0 else "SHORT"
                size = abs(amt)
                self.positions[symbol] = {
                    "side": side,
                    "size": size,
                    "entry_price": entry_price,
                }
                count += 1
            return count
        except Exception as e:
            print(f"⚠️ sync_positions_from_binance failed: {e}")
            return 0

    def close_all_positions(self) -> int:
        """Close ALL open positions — both in-memory cache and any live Binance positions.

        Querying Binance directly ensures we catch positions opened in a previous
        bot session that are not present in the in-memory cache.
        """
        closed = 0

        # 1. Close positions tracked in the in-memory cache.
        for symbol in list(self.positions.keys()):
            if self.close_position(symbol):
                closed += 1

        # 2. Also close any live Binance positions not in the cache
        #    (e.g. left over from a previous session after restart).
        try:
            live = self.client.futures_position_information()
            for pos in live:
                symbol = pos["symbol"]
                amt = float(pos["positionAmt"])
                if amt == 0 or symbol in self.positions:
                    continue
                # Position exists on Binance but not in our cache — close it.
                side = "SELL" if amt > 0 else "BUY"
                quantity = abs(amt)
                try:
                    self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=quantity,
                        reduceOnly=True,
                    )
                    closed += 1
                    print(f"✅ Closed orphaned position {symbol} (qty={quantity})")
                except Exception as e:
                    print(f"❌ Failed to close orphaned position {symbol}: {e}")
        except Exception as e:
            print(f"⚠️ Could not fetch live positions for orphan-close: {e}")

        return closed
