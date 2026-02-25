"""
Asymptotic Zero - Live Trading Bot

Main trading loop that connects all components:
- Morning screening
- Market data collection
- Agent decision making
- Trade execution
- Web dashboard state updates

v2: Added position reconciliation with Binance to prevent state divergence
"""

import logging
import sys
from pathlib import Path
import random
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import faulthandler
import pytz

faulthandler.enable()  # dump Python traceback on SIGABRT / SIGSEGV

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after sys.path is set so 'src' is resolvable
from src.utils.log_setup import get_logger

_log = get_logger("trading")
WIB = pytz.timezone("Asia/Jakarta")

from src.config.secrets import get_secret
from binance.client import Client

from src.database.logger import TradingLogger
from src.trading.screening import get_top_movers
from src.trading.market_data import fetch_all_coins
from src.data_pipeline.features.technical_indicators import TechnicalIndicators
from src.trading.state_builder import StateBuilder
from src.trading.trade_executor import TradeExecutor
from src.trading.guardrails import GuardrailManager

# DQNAgent (TensorFlow) is imported lazily inside __init__, AFTER the DB
# warmup connection.  TensorFlow bundles its own libssl.so; if it loads
# before psycopg2's first connect(), two OpenSSL instances clash → SIGSEGV.


class _DashboardLogHandler(logging.Handler):
    """Captures recent WARNING/ERROR log records for the web dashboard and persists to DB."""

    def __init__(self, maxlen: int = 50):
        super().__init__(level=logging.WARNING)
        self._records: list = []
        self._maxlen = maxlen
        self._logger: TradingLogger | None = None
        self._session_key: str = ""

    def set_logger(self, logger: TradingLogger, session_key: str = "") -> None:
        self._logger = logger
        self._session_key = session_key

    def emit(self, record: logging.LogRecord) -> None:
        source = record.name.split(".")[-1]
        message = self.format(record)
        self._records.append({
            "time": datetime.fromtimestamp(record.created, WIB).strftime("%H:%M:%S"),
            "level": record.levelname,
            "source": source,
            "message": message,
        })
        if len(self._records) > self._maxlen:
            self._records = self._records[-self._maxlen:]

        # Persist to DB
        if self._logger is not None:
            import traceback as _tb
            exc_text = ""
            if record.exc_info and record.exc_info[1]:
                exc_text = "".join(_tb.format_exception(*record.exc_info))
            self._logger.log_event(
                level=record.levelname,
                source=source,
                message=message,
                session_key=self._session_key,
                exception=exc_text,
            )

    @property
    def entries(self) -> list:
        return list(self._records)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self):
        """Initialize bot components."""
        print("Initializing Asymptotic Zero Trading Bot...")

        # Load secrets
        self.api_key = get_secret("BINANCE_API_KEY")
        self.api_secret = get_secret("BINANCE_SECRET")
        self.db_url = get_secret("DATABASE_URL")
        self.trading_mode = get_secret("TRADING_MODE")
        self.max_daily_loss_pct = float(get_secret("MAX_DAILY_LOSS_PCT", 5.0))
        self._daily_loss_limit = 0.0  # computed from initial_balance × pct at each screening
        self.max_daily_trades = int(get_secret("MAX_DAILY_TRADES", 10))
        self.STATE_DIM = 823
        self.ACTION_DIM = 63

        print(f"✅ Secrets loaded (mode: {self.trading_mode})")

        # Initialize Binance Client
        self.client = Client(
            self.api_key, self.api_secret, testnet=(self.trading_mode == "testnet"),
        )

        # Fix time offset
        server_time = self.client.get_server_time()
        local_time = int(time.time() * 1000)
        self.client.timestamp_offset = server_time["serverTime"] - local_time

        print(f"✅ Connected to Binance {self.trading_mode}")

        # Initialize trading logger HERE — before importing TensorFlow.
        # TradingLogger.__init__ calls _warmup_connection() which makes the
        # very first psycopg2.connect() call, initialising libpq + its OpenSSL.
        # TensorFlow bundles its own libssl.so; loading TF after this point is
        # safe because the system OpenSSL is already fully initialised.
        # If TF loads first, two OpenSSL instances clash → SIGSEGV on connect().
        self.logger = TradingLogger(self.db_url)
        self.current_session_key: str = ""
        self.open_trade_keys: dict = {}
        self._last_close_exit_prices: dict = {}

        # Activity log, PnL history, recent trades, and market features for web dashboard
        self._activity_log: list = []
        self._recent_trades: list = []
        self._session_started: str = ""
        self._pnl_history: list = []
        self._last_market_features: dict = {}

        # System error log — captures WARNING/ERROR from ALL Python loggers
        self._dashboard_log_handler = _DashboardLogHandler(maxlen=50)
        self._dashboard_log_handler.set_logger(self.logger)
        logging.getLogger().addHandler(self._dashboard_log_handler)

        print("✅ DB logger initialized (libpq/OpenSSL warmed up)")

        # Load trained agent — TensorFlow imports HERE, after libpq is warm.
        from src.agent.iqn_agent import IQNAgent

        self.agent = IQNAgent(state_dim=self.STATE_DIM, action_size=self.ACTION_DIM)
        # Warm up networks - handle LSTM vs flat state
        # Warm up networks - handle LSTM vs flat state
        if self.agent.lstm_enabled:
            # LSTM mode: create sequence inputs
            dummy_short = np.zeros((1, 10, self.STATE_DIM))
            dummy_medium = np.zeros((1, 20, self.STATE_DIM))
            dummy_long = np.zeros((1, 30, self.STATE_DIM))
            self.agent.online_network(dummy_short, dummy_medium, dummy_long, training=False)
        self.agent.load("checkpoints/best")
        self.agent.epsilon = 0.0  # live trading is always deterministic

        print(f"✅ Agent loaded (deterministic mode, epsilon forced to 0)")

        # Initialize feature calculator — derive feature count like the environment does
        self.feature_calculator = TechnicalIndicators()
        self._feature_names = self.feature_calculator.get_feature_names()
        self._num_market_features = len(self._feature_names)
        print(f"✅ Feature calculator initialized ({self._num_market_features} features per coin)")

        # Initialize state builder (num_features from TA config, matches environment.py)
        self.state_builder = StateBuilder(num_coins=20, num_features=self._num_market_features)
        print("✅ State builder initialized")

        # Initialize trade executor
        self.executor = TradeExecutor(
            client=self.client, leverage=10, position_size_pct=0.20, max_positions=3
        )
        # Sync any positions opened in a previous session so the in-memory cache
        # is accurate from the start (close_all_positions iterates this cache).
        n_synced = self.executor.sync_positions_from_binance()
        if n_synced:
            print(f"✅ Trade executor initialized — synced {n_synced} existing position(s) from Binance")
        else:
            print("✅ Trade executor initialized")

        # Initialize guardrails
        self.guardrails = GuardrailManager(max_positions=3)
        print("✅ Guardrails initialized (5-step cooldown, 3-step min hold)")

        # Trading state
        self.current_balance = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_coins = None
        self.max_positions = 3
        self.initial_balance = 0.0

        # Today's PnL baseline (resets at midnight, independent of session rescreening)
        self._today_date: str = ""
        self._today_initial_balance: float = 0.0
        self._realized_pnl_today: float = 0.0  # sum of closed-trade PnL today only

        # LSTM state buffer — matches environment.state_buffer
        self.lstm_enabled = self.agent.lstm_enabled
        self.sequence_lengths = [10, 20, 30]
        self.state_buffer: list = []

        # Portfolio tracking for portfolio feature vector (mirrors PositionManager internals)
        self._step_pnls: list = []       # realized PnL per closed trade (last 100)
        self._peak_balance: float = 0.0  # for drawdown computation

        # Win/loss counters (used for DB session logging)
        self.winning_trades = 0
        self.losing_trades = 0

        # Sync positions with Binance on startup
        print("🔄 Syncing positions with Binance...")
        try:
            actual_positions = self._sync_positions_with_binance()
            print(f"✅ Position sync complete: {len(actual_positions)} open positions")
            if actual_positions:
                print("   Open positions found on Binance:")
                for sym, pos in actual_positions.items():
                    print(f"     - {sym}: {pos['side']} {pos['size']} @ ${pos['entry_price']:.2f}")
        except Exception as e:
            print(f"⚠️  Initial position sync failed: {e}")
            print("   Continuing with empty position state...")

        print("\n🤖 Bot initialized successfully!\n")

    def _sync_positions_with_binance(self) -> dict:
        """
        Query Binance for actual open positions and reconcile with internal state.
        
        This is the source of truth. Call this:
        - On startup
        - Before every trading decision
        - After every order execution
        
        Returns:
            Dict of actual positions from Binance
        """
        try:
            # Query Binance futures positions
            binance_positions = self.client.futures_position_information()
            
            # Filter to only positions with non-zero size
            actual_positions = {}
            for pos in binance_positions:
                size = float(pos['positionAmt'])
                if abs(size) > 0.0001:  # Ignore dust
                    symbol = pos['symbol']
                    actual_positions[symbol] = {
                        'symbol': symbol,
                        'side': 'LONG' if size > 0 else 'SHORT',
                        'size': abs(size),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'leverage': int(pos['leverage']),
                    }
            
            # Get bot's internal positions
            internal_positions = self.executor.get_position_info()['positions']
            internal_symbols = set(internal_positions.keys())
            actual_symbols = set(actual_positions.keys())
            
            # Find discrepancies
            ghost_positions = internal_symbols - actual_symbols
            unknown_positions = actual_symbols - internal_symbols
            
            # Log and fix ghost positions (bot thinks it's open, but it's not)
            if ghost_positions:
                _log.warning(f"👻 Ghost positions detected (removing from internal state): {ghost_positions}")
                for symbol in ghost_positions:
                    # Remove from executor's internal tracking
                    if hasattr(self.executor, 'positions'):
                        self.executor.positions.pop(symbol, None)
                    # Remove from guardrails
                    self.guardrails.record_position_closed(symbol)
            
            # Log unknown positions (open on Binance but bot doesn't know)
            if unknown_positions:
                _log.warning(f"🔍 Unknown positions detected on Binance (adding to internal state): {unknown_positions}")
                # Add them to internal state
                for symbol in unknown_positions:
                    pos = actual_positions[symbol]
                    if hasattr(self.executor, 'positions'):
                        self.executor.positions[symbol] = pos
                    # Record in guardrails
                    self.guardrails.record_position_opened(symbol)
            
            # Update executor's internal state with Binance truth
            if hasattr(self.executor, 'positions'):
                self.executor.positions = actual_positions.copy()
            
            return actual_positions
            
        except Exception as e:
            _log.error(f"Position sync with Binance failed: {e}")
            # Return empty dict on failure — better to be cautious
            return {}

    def log_event(self, level: str, message: str, extra_data: dict = None):
        """Log event to file."""
        if level == "ERROR":
            _log.error(message, extra=extra_data or {})

    def _log_activity(self, message: str) -> None:
        """Append a timestamped event to the rolling activity log and persist to DB."""
        self._activity_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
        })
        if len(self._activity_log) > 50:
            self._activity_log = self._activity_log[-50:]
        # Persist to DB
        if self.current_session_key:
            self.logger.log_activity(self.current_session_key, message)

    def _record_close_to_db(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        portfolio_value: float,
        action_id: int = 0,
    ) -> None:
        """Record a position close to the DB trades table.

        Handles both cases: position was opened in this session (has a trade key
        in open_trade_keys) or was inherited from a prior session / Binance sync
        (no trade key — creates a standalone close record).
        """
        if not self.current_session_key:
            return
        try:
            trade_key = self.open_trade_keys.pop(symbol, None)
            if trade_key:
                self.logger.log_trade_close(
                    trade_key=trade_key,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                )
            else:
                # No matching open record — create a standalone close entry
                all_symbols = (
                    (self.current_coins or {}).get("gainers", [])
                    + (self.current_coins or {}).get("losers", [])
                )
                coin_rank = all_symbols.index(symbol) if symbol in all_symbols else 0
                is_gainer = coin_rank < 10
                open_count = self.executor.get_position_info()["count"]
                self.logger.log_trade_close_standalone(
                    session_key=self.current_session_key,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    portfolio_value=portfolio_value,
                    open_positions=open_count,
                    coin_rank=coin_rank,
                    is_gainer=is_gainer,
                )
        except Exception as e:
            _log.error(f"DB _record_close_to_db failed for {symbol}: {e}")

    def _write_metrics(
        self,
        step: int,
        current_balance: float,
        action_str: str = "",
        avg_q: float = 0.0,
        today_pnl: float = 0.0,
        record_history: bool = True,
    ) -> None:
        """Push live bot state to bot_heartbeat (DB) for the web dashboard."""
        try:
            winning = self.winning_trades
            losing = self.losing_trades
            g_status = self.guardrails.get_status()

            # PnL snapshot — one per step when record_history is True
            if record_history and self.current_session_key:
                pnl = float(current_balance - self.initial_balance)
                self.logger.log_pnl_snapshot(
                    self.current_session_key, float(current_balance), pnl,
                )

            # Build per-coin table for web dashboard
            coin_table = []
            try:
                if self.current_coins and self._last_market_features:
                    all_syms = (
                        self.current_coins.get("gainers", [])
                        + self.current_coins.get("losers", [])
                    )
                    q_vals = getattr(self.agent, "last_q_values", None)
                    ta_cols = [
                        "close", "rsi_14", "macd_histogram",
                        "bb_pctb_20", "volume_ratio", "atr_14", "adx_14",
                    ]
                    gainers_pct = self.current_coins.get("gainers_pct", [])
                    losers_pct = self.current_coins.get("losers_pct", [])
                    for coin_idx, symbol in enumerate(all_syms[:20]):
                        is_gainer = coin_idx < 10
                        local_idx = coin_idx % 10
                        # TA values
                        ta: dict = {}
                        df = self._last_market_features.get(symbol)
                        if df is not None:
                            for col in ta_cols:
                                try:
                                    val = float(df[col][-1]) if col in df.columns else None
                                    ta[col] = None if (val is None or val != val) else round(val, 6)
                                except Exception:
                                    ta[col] = None
                        # Q-values (masked actions return -1e9 from network)
                        q_hold = q_long = q_short = q_close = None
                        if q_vals is not None and len(q_vals) >= 63:
                            q_hold = round(float(q_vals[0]), 3)  # HOLD always valid
                            if is_gainer:
                                rl, rs, rc = (
                                    float(q_vals[1 + local_idx]),
                                    float(q_vals[11 + local_idx]),
                                    float(q_vals[21 + local_idx]),
                                )
                            else:
                                rl, rs, rc = (
                                    float(q_vals[31 + local_idx]),
                                    float(q_vals[41 + local_idx]),
                                    float(q_vals[51 + local_idx]),
                                )
                            q_long  = round(rl, 3) if rl  > -1e6 else None
                            q_short = round(rs, 3) if rs  > -1e6 else None
                            q_close = round(rc, 3) if rc  > -1e6 else None
                        # 24h change %
                        if is_gainer:
                            pct = float(gainers_pct[coin_idx]) if coin_idx < len(gainers_pct) else 0.0
                        else:
                            pct = float(losers_pct[local_idx]) if local_idx < len(losers_pct) else 0.0
                        coin_table.append({
                            "symbol": symbol,
                            "type": "gainer" if is_gainer else "loser",
                            "rank": local_idx,
                            "change_24h": round(pct, 2),
                            "ta": ta,
                            "q_hold": q_hold,
                            "q_long": q_long,
                            "q_short": q_short,
                            "q_close": q_close,
                        })
            except Exception as e:
                _log.warning(f"coin_table build failed: {e}")

            self.logger.update_heartbeat({
                "session_key": self.current_session_key,
                "status": "running",
                "session_started": self._session_started,
                "agent_status": "active",
                "epsilon": float(self.agent.epsilon),
                "current_step": step,
                "last_action": action_str,
                "avg_q": float(avg_q),
                "current_balance": float(current_balance),
                "initial_balance": float(self.initial_balance),
                "trade_count": self.daily_trades,
                "winning_trades": winning,
                "losing_trades": losing,
                "guardrail_status": {
                    "cooldown_remaining": int(g_status["cooldown_remaining"]),
                    "daily_trades": self.daily_trades,
                    "daily_pnl": float(current_balance - self.initial_balance),
                },
                "coin_table": coin_table,
                "screened_coins": self.current_coins,
                "trading_mode": self.trading_mode,
            })
        except Exception as e:
            _log.error(f"_write_metrics failed: {e}")

    def _retry(
        self, fn, *args, retries: int = 3, delay: float = 5.0, label: str = "", **kwargs
    ):
        """
        Call fn(*args, **kwargs) with automatic retries on failure.

        Args:
            fn:      The function to call
            retries: Max number of attempts (default 3)
            delay:   Seconds to wait between attempts, doubles each retry
            label:   Human-readable name shown on dashboard on failure

        Returns:
            Result of fn, or raises the last exception if all retries fail
        """
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                msg = (
                    f"{label or fn.__name__} failed (attempt {attempt}/{retries}): {e}"
                )
                _log.error(msg)
                if attempt < retries:
                    # Exponential backoff: 5s, 10s, 20s
                    wait = delay * (2 ** (attempt - 1))
                    time.sleep(wait)
                    # Re-sync Binance time offset before retry
                    try:
                        server_time = self.client.get_server_time()
                        self.client.timestamp_offset = server_time["serverTime"] - int(
                            time.time() * 1000
                        )
                    except Exception:
                        pass
        raise last_error

    def _decode_action(self, action: int) -> str:
        """Decode action number to human-readable string."""
        match action:
            case 0:
                return "HOLD"
            case n if 1 <= n <= 10:
                coin_idx = n - 1
                symbol = self.current_coins["gainers"][coin_idx]
                return f"LONG {symbol}"
            case n if 11 <= n <= 20:
                coin_idx = n - 11
                symbol = self.current_coins["gainers"][coin_idx]
                return f"SHORT {symbol}"
            case n if 21 <= n <= 30:
                coin_idx = n - 21
                symbol = self.current_coins["gainers"][coin_idx]
                return f"CLOSE {symbol}"
            case n if 31 <= n <= 40:
                coin_idx = n - 31
                symbol = self.current_coins["losers"][coin_idx]
                return f"LONG {symbol}"
            case n if 41 <= n <= 50:
                coin_idx = n - 41
                symbol = self.current_coins["losers"][coin_idx]
                return f"SHORT {symbol}"
            case n if 51 <= n <= 60:
                coin_idx = n - 51
                symbol = self.current_coins["losers"][coin_idx]
                return f"CLOSE {symbol}"
            case 61:
                return "CLOSE_ALL"
            case 62:
                return "CLOSE_WORST"
            case _:
                return f"UNKNOWN ({action})"

    def _execute_action(self, action: int) -> tuple[str, str, float]:
        """
        Execute the agent's chosen action with guardrails.

        Returns:
            (symbol, side, realized_pnl) -- used to record trade in dashboard
        """
        balance = self.executor.get_balance()
        current_positions = self.executor.get_position_info()["count"]

        match action:
            case 0:
                return "", "", 0.0

            case n if 1 <= n <= 10:
                symbol = self.current_coins["gainers"][n - 1]
                allowed, reason = self.guardrails.can_open_position(
                    symbol, current_positions
                )
                if not allowed:
                    return "", "", 0.0
                result = self.executor.open_long(symbol, balance)
                if result:
                    self.guardrails.record_position_opened(symbol)
                    return symbol, "LONG", 0.0
                return "", "", 0.0

            case n if 11 <= n <= 20:
                symbol = self.current_coins["gainers"][n - 11]
                allowed, reason = self.guardrails.can_open_position(
                    symbol, current_positions
                )
                if not allowed:
                    return "", "", 0.0
                result = self.executor.open_short(symbol, balance)
                if result:
                    self.guardrails.record_position_opened(symbol)
                    return symbol, "SHORT", 0.0
                return "", "", 0.0

            case n if 21 <= n <= 30:
                symbol = self.current_coins["gainers"][n - 21]
                allowed, reason = self.guardrails.can_close_position(symbol)
                if not allowed:
                    return "", "", 0.0
                result = self.executor.close_position(symbol)
                if result:
                    self.guardrails.record_position_closed(symbol)
                    self._last_close_exit_prices[symbol] = result.get("exit_price", 0.0)
                    self._recent_trades.append({
                        "symbol": symbol,
                        "side": result["side"],
                        "entry_price": round(result.get("entry_price", 0.0), 6),
                        "exit_price": round(result.get("exit_price", 0.0), 6),
                        "pnl": round(result["pnl"], 4),
                        "pnl_pct": round(result.get("pnl_pct", 0.0), 2),
                        "timestamp": datetime.now(WIB).isoformat(),
                    })
                    if len(self._recent_trades) > 20:
                        self._recent_trades = self._recent_trades[-20:]
                    return symbol, result["side"], result["pnl"]
                return "", "", 0.0

            case n if 31 <= n <= 40:
                symbol = self.current_coins["losers"][n - 31]
                allowed, reason = self.guardrails.can_open_position(
                    symbol, current_positions
                )
                if not allowed:
                    return "", "", 0.0
                result = self.executor.open_long(symbol, balance)
                if result:
                    self.guardrails.record_position_opened(symbol)
                    return symbol, "LONG", 0.0
                return "", "", 0.0

            case n if 41 <= n <= 50:
                symbol = self.current_coins["losers"][n - 41]
                allowed, reason = self.guardrails.can_open_position(
                    symbol, current_positions
                )
                if not allowed:
                    return "", "", 0.0
                result = self.executor.open_short(symbol, balance)
                if result:
                    self.guardrails.record_position_opened(symbol)
                    return symbol, "SHORT", 0.0
                return "", "", 0.0

            case n if 51 <= n <= 60:
                symbol = self.current_coins["losers"][n - 51]
                allowed, reason = self.guardrails.can_close_position(symbol)
                if not allowed:
                    return "", "", 0.0
                result = self.executor.close_position(symbol)
                if result:
                    self.guardrails.record_position_closed(symbol)
                    self._last_close_exit_prices[symbol] = result.get("exit_price", 0.0)
                    self._recent_trades.append({
                        "symbol": symbol,
                        "side": result["side"],
                        "entry_price": round(result.get("entry_price", 0.0), 6),
                        "exit_price": round(result.get("exit_price", 0.0), 6),
                        "pnl": round(result["pnl"], 4),
                        "pnl_pct": round(result.get("pnl_pct", 0.0), 2),
                        "timestamp": datetime.now(WIB).isoformat(),
                    })
                    if len(self._recent_trades) > 20:
                        self._recent_trades = self._recent_trades[-20:]
                    return symbol, result["side"], result["pnl"]
                return "", "", 0.0

            case 61:
                # CLOSE_ALL bypasses guardrails (emergency exit)
                # Close each position individually so we capture PnL
                positions_snapshot = dict(self.executor.positions)
                now = datetime.now(WIB).isoformat()
                for sym, pos in positions_snapshot.items():
                    result = self.executor.close_position(sym)
                    if result:
                        self.guardrails.record_position_closed(sym)
                        self._log_activity(f"CLOSE_ALL: {sym} ({pos['side']}) closed → PnL {result['pnl']:+.2f}")
                        self._recent_trades.append({
                            "symbol": sym,
                            "side": result["side"],
                            "entry_price": round(result.get("entry_price", 0.0), 6),
                            "exit_price": round(result.get("exit_price", 0.0), 6),
                            "pnl": round(result["pnl"], 4),
                            "pnl_pct": round(result.get("pnl_pct", 0.0), 2),
                            "timestamp": now,
                        })
                        self._record_close_to_db(
                            symbol=sym,
                            side=result.get("side", pos["side"]),
                            entry_price=result.get("entry_price", pos.get("entry_price", 0.0)),
                            exit_price=result.get("exit_price", 0.0),
                            pnl=result["pnl"],
                            pnl_pct=result.get("pnl_pct", 0.0),
                            portfolio_value=balance,
                            action_id=61,
                        )
                    else:
                        self._log_activity(f"CLOSE_ALL: {sym} ({pos['side']}) FAILED to close")
                for symbol in list(self.guardrails.position_history.keys()):
                    self.guardrails.record_position_closed(symbol)
                if len(self._recent_trades) > 20:
                    self._recent_trades = self._recent_trades[-20:]
                return "ALL", "CLOSE", 0.0

            case 62:
                # CLOSE_WORST
                positions = self.executor.positions
                if not positions:
                    return "", "", 0.0

                worst_symbol = None
                worst_pnl = float("inf")

                for symbol, pos in positions.items():
                    try:
                        current_price = self.executor.get_current_price(symbol)
                        entry = pos["entry_price"]
                        size = pos["size"]
                        if pos["side"] == "LONG":
                            unpnl = (current_price - entry) * size
                        else:
                            unpnl = (entry - current_price) * size
                    except Exception as e:
                        _log.warning(
                            f"Failed to get price for {symbol} during CLOSE_WORST: {e}"
                        )
                        unpnl = 0.0  # assume breakeven so we still consider it

                    if unpnl < worst_pnl:
                        worst_pnl = unpnl
                        worst_symbol = symbol

                if worst_symbol:
                    result = self.executor.close_position(worst_symbol)
                    if result:
                        self.guardrails.record_position_closed(worst_symbol)
                        self._last_close_exit_prices[worst_symbol] = result.get(
                            "exit_price", 0.0
                        )
                        self._recent_trades.append({
                            "symbol": worst_symbol,
                            "side": result["side"],
                            "entry_price": round(result.get("entry_price", 0.0), 6),
                            "exit_price": round(result.get("exit_price", 0.0), 6),
                            "pnl": round(result["pnl"], 4),
                            "pnl_pct": round(result.get("pnl_pct", 0.0), 2),
                            "timestamp": datetime.now(WIB).isoformat(),
                        })
                        if len(self._recent_trades) > 20:
                            self._recent_trades = self._recent_trades[-20:]
                        return worst_symbol, result["side"], result["pnl"]
                return "", "", 0.0

            case _:
                return "", "", 0.0

    def _build_state(
        self,
        market_features: dict,
        current_balance: float,
        step: int = 0,
        pnl_today: float = 0.0,
    ) -> np.ndarray:
        """Build state vector matching training environment exactly."""
        initial_balance = 10000.0  # Must match training config
        max_positions = 3

        all_symbols = self.current_coins["gainers"] + self.current_coins["losers"]

        # --- Market Features (20 × num_features) ---
        # Use the same feature column list as the environment (_get_market_features)
        market_features_dict = {}
        for coin_idx, symbol in enumerate(all_symbols):
            if symbol in market_features:
                df = market_features[symbol]
                values = [
                    float(df[col][-1]) if col in df.columns else 0.0
                    for col in self._feature_names
                ]
                market_features_dict[coin_idx] = np.array(values, dtype=np.float32)

        # --- Position Features (20, 8) — matches PositionManager.get_position_features() ---
        position_features = np.zeros((20, 8), dtype=np.float32)
        positions = self.executor.get_position_info()["positions"]
        total_portfolio = max(current_balance, 1.0)

        for coin_idx, symbol in enumerate(all_symbols):
            if symbol in positions:
                pos = positions[symbol]
                current_price = self.executor.get_current_price(symbol)
                entry = pos["entry_price"]
                size_usdt = pos["size"] * entry

                if pos["side"] == "LONG":
                    unrealized_pnl = (current_price - entry) / entry * size_usdt
                else:
                    unrealized_pnl = (entry - current_price) / entry * size_usdt

                opened_step = self.guardrails.position_history.get(symbol, {}).get(
                    "opened_step", self.guardrails.current_step
                )
                steps_held = self.guardrails.current_step - opened_step

                position_features[coin_idx, 0] = 1.0
                position_features[coin_idx, 1] = 1.0 if pos["side"] == "LONG" else -1.0
                position_features[coin_idx, 2] = size_usdt / initial_balance
                position_features[coin_idx, 3] = unrealized_pnl / size_usdt if size_usdt > 0 else 0.0
                position_features[coin_idx, 4] = (current_price - entry) / entry if entry > 0 else 0.0
                position_features[coin_idx, 5] = min(steps_held / 88.0, 1.0)
                position_features[coin_idx, 6] = min(steps_held / 288.0, 1.0)   # [6] hold_duration/day
                position_features[coin_idx, 7] = size_usdt / total_portfolio      # [7] size_fraction

        # --- Portfolio Features (8,) — matches PositionManager.get_portfolio_features() ---
        num_positions = len(positions)
        total_unrealized = sum(
            (
                (self.executor.get_current_price(sym) - pos["entry_price"])
                / pos["entry_price"]
                * pos["size"]
                * pos["entry_price"]
                if pos["side"] == "LONG"
                else (pos["entry_price"] - self.executor.get_current_price(sym))
                / pos["entry_price"]
                * pos["size"]
                * pos["entry_price"]
            )
            for sym, pos in positions.items()
        ) if positions else 0.0

        if len(self._step_pnls) >= 3:
            pnls_arr = np.array(self._step_pnls[-20:])
            rolling_sharpe = float(np.clip(np.mean(pnls_arr) / (np.std(pnls_arr) + 1e-8), -3.0, 3.0))
            port_vol = float(np.clip(np.std(pnls_arr) / initial_balance, 0.0, 1.0))
        else:
            rolling_sharpe = 0.0
            port_vol = 0.0

        drawdown = float(np.clip(
            (self._peak_balance - current_balance) / max(self._peak_balance, 1.0), 0.0, 1.0
        )) if self._peak_balance > 0 else 0.0

        portfolio_features = np.array([
            current_balance / initial_balance,   # [0] portfolio value
            current_balance / initial_balance,   # [1] cash (approx — no separate tracking)
            num_positions / max_positions,        # [2] slots used
            total_unrealized / initial_balance,   # [3] total unrealized PnL
            0.0,                                  # [4] fees (not tracked separately live)
            rolling_sharpe,                       # [5] rolling Sharpe
            drawdown,                             # [6] max drawdown pct
            port_vol,                             # [7] portfolio volatility
        ], dtype=np.float32)

        # --- Coin Metadata (20 × 6) ---
        coin_metadata = {}
        for coin_idx in range(20):
            is_gainer = coin_idx < 10
            rank = coin_idx % 10
            daily_change = (
                self.current_coins["gainers_pct"][coin_idx]
                if is_gainer
                else self.current_coins["losers_pct"][coin_idx - 10]
            )
            coin_metadata[coin_idx] = {
                "is_gainer": is_gainer,
                "rank": rank,
                "daily_change": daily_change,
            }

        # --- Regime Features (8,) — computed from live market data ---
        regime_features = self._compute_regime_features(market_features, step)

        # --- Guardrail Features (5,) — current constraint state ---
        guardrail_features = self._compute_guardrail_features(current_balance, pnl_today)

        # --- Time (candles_per_day=288, warmup=0) ---
        total_steps = 288
        current_relative_step = min(step, total_steps)

        return self.state_builder.build_state(
            market_features=market_features_dict,
            position_features=position_features,
            portfolio_features=portfolio_features,
            coin_metadata=coin_metadata,
            regime_features=regime_features,
            guardrail_features=guardrail_features,
            current_step=current_relative_step,
            total_steps=total_steps,
        )

    def _get_enriched_positions(self) -> dict:
        """Get positions with live unrealized PnL for dashboard display."""
        positions = self.executor.get_position_info()["positions"]
        enriched = {}
        for sym, pos in positions.items():
            current_price = self.executor.get_current_price(sym)
            entry = pos["entry_price"]
            size_usdt = pos["size"] * entry
            if pos["side"] == "LONG":
                upnl = (current_price - entry) / entry * size_usdt
            else:
                upnl = (entry - current_price) / entry * size_usdt
            enriched[sym] = {**pos, "unrealized_pnl": upnl}
        return enriched

    def _compute_regime_features(self, market_features: dict, step: int) -> np.ndarray:
        """
        Compute 8 regime features from live market data.
        Mirrors environment._compute_regime_features() but driven by live DataFrames
        (symbol → polars DataFrame with all TA columns) instead of episode_data.
        """
        vol_lb = 24  # volatility lookback candles (matches config regime.volatility_lookback)

        all_returns: list = []
        all_adx: list = []
        step_returns: list = []

        for df in market_features.values():
            n_rows = len(df)
            if n_rows < 2:
                continue

            closes = df["close"].to_numpy().astype(np.float64)

            # Rolling vol: last vol_lb candles
            start = max(0, n_rows - vol_lb)
            window_closes = closes[start:]
            if len(window_closes) >= 2:
                rets = np.diff(np.log(window_closes + 1e-8))
                all_returns.extend(rets.tolist())

            # ADX from latest row
            if "adx_14" in df.columns:
                adx_val = float(df["adx_14"][-1])
                if not np.isnan(adx_val):
                    all_adx.append(adx_val / 100.0)

            # Step return (last candle vs previous)
            if n_rows >= 2:
                step_returns.append(float(closes[-1]) / float(closes[-2]) - 1)

        all_returns_arr = np.array(all_returns) if all_returns else np.array([0.0])

        # 1. Average ADX (trend strength)
        avg_adx = float(np.mean(all_adx)) if all_adx else 0.0

        # 2. Realized volatility, normalized vs typical 0.005 per 5m candle
        realized_vol = float(np.std(all_returns_arr)) if len(all_returns_arr) > 1 else 0.0
        vol_normalized = float(np.clip(realized_vol / 0.005, 0, 5) / 5.0)

        # 3. Market breadth — fraction of coins with positive step return
        breadth = float(np.mean([r > 0 for r in step_returns])) if step_returns else 0.5

        # 4. Average absolute return this step (session heat)
        avg_abs_return = float(np.clip(
            np.mean(np.abs(step_returns)) / 0.02, 0, 1
        )) if step_returns else 0.0

        # 5. Trend consistency — fraction of returns matching overall direction
        overall_sign = np.sign(np.mean(all_returns_arr)) if len(all_returns_arr) > 0 else 0
        consistency = float(np.mean(
            np.sign(all_returns_arr) == overall_sign
        )) if len(all_returns_arr) > 0 else 0.5

        # 6. Vol ratio: recent window vs full-history vol
        full_returns: list = []
        for df in market_features.values():
            closes = df["close"].to_numpy().astype(np.float64)
            if len(closes) >= 2:
                full_returns.extend(np.diff(np.log(closes + 1e-8)).tolist())
        full_vol = float(np.std(full_returns)) if len(full_returns) > 1 else realized_vol
        vol_ratio = float(np.clip(realized_vol / (full_vol + 1e-8), 0, 3) / 3.0)

        # 7. Intraday momentum persistence (early-move retention)
        persistence = 0.5
        if market_features:
            df = next(iter(market_features.values()))
            n_rows = len(df)
            closes = df["close"].to_numpy().astype(np.float64)
            check_step = 30
            if n_rows > check_step + 1:
                early_move = closes[check_step] / (closes[0] + 1e-8) - 1
                recent_move = closes[-1] / (closes[0] + 1e-8) - 1
                if abs(early_move) > 1e-8:
                    persistence = float(np.clip(recent_move / early_move, -1, 2) / 2.0 + 0.5)

        # 8. Fraction of trading day elapsed
        day_progress = float(np.clip(step / 288.0, 0.0, 1.0))

        return np.array([
            avg_adx, vol_normalized, breadth, avg_abs_return,
            consistency, vol_ratio, float(np.clip(persistence, 0, 1)), day_progress,
        ], dtype=np.float32)

    def _compute_guardrail_features(
        self, current_balance: float, pnl_today: float
    ) -> np.ndarray:
        """
        Compute 5 guardrail state features.
        Mirrors environment._compute_guardrail_features() using live GuardrailManager state.
        """
        positions = self.executor.get_position_info()["positions"]
        num_positions = len(positions)

        # 1. Cooldown remaining — always 0 (removed from live guardrails;
        #    agent learned trading frequency via reward shaping during training)
        cooldown_norm = 0.0

        # 2. Average hold remaining — always 0 (same reason)
        avg_hold_remaining = 0.0

        # 3. Position slots used
        slots_used = num_positions / max(self.max_positions, 1)

        # 4. Daily trades used
        trades_norm = self.daily_trades / max(self.max_daily_trades, 1)

        # 5. Daily loss used (negative pnl_today / max loss)
        loss_norm = float(np.clip(max(0.0, -self._realized_pnl_today) / max(self._daily_loss_limit, 1.0), 0.0, 1.0))

        return np.array([
            float(np.clip(cooldown_norm,      0, 1)),
            float(np.clip(avg_hold_remaining, 0, 1)),
            float(np.clip(slots_used,         0, 1)),
            float(np.clip(trades_norm,        0, 1)),
            float(np.clip(loss_norm,          0, 1)),
        ], dtype=np.float32)

    def _build_action_mask(self) -> np.ndarray:
        """
        Build a (63,) boolean action mask matching environment.get_action_mask()
        with num_coins_per_side=10, num_sizes=1.

        Action layout (n=10, s=1):
          0      : HOLD
          1-10   : LONG  gainer[0-9]
          11-20  : SHORT gainer[0-9]
          21-30  : CLOSE gainer[0-9]
          31-40  : LONG  loser[0-9]
          41-50  : SHORT loser[0-9]
          51-60  : CLOSE loser[0-9]
          61     : CLOSE_ALL
          62     : CLOSE_WORST
        """
        n = 10
        mask = np.zeros(self.ACTION_DIM, dtype=bool)
        mask[0] = True  # HOLD always valid

        positions = self.executor.get_position_info()["positions"]
        num_positions = len(positions)
        status = self.guardrails.get_status()
        cooldown_remaining = status["cooldown_remaining"]
        position_ages = status["position_ages"]  # {symbol: steps_held}

        # Circuit breaker: if daily loss >= max, only HOLD/CLOSE allowed
        pnl_today = (self.current_balance or 0.0) - self.initial_balance
        circuit_breaker = (self.initial_balance > 0) and (-self._realized_pnl_today >= self._daily_loss_limit)
        trade_limit_hit = self.daily_trades >= self.max_daily_trades

        can_open = (
            not circuit_breaker
            and not trade_limit_hit
            and num_positions < self.max_positions
            and cooldown_remaining == 0
        )

        all_symbols = self.current_coins["gainers"] + self.current_coins["losers"]

        for coin_idx, symbol in enumerate(all_symbols):
            is_gainer = coin_idx < n
            local_idx = coin_idx % n

            # Open actions
            if can_open and symbol not in positions:
                if is_gainer:
                    mask[1 + local_idx] = True   # LONG gainer
                    mask[11 + local_idx] = True  # SHORT gainer
                else:
                    mask[31 + local_idx] = True  # LONG loser
                    mask[41 + local_idx] = True  # SHORT loser

            # Close actions — always allowed (no min-hold in live trading)
            if symbol in positions:
                if is_gainer:
                    mask[21 + local_idx] = True  # CLOSE gainer
                else:
                    mask[51 + local_idx] = True  # CLOSE loser

        # CLOSE_ALL (61) and CLOSE_WORST (62)
        if positions:
            mask[61] = True
            mask[62] = True

        return mask

    def _sync_positions_with_binance(self) -> dict:
        """Query Binance for actual open positions and reconcile with internal state.

        Detects and fixes two types of divergence:
          - Ghost positions: bot thinks it's open, Binance says it's closed
          - Unknown positions: open on Binance but bot has no record

        Returns the live Binance positions dict (symbol → position info).
        """
        binance_positions = self.client.futures_position_information()
        actual: dict = {}
        for pos in binance_positions:
            amt = float(pos["positionAmt"])
            if abs(amt) > 0.0001:
                sym = pos["symbol"]
                actual[sym] = {
                    "side": "LONG" if amt > 0 else "SHORT",
                    "size": abs(amt),
                    "entry_price": float(pos["entryPrice"]),
                    "unrealized_pnl": float(pos["unRealizedProfit"]),
                    "leverage": int(pos["leverage"]),
                }

        internal = set(self.executor.positions.keys())
        actual_syms = set(actual.keys())

        # Ghost positions — remove from executor and guardrails
        for sym in internal - actual_syms:
            _log.warning(f"👻 Ghost position removed: {sym}")
            self.executor.positions.pop(sym, None)
            self.guardrails.record_position_closed(sym)

        # Unknown positions — add to executor and guardrails
        for sym in actual_syms - internal:
            _log.warning(f"🔍 Unknown position adopted: {sym}")
            self.executor.positions[sym] = actual[sym]
            if sym not in self.guardrails.position_history:
                self.guardrails.record_position_opened(sym)

        return actual

    def _wait_for_next_interval(self, reason: str = ""):
        """Sit out the rest of the current 5-minute interval."""
        if reason:
            _log.info(f"Waiting for next interval: {reason}")
        now = datetime.now(WIB)
        seconds_into_interval = (now.minute % 5) * 60 + now.second
        seconds_to_wait = max(300 - seconds_into_interval, 10)
        time.sleep(seconds_to_wait)

    def run(self):
        """Main trading loop."""
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

        last_screening_time = None
        step = 0

        while True:
            try:
                now_wib = datetime.now(WIB)
                current_date = now_wib.date()
                current_hour = now_wib.hour

                # -- Morning Screening -----------------------------------------
                is_7am_window = current_hour == 7
                already_screened_this_hour = (
                    last_screening_time is not None
                    and now_wib - last_screening_time < timedelta(hours=1)
                )
                should_screen = last_screening_time is None or (
                    is_7am_window and not already_screened_this_hour
                )

                if should_screen:
                    # Close all before re-screening (not first run)
                    if last_screening_time is not None:
                        # Sync with Binance first to remove ghost positions
                        # that were liquidated or closed externally overnight.
                        try:
                            self._sync_positions_with_binance()
                        except Exception as e:
                            _log.warning(f"Pre-close sync failed: {e}")

                        for sym in list(self.executor.positions.keys()):
                            pos = self.executor.positions.get(sym)
                            if pos is None:
                                continue
                            # Retry up to 3 times — 07:00 WIB = 00:00 UTC
                            # (Binance daily settlement) can cause transient rejects.
                            result = None
                            for attempt in range(3):
                                result = self.executor.close_position(sym)
                                if result is not None:
                                    break
                                if attempt < 2:
                                    time.sleep(5)
                            if result:
                                self._log_activity(
                                    f"Re-screening: closed {sym} ({pos['side']})"
                                    + f" → PnL {result['pnl']:+.2f}"
                                )
                                self._record_close_to_db(
                                    symbol=sym,
                                    side=result.get("side", pos["side"]),
                                    entry_price=result.get("entry_price", pos.get("entry_price", 0.0)),
                                    exit_price=result.get("exit_price", 0.0),
                                    pnl=result["pnl"],
                                    pnl_pct=result.get("pnl_pct", 0.0),
                                    portfolio_value=self.initial_balance,
                                )
                            else:
                                self._log_activity(
                                    f"Re-screening: FAILED to close {sym}"
                                )
                                # Force-remove from cache so the bot doesn't
                                # keep a stale position forever.
                                self.executor.positions.pop(sym, None)
                        for sym in list(self.guardrails.position_history.keys()):
                            self.guardrails.record_position_closed(sym)

                    # Screen top movers -- retry 3x, keep old list if all attempts fail
                    try:
                        new_coins = self._retry(
                            get_top_movers,
                            self.client,
                            retries=3,
                            delay=10.0,
                            label="Screening",
                        )
                        self.current_coins = new_coins
                    except Exception:
                        # If first run and screening fails, can't trade -- wait and retry
                        if self.current_coins is None:
                            self._wait_for_next_interval("Screening failed -- retrying")
                            continue
                        # Otherwise keep yesterday's list and log it
                        _log.error(
                            "Screening failed after 3 attempts -- keeping previous coin list"
                        )

                    # Reset counters
                    self.daily_trades = 0
                    self.daily_pnl = 0.0
                    self.guardrails.reset_daily()
                    self._step_pnls = []
                    self._peak_balance = 0.0
                    self.state_buffer = []
                    self._pnl_history = []
                    self.winning_trades = 0
                    self.losing_trades = 0

                    # Get initial balance with retry.
                    # Use get_equity() (wallet + unrealized) so the baseline
                    # matches current_balance in the step loop, preventing
                    # inherited unrealized losses from immediately triggering
                    # the daily loss limit on restart.
                    try:
                        self.initial_balance = self._retry(
                            self.executor.get_equity,
                            retries=3,
                            delay=5.0,
                            label="get_equity (screening)",
                        )
                    except Exception:
                        self._wait_for_next_interval("Balance fetch failed")
                        continue

                    # Recompute dollar loss limit from current equity each session
                    self._daily_loss_limit = self.initial_balance * self.max_daily_loss_pct / 100.0
                    _log.info(
                        f"Daily loss limit: {self.max_daily_loss_pct}% of "
                        f"${self.initial_balance:.2f} = ${self._daily_loss_limit:.2f}"
                    )

                    # Close previous session in DB (if not first run)
                    if self.current_session_key:
                        self.logger.close_session(
                            self.current_session_key,
                            final_balance=self.initial_balance,
                            total_pnl=self.daily_pnl,
                            total_trades=self.daily_trades,
                            winning_trades=self.winning_trades,
                            losing_trades=self.losing_trades,
                        )

                    # Open new session in DB
                    self.current_session_key = self.logger.open_session(
                        mode=self.trading_mode,
                        initial_balance=self.initial_balance,
                        screened_coins=self.current_coins,
                    )
                    self._dashboard_log_handler.set_logger(self.logger, self.current_session_key)
                    self.open_trade_keys = {}
                    self._session_started = now_wib.isoformat()

                    last_screening_time = now_wib
                    step = 0

                    gainers_str = " ".join(self.current_coins["gainers"][:10])
                    losers_str = " ".join(self.current_coins["losers"][:10])
                    self._log_activity(
                        f"Screened — gainers: {gainers_str} | losers: {losers_str}"
                    )

                    # Handle positions synced from Binance at startup.
                    # Positions not in the new screened list are invisible to the
                    # agent (no state features, no close actions) — close them now.
                    # Positions that ARE in the list get registered with guardrails
                    # so the agent sees them correctly in the state.
                    all_screened = set(
                        self.current_coins["gainers"] + self.current_coins["losers"]
                    )
                    now_iso = now_wib.isoformat()
                    for sym, pos in list(self.executor.positions.items()):
                        if sym not in all_screened:
                            result = self.executor.close_position(sym)
                            self._log_activity(
                                f"Startup: closed {sym} ({pos['side']}) — not in screened list"
                                + (f" → PnL {result['pnl']:+.2f}" if result else " (close failed)")
                            )
                            if result:
                                self._recent_trades.append({
                                    "symbol": sym,
                                    "side": pos["side"],
                                    "entry_price": round(pos.get("entry_price", 0.0), 6),
                                    "exit_price": round(result.get("exit_price", 0.0), 6),
                                    "pnl": round(result["pnl"], 4),
                                    "pnl_pct": round(result.get("pnl_pct", 0.0), 2),
                                    "timestamp": now_iso,
                                })
                        else:
                            # Screened position: register with guardrails at step 0
                            # so min-hold and cooldown apply correctly going forward
                            if sym not in self.guardrails.position_history:
                                self.guardrails.record_position_opened(sym)
                    if len(self._recent_trades) > 20:
                        self._recent_trades = self._recent_trades[-20:]

                    self._write_metrics(0, self.initial_balance, "Screened")

                # -- Step ------------------------------------------------------
                step += 1
                self.guardrails.step()

                # Sync positions with Binance BEFORE every decision to catch
                # ghost / unknown positions from previous steps or manual trades.
                try:
                    self._sync_positions_with_binance()
                except Exception as e:
                    _log.warning(f"Pre-step position sync failed: {e}")

                # Signal the dashboard that we are working on this step.
                # record_history=False: balance is stale here (equity not fetched yet),
                # so don't add a misleading $0.00 point to pnl_history.
                self._write_metrics(step, self.initial_balance, f"Step {step}: fetching data…",
                                    record_history=False)

                # Get equity (wallet + unrealized) for each step.
                # initial_balance uses get_balance() (wallet only) so that
                # session_pnl = equity_now - wallet_at_start = unrealized + closed_trade_pnl
                try:
                    current_balance = self._retry(
                        self.executor.get_equity,
                        retries=3,
                        delay=5.0,
                        label="get_equity",
                    )
                except Exception:
                    self._wait_for_next_interval("Balance fetch failed -- skipping step")
                    continue

                self.current_balance = current_balance  # keep instance var in sync for _build_action_mask
                pnl_today = current_balance - self.initial_balance

                # Today's PnL: resets at midnight using same wallet baseline as session.
                # _today_initial_balance holds wallet at start-of-day (not equity),
                # so today_pnl = equity_now - wallet_at_day_start = unrealized + realized.
                today_str = now_wib.strftime("%Y-%m-%d")
                if today_str != self._today_date:
                    self._today_date = today_str
                    self._today_initial_balance = self.initial_balance
                    self._realized_pnl_today = 0.0  # fresh day — reset realized counter
                today_pnl = current_balance - self._today_initial_balance

                # Track peak balance for drawdown computation
                if self._peak_balance <= 0:
                    self._peak_balance = current_balance
                else:
                    self._peak_balance = max(self._peak_balance, current_balance)

                # -- Daily Loss Limit ------------------------------------------
                if self._realized_pnl_today <= -self._daily_loss_limit:
                    now_iso = now_wib.isoformat()
                    # Close each position individually and log actual outcome
                    for sym in list(self.executor.positions.keys()):
                        pos = self.executor.positions.get(sym)
                        if pos is None:
                            continue
                        result = self.executor.close_position(sym)
                        if result:
                            self._log_activity(
                                f"Daily loss limit hit: closed {sym} ({pos['side']})"
                                + f" → PnL {result['pnl']:+.2f}"
                            )
                            self._recent_trades.append({
                                "symbol": sym,
                                "side": pos["side"],
                                "entry_price": round(pos.get("entry_price", 0.0), 6),
                                "exit_price": round(result.get("exit_price", 0.0), 6),
                                "pnl": round(result["pnl"], 4),
                                "pnl_pct": round(result.get("pnl_pct", 0.0), 2),
                                "timestamp": now_iso,
                            })
                            self._record_close_to_db(
                                symbol=sym,
                                side=result.get("side", pos["side"]),
                                entry_price=result.get("entry_price", pos.get("entry_price", 0.0)),
                                exit_price=result.get("exit_price", 0.0),
                                pnl=result["pnl"],
                                pnl_pct=result.get("pnl_pct", 0.0),
                                portfolio_value=current_balance,
                            )
                        else:
                            self._log_activity(
                                f"Daily loss limit hit: FAILED to close {sym} ({pos['side']})"
                            )
                    if len(self._recent_trades) > 20:
                        self._recent_trades = self._recent_trades[-20:]
                    self._write_metrics(
                        step, current_balance,
                        f"Daily loss limit hit (${pnl_today:.2f}) — closed all positions",
                        0.0, pnl_today,
                    )
                    # Wait until next 7 AM — heartbeat every 5 min so the
                    # dashboard shows the bot is alive, just benched for the day.
                    _heartbeat_msgs = [
                        "🛌 Sleeping it off… back at 7 AM",
                        "☕ Taking a breather… markets will still be there tomorrow",
                        "🧘 Meditating on today's losses… resumes at 7 AM",
                        "🌙 Lights out for today — see you at dawn",
                        "🤕 Licking wounds… trading resumes at 7 AM",
                        "📵 Do Not Disturb — daily limit reached, resuming at 7 AM",
                        "🔋 Recharging… back in action at 7 AM",
                        "🎯 Tomorrow is another day — standing by until 7 AM",
                        "💤 Zzzz… the market can wait",
                        "🏖️ On vacation (involuntarily) — back at 7 AM",
                        "🍵 Brewing tea and regretting life choices…",
                        "🐢 Slow and steady… but first, a nap",
                        "🎮 Playing video games until the market reopens",
                        "🌊 Riding out the storm — resumes at 7 AM",
                        "🙃 Everything is fine. Totally fine. Resuming at 7 AM",
                        "📉 What goes down must come up… probably… at 7 AM",
                        "🤖 Bot.exe has stopped trading and is now coping",
                        "🧊 Keeping it cool after a rough session",
                        "🦥 Maximum chill mode activated — until 7 AM",
                        "🪫 Running on empty — recharging overnight",
                        "🌚 Gone dark for the night. See you at sunrise",
                        "🎵 Playing sad violin music until 7 AM",
                        "🍕 Stress-eating virtual pizza until morning",
                        "🔕 Muted. Daily limit hit. Not taking calls",
                        "💸 Watching the PnL like a horror movie… paused at 7 AM",
                        "🧯 Fire's out. Bot is cooling down until dawn",
                        "🪞 Reflecting on decisions. Will do better tomorrow",
                        "⛔ No more trades today. The market has been grounded",
                        "🎲 Lesson learned: not every candle is a friend",
                        "🌛 Goodnight, cruel market — see you at 7 AM",
                        "🐣 Going back into the egg until morning",
                        "🏳️ Waving the white flag for today — back at 7 AM",
                        "🚫 Closed for emotional damage. Reopens at 7 AM",
                        "⏸️ Trading paused. Dignity partially intact",
                        "🧠 Running post-mortem analysis… aka lying in bed thinking",
                        "😮‍💨 Exhaling deeply… resuming at 7 AM",
                        "🛸 Beaming out of this dimension until 7 AM",
                        "🌴 Out of office — daily loss limit reached",
                        "🔐 Wallet locked for its own protection until dawn",
                        "🐻 Hibernating until the bulls show up at 7 AM",
                        "🌀 In the void… will return at 7 AM",
                        "⚰️ Today's PnL RIP. Tomorrow is a new chapter",
                        "🤞 Fingers crossed for tomorrow — standing by until 7 AM",
                        "🦔 Curled up in a ball until market hours",
                        "🎬 That's a wrap on today — scene resumes at 7 AM",
                        "🍀 Saving all the luck for tomorrow",
                        "🧸 Hugging the stop-loss plushie until morning",
                        "💡 Idea: maybe trade less tomorrow. Thinking about it until 7 AM",
                        "🌤️ Waiting for clearer skies at 7 AM",
                        "📖 Reading 'How to Not Lose Money' until dawn",
                    ]
                    while True:
                        time.sleep(300)
                        now = datetime.now(WIB)
                        resume_in = f"resumes at 07:00 ({(7 - now.hour) % 24}h away)"
                        msg = random.choice(_heartbeat_msgs)
                        self.logger.update_heartbeat({
                            "session_key": self.current_session_key,
                            "status": "sleeping",
                            "last_action": f"{msg} — {resume_in}",
                            "current_balance": float(current_balance),
                            "initial_balance": float(self.initial_balance),
                            "current_step": step,
                            "trade_count": self.daily_trades,
                            "winning_trades": self.winning_trades,
                            "losing_trades": self.losing_trades,
                            "trading_mode": self.trading_mode,
                        })
                        if now.hour == 7 and (
                            last_screening_time is None
                            or now - last_screening_time > timedelta(hours=1)
                        ):
                            break
                    continue

                # -- Market Data -----------------------------------------------
                all_symbols = (
                    self.current_coins["gainers"] + self.current_coins["losers"]
                )
                try:
                    market_data = self._retry(
                        fetch_all_coins,
                        self.client,
                        all_symbols,
                        interval="5m",
                        limit=300,
                        retries=1,
                        delay=5.0,
                        label="fetch_all_coins",
                    )
                except Exception:
                    # Can't trade without market data -- HOLD this step
                    self._wait_for_next_interval("Market data fetch failed -- HOLD")
                    continue

                # -- Features --------------------------------------------------
                market_features = {}
                for symbol, df in market_data.items():
                    try:
                        market_features[symbol] = self.feature_calculator.calculate(
                            df
                        )
                    except Exception as e:
                        # One bad coin doesn't stop the rest
                        _log.error(f"Feature calculation failed for {symbol}: {e}")
                self._last_market_features = market_features

                # -- Build State -----------------------------------------------
                try:
                    state = self._build_state(
                        market_features, current_balance, step, pnl_today
                    )
                except Exception as e:
                    _log.error(f"State build failed: {e}")
                    self._wait_for_next_interval("State build failed -- HOLD")
                    continue

                # Update LSTM state buffer (rolling window of last 30 flat states)
                self.state_buffer.append(state)
                if len(self.state_buffer) > self.sequence_lengths[-1]:
                    self.state_buffer = self.state_buffer[-self.sequence_lengths[-1]:]

                # -- Action Mask -----------------------------------------------
                action_mask = self._build_action_mask()

                # -- Agent Input (LSTM sequences or flat state) ----------------
                if self.lstm_enabled:
                    agent_state = self.state_builder.build_state_sequences(
                        self.state_buffer, self.sequence_lengths
                    )
                else:
                    agent_state = state

                # -- Agent Decision --------------------------------------------
                action = self.agent.select_action(
                    agent_state, action_mask, deterministic=True
                )
                action_str = self._decode_action(action)

                avg_q = float(getattr(self.agent, "last_avg_q", 0.0))

                # -- Execute Trade ---------------------------------------------
                old_count = self.executor.get_position_info()["count"]
                try:
                    trade_symbol, trade_side, trade_pnl = self._execute_action(
                        action
                    )
                except Exception as e:
                    # Order failed -- log it but don't crash the bot
                    _log.error(f"Trade execution failed: {e}")
                    trade_symbol, trade_side, trade_pnl = "", "", 0.0

                # Sync with Binance after execution to verify the order filled
                # and reconcile any state divergence before counting positions.
                try:
                    time.sleep(0.5)  # give exchange 500 ms to settle
                    self._sync_positions_with_binance()
                except Exception as e:
                    _log.warning(f"Post-trade position sync failed: {e}")
                new_count = self.executor.get_position_info()["count"]

                # Accumulate realized PnL (closes only — trade_pnl is 0 on opens)
                if trade_pnl != 0.0:
                    self._realized_pnl_today += trade_pnl

                if trade_symbol and new_count != old_count:
                    self.daily_trades += 1
                    if trade_pnl > 0:
                        self.winning_trades += 1
                    elif trade_pnl < 0:
                        self.losing_trades += 1

                    # -- DB logging ------------------------------------------
                    if new_count > old_count:
                        # Position opened: fetch fill details from executor
                        try:
                            pos = self.executor.get_position_info()[
                                "positions"
                            ].get(trade_symbol, {})
                            all_symbols = (
                                self.current_coins["gainers"]
                                + self.current_coins["losers"]
                            )
                            coin_rank = (
                                all_symbols.index(trade_symbol)
                                if trade_symbol in all_symbols
                                else 0
                            )
                            is_gainer = coin_rank < 10
                            trade_key = self.logger.log_trade_open(
                                session_key=self.current_session_key,
                                symbol=trade_symbol,
                                side=trade_side,
                                action_id=action,
                                entry_price=pos.get("entry_price", 0.0),
                                quantity=pos.get("size", 0.0),
                                portfolio_value=current_balance,
                                open_positions=new_count,
                                avg_q=avg_q,
                                coin_rank=coin_rank,
                                is_gainer=is_gainer,
                            )
                            self.open_trade_keys[trade_symbol] = trade_key
                        except Exception as e:
                            _log.error(f"DB log_trade_open failed: {e}")

                    elif new_count < old_count:
                        # Track realized PnL for portfolio feature rolling stats
                        if trade_pnl != 0.0:
                            self._step_pnls.append(trade_pnl)
                            if len(self._step_pnls) > 100:
                                self._step_pnls = self._step_pnls[-100:]

                        # Position closed: record to DB
                        exit_price = self._last_close_exit_prices.pop(
                            trade_symbol, 0.0
                        )
                        pnl_pct = (
                            (trade_pnl / current_balance * 100)
                            if current_balance > 0
                            else 0.0
                        )
                        self._record_close_to_db(
                            symbol=trade_symbol,
                            side=trade_side,
                            entry_price=0.0,
                            exit_price=exit_price,
                            pnl=trade_pnl,
                            pnl_pct=pnl_pct,
                            portfolio_value=current_balance,
                            action_id=action,
                        )

                # -- Write state for web dashboard --------------------------------
                if trade_symbol and trade_symbol != "ALL":
                    # Trade executed successfully
                    self._log_activity(
                        f"Step {step}: {action_str}"
                        + (f" → PnL {trade_pnl:+.2f}" if trade_pnl else " → opened")
                    )
                elif trade_symbol == "ALL":
                    # CLOSE_ALL (per-symbol entries already logged inside _execute_action)
                    self._log_activity(f"Step {step}: {action_str}")
                elif action_str == "HOLD":
                    self._log_activity(f"Step {step}: HOLD")
                else:
                    # Agent chose a trade action but it was blocked or failed
                    self._log_activity(f"Step {step}: {action_str} → no trade")
                self._write_metrics(step, current_balance, action_str, avg_q, today_pnl)

                # -- Countdown -------------------------------------------------
                now = datetime.now(WIB)
                seconds_into_interval = (now.minute % 5) * 60 + now.second
                seconds_to_wait = 300 - seconds_into_interval
                time.sleep(seconds_to_wait)

            except KeyboardInterrupt:
                raise  # Propagate so outer handler can clean up

            except Exception as e:
                # Unexpected error -- log it, wait out the interval, then continue
                import traceback

                _log.error(f"Unexpected loop error: {e}\n{traceback.format_exc()}")
                self._wait_for_next_interval("Recovering from error")

        # -- Cleanup ------------------------------------
        print("\n🛑 Bot stopped.")
        print(f"  Steps completed : {step}")
        print(f"  Trades executed : {self.daily_trades}")
        if self.executor.get_position_info()["count"] > 0:
            print("  Closing all open positions...")
            self.executor.close_all_positions()

        # Persist final session summary to DB then flush the write queue
        if self.current_session_key:
            try:
                final_bal = self.executor.get_balance()
            except Exception:
                final_bal = self.initial_balance
            self.logger.close_session(
                self.current_session_key,
                final_balance=final_bal,
                total_pnl=final_bal - self.initial_balance,
                total_trades=self.daily_trades,
                winning_trades=self.winning_trades,
                losing_trades=self.losing_trades,
            )
        self.logger.mark_offline()
        self.logger.shutdown(timeout=5.0)
        print("Done.")


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
