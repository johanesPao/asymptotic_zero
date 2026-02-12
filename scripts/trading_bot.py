"""
Asymptotic Zero - Live Trading Bot

Main trading loop that connects all components:
- Morning screening
- Market data collection
- Agent decision making
- Trade execution
- Live terminal dashboard

v2: Added position reconciliation with Binance to prevent state divergence
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import numpy as np
import faulthandler

faulthandler.enable()  # dump Python traceback on SIGABRT / SIGSEGV

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after sys.path is set so 'src' is resolvable
from src.utils.log_setup import get_logger

_log = get_logger("trading")

from src.config.secrets import get_secret
from binance.client import Client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import TradingSession, SystemLog
from src.database.logger import TradingLogger
from src.trading.screening import get_top_movers
from src.trading.market_data import fetch_all_coins
from src.data_pipeline.features.technical_indicators import TechnicalIndicators
from src.trading.state_builder import StateBuilder
from src.trading.trade_executor import TradeExecutor
from src.trading.guardrails import GuardrailManager
from src.trading.live_dashboard import LiveTradingDashboard

# DQNAgent (TensorFlow) is imported lazily inside __init__, AFTER the DB
# warmup connection.  TensorFlow bundles its own libssl.so; if it loads
# before psycopg2's first connect(), two OpenSSL instances clash → SIGSEGV.
from rich.live import Live


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
        self.max_daily_loss = float(get_secret("MAX_DAILY_LOSS", 50))
        self.max_daily_trades = int(get_secret("MAX_DAILY_TRADES", 10))
        self.STATE_DIM = 3907
        self.ACTION_DIM = 63

        print(f"✅ Secrets loaded (mode: {self.trading_mode})")

        # Initialize Binance Client
        self.client = Client(
            self.api_key, self.api_secret, testnet=(self.trading_mode == "testnet")
        )

        # Fix time offset
        server_time = self.client.get_server_time()
        local_time = int(time.time() * 1000)
        self.client.timestamp_offset = server_time["serverTime"] - local_time

        print(f"✅ Connected to Binance {self.trading_mode}")

        # Initialize database
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

        print("✅ Connected to database")

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

        print("✅ DB logger initialized (libpq/OpenSSL warmed up)")

        # Load trained agent — TensorFlow imports HERE, after libpq is warm.
        from src.agent.dqn_agent import DQNAgent

        self.agent = DQNAgent(state_dim=self.STATE_DIM, action_dim=self.ACTION_DIM)
        dummy_state = np.zeros((1, self.STATE_DIM))
        self.agent.q_network(dummy_state)
        self.agent.target_network(dummy_state)
        self.agent.load("checkpoints/best")

        print(f"✅ Agent loaded (epsilon: {self.agent.epsilon:.3f})")

        # Initialize feature calculator
        self.feature_calculator = TechnicalIndicators()
        print("✅ Feature calculator initialized")

        # Initialize state builder
        self.state_builder = StateBuilder(num_coins=20, num_features=186)

        # Initialize trade executor
        self.executor = TradeExecutor(
            client=self.client, leverage=10, position_size_pct=0.20, max_positions=3
        )
        print("✅ Trade executor initialized")

        # Initialize guardrails
        self.guardrails = GuardrailManager(
            cooldown_steps=5, min_hold_steps=3, max_positions=3
        )
        print("✅ Guardrails initialized (5-step cooldown, 3-step min hold)")

        # Trading state
        self.current_balance = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_coins = None
        self.max_positions = 3
        self.initial_balance = 0.0

        # Initialize dashboard
        self.dashboard = LiveTradingDashboard()
        self.dashboard.update_config(
            mode=self.trading_mode,
            max_daily_loss=self.max_daily_loss,
            max_daily_trades=self.max_daily_trades,
            epsilon=self.agent.epsilon,
        )

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
                self.dashboard.last_error = msg
                self.dashboard.error_count += 1
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

            case n if 21 <= n <= 30:
                symbol = self.current_coins["gainers"][n - 21]
                allowed, reason = self.guardrails.can_close_position(symbol)
                if not allowed:
                    return "", "", 0.0
                result = self.executor.close_position(symbol)
                if result:
                    self.guardrails.record_position_closed(symbol)
                    self._last_close_exit_prices[symbol] = result.get("exit_price", 0.0)
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

            case n if 51 <= n <= 60:
                symbol = self.current_coins["losers"][n - 51]
                allowed, reason = self.guardrails.can_close_position(symbol)
                if not allowed:
                    return "", "", 0.0
                result = self.executor.close_position(symbol)
                if result:
                    self.guardrails.record_position_closed(symbol)
                    self._last_close_exit_prices[symbol] = result.get("exit_price", 0.0)
                    return symbol, result["side"], result["pnl"]
                return "", "", 0.0

            case 61:
                # CLOSE_ALL bypasses guardrails (emergency exit)
                closed = self.executor.close_all_positions()
                for symbol in list(self.guardrails.position_history.keys()):
                    self.guardrails.record_position_closed(symbol)
                return "ALL", "CLOSE", 0.0

            case 62:
                return "", "", 0.0

            case _:
                return "", "", 0.0

    def _build_state(self, market_features: dict, current_balance: float) -> np.ndarray:
        """Build state vector that matches training environment exactly."""
        initial_balance = 10000.0  # Must match training config
        max_positions = 3

        # --- Market Features ---
        market_features_dict = {}
        all_symbols = self.current_coins["gainers"] + self.current_coins["losers"]

        for coin_idx, symbol in enumerate(all_symbols):
            if symbol in market_features:
                df = market_features[symbol]
                feature_cols = [
                    col
                    for col in df.columns
                    if col
                    not in ["timestamp", "open", "high", "low", "close", "volume"]
                ]
                latest_features = df.select(feature_cols).row(-1)
                market_features_dict[coin_idx] = np.array(
                    latest_features, dtype=np.float32
                )

        # --- Position Features - MATCHES position_manager.get_position_features() ---
        position_features = np.zeros((20, 6), dtype=np.float32)
        positions = self.executor.get_position_info()["positions"]

        for coin_idx, symbol in enumerate(all_symbols):
            if symbol in positions:
                pos = positions[symbol]
                current_price = self.executor.get_current_price(symbol)
                entry = pos["entry_price"]
                size = pos["size"] * entry  # Convert quantity to USDT value

                if pos["side"] == "LONG":
                    unrealized_pnl = (current_price - entry) / entry * size
                else:
                    unrealized_pnl = (entry - current_price) / entry * size

                holding_time = self.guardrails.position_history.get(symbol, {}).get(
                    "opened_step", self.guardrails.current_step
                )
                steps_held = self.guardrails.current_step - holding_time

                position_features[coin_idx, 0] = 1.0
                position_features[coin_idx, 1] = 1.0 if pos["side"] == "LONG" else -1.0
                position_features[coin_idx, 2] = size / initial_balance
                position_features[coin_idx, 3] = (
                    unrealized_pnl / size if size > 0 else 0.0
                )
                position_features[coin_idx, 4] = (
                    (current_price - entry) / entry if entry > 0 else 0.0
                )
                position_features[coin_idx, 5] = min(steps_held / 88.0, 1.0)

        # --- Portfolio Features - MATCHES position_manager.get_portfolio_features() ---
        num_positions = len(positions)
        total_unrealized = (
            sum(
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
            )
            if positions
            else 0.0
        )

        portfolio_features = np.array(
            [
                current_balance / initial_balance,
                current_balance / initial_balance,
                num_positions / max_positions,
                total_unrealized / initial_balance,
                num_positions / max_positions,
            ],
            dtype=np.float32,
        )

        # --- Time Features - MATCHES environment._build_state() ---
        # Training: candles_per_day=288, warmup=200 -> 88 tradeable steps
        total_steps = 88
        current_relative_step = min(self.guardrails.current_step, total_steps)

        time_array = np.array(
            [
                current_relative_step / total_steps,
                1.0 - (current_relative_step / total_steps),
            ],
            dtype=np.float32,
        )

        # --- Coin Metadata ---
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

        return self.state_builder.build_state(
            market_features=market_features_dict,
            position_features=position_features,
            portfolio_features=portfolio_features,
            coin_metadata=coin_metadata,
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

    def _wait_for_next_interval(self, live, reason: str = ""):
        """Sit out the rest of the current 5-minute interval, updating dashboard each second."""
        now = datetime.now()
        seconds_into_interval = (now.minute % 5) * 60 + now.second
        seconds_to_wait = max(300 - seconds_into_interval, 10)
        next_interval = now + timedelta(seconds=seconds_to_wait)
        self.dashboard.update_step(self.dashboard.current_step, next_interval)
        for remaining in range(seconds_to_wait, 0, -1):
            self.dashboard.update_countdown(remaining)
            live.update(self.dashboard.generate_dashboard())
            time.sleep(1)

    def run(self):
        """Main trading loop with live dashboard."""
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)

        last_screening_time = None
        step = 0

        with Live(
            self.dashboard.generate_dashboard(),
            console=self.dashboard.console,
            refresh_per_second=2,
            screen=True,
        ) as live:
            self.dashboard.start_keyboard_listener()
            while True:
                try:
                    current_date = datetime.now().date()
                    current_hour = datetime.now().hour

                    # -- Morning Screening -----------------------------------------
                    now = datetime.now()
                    is_7am_window = current_hour == 7
                    already_screened_this_hour = (
                        last_screening_time is not None
                        and now - last_screening_time < timedelta(hours=1)
                    )
                    should_screen = last_screening_time is None or (
                        is_7am_window and not already_screened_this_hour
                    )

                    if should_screen:
                        # Close all before re-screening (not first run)
                        if last_screening_time is not None:
                            self.executor.close_all_positions()
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
                                self._wait_for_next_interval(
                                    live, "Screening failed -- retrying"
                                )
                                continue
                            # Otherwise keep yesterday's list and log it
                            _log.error(
                                "Screening failed after 3 attempts -- keeping previous coin list"
                            )

                        # Reset counters
                        self.daily_trades = 0
                        self.daily_pnl = 0.0
                        self.guardrails.reset_daily()

                        # Get initial balance with retry
                        try:
                            self.initial_balance = self._retry(
                                self.executor.get_balance,
                                retries=3,
                                delay=5.0,
                                label="get_balance (screening)",
                            )
                        except Exception:
                            self._wait_for_next_interval(live, "Balance fetch failed")
                            continue

                        # Push to dashboard
                        self.dashboard.update_screening(self.current_coins)
                        self.dashboard.update_portfolio(
                            self.initial_balance, self.initial_balance, 0.0
                        )
                        self.dashboard.winning_trades = 0
                        self.dashboard.losing_trades = 0

                        # Close previous session in DB (if not first run)
                        if self.current_session_key:
                            self.logger.close_session(
                                self.current_session_key,
                                final_balance=self.initial_balance,
                                total_pnl=self.daily_pnl,
                                total_trades=self.daily_trades,
                                winning_trades=self.dashboard.winning_trades,
                                losing_trades=self.dashboard.losing_trades,
                            )

                        # Open new session in DB
                        self.current_session_key = self.logger.open_session(
                            mode=self.trading_mode,
                            initial_balance=self.initial_balance,
                            screened_coins=self.current_coins,
                        )
                        self.open_trade_keys = {}

                        last_screening_time = now
                        step = 0

                    # -- Step ------------------------------------------------------
                    step += 1
                    self.guardrails.step()

                    # Sync positions with Binance BEFORE every decision
                    try:
                        actual_positions = self._sync_positions_with_binance()
                    except Exception as e:
                        _log.error(f"Position sync failed before step: {e}")
                        # Continue with internal state if sync fails

                    # Get balance -- if this fails after retries, skip the whole step
                    try:
                        current_balance = self._retry(
                            self.executor.get_balance,
                            retries=3,
                            delay=5.0,
                            label="get_balance",
                        )
                    except Exception:
                        self._wait_for_next_interval(
                            live, "Balance fetch failed -- skipping step"
                        )
                        continue

                    pnl_today = current_balance - self.initial_balance

                    self.dashboard.update_portfolio(
                        current_balance, self.initial_balance, pnl_today
                    )
                    self.dashboard.update_step(step, datetime.now())
                    guardrail_status = self.guardrails.get_status()
                    self.dashboard.update_guardrails(
                        guardrail_status["current_step"],
                        guardrail_status["cooldown_remaining"],
                    )
                    live.update(self.dashboard.generate_dashboard())

                    # -- Daily Loss Limit ------------------------------------------
                    if pnl_today <= -self.max_daily_loss:
                        self.executor.close_all_positions()
                        # Wait until next 7 AM
                        while True:
                            time.sleep(300)
                            if datetime.now().hour == 7 and (
                                last_screening_time is None
                                or datetime.now() - last_screening_time
                                > timedelta(hours=1)
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
                            retries=3,
                            delay=5.0,
                            label="fetch_all_coins",
                        )
                    except Exception:
                        # Can't trade without market data -- HOLD this step
                        self._wait_for_next_interval(
                            live, "Market data fetch failed -- HOLD"
                        )
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

                    # -- Build State -----------------------------------------------
                    try:
                        state = self._build_state(market_features, current_balance)
                    except Exception as e:
                        _log.error(f"State build failed: {e}")
                        self.dashboard.last_error = f"State build failed: {e}"
                        self.dashboard.error_count += 1
                        self._wait_for_next_interval(live, "State build failed -- HOLD")
                        continue

                    # -- Agent Decision --------------------------------------------
                    action = self.agent.select_action(state, training=False)
                    action_str = self._decode_action(action)

                    avg_q = float(getattr(self.agent, "last_avg_q", 0.0))
                    max_q = float(getattr(self.agent, "last_max_q", 0.0))
                    self.dashboard.update_agent(action, action_str, avg_q, max_q)
                    live.update(self.dashboard.generate_dashboard())

                    # -- Execute Trade ---------------------------------------------
                    old_count = len(actual_positions) if 'actual_positions' in locals() else self.executor.get_position_info()["count"]
                    try:
                        trade_symbol, trade_side, trade_pnl = self._execute_action(
                            action
                        )
                    except Exception as e:
                        # Order failed -- log it but don't crash the bot
                        _log.error(f"Trade execution failed: {e}")
                        self.dashboard.last_error = f"Order failed: {e}"
                        self.dashboard.error_count += 1
                        trade_symbol, trade_side, trade_pnl = "", "", 0.0

                    # Sync with Binance to verify order actually filled
                    time.sleep(0.5)  # Give exchange 500ms to settle
                    try:
                        actual_positions = self._sync_positions_with_binance()
                        new_count = len(actual_positions)
                    except Exception as e:
                        _log.error(f"Post-order position sync failed: {e}")
                        new_count = self.executor.get_position_info()["count"]

                    if trade_symbol and new_count != old_count:
                        self.daily_trades += 1
                        self.dashboard.record_trade(
                            trade_symbol, trade_side, trade_pnl, action_str
                        )

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
                            # Position closed: update the open trade row
                            try:
                                trade_key = self.open_trade_keys.pop(trade_symbol, None)
                                exit_price = self._last_close_exit_prices.pop(
                                    trade_symbol, 0.0
                                )
                                if trade_key:
                                    self.logger.log_trade_close(
                                        trade_key=trade_key,
                                        exit_price=exit_price,
                                        pnl=trade_pnl,
                                        pnl_percent=(
                                            (trade_pnl / current_balance * 100)
                                            if current_balance > 0
                                            else 0.0
                                        ),
                                    )
                            except Exception as e:
                                _log.error(f"DB log_trade_close failed: {e}")

                    # Update positions display
                    self.dashboard.update_positions(self._get_enriched_positions())
                    self.dashboard.daily_trades = self.daily_trades
                    live.update(self.dashboard.generate_dashboard())

                    # -- Countdown -------------------------------------------------
                    now = datetime.now()
                    seconds_into_interval = (now.minute % 5) * 60 + now.second
                    seconds_to_wait = 300 - seconds_into_interval
                    next_interval = now + timedelta(seconds=seconds_to_wait)
                    self.dashboard.update_step(step, next_interval)

                    for remaining in range(seconds_to_wait, 0, -1):
                        self.dashboard.update_countdown(remaining)
                        live.update(self.dashboard.generate_dashboard())
                        time.sleep(1)

                except KeyboardInterrupt:
                    raise  # Propagate so outer handler can clean up

                except Exception as e:
                    # Unexpected error -- log it, wait out the interval, then continue
                    import traceback

                    _log.error(f"Unexpected loop error: {e}\n{traceback.format_exc()}")
                    self.dashboard.last_error = f"Loop error: {e}"
                    self.dashboard.error_count += 1
                    live.update(self.dashboard.generate_dashboard())
                    self._wait_for_next_interval(live, "Recovering from error")

        # -- Cleanup (after Live context exits) ------------------------------------
        self.dashboard.stop_keyboard_listener()
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
                winning_trades=self.dashboard.winning_trades,
                losing_trades=self.dashboard.losing_trades,
            )
        self.logger.shutdown(timeout=5.0)
        print("Done.")


if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
