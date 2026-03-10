"""
Trading Environment — v2

Major changes from v1:
  - Action masking: invalid actions are blocked at the source, not just post-filtered
  - Guardrails baked into the environment (cooldown, min hold, max loss, max trades)
  - Position sizing: 3 sizes (small/medium/large) as part of action space
  - Regime features: ADX, volatility percentile, breadth, momentum
  - Adversarial training: price noise, slippage spikes, flash crashes
  - Full transaction cost and funding rate integration with reward calculator

Action Space (143 discrete actions):
    0         : HOLD
    1–30      : OPEN LONG  gainer[0-9] × 3 sizes
    31–60     : OPEN SHORT gainer[0-9] × 3 sizes
    61–70     : CLOSE gainer[0-9]
    71–100    : OPEN LONG  loser[0-9]  × 3 sizes
    101–130   : OPEN SHORT loser[0-9]  × 3 sizes
    131–140   : CLOSE loser[0-9]
    141       : CLOSE_ALL
    142       : CLOSE_WORST
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml
import logging

from .position_manager import PositionManager, PositionSide
from .reward_calculator import RewardCalculator
from .state_builder import StateBuilder
from ..data_pipeline.features import TechnicalIndicators

logger = logging.getLogger(__name__)

# Action layout constants — derived from num_coins_per_side=10, num_sizes=3
# These map cleanly to the 143-action space defined in trading.yaml
_N  = 10   # coins per side
_S  = 3    # sizes per open action
_NS = _N * _S  # 30 slots per open-type (e.g. all LONG gainer variants)


class TradingEnvironment:
    """
    Gym-style trading environment for cryptocurrency futures.
    v2: Full action masking, guardrails baked in, IQN-ready.
    """

    def __init__(
        self,
        config_path:     str = "config/trading.yaml",
        data_directory:  str = "data/volatility",
        features_config: str = "config/features.yaml",
    ):
        self.config_path    = Path(config_path)
        self.data_directory = Path(data_directory)
        self.features_config = Path(features_config)

        self.config = self._load_config()

        # Extract config sections
        portfolio_cfg  = self.config.get("portfolio",   {})
        costs_cfg      = self.config.get("costs",       {})
        actions_cfg    = self.config.get("actions",     {})
        episode_cfg    = self.config.get("episode",     {})
        guard_cfg      = self.config.get("guardrails",  {})
        adv_cfg        = self.config.get("adversarial", {})

        # ── Episode ────────────────────────────────────────────────────────────
        self.candles_per_day    = episode_cfg.get("candles_per_day",  288)
        self.warmup_candles     = episode_cfg.get("warmup_candles",   0)
        self.historical_candles = episode_cfg.get("historical_candles", 0)
        self.force_close_at_end = episode_cfg.get("force_close_at_end", True)

        # ── Action space ───────────────────────────────────────────────────────
        self.num_coins_per_side = actions_cfg.get("num_coins_per_side", 10)
        self.num_coins          = self.num_coins_per_side * 2
        self.num_sizes          = actions_cfg.get("num_sizes", 3)
        self.enable_close_all   = actions_cfg.get("enable_close_all",   True)
        self.enable_close_worst = actions_cfg.get("enable_close_worst", True)

        # Position size mapping (small/medium/large)
        sizes_cfg = portfolio_cfg.get("position_sizes", {})
        
        if self.num_sizes == 1:
            # Single size mode: use "medium" as the standard size
            self.position_sizes = [sizes_cfg.get("medium", 0.20)]
        else:
            # Multi-size mode
            self.position_sizes = [
                sizes_cfg.get("small",  0.10),
                sizes_cfg.get("medium", 0.20),
                sizes_cfg.get("large",  0.30),
            ]

        # Total action count:
        #   1 (HOLD)
        # + num_coins_per_side × num_sizes × 2  (LONG gainers + SHORT gainers)
        # + num_coins_per_side                   (CLOSE gainers)
        # + num_coins_per_side × num_sizes × 2  (LONG losers + SHORT losers)
        # + num_coins_per_side                   (CLOSE losers)
        # + 2                                    (CLOSE_ALL + CLOSE_WORST)
        n, s = self.num_coins_per_side, self.num_sizes
        self.action_size = 1 + n*s*2 + n + n*s*2 + n + 2  # = 143

        # ── Guardrails ─────────────────────────────────────────────────────────
        self.cooldown_steps     = guard_cfg.get("cooldown_steps",    5)
        self.min_hold_steps     = guard_cfg.get("min_hold_steps",    3)
        self.max_daily_loss_pct = guard_cfg.get("max_daily_loss_pct", 0.15)
        self.max_daily_trades   = guard_cfg.get("max_daily_trades",  50)

        # ── Adversarial training ───────────────────────────────────────────────
        self.adversarial_enabled      = adv_cfg.get("enabled", True)
        self.price_noise_std          = adv_cfg.get("price_noise_std",      0.0002)
        self.slippage_spike_prob      = adv_cfg.get("slippage_spike_prob",  0.02)
        self.slippage_spike_mult      = adv_cfg.get("slippage_spike_mult",  5.0)
        self.data_gap_prob            = adv_cfg.get("data_gap_prob",        0.01)
        self.flash_crash_prob         = adv_cfg.get("flash_crash_prob",     0.001)
        self.flash_crash_magnitude    = adv_cfg.get("flash_crash_magnitude", 0.05)

        # ── Components ────────────────────────────────────────────────────────
        initial_balance = portfolio_cfg.get("initial_balance", 10000.0)
        self.initial_balance = initial_balance
        self.leverage        = portfolio_cfg.get("leverage", 10)

        self.position_manager = PositionManager(
            initial_balance    = initial_balance,
            max_positions      = portfolio_cfg.get("max_positions", 3),
            position_size_pct  = self.position_sizes[1 if self.num_sizes > 1 else 0],  # default = medium
            taker_fee_pct      = costs_cfg.get("taker_fee_pct", 0.04),
        )

        self.reward_calculator = RewardCalculator(
            config_path     = str(self.config_path),
            initial_balance = initial_balance,
        )

        self.technical_indicators = TechnicalIndicators(
            config_path      = str(self.features_config),
            data_directory   = str(self.data_directory / "5m"),
        )

        feature_names     = self.technical_indicators.get_feature_names()
        self.num_features = len(feature_names)

        self.state_builder = StateBuilder(
            config_path  = str(self.config_path),
            num_coins    = self.num_coins,
            num_features = self.num_features,
        )

        # LSTM state buffer
        lstm_cfg = self.config.get("lstm", {})
        self.lstm_enabled = lstm_cfg.get("enabled", False)
        if self.lstm_enabled:
            self.sequence_lengths = lstm_cfg.get("sequence_lengths", [10, 20, 30])
            self.max_sequence_length = max(self.sequence_lengths)
            self.state_buffer = []
            self._prev_episode_states = []  # cross-episode LSTM lookback
            logger.info(f"LSTM enabled: maintaining state buffer (max {self.max_sequence_length} steps)")
        else:
            self.sequence_lengths = None
            self.max_sequence_length = 0
            self.state_buffer = None
            self._prev_episode_states = None

        # ── Data ──────────────────────────────────────────────────────────────
        self.top_movers_file = self.data_directory / "top_movers.parquet"
        self.top_movers_df   = self._load_top_movers()
        self.available_dates = self._get_available_dates()

        # Optionally filter dates by regime label for curriculum
        self._curriculum_dates: Optional[List[str]] = None

        # ── Episode state ──────────────────────────────────────────────────────
        self.current_date:   Optional[str]           = None
        self.current_step:   int                     = 0
        self.episode_data:   Dict[int, pl.DataFrame] = {}
        self.coin_symbols:   Dict[int, str]          = {}
        self.coin_metadata:  Dict[int, Dict]         = {}
        self.done:           bool                    = False

        # Guardrail state
        self._cooldown_remaining: int         = 0
        self._hold_steps:         Dict[int, int] = {}  # coin_idx -> steps held
        self._daily_trades:       int         = 0
        self._daily_loss:         float       = 0.0

        logger.info(
            f"TradingEnvironment v2 initialized | "
            f"actions={self.action_size} | "
            f"state={self.state_builder.get_state_dim()} | "
            f"dates={len(self.available_dates)}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Config / data loading
    # ══════════════════════════════════════════════════════════════════════════

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _load_top_movers(self) -> pl.DataFrame:
        if not self.top_movers_file.exists():
            raise FileNotFoundError(f"Top movers file not found: {self.top_movers_file}")
        return pl.read_parquet(self.top_movers_file)

    def _get_available_dates(self) -> List[str]:
        dates = self.top_movers_df["date"].unique().sort().to_list()
        return [str(d) for d in dates]

    def set_curriculum_dates(self, dates: List[str]):
        """Restrict episode sampling to a specific list of dates (curriculum stage)."""
        self._curriculum_dates = dates
        logger.info(f"Curriculum: {len(dates)} dates available for this stage")

    def clear_curriculum_dates(self):
        """Remove curriculum restriction — use all available dates."""
        self._curriculum_dates = None

    def set_guardrails(self, guardrail_overrides: dict):
        """
        Dynamically update guardrail values (e.g., per curriculum stage).

        Args:
            guardrail_overrides: Dict with any of:
                cooldown_steps, min_hold_steps, max_daily_loss_pct, max_daily_trades
        """
        if "cooldown_steps" in guardrail_overrides:
            self.cooldown_steps = guardrail_overrides["cooldown_steps"]
        if "min_hold_steps" in guardrail_overrides:
            self.min_hold_steps = guardrail_overrides["min_hold_steps"]
        if "max_daily_loss_pct" in guardrail_overrides:
            self.max_daily_loss_pct = guardrail_overrides["max_daily_loss_pct"]
        if "max_daily_trades" in guardrail_overrides:
            self.max_daily_trades = guardrail_overrides["max_daily_trades"]
        logger.info(
            f"Guardrails updated: cooldown={self.cooldown_steps} | "
            f"min_hold={self.min_hold_steps} | "
            f"max_loss={self.max_daily_loss_pct:.0%} | "
            f"max_trades={self.max_daily_trades}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Episode data loading
    # ══════════════════════════════════════════════════════════════════════════

    def _load_episode_data(self, date: str) -> bool:
        date_obj  = datetime.strptime(date, "%Y-%m-%d").date()
        date_data = self.top_movers_df.filter(pl.col("date") == date_obj)

        if len(date_data) == 0:
            return False

        row        = date_data.row(0, named=True)
        gainers    = row["gainers"]
        losers     = row["losers"]
        gainers_pct = row["gainers_pct"]
        losers_pct  = row["losers_pct"]

        self.episode_data  = {}
        self.coin_symbols  = {}
        self.coin_metadata = {}
        actual_index       = 0

        for i, (symbol, pct) in enumerate(zip(gainers, gainers_pct)):
            if i >= self.num_coins_per_side:
                break
            data_file = self.data_directory / "5m" / symbol / f"{date}.parquet"
            if not data_file.exists():
                continue
            df = pl.read_parquet(data_file)
            df = self._calculate_with_historical(df, symbol, date)
            self.episode_data[actual_index]  = df
            self.coin_symbols[actual_index]  = symbol
            self.coin_metadata[actual_index] = {
                "is_gainer":    True,
                "rank":         i + 1,
                "daily_change": pct,
            }
            # Add coin profile features
            self._enrich_coin_metadata(actual_index, symbol, date)
            actual_index += 1

        for i, (symbol, pct) in enumerate(zip(losers, losers_pct)):
            if i >= self.num_coins_per_side:
                break
            data_file = self.data_directory / "5m" / symbol / f"{date}.parquet"
            if not data_file.exists():
                continue
            df = pl.read_parquet(data_file)
            df = self._calculate_with_historical(df, symbol, date)
            self.episode_data[actual_index]  = df
            self.coin_symbols[actual_index]  = symbol
            self.coin_metadata[actual_index] = {
                "is_gainer":    False,
                "rank":         i + 1,
                "daily_change": pct,
            }
            # Add coin profile features
            self._enrich_coin_metadata(actual_index, symbol, date)
            actual_index += 1

        if not self.episode_data:
            return False

        # Compute relative volume ranks across all loaded coins
        self._compute_volume_ranks()

        logger.debug(f"Loaded {len(self.episode_data)} coins for {date}")
        return True

    def _enrich_coin_metadata(
        self, coin_index: int, symbol: str, date: str
    ) -> None:
        """
        Add coin profile features to metadata (non-identity-based).

        Computes:
          - historical_volatility: std of 5m returns from the episode data
          - volume_rank: percentile rank of avg volume among loaded coins
          - avg_price_range: average high-low range as fraction of close
        """
        meta = self.coin_metadata.get(coin_index, {})
        df = self.episode_data.get(coin_index)

        if df is not None and len(df) > 10:
            # Historical volatility: std of 5m log returns
            closes = df["close"].to_numpy()
            returns = np.diff(np.log(closes + 1e-10))
            hist_vol = float(np.std(returns)) if len(returns) > 1 else 0.0
            meta["historical_volatility"] = np.clip(hist_vol, 0.0, 0.5)

            # Average volume (normalized later by rank across coins)
            if "volume" in df.columns:
                avg_vol = float(df["volume"].mean())
            else:
                avg_vol = 0.0
            meta["avg_volume"] = avg_vol

            # Average price range (high-low)/close — proxy for intraday volatility
            if "high" in df.columns and "low" in df.columns:
                ranges = ((df["high"] - df["low"]) / (df["close"] + 1e-10)).to_numpy()
                meta["avg_price_range"] = float(np.clip(np.mean(ranges), 0.0, 0.5))
            else:
                meta["avg_price_range"] = 0.0
        else:
            meta["historical_volatility"] = 0.0
            meta["avg_volume"] = 0.0
            meta["avg_price_range"] = 0.0

        self.coin_metadata[coin_index] = meta

    def _compute_volume_ranks(self) -> None:
        """Compute relative volume ranks across all loaded coins for this episode."""
        volumes = []
        indices = []
        for idx, meta in self.coin_metadata.items():
            volumes.append(meta.get("avg_volume", 0.0))
            indices.append(idx)

        if not volumes:
            return

        # Rank by volume (0 = lowest, 1 = highest)
        sorted_order = np.argsort(volumes)
        n = len(volumes)
        for rank, order_idx in enumerate(sorted_order):
            coin_idx = indices[order_idx]
            self.coin_metadata[coin_idx]["volume_rank"] = rank / max(n - 1, 1)

    def _calculate_with_historical(
        self, df: pl.DataFrame, symbol: str, date: str
    ) -> pl.DataFrame:
        """
        Compute TAs on current-day data, optionally prepending historical candles
        so indicators converge from step 0.

        If historical_candles > 0:
          1. Load OHLCV columns from prior date files
          2. Concatenate [historical] + [current_day]
          3. Compute TAs on combined data
          4. Slice to return only the current day's rows (fully converged)
        """
        if self.historical_candles <= 0:
            return self.technical_indicators.calculate(df)

        current_len = len(df)
        ohlcv_cols = ["open_time", "open", "high", "low", "close", "volume"]
        keep_cols = [c for c in ohlcv_cols if c in df.columns]
        current_ohlcv = df.select(keep_cols)

        hist_df = self._load_historical_candles(symbol, date)
        if hist_df is not None and len(hist_df) > 0:
            hist_ohlcv = hist_df.select([c for c in keep_cols if c in hist_df.columns])
            combined = pl.concat([hist_ohlcv, current_ohlcv])
        else:
            combined = current_ohlcv

        combined = self.technical_indicators.calculate(combined)
        return combined.tail(current_len)

    def _load_historical_candles(
        self, symbol: str, current_date: str
    ) -> Optional[pl.DataFrame]:
        """Load candles from the most recent prior date file(s) for TA convergence."""
        symbol_dir = self.data_directory / "5m" / symbol
        if not symbol_dir.exists():
            return None

        # Find all date files for this symbol, sorted
        date_files = sorted(symbol_dir.glob("*.parquet"))
        prior_files = [f for f in date_files if f.stem < current_date]

        if not prior_files:
            return None

        # Load from most recent prior file(s) until we have enough candles
        needed = self.historical_candles
        frames = []
        for f in reversed(prior_files):
            df = pl.read_parquet(f)
            frames.insert(0, df)
            needed -= len(df)
            if needed <= 0:
                break

        combined = pl.concat(frames)
        return combined.tail(self.historical_candles)

    # ══════════════════════════════════════════════════════════════════════════
    # Action parsing — new 143-action layout
    # ══════════════════════════════════════════════════════════════════════════

    def _parse_action(
        self, action: int
    ) -> Tuple[str, int, Optional[PositionSide], float]:
        """
        Parse action integer into (action_type, coin_index, side, size_pct).

        Layout (n=10 coins/side, s=3 sizes):
          0          : HOLD
          1–30       : LONG  gainer — 10 coins × 3 sizes, row-major [coin][size]
          31–60      : SHORT gainer
          61–70      : CLOSE gainer (no size variant)
          71–100     : LONG  loser
          101–130    : SHORT loser
          131–140    : CLOSE loser
          141        : CLOSE_ALL
          142        : CLOSE_WORST
        """
        n, s = self.num_coins_per_side, self.num_sizes

        if action == 0:
            return "hold", -1, None, 0.0

        # Helper: decode a block of n*s open actions
        # Returns (coin_index_within_side, size_index) from a block offset
        def _decode_open(offset: int):
            o    = offset - 1          # 0-based offset within block
            coin = o // s              # 0-based coin index within side
            size = o % s               # 0 = small, 1 = medium, 2 = large
            return coin, size

        # LONG gainer: 1 – n*s
        if 1 <= action <= n * s:
            coin, sz = _decode_open(action)
            return "open", coin, PositionSide.LONG, self.position_sizes[sz]

        # SHORT gainer: n*s+1 – 2*n*s
        if n*s + 1 <= action <= 2*n*s:
            coin, sz = _decode_open(action - n*s)
            return "open", coin, PositionSide.SHORT, self.position_sizes[sz]

        # CLOSE gainer: 2*n*s+1 – 2*n*s+n
        close_g_start = 2 * n * s + 1
        if close_g_start <= action <= close_g_start + n - 1:
            coin = action - close_g_start
            return "close", coin, None, 0.0

        # LONG loser: close_g_start+n – close_g_start+n+n*s-1
        long_l_start = close_g_start + n
        if long_l_start <= action <= long_l_start + n*s - 1:
            coin, sz = _decode_open(action - long_l_start + 1)
            return "open", n + coin, PositionSide.LONG, self.position_sizes[sz]

        # SHORT loser
        short_l_start = long_l_start + n * s
        if short_l_start <= action <= short_l_start + n*s - 1:
            coin, sz = _decode_open(action - short_l_start + 1)
            return "open", n + coin, PositionSide.SHORT, self.position_sizes[sz]

        # CLOSE loser
        close_l_start = short_l_start + n * s
        if close_l_start <= action <= close_l_start + n - 1:
            coin = (action - close_l_start) + n
            return "close", coin, None, 0.0

        # CLOSE_ALL
        if action == close_l_start + n and self.enable_close_all:
            return "close_all", -1, None, 0.0

        # CLOSE_WORST
        if action == close_l_start + n + 1 and self.enable_close_worst:
            return "close_worst", -1, None, 0.0

        return "invalid", -1, None, 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Action masking — the core of v2
    # ══════════════════════════════════════════════════════════════════════════

    def get_action_mask(self) -> np.ndarray:
        """
        Return a boolean mask of valid actions given current environment state.

        True  = action is valid and can be selected
        False = action is structurally impossible right now

        The IQN agent applies this mask before argmax to guarantee
        it never selects an invalid action.
        """
        n, s = self.num_coins_per_side, self.num_sizes
        mask = np.zeros(self.action_size, dtype=bool)

        # HOLD is always valid
        mask[0] = True

        # Daily loss circuit breaker — if triggered, only HOLD and CLOSE are valid
        daily_loss_limit   = self.initial_balance * self.max_daily_loss_pct
        circuit_breaker_on = self._daily_loss >= daily_loss_limit

        # Daily trade limit
        trade_limit_hit = self._daily_trades >= self.max_daily_trades

        positions          = self.position_manager.positions
        num_positions      = len(positions)
        max_positions      = self.position_manager.max_positions
        can_open_new       = (
            not circuit_breaker_on
            and not trade_limit_hit
            and num_positions < max_positions
            and self._cooldown_remaining == 0
        )

        # ── OPEN actions ───────────────────────────────────────────────────────
        if can_open_new:
            for coin_idx in range(self.num_coins):
                # Can't open if coin has no data
                if coin_idx not in self.coin_symbols:
                    continue
                # Can't open if already have a position in this coin
                if coin_idx in positions:
                    continue

                # Gainer side
                if coin_idx < n:
                    base_long  = 1          + coin_idx * s
                    base_short = n*s + 1    + coin_idx * s
                    for sz in range(s):
                        mask[base_long  + sz] = True
                        mask[base_short + sz] = True
                # Loser side
                else:
                    loser_idx  = coin_idx - n
                    long_l_start  = 2*n*s + n + 1
                    short_l_start = long_l_start + n*s
                    base_long     = long_l_start  + loser_idx * s
                    base_short    = short_l_start + loser_idx * s
                    for sz in range(s):
                        mask[base_long  + sz] = True
                        mask[base_short + sz] = True

        # ── CLOSE actions ──────────────────────────────────────────────────────
        close_g_start = 2 * n * s + 1
        close_l_start = 2 * n * s + n + 2 * n * s + 1

        for coin_idx, pos in positions.items():
            # Enforce min hold — can't close until held long enough
            steps_held = self._hold_steps.get(coin_idx, 0)
            if steps_held < self.min_hold_steps:
                continue

            if coin_idx < n:
                mask[close_g_start + coin_idx] = True
            else:
                mask[close_l_start + (coin_idx - n)] = True

        # CLOSE_ALL: valid only if there are closable positions
        close_all_idx   = close_l_start + n
        close_worst_idx = close_all_idx + 1

        has_closable = any(
            self._hold_steps.get(c, 0) >= self.min_hold_steps
            for c in positions
        )
        if has_closable and self.enable_close_all:
            mask[close_all_idx] = True
        if has_closable and self.enable_close_worst:
            mask[close_worst_idx] = True

        return mask

    # ══════════════════════════════════════════════════════════════════════════
    # Price helpers with adversarial perturbations
    # ══════════════════════════════════════════════════════════════════════════

    def _get_current_prices(self) -> Dict[int, float]:
        prices = {}
        for coin_idx, df in self.episode_data.items():
            row = self.current_step if self.current_step < len(df) else len(df) - 1
            price = float(df["close"][row])

            if self.adversarial_enabled:
                # Data gap — return last price instead of current
                if np.random.random() < self.data_gap_prob and self.current_step > 0:
                    prev = max(0, self.current_step - 1)
                    price = float(df["close"][prev if prev < len(df) else len(df)-1])

                # Gaussian price noise
                price *= (1.0 + np.random.normal(0, self.price_noise_std))

                # Flash crash — instant drop on this coin
                if np.random.random() < self.flash_crash_prob:
                    price *= (1.0 - self.flash_crash_magnitude)

            prices[coin_idx] = max(price, 1e-8)  # guard against zero/negative

        return prices

    def _get_execution_slippage(self) -> float:
        """Return slippage multiplier for this trade (normal or spike)."""
        if self.adversarial_enabled and np.random.random() < self.slippage_spike_prob:
            return self.slippage_spike_mult
        return 1.0

    def _get_market_features(self) -> Dict[int, np.ndarray]:
        feature_cols = self.technical_indicators.get_feature_names()
        features = {}
        for coin_idx, df in self.episode_data.items():
            # Use previous candle's TA values (current candle is not yet closed)
            row_idx = min(max(self.current_step - 1, 0), len(df) - 1)
            row     = df[row_idx]
            values  = [
                row[col].item() if col in df.columns else 0.0
                for col in feature_cols
            ]
            features[coin_idx] = np.array(values)
        return features

    # ══════════════════════════════════════════════════════════════════════════
    # Regime feature computation
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_regime_features(self, prices: Dict[int, float]) -> np.ndarray:
        """
        Compute 8 market regime features from current episode state.
        These are coin-agnostic and describe the overall market session character.
        """
        regime_cfg = self.config.get("regime", {})
        vol_lb     = regime_cfg.get("volatility_lookback", 24)

        # Collect recent returns across all coins
        all_returns = []
        all_adx     = []

        for coin_idx, df in self.episode_data.items():
            # Use data up to previous candle only (current candle not yet closed)
            end   = min(self.current_step, len(df))
            start = max(0, end - vol_lb)
            if end - start < 2:
                continue

            closes = df["close"][start:end].to_numpy().astype(np.float64)
            rets   = np.diff(np.log(closes + 1e-8))
            all_returns.extend(rets.tolist())

            # ADX from DataFrame if pre-computed, else approximate via |return| trend
            if "adx_14" in df.columns:
                adx_val = float(df["adx_14"][min(max(self.current_step - 1, 0), len(df)-1)])
                if not np.isnan(adx_val):
                    all_adx.append(adx_val / 100.0)  # normalize 0–100 → 0–1

        all_returns = np.array(all_returns) if all_returns else np.array([0.0])

        # 1. Average ADX (trend strength)
        avg_adx = float(np.mean(all_adx)) if all_adx else 0.0

        # 2. Rolling realized volatility (std of log-returns)
        realized_vol = float(np.std(all_returns)) if len(all_returns) > 1 else 0.0
        # Normalize relative to a "typical" crypto vol of 0.005 per 5m candle
        vol_normalized = np.clip(realized_vol / 0.005, 0, 5) / 5.0

        # 3. Market breadth — fraction of coins with positive return last closed candle
        step_returns = []
        for coin_idx, df in self.episode_data.items():
            if self.current_step >= 2 and self.current_step - 1 < len(df):
                r = float(df["close"][self.current_step - 1]) / float(df["close"][self.current_step - 2]) - 1
                step_returns.append(r)
        breadth = float(np.mean([r > 0 for r in step_returns])) if step_returns else 0.5

        # 4. Average absolute return of screened coins this step (session heat)
        avg_abs_return = float(np.mean(np.abs(step_returns))) if step_returns else 0.0
        avg_abs_return = np.clip(avg_abs_return / 0.02, 0, 1)  # normalize vs 2% step

        # 5. Trend consistency — fraction of recent returns with same sign as overall
        overall_sign = np.sign(np.mean(all_returns)) if len(all_returns) > 0 else 0
        consistency  = float(np.mean(np.sign(all_returns) == overall_sign)) if len(all_returns) > 0 else 0.5

        # 6. Session volatility vs historical average (approx: use std vs window std)
        # Use ratio of current window vol to full-episode vol approximation
        full_returns = []
        for coin_idx, df in self.episode_data.items():
            end = min(self.current_step, len(df))
            if end >= 2:
                closes = df["close"][:end].to_numpy().astype(np.float64)
                full_returns.extend(np.diff(np.log(closes + 1e-8)).tolist())
        full_vol = float(np.std(full_returns)) if len(full_returns) > 1 else realized_vol
        vol_ratio = np.clip(realized_vol / (full_vol + 1e-8), 0, 3) / 3.0

        # 7. Intraday momentum persistence
        # How much of the early move (first 30 candles after warmup) has been retained
        persistence = 0.5
        warmup = self.warmup_candles
        check_step = warmup + 30
        for coin_idx, df in self.episode_data.items():
            if check_step < len(df) and self.current_step > check_step:
                early_move  = float(df["close"][check_step]) / float(df["close"][warmup]) - 1
                recent_move = float(df["close"][min(self.current_step - 1, len(df)-1)]) / float(df["close"][warmup]) - 1
                if abs(early_move) > 1e-8:
                    persistence = np.clip(recent_move / early_move, -1, 2) / 2.0 + 0.5
                break  # approximate from first coin; sufficient for a shaping signal

        # 8. Fraction of day elapsed (time in regime context)
        tradeable_steps = self.candles_per_day - self.warmup_candles
        day_progress    = (self.current_step - self.warmup_candles) / max(tradeable_steps, 1)

        return np.array([
            avg_adx,
            vol_normalized,
            breadth,
            avg_abs_return,
            consistency,
            vol_ratio,
            float(np.clip(persistence, 0, 1)),
            float(np.clip(day_progress, 0, 1)),
        ], dtype=np.float32)

    def _compute_guardrail_features(self) -> np.ndarray:
        """
        Compute 5 guardrail state features.
        Tells the agent its own constraint situation so it can plan ahead.
        """
        positions     = self.position_manager.positions
        max_positions = self.position_manager.max_positions

        # 1. Cooldown remaining (normalized 0-1)
        cooldown_norm = self._cooldown_remaining / max(self.cooldown_steps, 1)

        # 2. Min hold remaining — average across open positions (normalized 0-1)
        if positions:
            holds_remaining = [
                max(0, self.min_hold_steps - self._hold_steps.get(c, 0))
                for c in positions
            ]
            avg_hold_remaining = np.mean(holds_remaining) / max(self.min_hold_steps, 1)
        else:
            avg_hold_remaining = 0.0

        # 3. Position slots used (normalized 0-1)
        slots_used = len(positions) / max(max_positions, 1)

        # 4. Daily trades used (normalized 0-1)
        trades_norm = self._daily_trades / max(self.max_daily_trades, 1)

        # 5. Daily loss used (normalized 0-1)
        loss_limit = self.initial_balance * self.max_daily_loss_pct
        loss_norm  = min(self._daily_loss / max(loss_limit, 1.0), 1.0)

        return np.array([
            float(np.clip(cooldown_norm,       0, 1)),
            float(np.clip(avg_hold_remaining,  0, 1)),
            float(np.clip(slots_used,          0, 1)),
            float(np.clip(trades_norm,         0, 1)),
            float(np.clip(loss_norm,           0, 1)),
        ], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Action execution
    # ══════════════════════════════════════════════════════════════════════════

    def _execute_action(
        self,
        action_type: str,
        coin_index:  int,
        side:        Optional[PositionSide],
        size_pct:    float,
        prices:      Dict[int, float],
    ) -> Tuple[bool, float, float, str]:
        """
        Execute a trading action.

        Returns:
            (success, realized_pnl, trade_notional, message)
        """
        slip = self._get_execution_slippage()
        # Slippage magnitude (always positive, small value like 0.000005)
        slip_mag = (slip - 1) * self.slippage_spike_mult * 0.01

        if action_type == "hold":
            return True, 0.0, 0.0, "Hold"

        elif action_type == "open":
            if coin_index not in self.coin_symbols:
                return False, 0.0, 0.0, f"Invalid coin: {coin_index}"
            if coin_index not in prices:
                return False, 0.0, 0.0, f"No price for coin {coin_index}"

            # Temporarily adjust position_size_pct for this trade
            self.position_manager.position_size_pct = size_pct
            symbol    = self.coin_symbols[coin_index]
            is_gainer = self.coin_metadata[coin_index]["is_gainer"]

            # Apply directional slippage (always makes execution worse)
            # LONG open → price moves UP; SHORT open → price moves DOWN
            if side == PositionSide.LONG:
                exec_price = prices[coin_index] * (1 + slip_mag)
            else:
                exec_price = prices[coin_index] * (1 - slip_mag)
            success, message = self.position_manager.open_position(
                coin_index = coin_index,
                symbol     = symbol,
                side       = side,
                price      = exec_price,
                is_gainer  = is_gainer,
            )
            if success:
                notional = self.initial_balance * size_pct * self.leverage
                self._hold_steps[coin_index] = 0
                self._daily_trades += 1
                return True, 0.0, notional, message
            return False, 0.0, 0.0, message

        elif action_type == "close":
            if coin_index not in prices:
                return False, 0.0, 0.0, f"No price for {coin_index}"
            notional = self.initial_balance * self.position_manager.position_size_pct * self.leverage
            # Directional slippage on close (opposite direction to open)
            pos = self.position_manager.positions.get(coin_index)
            if pos and pos.side == PositionSide.LONG:
                exec_price = prices[coin_index] * (1 - slip_mag)  # LONG close → price DOWN
            else:
                exec_price = prices[coin_index] * (1 + slip_mag)  # SHORT close → price UP
            success, message, pnl = self.position_manager.close_position(
                coin_index = coin_index,
                price      = exec_price,
            )
            if success:
                self._hold_steps.pop(coin_index, None)
                self._cooldown_remaining = self.cooldown_steps
                if pnl < 0:
                    self._daily_loss += abs(pnl)
            return success, pnl if success else 0.0, notional, message

        elif action_type == "close_all":
            notional     = self.initial_balance * self.position_manager.position_size_pct * self.leverage
            # Apply directional slippage to each position's close price
            slipped_prices = {}
            for ci, pos in self.position_manager.positions.items():
                if ci in prices:
                    if pos.side == PositionSide.LONG:
                        slipped_prices[ci] = prices[ci] * (1 - slip_mag)
                    else:
                        slipped_prices[ci] = prices[ci] * (1 + slip_mag)
            num_closed, total_pnl = self.position_manager.close_all_positions(slipped_prices)
            if num_closed > 0:
                self._hold_steps.clear()
                self._cooldown_remaining = self.cooldown_steps
                if total_pnl < 0:
                    self._daily_loss += abs(total_pnl)
            return True, total_pnl, notional * num_closed, f"Closed {num_closed} positions"

        elif action_type == "close_worst":
            notional = self.initial_balance * self.position_manager.position_size_pct * self.leverage
            # Apply directional slippage to the worst position's close
            slipped_prices = {}
            for ci, pos in self.position_manager.positions.items():
                if ci in prices:
                    if pos.side == PositionSide.LONG:
                        slipped_prices[ci] = prices[ci] * (1 - slip_mag)
                    else:
                        slipped_prices[ci] = prices[ci] * (1 + slip_mag)
            success, pnl, closed_coin = self.position_manager.close_worst_position(slipped_prices)
            if success:
                self._hold_steps.pop(closed_coin, None)
                self._cooldown_remaining = self.cooldown_steps
                if pnl < 0:
                    self._daily_loss += abs(pnl)
            return success, pnl if success else 0.0, notional, "Closed worst"

        return False, 0.0, 0.0, "Invalid action"

    # ══════════════════════════════════════════════════════════════════════════
    # State building
    # ══════════════════════════════════════════════════════════════════════════

    def _build_state(self, prices: Optional[Dict[int, float]] = None) -> np.ndarray:
        if prices is None:
            prices = self._get_current_prices()

        market_features   = self._get_market_features()
        position_features = self.position_manager.get_position_features(self.num_coins)
        portfolio_features = self.position_manager.get_portfolio_features()
        regime_features    = self._compute_regime_features(prices)
        guardrail_features = self._compute_guardrail_features()

        tradeable_steps       = self.candles_per_day - self.warmup_candles
        current_relative_step = self.current_step - self.warmup_candles

        return self.state_builder.build_state(
            market_features    = market_features,
            position_features  = position_features,
            portfolio_features = portfolio_features,
            coin_metadata      = self.coin_metadata,
            regime_features    = regime_features,
            guardrail_features = guardrail_features,
            current_step       = current_relative_step,
            total_steps        = tradeable_steps,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public Gym interface
    # ══════════════════════════════════════════════════════════════════════════

    def reset(
        self,
        date:        Optional[str] = None,
        random_date: bool          = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment for a new episode.

        Returns:
            (initial_state, action_mask) — both needed by the IQN agent
        """
        # Pick date from curriculum pool if active, else all dates
        pool = self._curriculum_dates if self._curriculum_dates else self.available_dates

        if date is not None:
            self.current_date = date
        elif random_date:
            self.current_date = str(np.random.choice(pool))
        else:
            self.current_date = pool[0]

        # Load data with fallback
        if not self._load_episode_data(self.current_date):
            for fallback in pool:
                if self._load_episode_data(fallback):
                    self.current_date = fallback
                    break
            else:
                raise RuntimeError("Could not load data for any available date")

        # Reset all components
        self.position_manager.reset()
        self.reward_calculator.reset()

        self.current_step         = self.warmup_candles
        self.done                 = False
        self._cooldown_remaining  = 0
        self._hold_steps          = {}
        self._daily_trades        = 0
        self._daily_loss          = 0.0

        self.position_manager.set_step(self.current_step)
        prices = self._get_current_prices()
        self.position_manager.update_positions(prices)

        state = self._build_state(prices)
        if self.lstm_enabled:
            # Pre-populate with previous episode's tail states (cross-episode lookback)
            if self._prev_episode_states:
                self.state_buffer = list(self._prev_episode_states)
            else:
                self.state_buffer = []
            self.state_buffer.append(state)
            if len(self.state_buffer) > self.max_sequence_length:
                self.state_buffer = self.state_buffer[-self.max_sequence_length:]
        mask  = self.get_action_mask()

        return state, mask

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, np.ndarray, Dict[str, Any]]:
        """
        Execute one step.

        Returns:
            (next_state, reward, done, next_action_mask, info)
        """
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        prices = self._get_current_prices()
        self.position_manager.set_step(self.current_step)
        self.position_manager.update_positions(prices)

        # Validate action against mask (training safety net)
        mask = self.get_action_mask()
        if not mask[action]:
            # Agent tried an invalid action — penalise and stay
            reward_info = self.reward_calculator.calculate_step_reward(
                action_type   = "invalid",
                action_valid  = False,
                trade_notional = 0.0,
            )
            next_state = self._build_state(prices)
            next_mask  = self.get_action_mask()
            info = {
                "action": action, "action_type": "invalid",
                "action_success": False, "pnl": 0.0,
                "reward_breakdown": {"total": reward_info.total},
                "masked": True,
            }
            return next_state, reward_info.total, self.done, next_mask, info

        # Increment hold counters for all open positions
        for coin_idx in list(self.position_manager.positions.keys()):
            self._hold_steps[coin_idx] = self._hold_steps.get(coin_idx, 0) + 1

        # Decrement cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Parse and execute action
        action_type, coin_index, side, size_pct = self._parse_action(action)

        # Snapshot position side before close removes it (for action logging)
        _closed_side = None
        if action_type == "close" and coin_index in self.position_manager.positions:
            _closed_side = self.position_manager.positions[coin_index].side
        elif action_type == "close_worst" and self.position_manager.positions:
            # Identify worst position before it's closed
            worst_coin = min(
                self.position_manager.positions,
                key=lambda c: self.position_manager.positions[c].unrealized_pnl,
            )
            _closed_side = self.position_manager.positions[worst_coin].side

        success, pnl, notional, message = self._execute_action(
            action_type, coin_index, side, size_pct, prices
        )

        # Current state for reward context
        unrealized_pnl = sum(
            p.unrealized_pnl for p in self.position_manager.positions.values()
        )
        position_count = len(self.position_manager.positions)

        # Total notional of all open positions (for funding rate)
        total_open_notional = sum(
            p.size * self.leverage
            for p in self.position_manager.positions.values()
        )

        # Calculate reward
        reward_info = self.reward_calculator.calculate_step_reward(
            action_type         = action_type if success else "invalid",
            action_valid        = success,
            pnl                 = pnl,
            unrealized_pnl      = unrealized_pnl,
            position_count      = position_count,
            trade_notional      = notional,
            total_open_notional = total_open_notional,
        )

        # Advance time
        self.current_step += 1

        # Check episode end
        if self.current_step >= self.candles_per_day:
            self.done = True

            if self.force_close_at_end:
                final_prices      = self._get_current_prices()
                num_closed, final_pnl = self.position_manager.close_all_positions(final_prices)
                if num_closed > 0:
                    fc_notional = self.initial_balance * self.position_manager.position_size_pct * self.leverage * num_closed
                    fc_reward = self.reward_calculator.calculate_step_reward(
                        action_type    = "force_close",
                        action_valid   = True,
                        pnl            = final_pnl,
                        unrealized_pnl = 0,
                        position_count = 0,
                        trade_notional = fc_notional,
                    )
                    reward_info.total += fc_reward.total

            stats      = self.position_manager.get_statistics()
            end_reward = self.reward_calculator.calculate_episode_end_reward(
                final_pnl    = stats["total_pnl"],
                total_trades = stats["total_trades"],
                win_rate     = stats["win_rate"],
            )
            reward_info.total += end_reward.total

        next_state = self._build_state()
        if self.lstm_enabled:
            self.state_buffer.append(next_state)
            if len(self.state_buffer) > self.max_sequence_length:
                self.state_buffer = self.state_buffer[-self.max_sequence_length:]
            # Save tail for cross-episode lookback on next reset
            if self.done:
                self._prev_episode_states = list(self.state_buffer[-self.max_sequence_length:])
        next_mask  = self.get_action_mask()

        # Enrich action_type with side (e.g. "open_long", "close_short")
        display_action_type = action_type
        if action_type == "open" and side is not None:
            display_action_type = f"open_{side.name.lower()}"
        elif action_type in ("close", "close_worst") and _closed_side is not None and success:
            display_action_type = f"{action_type}_{_closed_side.name.lower()}"

        info = {
            "date":            self.current_date,
            "step":            self.current_step,
            "action":          action,
            "action_type":     display_action_type,
            "action_success":  success,
            "action_message":  message,
            "pnl":             pnl,
            "size_pct":        size_pct,
            "masked":          False,
            "cooldown":        self._cooldown_remaining,
            "daily_trades":    self._daily_trades,
            "reward_breakdown": {
                "realized_pnl":       reward_info.realized_pnl,
                "unrealized_shaping": reward_info.unrealized_shaping,
                "cost_penalty":       reward_info.cost_penalty,
                "funding_penalty":    reward_info.funding_penalty,
                "invalid_penalty":    reward_info.invalid_penalty,
                "holding_penalty":    reward_info.holding_penalty,
                "patience_bonus":     reward_info.patience_bonus,
                "total":              reward_info.total,
            },
            "portfolio_value": self.position_manager.get_portfolio_value(),
            "num_positions":   position_count,
            "unrealized_pnl":  unrealized_pnl,
        }

        return next_state, reward_info.total, self.done, next_mask, info

    # ══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ══════════════════════════════════════════════════════════════════════════

    def get_action_space_size(self) -> int:
        return self.action_size

    def get_state_space_size(self) -> int:
        return self.state_builder.get_state_dim()

    def get_episode_statistics(self) -> Dict:
        stats               = self.position_manager.get_statistics()
        stats["date"]       = self.current_date
        stats["final_step"] = self.current_step
        stats["reward_stats"] = self.reward_calculator.get_statistics()
        return stats

    def render(self, mode: str = "human"):
        if mode != "human":
            return
        print(f"\n{'='*60}")
        print(f"Date: {self.current_date} | Step: {self.current_step}/{self.candles_per_day}")
        print(f"Cooldown: {self._cooldown_remaining} | Daily trades: {self._daily_trades}")
        pv   = self.position_manager.get_portfolio_value()
        cash = self.position_manager.cash
        print(f"Portfolio: ${pv:.2f} | Cash: ${cash:.2f}")
        for coin_idx, pos in self.position_manager.positions.items():
            held = self._hold_steps.get(coin_idx, 0)
            side_str = "LONG" if pos.side == PositionSide.LONG else "SHORT"
            print(f"  [{coin_idx}] {pos.symbol}: {side_str} | held {held} steps | PnL ${pos.unrealized_pnl:.2f}")
        print(f"{'='*60}")

    def get_state_sequences(self) -> Optional[dict]:
        """Get multi-scale sequences for LSTM agent."""
        if not self.lstm_enabled:
            return None

        return self.state_builder.build_state_sequences(
            self.state_buffer,
            self.sequence_lengths
        )

    def close(self):
        pass


def make_env(
    config_path:    str = "config/trading.yaml",
    data_directory: str = "data/volatility",
) -> TradingEnvironment:
    return TradingEnvironment(config_path=config_path, data_directory=data_directory)
