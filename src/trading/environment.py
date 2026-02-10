"""
Trading Environment

Gym-style environment for DQN cryptocurrency trading.
Implements the standard Gym interface: reset(), step(), render()

Episode Structure:
- One episode = one trading day
- Each step = one 5-minute candle
- Agent observes state, takes action, receives reward
- Episode ends at end of day (all positions closed)
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


class TradingEnvironment:
    """
    Gym-style trading environment for cryptocurrency futures.

    The environment simulates intraday trading on volatile cryptocurrencies
    using 5-minute candle data. The agent can go long or short on any of
    the top 10 gainers or top 10 losers of the day.

    Action Space (63 discrete actions):
        0: HOLD
        1-10: LONG gainer[0-9]
        11-20: SHORT gainer[0-9]
        21-30: CLOSE gainer[0-9]
        31-40: LONG loser[0-9]
        41-50: SHORT loser[0-9]
        51-60: CLOSE loser[0-9]
        61: CLOSE_ALL
        62: CLOSE_WORST

    State Space:
        - Market features: 20 coins × 186 indicators = 3720
        - Position features: 20 coins × 6 features = 120
        - Portfolio features: 5
        - Metadata features: 20 coins × 3 = 60
        - Time features: 2
        - Total: 3907 features
    """

    def __init__(
        self,
        config_path: str = "config/trading.yaml",
        data_directory: str = "data/volatility",
        features_config: str = "config/features.yaml",
    ):
        """
        Initialize the trading environment.

        Args:
            config_path: Path to trading configuration
            data_directory: Base directory for volatility data
            features_config: Path to features configuration
        """
        self.config_path = Path(config_path)
        self.data_directory = Path(data_directory)
        self.features_config = Path(features_config)

        # Load configuration
        self.config = self._load_config()

        # Extract settings
        portfolio_cfg = self.config.get("portfolio", {})
        costs_cfg = self.config.get("costs", {})
        actions_cfg = self.config.get("actions", {})
        episode_cfg = self.config.get("episode", {})

        # Episode settings
        self.candles_per_day = episode_cfg.get("candles_per_day", 288)
        self.warmup_candles = episode_cfg.get("warmup_candles", 200)
        self.force_close_at_end = episode_cfg.get("force_close_at_end", True)

        # Action settings
        self.num_coins_per_side = actions_cfg.get("num_coins_per_side", 10)
        self.num_coins = self.num_coins_per_side * 2  # Gainers + Losers
        self.enable_close_all = actions_cfg.get("enable_close_all", True)
        self.enable_close_worst = actions_cfg.get("enable_close_worst", True)

        # Calculate action space size
        # 0: HOLD
        # 1-10: LONG gainer, 11-20: SHORT gainer, 21-30: CLOSE gainer
        # 31-40: LONG loser, 41-50: SHORT loser, 51-60: CLOSE loser
        # 61: CLOSE_ALL, 62: CLOSE_WORST
        self.action_size = 1 + (self.num_coins_per_side * 6)  # 61
        if self.enable_close_all:
            self.action_size += 1  # 62
        if self.enable_close_worst:
            self.action_size += 1  # 63

        # Initialize components
        self.position_manager = PositionManager(
            initial_balance=portfolio_cfg.get("initial_balance", 10000.0),
            max_positions=portfolio_cfg.get("max_positions", 3),
            position_size_pct=portfolio_cfg.get("position_size_pct", 0.20),
            taker_fee_pct=costs_cfg.get("taker_fee_pct", 0.04),
            slippage_pct=costs_cfg.get("slippage_pct", 0.01),
        )

        self.reward_calculator = RewardCalculator(
            config_path=str(self.config_path),
            initial_balance=portfolio_cfg.get("initial_balance", 10000.0),
        )

        self.technical_indicators = TechnicalIndicators(
            config_path=str(self.features_config),
            data_directory=str(self.data_directory / "5m"),
        )

        # Get feature count from calculator
        feature_names = self.technical_indicators.get_feature_names()
        self.num_features = len(feature_names)

        self.state_builder = StateBuilder(
            config_path=str(self.config_path),
            num_coins=self.num_coins,
            num_features=self.num_features,
        )

        # Load top movers index
        self.top_movers_file = self.data_directory / "top_movers.parquet"
        self.top_movers_df = self._load_top_movers()
        self.available_dates = self._get_available_dates()

        # Episode state
        self.current_date: Optional[str] = None
        self.current_step: int = 0
        self.episode_data: Dict[int, pl.DataFrame] = {}  # coin_index -> DataFrame
        self.coin_symbols: Dict[int, str] = {}  # coin_index -> symbol
        self.coin_metadata: Dict[int, Dict] = {}  # coin_index -> metadata
        self.done: bool = False

        logger.info("Environment initialized:")
        logger.info(f"  Action space: {self.action_size}")
        logger.info(f"  State space: {self.state_builder.get_state_dim()}")
        logger.info(f"  Available dates: {len(self.available_dates)}")
        logger.info(f"  Features per coin: {self.num_features}")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_top_movers(self) -> pl.DataFrame:
        """Load top movers index."""
        if not self.top_movers_file.exists():
            raise FileNotFoundError(
                f"Top movers file not found: {self.top_movers_file}"
            )

        return pl.read_parquet(self.top_movers_file)

    def _get_available_dates(self) -> List[str]:
        """Get list of dates with data available."""
        dates = self.top_movers_df["date"].unique().sort().to_list()
        return [str(d) for d in dates]

    def _load_episode_data(self, date: str) -> bool:
        """
        Load all data for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            True if data loaded successfully
        """
        # Get top movers for this date
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        date_data = self.top_movers_df.filter(pl.col("date") == date_obj)

        if len(date_data) == 0:
            logger.warning(f"No top movers data for date {date}")
            return False

        row = date_data.row(0, named=True)
        gainers = row["gainers"]
        losers = row["losers"]
        gainers_pct = row["gainers_pct"]
        losers_pct = row["losers_pct"]

        self.episode_data = {}
        self.coin_symbols = {}
        self.coin_metadata = {}

        # Track actual index (compacted, no gaps)
        actual_index = 0

        # Load gainers (indices 0-9)
        for i, (symbol, pct) in enumerate(zip(gainers, gainers_pct)):
            if i >= self.num_coins_per_side:
                break

            data_file = self.data_directory / "5m" / symbol / f"{date}.parquet"
            if not data_file.exists():
                logger.debug(f"Missing data file: {data_file}")
                continue

            df = pl.read_parquet(data_file)
            df = self.technical_indicators.calculate(df)

            # Use compacted index (no gaps)
            self.episode_data[actual_index] = df
            self.coin_symbols[actual_index] = symbol
            self.coin_metadata[actual_index] = {
                "is_gainer": True,
                "rank": i + 1,
                "daily_change": pct,
            }
            actual_index += 1

        # Load losers (continue compacting from where gainers left off)
        for i, (symbol, pct) in enumerate(zip(losers, losers_pct)):
            if i >= self.num_coins_per_side:
                break

            data_file = self.data_directory / "5m" / symbol / f"{date}.parquet"
            if not data_file.exists():
                logger.debug(f"Missing data file: {data_file}")
                continue

            df = pl.read_parquet(data_file)
            df = self.technical_indicators.calculate(df)

            # Use compacted index (no gaps)
            self.episode_data[actual_index] = df
            self.coin_symbols[actual_index] = symbol
            self.coin_metadata[actual_index] = {
                "is_gainer": False,
                "rank": i + 1,
                "daily_change": pct,
            }
            actual_index += 1

        if not self.episode_data:
            logger.warning(f"No data loaded for date {date}")
            return False

        logger.debug(f"Loaded {len(self.episode_data)} coins for date {date}")
        return True

    def _get_current_prices(self) -> Dict[int, float]:
        """
        Get current prices for all coins.

        For coins that have run out of data, use their last available price
        to allow positions to be closed properly.
        """
        prices = {}
        for coin_idx, df in self.episode_data.items():
            if self.current_step < len(df):
                # Normal case: use current price
                prices[coin_idx] = df["close"][self.current_step]
            else:
                # Coin data ended: use last available price
                # This allows positions to be closed even when data ends early
                prices[coin_idx] = df["close"][-1]

        return prices

    def _get_market_features(self) -> Dict[int, np.ndarray]:
        """
        Get market features for all coins at current step.

        For coins that have run out of data, use their last available features.
        """
        features = {}

        # Get list of feature columns (excluding OHLCV)
        feature_cols = self.technical_indicators.get_feature_names()

        for coin_idx, df in self.episode_data.items():
            # Determine which row to use
            if self.current_step < len(df):
                row_idx = self.current_step
            else:
                # Coin data ended: use last available row
                row_idx = len(df) - 1

            # Extract feature values
            row = df[row_idx]
            feature_values = []
            for col in feature_cols:
                if col in df.columns:
                    feature_values.append(row[col].item())
                else:
                    feature_values.append(0.0)
            features[coin_idx] = np.array(feature_values)

        return features

    def _parse_action(self, action: int) -> Tuple[str, int, Optional[PositionSide]]:
        """
        Parse action integer into action type and target.

        Args:
            action: Action integer

        Returns:
            (action_type, coin_index, side) tuple
        """
        n = self.num_coins_per_side

        if action == 0:
            return "hold", -1, None

        # LONG gainer (1-10)
        if 1 <= action <= n:
            return "open", action - 1, PositionSide.LONG

        # SHORT gainer (11-20)
        if n + 1 <= action <= 2 * n:
            return "open", action - n - 1, PositionSide.SHORT

        # CLOSE gainer (21-30)
        if 2 * n + 1 <= action <= 3 * n:
            return "close", action - 2 * n - 1, None

        # LONG loser (31-40)
        if 3 * n + 1 <= action <= 4 * n:
            return "open", n + (action - 3 * n - 1), PositionSide.LONG

        # SHORT loser (41-50)
        if 4 * n + 1 <= action <= 5 * n:
            return "open", n + (action - 4 * n - 1), PositionSide.SHORT

        # CLOSE loser (51-60)
        if 5 * n + 1 <= action <= 6 * n:
            return "close", n + (action - 5 * n - 1), None

        # CLOSE_ALL (61)
        if action == 6 * n + 1 and self.enable_close_all:
            return "close_all", -1, None

        # CLOSE_WORST (62)
        if action == 6 * n + 2 and self.enable_close_worst:
            return "close_worst", -1, None

        return "invalid", -1, None

    def _execute_action(
        self,
        action_type: str,
        coin_index: int,
        side: Optional[PositionSide],
        prices: Dict[int, float],
    ) -> Tuple[bool, float, str]:
        """
        Execute a trading action.

        Args:
            action_type: Type of action
            coin_index: Target coin index (-1 for portfolio actions)
            side: Position side for open actions
            prices: Current prices for all coins

        Returns:
            (success, pnl, message) tuple
        """
        if action_type == "hold":
            return True, 0.0, "Hold"

        elif action_type == "open":
            if coin_index not in self.coin_symbols:
                return False, 0.0, f"Invalid coin index: {coin_index}"

            if coin_index not in prices:
                return False, 0.0, f"No price for coin {coin_index}"

            symbol = self.coin_symbols[coin_index]
            is_gainer = self.coin_metadata[coin_index]["is_gainer"]

            success, message = self.position_manager.open_position(
                coin_index=coin_index,
                symbol=symbol,
                side=side,
                price=prices[coin_index],
                is_gainer=is_gainer,
            )
            return success, 0.0, message

        elif action_type == "close":
            if coin_index not in prices:
                return False, 0.0, f"No price for coin {coin_index}"

            success, message, pnl = self.position_manager.close_position(
                coin_index=coin_index,
                price=prices[coin_index],
            )
            return success, pnl, message

        elif action_type == "close_all":
            num_closed, total_pnl = self.position_manager.close_all_positions(prices)
            return True, total_pnl, f"Closed {num_closed} positions"

        elif action_type == "close_worst":
            success, pnl = self.position_manager.close_worst_position(prices)
            return success, pnl, "Closed worst position" if success else "No positions"

        return False, 0.0, "Invalid action"

    def reset(
        self,
        date: Optional[str] = None,
        random_date: bool = True,
    ) -> np.ndarray:
        """
        Reset environment for a new episode.

        Args:
            date: Specific date to use (YYYY-MM-DD)
            random_date: If True and date is None, pick random date

        Returns:
            Initial state observation
        """
        # Select date
        if date is not None:
            self.current_date = date
        elif random_date:
            self.current_date = np.random.choice(self.available_dates)
        else:
            self.current_date = self.available_dates[0]

        # Load episode data
        if not self._load_episode_data(self.current_date):
            # Try another date if loading fails
            for fallback_date in self.available_dates:
                if self._load_episode_data(fallback_date):
                    self.current_date = fallback_date
                    break
            else:
                raise RuntimeError("Could not load data for any date")

        # Reset components
        self.position_manager.reset()
        self.reward_calculator.reset()

        # Start after warmup period
        self.current_step = self.warmup_candles
        self.done = False

        # Update position manager step
        self.position_manager.set_step(self.current_step)

        # Update positions with initial prices
        prices = self._get_current_prices()
        self.position_manager.update_positions(prices)

        # Build initial state
        state = self._build_state()

        logger.debug(
            f"Episode reset: date={self.current_date}, step={self.current_step}"
        )

        return state

    def _build_state(self) -> np.ndarray:
        """Build current state observation."""
        market_features = self._get_market_features()
        position_features = self.position_manager.get_position_features(self.num_coins)
        portfolio_features = self.position_manager.get_portfolio_features()

        # Calculate total tradeable steps
        total_steps = self.candles_per_day - self.warmup_candles
        current_relative_step = self.current_step - self.warmup_candles

        state = self.state_builder.build_state(
            market_features=market_features,
            position_features=position_features,
            portfolio_features=portfolio_features,
            coin_metadata=self.coin_metadata,
            current_step=current_relative_step,
            total_steps=total_steps,
        )

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0 to action_size-1)

        Returns:
            (next_state, reward, done, info) tuple
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Get current prices
        prices = self._get_current_prices()

        # Update position manager
        self.position_manager.set_step(self.current_step)
        self.position_manager.update_positions(prices)

        # Parse and execute action
        action_type, coin_index, side = self._parse_action(action)
        success, pnl, message = self._execute_action(
            action_type, coin_index, side, prices
        )

        # Calculate reward using reward calculator
        unrealized_pnl = sum(
            p.unrealized_pnl for p in self.position_manager.positions.values()
        )
        position_count = len(self.position_manager.positions)

        reward_info = self.reward_calculator.calculate_step_reward(
            action_type=action_type if success else "invalid",
            action_valid=success,
            pnl=pnl,
            unrealized_pnl=unrealized_pnl,
            position_count=position_count,
        )

        # Advance time
        self.current_step += 1

        # Check if episode is done
        if self.current_step >= self.candles_per_day:
            self.done = True

            # Force close all positions at end of day
            if self.force_close_at_end:
                final_prices = self._get_current_prices()
                num_closed, final_pnl = self.position_manager.close_all_positions(
                    final_prices
                )
                if num_closed > 0:
                    # CRITICAL: Add force-close PnL to rewards!
                    force_close_reward = self.reward_calculator.calculate_step_reward(
                        action_type="force_close",
                        action_valid=True,
                        pnl=final_pnl,
                        unrealized_pnl=0,
                        position_count=0,
                    )
                    reward_info.total += force_close_reward.total
                    
                    logger.debug(
                        f"Force closed {num_closed} positions, PnL: {final_pnl:.2f}, Reward: {force_close_reward.total:.2f}"
                    )

            # Episode end reward
            stats = self.position_manager.get_statistics()
            end_reward = self.reward_calculator.calculate_episode_end_reward(
                final_pnl=stats["total_pnl"],
                total_trades=stats["total_trades"],
                win_rate=stats["win_rate"],
            )
            reward_info.total += end_reward.total

        # Build next state
        next_state = self._build_state()

        # Info dictionary
        info = {
            "date": self.current_date,
            "step": self.current_step,
            "action": action,
            "action_type": action_type,
            "action_success": success,
            "action_message": message,
            "pnl": pnl,
            "reward_breakdown": {
                "base": reward_info.base_reward,
                "pnl": reward_info.pnl_reward,
                "penalty": reward_info.penalty,
                "bonus": reward_info.bonus,
                "total": reward_info.total,
            },
            "portfolio_value": self.position_manager.get_portfolio_value(),
            "num_positions": len(self.position_manager.positions),
            "unrealized_pnl": unrealized_pnl,
        }

        return next_state, reward_info.total, self.done, info

    def get_action_space_size(self) -> int:
        """Get size of action space."""
        return self.action_size

    def get_state_space_size(self) -> int:
        """Get size of state space."""
        return self.state_builder.get_state_dim()

    def get_episode_statistics(self) -> Dict:
        """Get statistics for current/last episode."""
        stats = self.position_manager.get_statistics()
        stats["date"] = self.current_date
        stats["final_step"] = self.current_step
        stats["reward_stats"] = self.reward_calculator.get_statistics()
        return stats

    def render(self, mode: str = "human"):
        """
        Render the environment state.

        Args:
            mode: Render mode ("human" for console output)
        """
        if mode != "human":
            return

        print(f"\n{'='*60}")
        print(
            f"Date: {self.current_date} | Step: {self.current_step}/{self.candles_per_day}"
        )
        print(f"{'='*60}")

        # Portfolio
        pv = self.position_manager.get_portfolio_value()
        cash = self.position_manager.cash
        print(f"Portfolio: ${pv:.2f} | Cash: ${cash:.2f}")

        # Positions
        print(
            f"\nPositions ({len(self.position_manager.positions)}/{self.position_manager.max_positions}):"
        )
        for coin_idx, pos in self.position_manager.positions.items():
            side_str = "LONG" if pos.side == PositionSide.LONG else "SHORT"
            pnl_str = (
                f"+{pos.unrealized_pnl:.2f}"
                if pos.unrealized_pnl >= 0
                else f"{pos.unrealized_pnl:.2f}"
            )
            print(
                f"  [{coin_idx}] {pos.symbol}: {side_str} @ {pos.entry_price:.6f} | PnL: {pnl_str}"
            )

        if not self.position_manager.positions:
            print("  (no positions)")

        print(f"{'='*60}\n")

    def close(self):
        """Clean up environment resources."""
        pass


# Convenience function for creating environment
def make_env(
    config_path: str = "config/trading.yaml",
    data_directory: str = "data/volatility",
) -> TradingEnvironment:
    """
    Create a trading environment.

    Args:
        config_path: Path to trading configuration
        data_directory: Path to data directory

    Returns:
        TradingEnvironment instance
    """
    return TradingEnvironment(
        config_path=config_path,
        data_directory=data_directory,
    )
