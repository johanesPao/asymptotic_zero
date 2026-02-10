"""
State Builder

Builds the state representation for the DQN agent.
Combines technical indicators, position information and portfolio state
into a normalized feature vector.

State Components:
1. Market features (technical indicators per coin)
2. Position features (current position)
3. Portfolio features (cash, total value, etc.)
4. Coin metadata (gainer/loser, rank, daily change)
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class StateBuilder:
    """
    Builds state representation for the DQN agent.

    The state combines:
    - Technical indicators for each coin (from features model)
    - Position information (from position manager)
    - Portfolio-level features
    - Coin metadata (gainer/loser status, rank)
    """

    def __init__(
        self,
        config_path: str = "config/trading.yaml",
        num_coins: int = 20,
        num_features: int = 186,
    ):
        """
        Initialize state builder.

        Args:
            config_path: Path to trading configuration
            num_coins: Number of coins (gainers + losers)
            num_features: Number of technical indicator features per coin
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.num_coins = num_coins
        self.num_features = num_features

        # Normalization settings
        norm_config = self.config.get("normalization", {})
        self.norm_method = norm_config.get("method", "zscore")
        self.clip_outliers = norm_config.get("clip_outliers", True)
        self.clip_std = norm_config.get("clip_std", 3.0)

        # Features dimensions
        self.market_features_dim = num_coins * num_features
        self.position_features_dim = num_coins * 6
        self.portfolio_features_dim = 5
        self.metadata_features_dim = num_coins * 3
        self.time_features_dim = 2

        self.total_state_dim = (
            self.market_features_dim
            + self.position_features_dim
            + self.portfolio_features_dim
            + self.metadata_features_dim
            + self.time_features_dim
        )

        # Running statistics for normalization
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_state_dim(self) -> int:
        """Get total state dimension."""
        return self.total_state_dim

    def build_state(
        self,
        market_features: Dict[int, np.ndarray],
        position_features: np.ndarray,
        portfolio_features: np.ndarray,
        coin_metadata: Dict[int, Dict],
        current_step: int,
        total_steps: int,
    ) -> np.ndarray:
        """
        Build complete state vector.

        Args:
            market_features: Dict of coin_index -> feature array
            position_features: Array of shape (num_coins, 6)
            portfolio_features: Array of shape (5,)
            coin_metadata: Dict of coin_index -> {is_gainer, rank, daily_change}
            current_step: Current candle index
            total_steps: Total candles in episode

        Returns:
            Flattened state vector
        """
        # 1. Market features (num_coins * num_features)
        market_array = np.zeros((self.num_coins, self.num_features))
        for coin_idx, features in market_features.items():
            if coin_idx < self.num_coins and features is not None:
                # Handle NaN values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                market_array[coin_idx] = features[: self.num_features]

        # 2. Position features already in correct shape (num_coins, 6)
        position_array = np.nan_to_num(position_features, nan=0.0)

        # 3. Portfolio features (5,)
        portfolio_array = np.nan_to_num(portfolio_features, nan=0.0)

        # 4. Metadata features (num_coins, 3)
        metadata_array = np.zeros((self.num_coins, 3))
        for coin_idx, meta in coin_metadata.items():
            if coin_idx < self.num_coins:
                metadata_array[coin_idx, 0] = (
                    1.0 if meta.get("is_gainer", False) else 0.0
                )
                metadata_array[coin_idx, 1] = (
                    meta.get("rank", 0) / 10.0
                )  # Normalized rank
                metadata_array[coin_idx, 2] = np.clip(
                    meta.get("daily_change", 0.0) / 100.0,  # Normalized percentage
                    -1.0,
                    1.0,
                )

        # 5. Time features (2,)
        time_array = np.array(
            [
                current_step / total_steps,  # Progress through day
                1.0 - (current_step / total_steps),  # Time remaining
            ]
        )

        # Flattened and concatenate
        state = np.concatenate(
            [
                market_array.flatten(),
                position_array.flatten(),
                portfolio_array,
                metadata_array.flatten(),
                time_array,
            ]
        )

        # Normalize
        state = self._normalize(state)

        return state.astype(np.float32)

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state vector.

        Args:
            state: Raw state vector

        Returns:
            Normalized state vector
        """
        match self.norm_method:
            case "zscore":
                # Use running statistics if available
                if self._feature_means is not None and self._feature_stds is not None:
                    state = (state - self._feature_means) / (self._feature_stds + 1e-8)

                # Clip outliers
                if self.clip_outliers:
                    state = np.clip(state, -self.clip_std, self.clip_std)
            case "minmax":
                # Simple min-max to [-1, 1]
                state_min = np.min(state)
                state_max = np.max(state)
                if state_max - state_min > 1e-8:
                    state = 2 * (state - state_min) / (state_max - state_min) - 1
            case "robust":
                # Median-based normalization (robust to outliers)
                median = np.median(state)
                mad = np.median(np.abs(state - median)) + 1e-8
                state = (state - median) / mad

                if self.clip_outliers:
                    state = np.clip(state, -self.clip_std, self.clip_std)

        return state

    def update_statistics(self, states: List[np.ndarray]):
        """
        Update running statistics for normalization.

        Args:
            states: List of state vectors to compute statistics from
        """
        if not states:
            return

        states_array = np.array(states)
        self._feature_means = np.mean(states_array, axis=0)
        self._feature_stds = np.std(states_array, axis=0)

        # Prevent division by zero
        self._feature_stds = np.maximum(self._feature_stds, 1e-8)

        logger.info(f"Updated normalization statistics from {len(states)} states")

    def save_statistics(self, path: str):
        """Save normalization statistics to file."""
        if self._feature_means is None or self._feature_stds is None:
            logger.warning("No statistics to save")
            return

        np.savez(path, means=self._feature_means, stds=self._feature_stds)
        logger.info(f"Saved normalization statistics to {path}")

    def load_statistics(self, path: str):
        """Load normalization statistics from file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Statistics file not found: {path}")
            return

        data = np.load(path)
        self._feature_means = data["means"]
        self._feature_stds = data["stds"]
        logger.info(f"Loaded normalization statistics from {path}")

    def get_feature_info(self) -> Dict:
        """Get information about state features."""
        return {
            "total_dim": self.total_state_dim,
            "market_features": {
                "dim": self.market_features_dim,
                "shape": (self.num_coins, self.num_features),
            },
            "position_features": {
                "dim": self.position_features_dim,
                "shape": (self.num_coins, 6),
            },
            "portfolio_features": {"dim": self.portfolio_features_dim, "shape": (5,)},
            "metadata_features": {
                "dim": self.metadata_features_dim,
                "shape": (self.num_coins, 3),
            },
            "time_features": {"dim": self.time_features_dim, "shape": (2,)},
        }
