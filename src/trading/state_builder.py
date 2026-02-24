"""
State Builder — v2

Changes from v1:
  - Accepts regime_features (8) and guardrail_features (5) as explicit inputs
  - Position features expanded from 6 → 8 per coin (+hold_duration, +size_fraction)
  - Portfolio features expanded from 5 → 8 (+rolling_sharpe, +max_drawdown, +vol)
  - Metadata features expanded from 3 → 6 per coin (+hist_vol, +volume_rank, +price_range)
  - No coin identity in state — only behavioral / market characteristics

State dimensions:
  Market features:    20 × 26  =  520
  Position features:  20 × 8   =  160
  Portfolio features:           =    8
  Metadata features:  20 × 6   =  120
  Regime features:              =    8
  Guardrail features:           =    5
  Time features:                =    2
  ────────────────────────────────────
  Total:                         =  823
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


class StateBuilder:
    """
    Builds the flat state vector fed to the IQN agent.

    Called once per step by TradingEnvironment._build_state().
    All normalization happens here so the agent always receives
    values in a consistent numerical range.
    """

    def __init__(
        self,
        config_path:  str = "config/trading.yaml",
        num_coins:    int = 20,
        num_features: int = 186,
    ):
        self.config_path  = Path(config_path)
        self.config       = self._load_config()
        self.num_coins    = num_coins
        self.num_features = num_features

        norm_cfg           = self.config.get("normalization", {})
        self.norm_method   = norm_cfg.get("method",        "zscore")
        self.clip_outliers = norm_cfg.get("clip_outliers", True)
        self.clip_std      = norm_cfg.get("clip_std",      3.0)

        # Feature block dimensions
        self.market_dim    = num_coins * num_features   # 3720
        self.position_dim  = num_coins * 8              # 160  (+hold_duration, +size)
        self.portfolio_dim = 8                          # (+sharpe, +drawdown, +vol)
        self.metadata_dim  = num_coins * 6              # 120 (+hist_vol, +vol_rank, +range)
        self.regime_dim    = 8                          # NEW
        self.guardrail_dim = 5                          # NEW
        self.time_dim      = 2

        self.total_state_dim = (
            self.market_dim
            + self.position_dim
            + self.portfolio_dim
            + self.metadata_dim
            + self.regime_dim
            + self.guardrail_dim
            + self.time_dim
        )

        # Running statistics for z-score normalization
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds:  Optional[np.ndarray] = None

        logger.info(f"StateBuilder v2: total_dim={self.total_state_dim}")

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}")
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def get_state_dim(self) -> int:
        return self.total_state_dim

    # ──────────────────────────────────────────────────────────────────────────

    def build_state(
        self,
        market_features:    Dict[int, np.ndarray],
        position_features:  np.ndarray,          # (num_coins, 6) from PositionManager
        portfolio_features: np.ndarray,          # (5,) from PositionManager
        coin_metadata:      Dict[int, Dict],
        regime_features:    np.ndarray,          # (8,) from environment
        guardrail_features: np.ndarray,          # (5,) from environment
        current_step:       int,
        total_steps:        int,
    ) -> np.ndarray:
        """
        Build complete flat state vector.

        Args:
            market_features:    coin_idx → 1D feature array (length num_features)
            position_features:  shape (num_coins, 6) — raw from PositionManager
            portfolio_features: shape (5,) — raw from PositionManager
            coin_metadata:      coin_idx → {is_gainer, rank, daily_change}
            regime_features:    shape (8,) — computed by environment
            guardrail_features: shape (5,) — computed by environment
            current_step:       steps elapsed since warmup
            total_steps:        total tradeable steps in episode

        Returns:
            Flat float32 array of length total_state_dim
        """
        # ── 1. Market features (num_coins × num_features) ─────────────────────
        market_array = np.zeros((self.num_coins, self.num_features), dtype=np.float32)
        for coin_idx, features in market_features.items():
            if coin_idx < self.num_coins and features is not None:
                f = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                market_array[coin_idx] = f[: self.num_features]

        # ── 2. Position features (num_coins × 8) ──────────────────────────────
        # PositionManager now returns (num_coins, 8) natively.
        pos_array = np.nan_to_num(position_features, nan=0.0).astype(np.float32)
        if pos_array.shape[1] < 8:
            pad = np.zeros((self.num_coins, 8 - pos_array.shape[1]), dtype=np.float32)
            pos_array = np.concatenate([pos_array, pad], axis=1)
        else:
            pos_array = pos_array[:, :8]

        # ── 3. Portfolio features (8,) ─────────────────────────────────────────
        # PositionManager now returns (8,) natively.
        portfolio_array = np.nan_to_num(portfolio_features, nan=0.0).astype(np.float32)
        if len(portfolio_array) < 8:
            portfolio_array = np.concatenate([portfolio_array, np.zeros(8 - len(portfolio_array))])
        else:
            portfolio_array = portfolio_array[:8]

        # ── 4. Metadata features (num_coins × 6) ──────────────────────────────
        metadata_array = np.zeros((self.num_coins, 6), dtype=np.float32)
        for coin_idx, meta in coin_metadata.items():
            if coin_idx < self.num_coins:
                metadata_array[coin_idx, 0] = 1.0 if meta.get("is_gainer", False) else 0.0
                metadata_array[coin_idx, 1] = meta.get("rank", 0) / 10.0
                metadata_array[coin_idx, 2] = float(np.clip(
                    meta.get("daily_change", 0.0) / 100.0, -1.0, 1.0
                ))
                # Coin profile features
                metadata_array[coin_idx, 3] = float(meta.get("historical_volatility", 0.0))
                metadata_array[coin_idx, 4] = float(meta.get("volume_rank", 0.0))
                metadata_array[coin_idx, 5] = float(meta.get("avg_price_range", 0.0))

        # ── 5. Regime features (8,) ────────────────────────────────────────────
        regime_array = np.nan_to_num(
            np.asarray(regime_features, dtype=np.float32), nan=0.0
        )

        # ── 6. Guardrail features (5,) ─────────────────────────────────────────
        guard_array = np.nan_to_num(
            np.asarray(guardrail_features, dtype=np.float32), nan=0.0
        )

        # ── 7. Time features (2,) ─────────────────────────────────────────────
        progress   = current_step / max(total_steps, 1)
        time_array = np.array([
            float(np.clip(progress,       0, 1)),
            float(np.clip(1.0 - progress, 0, 1)),
        ], dtype=np.float32)

        # ── Concatenate all blocks ─────────────────────────────────────────────
        state = np.concatenate([
            market_array.flatten(),
            pos_array.flatten(),
            portfolio_array,
            metadata_array.flatten(),
            regime_array,
            guard_array,
            time_array,
        ])

        # ── Normalize ─────────────────────────────────────────────────────────
        state = self._normalize(state)

        return state.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        match self.norm_method:
            case "zscore":
                if self._feature_means is not None and self._feature_stds is not None:
                    state = (state - self._feature_means) / (self._feature_stds + 1e-8)
                if self.clip_outliers:
                    state = np.clip(state, -self.clip_std, self.clip_std)

            case "minmax":
                lo, hi = np.min(state), np.max(state)
                if hi - lo > 1e-8:
                    state = 2.0 * (state - lo) / (hi - lo) - 1.0

            case "robust":
                median = np.median(state)
                mad    = np.median(np.abs(state - median)) + 1e-8
                state  = (state - median) / mad
                if self.clip_outliers:
                    state = np.clip(state, -self.clip_std, self.clip_std)

        return state

    def update_statistics(self, states: List[np.ndarray]):
        if not states:
            return
        arr = np.array(states)
        self._feature_means = np.mean(arr, axis=0)
        self._feature_stds  = np.std(arr,  axis=0)
        self._feature_stds  = np.maximum(self._feature_stds, 1e-8)
        logger.info(f"Updated normalization stats from {len(states)} states")

    def save_statistics(self, path: str):
        if self._feature_means is None:
            logger.warning("No statistics to save")
            return
        np.savez(path, means=self._feature_means, stds=self._feature_stds)
        logger.info(f"Saved normalization stats → {path}")

    def load_statistics(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.warning(f"Statistics file not found: {path}")
            return
        data = np.load(p)
        self._feature_means = data["means"]
        self._feature_stds  = data["stds"]
        logger.info(f"Loaded normalization stats ← {path}")

    def get_feature_info(self) -> Dict:
        return {
            "total_dim":         self.total_state_dim,
            "market_features":   {"dim": self.market_dim,    "shape": (self.num_coins, self.num_features)},
            "position_features": {"dim": self.position_dim,  "shape": (self.num_coins, 8)},
            "portfolio_features":{"dim": self.portfolio_dim, "shape": (8,)},
            "metadata_features": {"dim": self.metadata_dim,  "shape": (self.num_coins, 6)},
            "regime_features":   {"dim": self.regime_dim,    "shape": (8,)},
            "guardrail_features":{"dim": self.guardrail_dim, "shape": (5,)},
            "time_features":     {"dim": self.time_dim,      "shape": (2,)},
        }
    
    def build_state_sequences(
        self,
        state_buffer: list,
        sequence_lengths: list = [10, 20, 30]
    ) -> dict:
        """
        Build multi-scale state sequences from buffer for LSTM.
        
        Args:
            state_buffer: List of state arrays, oldest to newest
            sequence_lengths: [short, medium, long] timescales
        
        Returns:
            dict: {'short': (10, state_dim), 'medium': (20, state_dim), 'long': (30, state_dim)}
        """
        max_len = max(sequence_lengths)
        
        if len(state_buffer) < max_len:
            # Pad with zeros if buffer not full yet
            padding_needed = max_len - len(state_buffer)
            zero_state = np.zeros(self.total_state_dim, dtype=np.float32)
            padded_buffer = [zero_state] * padding_needed + state_buffer
        else:
            padded_buffer = state_buffer
        
        # Extract last N states for each timescale
        sequences = {}
        for name, length in zip(['short', 'medium', 'long'], sequence_lengths):
            seq = np.array(padded_buffer[-length:], dtype=np.float32)
            sequences[name] = seq
        
        return sequences
