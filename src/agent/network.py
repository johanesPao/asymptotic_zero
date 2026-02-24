"""
IQN Network — Implicit Quantile Network

Replaces the previous Double DQN + LSTM architecture.

What IQN does differently from DDQN:
  - DDQN learns: Q(s, a) = expected return for action a in state s
  - IQN  learns: Z(s, a, τ) = the full return DISTRIBUTION, parameterized
                               by risk level τ (a random quantile in [0, 1])

At inference, we sample many τ values and take their average to get a
stable estimate. This gives us a richer signal than a single expected value.

Architecture:
  1. Trunk (MLP):     state → shared embedding
  2. Quantile embed:  τ → same-dim vector via cosine embedding
  3. Elementwise multiply trunk × quantile_embed → combined representation
  4. Output head:     combined → Q(s, a, τ) for all actions

Risk distortion (optional):
  - "neutral": sample τ uniformly → maximize expected return
  - "cvar":    sample τ from [0, alpha] → focus on worst outcomes
  - "wang":    apply Wang distortion → penalise variance

References:
  Dabney et al., 2018 — "Implicit Quantile Networks for Distributional RL"
  https://arxiv.org/abs/1806.06923
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class CosineBasisLayer(layers.Layer):
    """
    Maps scalar quantile τ ∈ [0, 1] to a vector of dimension `embedding_dim`
    using the cosine basis from the IQN paper:

        φ(τ)_i = ReLU( Σ_j cos(π × j × τ) × w_ij + b_i )

    This lets the network learn smooth, continuous functions of τ
    rather than treating each quantile level as a discrete input.
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_basis       = 64   # Number of cosine basis functions

        self.dense = layers.Dense(embedding_dim, activation="relu")

    def call(self, tau: tf.Tensor) -> tf.Tensor:
        """
        Args:
            tau: shape (batch, num_quantiles) — quantile levels

        Returns:
            shape (batch, num_quantiles, embedding_dim)
        """
        # tau: (B, N) → (B, N, 1)
        tau_expanded = tf.expand_dims(tau, axis=-1)

        # Build cosine basis: i = 1, 2, ..., n_basis
        # (1, 1, n_basis)
        i = tf.cast(
            tf.range(1, self.n_basis + 1), dtype=tf.float32
        )[tf.newaxis, tf.newaxis, :]

        # cos(π × i × τ): (B, N, n_basis)
        cos_features = tf.cos(np.pi * i * tau_expanded)

        # Apply learned linear → embedding_dim, then ReLU
        # Reshape for Dense: (B×N, n_basis) → dense → (B×N, embedding_dim)
        B = tf.shape(tau)[0]
        N = tf.shape(tau)[1]
        cos_flat    = tf.reshape(cos_features, (-1, self.n_basis))
        embed_flat  = self.dense(cos_flat)         # (B×N, embedding_dim)
        embed       = tf.reshape(embed_flat, (B, N, self.embedding_dim))

        return embed  # (B, N, embedding_dim)


class IQNNetwork(keras.Model):
    """
    Implicit Quantile Network.

    At each forward pass:
      1. Trunk encodes the state:   state → h  (batch, embedding_dim)
      2. Cosine layer encodes τ:    τ     → φ  (batch, N, embedding_dim)
      3. Elementwise multiply:      h × φ → z  (batch, N, embedding_dim)
      4. Output head:               z     → Q  (batch, N, action_size)

    The output Q(s, a, τ) represents the τ-th quantile of the return
    distribution for action a in state s.
    """

    def __init__(
        self,
        state_dim:     int,
        action_size:   int,
        config_path:   str = "config/trading.yaml",
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.action_size = action_size

        # Load config
        config       = self._load_config(config_path)
        net_cfg      = config.get("network", {})
        hidden_sizes = net_cfg.get("hidden_layers", [512, 256, 128])
        dropout_rate = net_cfg.get("dropout_rate",  0.1)
        use_ln       = net_cfg.get("use_layer_norm", True)

        self.embedding_dim          = net_cfg.get("embedding_dim",          64)
        self.num_quantiles_train    = net_cfg.get("num_quantiles_train",     8)
        self.num_quantiles_inference = net_cfg.get("num_quantiles_inference", 32)
        self.risk_distortion        = net_cfg.get("risk_distortion",       "neutral")
        self.cvar_alpha             = net_cfg.get("cvar_alpha",             0.25)
        self.q_clip                 = net_cfg.get("q_clip",                10.0)

        # ── Trunk (shared feature extractor) ──────────────────────────────────
        # Maps state → embedding of size hidden_sizes[-1]
        trunk_layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            trunk_layers.append(layers.Dense(h, activation=None))
            if use_ln:
                trunk_layers.append(layers.LayerNormalization())
            trunk_layers.append(layers.Activation("relu"))
            trunk_layers.append(layers.Dropout(dropout_rate))
            in_dim = h

        # Project to embedding_dim for the multiply step
        trunk_layers.append(layers.Dense(self.embedding_dim, activation="relu"))
        self.trunk = keras.Sequential(trunk_layers, name="trunk")

        # ── Quantile embedding ─────────────────────────────────────────────────
        self.cosine_embed = CosineBasisLayer(self.embedding_dim, name="quantile_embed")

        # ── Output head ────────────────────────────────────────────────────────
        # (batch × N, embedding_dim) → (batch × N, action_size)
        self.output_head = keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(action_size, activation=None),
        ], name="output_head")

        logger.info(
            f"IQNNetwork | state={state_dim} | actions={action_size} | "
            f"embed={self.embedding_dim} | τ_train={self.num_quantiles_train} | "
            f"τ_infer={self.num_quantiles_inference} | "
            f"risk={self.risk_distortion}"
        )

    def _load_config(self, config_path: str) -> dict:
        p = Path(config_path)
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f)

    # ──────────────────────────────────────────────────────────────────────────

    def _sample_quantiles(
        self,
        batch_size:   int,
        num_quantiles: int,
        training:     bool,
    ) -> tf.Tensor:
        """
        Sample τ values according to the risk distortion setting.

        Returns:
            shape (batch_size, num_quantiles)
        """
        match self.risk_distortion:
            case "neutral":
                # Uniform sampling — standard IQN
                tau = tf.random.uniform(
                    (batch_size, num_quantiles), minval=0.0, maxval=1.0
                )

            case "cvar":
                # Conditional Value at Risk — focus on worst α-fraction
                # Sample τ ∈ [0, alpha] only, so agent optimises the tail
                tau = tf.random.uniform(
                    (batch_size, num_quantiles), minval=0.0, maxval=self.cvar_alpha
                )

            case "wang":
                # Wang distortion: τ → Φ(Φ⁻¹(τ) + η)
                # η < 0 = risk-averse, η > 0 = risk-seeking
                # We use η = -0.75 (conservative)
                eta = -0.75
                u   = tf.random.uniform((batch_size, num_quantiles))
                # Approximate Φ⁻¹ via the probit approximation
                tau = 0.5 * (1.0 + tf.math.erf(
                    (self._probit(u) + eta) / tf.sqrt(2.0)
                ))

            case _:
                tau = tf.random.uniform(
                    (batch_size, num_quantiles), minval=0.0, maxval=1.0
                )

        return tf.cast(tau, tf.float32)

    @staticmethod
    def _probit(u: tf.Tensor) -> tf.Tensor:
        """Approximate probit (inverse CDF of standard normal)."""
        # Rational approximation — good to ~1e-4 accuracy
        u = tf.clip_by_value(u, 1e-6, 1 - 1e-6)
        return tf.sqrt(2.0) * tf.math.erfinv(2.0 * u - 1.0)

    # ──────────────────────────────────────────────────────────────────────────

    def call(
        self,
        state:         tf.Tensor,
        num_quantiles: int  = None,
        training:      bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.

        Args:
            state:         (batch, state_dim)
            num_quantiles: Override number of τ samples (None = use config)
            training:      Controls dropout

        Returns:
            q_values: (batch, num_quantiles, action_size)
                      The τ-th quantile of the return for each action
            tau:      (batch, num_quantiles) — the sampled quantile levels
        """
        if num_quantiles is None:
            num_quantiles = (
                self.num_quantiles_train if training
                else self.num_quantiles_inference
            )

        B = tf.shape(state)[0]

        # 1. Encode state via trunk: (B, embedding_dim)
        h = self.trunk(state, training=training)

        # 2. Sample quantiles: (B, N)
        tau = self._sample_quantiles(B, num_quantiles, training)

        # 3. Cosine quantile embedding: (B, N, embedding_dim)
        phi = self.cosine_embed(tau)

        # 4. Combine: (B, 1, embedding_dim) × (B, N, embedding_dim) → (B, N, embedding_dim)
        h_expanded = tf.expand_dims(h, axis=1)       # (B, 1, embedding_dim)
        combined   = h_expanded * phi                 # element-wise, broadcast over N

        # 5. Output head: reshape to (B×N, embedding_dim) → dense → reshape back
        BN         = B * num_quantiles
        combined_flat = tf.reshape(combined, (BN, self.embedding_dim))
        q_flat        = self.output_head(combined_flat, training=training)  # (B×N, actions)
        q_values      = tf.reshape(q_flat, (B, num_quantiles, self.action_size))
        q_values      = tf.clip_by_value(q_values, -self.q_clip, self.q_clip)

        return q_values, tau

    def q_mean(
        self,
        state:    tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Convenience: return mean Q-values across all quantiles.
        Shape: (batch, action_size)
        Used for greedy action selection at inference.
        """
        q_values, _ = self(state, training=training)
        return tf.reduce_mean(q_values, axis=1)  # average over quantile dim

    def masked_q_mean(
        self,
        state: tf.Tensor,
        mask:  tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Mean Q-values with invalid actions set to -∞.

        Args:
            state: (batch, state_dim)
            mask:  (batch, action_size) bool — True = valid

        Returns:
            (batch, action_size) with invalid actions at -1e9
        """
        q_mean = self.q_mean(state, training=training)
        mask_f = tf.cast(mask, tf.float32)
        return q_mean * mask_f + (1.0 - mask_f) * (-1e9)


# Alias for backward compatibility
DQNNetwork = IQNNetwork


class LSTMIQNNetwork(keras.Model):
    """
    Multi-Scale LSTM + IQN Network for temporal awareness.
    
    Processes three timescales simultaneously:
    - Short (10 steps = 50 min): immediate momentum, entry/exit timing
    - Medium (20 steps = 100 min): intraday trend direction
    - Long (30 steps = 150 min): broader context, multi-hour trends
    
    Architecture:
    1. Three parallel LSTMs process each timescale
    2. Concatenate LSTM outputs → combined temporal context (192-dim)
    3. Trunk processes combined context → shared embedding
    4. IQN quantile embedding and output (same as original IQN)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_size: int,
        config_path: str = "config/trading.yaml",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_size = action_size
        
        # Load config
        config = self._load_config(config_path)
        
        # LSTM config
        lstm_cfg = config.get("lstm", {})
        self.sequence_lengths = lstm_cfg.get("sequence_lengths", [10, 20, 30])
        lstm_hidden_sizes = lstm_cfg.get("hidden_sizes", [64, 64, 64])
        lstm_dropout = lstm_cfg.get("dropout_rate", 0.2)
        
        # IQN config
        net_cfg = config.get("network", {})
        hidden_sizes = net_cfg.get("hidden_layers", [512, 256, 128])
        dropout_rate = net_cfg.get("dropout_rate", 0.1)
        use_ln = net_cfg.get("use_layer_norm", True)
        
        self.embedding_dim = net_cfg.get("embedding_dim", 64)
        self.num_quantiles_train = net_cfg.get("num_quantiles_train", 8)
        self.num_quantiles_inference = net_cfg.get("num_quantiles_inference", 32)
        self.risk_distortion = net_cfg.get("risk_distortion", "neutral")
        self.cvar_alpha = net_cfg.get("cvar_alpha", 0.25)
        self.q_clip = net_cfg.get("q_clip", 10.0)

        logger.info(f"Building LSTM-IQN: sequences={self.sequence_lengths}, LSTM hidden={lstm_hidden_sizes}")
        
        # ── Three parallel LSTMs ──────────────────────────────────────
        self.lstm_short = layers.LSTM(
            lstm_hidden_sizes[0],
            return_sequences=False,
            dropout=lstm_dropout,
            name="lstm_short"
        )
        
        self.lstm_medium = layers.LSTM(
            lstm_hidden_sizes[1],
            return_sequences=False,
            dropout=lstm_dropout,
            name="lstm_medium"
        )
        
        self.lstm_long = layers.LSTM(
            lstm_hidden_sizes[2],
            return_sequences=False,
            dropout=lstm_dropout,
            name="lstm_long"
        )
        
        # Combined context dimension
        combined_dim = sum(lstm_hidden_sizes)  # 64 + 64 + 64 = 192
        
        # ── Trunk (shared embedding from combined context) ────────────
        trunk_layers = []
        in_dim = combined_dim
        for h in hidden_sizes:
            trunk_layers.append(layers.Dense(h, activation=None))
            if use_ln:
                trunk_layers.append(layers.LayerNormalization())
            trunk_layers.append(layers.Activation("relu"))
            trunk_layers.append(layers.Dropout(dropout_rate))
            in_dim = h
        
        # Project to embedding_dim for multiply step
        trunk_layers.append(layers.Dense(self.embedding_dim, activation="relu"))
        self.trunk = keras.Sequential(trunk_layers, name="trunk")
        
        # ── Quantile embedding (cosine basis) ─────────────────────────
        self.cosine_embed = CosineBasisLayer(
            embedding_dim=self.embedding_dim,
            name="quantile_embed"
        )
        
        # ── Output head ───────────────────────────────────────────────
        self.output_head = keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(action_size, activation=None),
        ], name="output_head")
        
        logger.info(
            f"LSTM-IQN | state={state_dim} | actions={action_size} | "
            f"embed={self.embedding_dim} | sequences={self.sequence_lengths}"
        )
    
    def _load_config(self, config_path: str) -> dict:
        p = Path(config_path)
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f)
    
    def _sample_quantiles(
        self,
        batch_size: int,
        num_quantiles: int,
        training: bool,
    ) -> tf.Tensor:
        """Sample τ values according to risk distortion."""
        match self.risk_distortion:
            case "neutral":
                tau = tf.random.uniform(
                    (batch_size, num_quantiles), minval=0.0, maxval=1.0
                )
            case "cvar":
                tau = tf.random.uniform(
                    (batch_size, num_quantiles), minval=0.0, maxval=self.cvar_alpha
                )
            case "wang":
                eta = -0.75
                u = tf.random.uniform((batch_size, num_quantiles))
                tau = 0.5 * (1.0 + tf.math.erf(
                    (IQNNetwork._probit(u) + eta) / tf.sqrt(2.0)
                ))
            case _:
                tau = tf.random.uniform(
                    (batch_size, num_quantiles), minval=0.0, maxval=1.0
                )
        return tf.cast(tau, tf.float32)
    
    def call(
        self,
        state_seq_short: tf.Tensor,   # (batch, 10, state_dim)
        state_seq_medium: tf.Tensor,  # (batch, 20, state_dim)
        state_seq_long: tf.Tensor,    # (batch, 30, state_dim)
        num_quantiles: int = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass with three timescale sequences.
        
        Returns:
            (q_values, tau) where:
                q_values: shape (batch, num_quantiles, action_size)
                tau: sampled quantile levels (batch, num_quantiles)
        """
        if num_quantiles is None:
            num_quantiles = (
                self.num_quantiles_train if training
                else self.num_quantiles_inference
            )
        
        B = tf.shape(state_seq_short)[0]
        
        # ── Process three timescales ──────────────────────────────────
        short_context = self.lstm_short(state_seq_short, training=training)    # (B, 64)
        medium_context = self.lstm_medium(state_seq_medium, training=training) # (B, 64)
        long_context = self.lstm_long(state_seq_long, training=training)       # (B, 64)
        
        # ── Concatenate temporal contexts ─────────────────────────────
        combined = tf.concat([short_context, medium_context, long_context], axis=-1)  # (B, 192)
        
        # ── Trunk: combined → shared embedding ────────────────────────
        h = self.trunk(combined, training=training)  # (B, embedding_dim)
        
        # ── Sample quantiles ──────────────────────────────────────────
        tau = self._sample_quantiles(B, num_quantiles, training)  # (B, N)
        
        # ── Quantile embedding ────────────────────────────────────────
        phi = self.cosine_embed(tau)  # (B, N, embedding_dim)
        
        # ── Element-wise multiply ─────────────────────────────────────
        h_expanded = tf.expand_dims(h, axis=1)  # (B, 1, embedding_dim)
        combined_embed = h_expanded * phi        # (B, N, embedding_dim)
        
        # ── Output head ───────────────────────────────────────────────
        BN = B * num_quantiles
        combined_flat = tf.reshape(combined_embed, (BN, self.embedding_dim))
        q_flat = self.output_head(combined_flat, training=training)  # (BN, actions)
        q_values = tf.reshape(q_flat, (B, num_quantiles, self.action_size))
        q_values = tf.clip_by_value(q_values, -self.q_clip, self.q_clip)

        return q_values, tau

    def q_mean(
        self,
        state_seq_short: tf.Tensor,
        state_seq_medium: tf.Tensor,
        state_seq_long: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Mean Q-values across quantiles for greedy action selection."""
        q_values, _ = self(state_seq_short, state_seq_medium, state_seq_long, training=training)
        return tf.reduce_mean(q_values, axis=1)
    
    def masked_q_mean(
        self,
        state_seq_short: tf.Tensor,
        state_seq_medium: tf.Tensor,
        state_seq_long: tf.Tensor,
        mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Mean Q-values with invalid actions masked."""
        q_mean = self.q_mean(state_seq_short, state_seq_medium, state_seq_long, training=training)
        mask_f = tf.cast(mask, tf.float32)
        return q_mean * mask_f + (1.0 - mask_f) * (-1e9)

