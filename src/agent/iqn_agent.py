"""
IQN Agent

Replaces the previous Double DQN agent.

Key algorithm differences:

  DDQN loss:
    TD_error  = r + γ × Q_target(s', argmax_a Q_online(s', a)) - Q_online(s, a)
    loss      = MSE(TD_error)

  IQN loss (Quantile Huber):
    For sampled quantiles τ  (online)  and τ' (target):
    TD_error_ij = r + γ × Z_target(s', a*, τ'_j) - Z_online(s, a, τ_i)
    loss        = Σ_i Σ_j  ρ_τ_i( TD_error_ij )

    where ρ_τ(u) is the asymmetric Huber loss:
      ρ_τ(u) = |τ - 𝟙(u < 0)| × L(u)
      L(u)   = 0.5u² if |u| ≤ κ, else κ(|u| - 0.5κ)   (Huber with κ=1)

    This trains the network to predict correct quantiles, not just the mean.
    The asymmetric weighting ensures τ_i = 0.1 places most weight on low returns
    (cautious) and τ_i = 0.9 places most weight on high returns (optimistic).

  Action masking in target computation:
    a* = argmax_a  mean_τ'[ Z_target(s', a, τ') ]  subject to  mask(s')[a] = True
    Invalid actions are set to -∞ before argmax, so they can never be chosen.

  Exploration:
    ε-greedy over mean Q-values, with action mask applied before argmax.
    Invalid actions are completely excluded from both exploration and exploitation.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import logging
from typing import Dict, Optional, Tuple

from .network import IQNNetwork
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class IQNAgent:
    """
    Implicit Quantile Network agent for cryptocurrency futures trading.

    Usage:
        agent = IQNAgent(state_dim=3963, action_size=143)
        state, mask = env.reset()
        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done, next_mask, info = env.step(action)
            agent.remember(state, action, reward, next_state, done, next_mask)
            agent.train()
            state, mask = next_state, next_mask
    """

    def __init__(
        self,
        state_dim:   int = 4023,
        action_size: int = 143,
        config_path: str = "config/trading.yaml",
    ):
        self.state_dim   = state_dim
        self.action_size = action_size
        self.config      = self._load_config(config_path)
        agent_cfg        = self.config.get("agent", {})

        # ── LSTM mode detection ────────────────────────────────────────────────
        lstm_cfg = self.config.get("lstm", {})
        self.lstm_enabled = lstm_cfg.get("enabled", False)
        self.sequence_lengths = lstm_cfg.get("sequence_lengths", [10, 20, 30]) if self.lstm_enabled else None
        
        # ── Hyperparameters ────────────────────────────────────────────────────
        self.learning_rate        = agent_cfg.get("learning_rate",        0.00005)
        self.gamma                = agent_cfg.get("gamma",                0.99)
        self.epsilon              = agent_cfg.get("epsilon_start",        1.0)
        self.epsilon_min          = agent_cfg.get("epsilon_end",          0.05)
        
        # Adaptive epsilon decay (percentage-based)
        self.epsilon_decay_percentage = agent_cfg.get("epsilon_decay_percentage", None)
        if self.epsilon_decay_percentage is not None:
            self.total_episodes = None  # Set via set_total_episodes()
            self.epsilon_decay_eps = None  # Calculated when total_episodes is set
            logger.info(f"Using adaptive epsilon decay: {self.epsilon_decay_percentage*100:.0f}% of total episodes")
        else:
            # Fallback to old fixed decay
            self.epsilon_decay_eps = agent_cfg.get("epsilon_decay_episodes", 3000)
            self.total_episodes = self.epsilon_decay_eps
        
        self.batch_size           = agent_cfg.get("batch_size",           64)
        self.min_replay_size      = agent_cfg.get("min_replay_size",      2000)
        self.target_update_freq   = agent_cfg.get("target_update_freq",   100)
        self.train_freq           = agent_cfg.get("train_freq",           4)
        self.gradient_clip        = agent_cfg.get("gradient_clip",        1.0)
        
        # Soft update config
        self.use_soft_update      = agent_cfg.get("use_soft_update",      False)
        self.target_update_tau    = agent_cfg.get("target_update_tau",    0.005)
        
        # LR Schedule config
        self.use_lr_schedule      = agent_cfg.get("use_lr_schedule",      False)
        
        replay_capacity           = agent_cfg.get("replay_capacity",      200_000)

        # Epsilon decay type: "linear" or "cosine"
        self.epsilon_decay_type = agent_cfg.get("epsilon_decay_type", "cosine")
        self._stage_episode_count = 0  # counter within current stage

        # PER config
        replay_cfg = self.config.get("replay", {})
        self.per_enabled = replay_cfg.get("per_enabled", True)
        per_alpha    = replay_cfg.get("per_alpha",     0.6)
        per_beta     = replay_cfg.get("per_beta_start", 0.4)
        per_beta_end = replay_cfg.get("per_beta_end",   1.0)
        per_epsilon  = replay_cfg.get("per_epsilon",    1e-6)

        # IQN-specific from network config
        net_cfg = self.config.get("network", {})
        self.num_quantiles_train     = net_cfg.get("num_quantiles_train",    8)
        self.num_quantiles_inference = net_cfg.get("num_quantiles_inference", 32)

        # Huber loss κ
        self.huber_kappa = 1.0

        # ── Networks (LSTM or snapshot-based) ─────────────────────────────────
        if self.lstm_enabled:
            from .network import LSTMIQNNetwork
            self.online_network = LSTMIQNNetwork(
                state_dim   = state_dim,
                action_size = action_size,
                config_path = config_path,
            )
            self.target_network = LSTMIQNNetwork(
                state_dim   = state_dim,
                action_size = action_size,
                config_path = config_path,
            )
            logger.info("Using LSTM-IQN with multi-scale temporal awareness")
            
            # Warm up with dummy sequences
            dummy_short = tf.zeros((1, self.sequence_lengths[0], state_dim))
            dummy_medium = tf.zeros((1, self.sequence_lengths[1], state_dim))
            dummy_long = tf.zeros((1, self.sequence_lengths[2], state_dim))
            self.online_network(dummy_short, dummy_medium, dummy_long, training=False)
            self.target_network(dummy_short, dummy_medium, dummy_long, training=False)
        else:
            self.online_network = IQNNetwork(
                state_dim   = state_dim,
                action_size = action_size,
                config_path = config_path,
            )
            self.target_network = IQNNetwork(
                state_dim   = state_dim,
                action_size = action_size,
                config_path = config_path,
            )
            logger.info("Using snapshot-based IQN")
            
            # Warm up with dummy state
            dummy = tf.zeros((1, state_dim))
            self.online_network(dummy, training=False)
            self.target_network(dummy, training=False)
        
        self._sync_target()

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.learning_rate,
            clipnorm      = self.gradient_clip,
        )

        # ── Replay buffer ──────────────────────────────────────────────────────
        self.replay = ReplayBuffer(
            capacity    = replay_capacity,
            state_dim   = state_dim,
            action_size = action_size,
            per_enabled = self.per_enabled,
            per_alpha   = per_alpha,
            per_beta    = per_beta,
            per_beta_end = per_beta_end,
            per_epsilon = per_epsilon,
        )

        # ── Counters ───────────────────────────────────────────────────────────
        self.train_step_count  = 0
        self.total_steps       = 0
        self.episodes_done     = 0
        
        # Epsilon step calculated after set_total_episodes() is called
        if self.epsilon_decay_eps is not None:
            self.epsilon_step = (self.epsilon - self.epsilon_min) / max(self.epsilon_decay_eps, 1)
        else:
            self.epsilon_step = 0  # Will be set in set_total_episodes()

        # ── Statistics ─────────────────────────────────────────────────────────
        self.loss_history:          list = []
        self.invalid_action_history: list = []

        logger.info(
            f"IQNAgent | state={state_dim} | actions={action_size} | "
            f"lr={self.learning_rate} | ε={self.epsilon:.2f}→{self.epsilon_min} | "
            f"γ={self.gamma} | replay={replay_capacity:,}"
        )

    def _load_config(self, config_path: str) -> dict:
        p = Path(config_path)
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f)
    
    def set_total_episodes(self, total_episodes: int):
        """
        Set total training episodes and calculate adaptive epsilon decay.
        Must be called before training starts if using adaptive epsilon.
        
        Args:
            total_episodes: Total number of episodes in curriculum
        """
        self.total_episodes = total_episodes
        if self.epsilon_decay_percentage is not None:
            self.epsilon_decay_eps = int(total_episodes * self.epsilon_decay_percentage)
            self.epsilon_step = (self.epsilon - self.epsilon_min) / max(self.epsilon_decay_eps, 1)
            logger.info(
                f"Epsilon decay configured: {self.epsilon_decay_eps}/{total_episodes} episodes "
                f"({self.epsilon_decay_percentage*100:.0f}%)"
            )
        else:
            logger.info(f"Using fixed epsilon decay: {self.epsilon_decay_eps} episodes")
        
        # Setup LR schedule if requested
        if self.use_lr_schedule:
            # Decay over full training duration (approx steps)
            # Assuming avg 288 steps/episode (24h * 12 steps/h)
            sim_steps_per_episode = 288
            decay_steps = self.total_episodes * sim_steps_per_episode
            
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=decay_steps,
                alpha=0.1  # End LR is 10% of initial
            )
            # Keras optimizers allow setting learning_rate to a schedule
            self.optimizer.learning_rate = lr_schedule
            logger.info(f"Using CosineDecay LR schedule over ~{decay_steps} steps")

    def reset_epsilon(self, stage_episodes: int, epsilon_start: float = None):
        """
        Reset epsilon for a new curriculum stage.

        Each stage gets its own full ε decay cycle from epsilon_start → epsilon_min
        over the given number of episodes. Supports cosine or linear schedule.

        Args:
            stage_episodes: Number of episodes in this stage
            epsilon_start:  Starting ε for this stage (None = use config default)
        """
        if epsilon_start is None:
            epsilon_start = self.config.get("agent", {}).get("epsilon_start", 1.0)

        self.epsilon = epsilon_start
        decay_eps = int(stage_episodes * (self.epsilon_decay_percentage or 0.85))
        self.epsilon_decay_eps = max(decay_eps, 1)
        self.epsilon_step = (self.epsilon - self.epsilon_min) / self.epsilon_decay_eps
        self._stage_episode_count = 0
        self._stage_epsilon_start = epsilon_start
        logger.info(
            f"Epsilon reset: {self.epsilon:.2f} → {self.epsilon_min:.2f} "
            f"over {self.epsilon_decay_eps}/{stage_episodes} episodes "
            f"(decay={self.epsilon_decay_type})"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Action selection
    # ══════════════════════════════════════════════════════════════════════════

    def select_action(
        self,
        state,  # Can be np.ndarray or dict (if LSTM)
        mask:  np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        Select action using ε-greedy over masked mean Q-values.

        Args:
            state:         (state_dim,) if snapshot-based, or dict with sequences if LSTM
            mask:          (action_size,) bool — True = valid action
            deterministic: If True, always greedy (use during evaluation)

        Returns:
            action integer in [0, action_size)
        """
        valid_actions = np.where(mask)[0]

        if len(valid_actions) == 0:
            # Should never happen (HOLD is always valid), safety fallback
            logger.warning("Empty action mask! Defaulting to HOLD (action 0)")
            return 0

        # ε-exploration: pick a random VALID action
        if not deterministic and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # Greedy: pick action with highest mean Q-value among valid actions
        mask_t = tf.expand_dims(tf.constant(mask, dtype=tf.float32), 0)
        
        if self.lstm_enabled:
            # state is dict: {'short': (10, state_dim), 'medium': (20, state_dim), 'long': (30, state_dim)}
            state_short = tf.expand_dims(tf.constant(state['short'], dtype=tf.float32), 0)
            state_medium = tf.expand_dims(tf.constant(state['medium'], dtype=tf.float32), 0)
            state_long = tf.expand_dims(tf.constant(state['long'], dtype=tf.float32), 0)
            q_mean = self.online_network.masked_q_mean(
                state_short, state_medium, state_long, mask_t, training=False
            )
        else:
            # state is array: (state_dim,)
            state_t = tf.expand_dims(tf.constant(state, dtype=tf.float32), 0)
            q_mean = self.online_network.masked_q_mean(state_t, mask_t, training=False)
        
        action = int(tf.argmax(q_mean[0]).numpy())
        self.last_q_values = q_mean[0].numpy()  # shape: (action_size,) for per-coin Q display
        return action

    # ══════════════════════════════════════════════════════════════════════════
    # Memory
    # ══════════════════════════════════════════════════════════════════════════

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        next_mask:  np.ndarray,
    ):
        """Store one transition in the replay buffer."""
        reward = float(np.clip(reward, -10.0, 10.0))
        self.replay.push(state, action, reward, next_state, done, next_mask)
        self.total_steps += 1

    # ══════════════════════════════════════════════════════════════════════════
    # Training — IQN quantile Huber loss
    # ══════════════════════════════════════════════════════════════════════════

    def train(self) -> Optional[float]:
        """
        Run one gradient update if conditions are met.

        Returns:
            loss value or None if skipped
        """
        if not self.replay.is_ready(self.min_replay_size):
            return None
        if self.total_steps % self.train_freq != 0:
            return None

        if self.lstm_enabled:
            # Sample with real LSTM sequences (no more fake tiling!)
            batch = self.replay.sample_with_sequences(
                self.batch_size, self.sequence_lengths
            )
            loss, td_errors = self._update_lstm(
                tf.constant(batch['seq_short'], dtype=tf.float32),
                tf.constant(batch['seq_medium'], dtype=tf.float32),
                tf.constant(batch['seq_long'], dtype=tf.float32),
                tf.constant(batch['actions'], dtype=tf.int32),
                tf.constant(batch['rewards'], dtype=tf.float32),
                tf.constant(batch['next_seq_short'], dtype=tf.float32),
                tf.constant(batch['next_seq_medium'], dtype=tf.float32),
                tf.constant(batch['next_seq_long'], dtype=tf.float32),
                tf.constant(batch['dones'], dtype=tf.float32),
                tf.constant(batch['next_masks'], dtype=tf.float32),
                tf.constant(batch['is_weights'], dtype=tf.float32),
            )
            # PER priority update
            if self.per_enabled:
                self.replay.update_priorities(
                    batch['indices'], td_errors.numpy()
                )
        else:
            # Standard flat-state sampling
            (states, actions, rewards, next_states,
             dones, next_masks, indices, is_weights) = self.replay.sample(self.batch_size)

            loss, td_errors = self._update(
                tf.constant(states, dtype=tf.float32),
                tf.constant(actions, dtype=tf.int32),
                tf.constant(rewards, dtype=tf.float32),
                tf.constant(next_states, dtype=tf.float32),
                tf.constant(dones, dtype=tf.float32),
                tf.constant(next_masks, dtype=tf.float32),
                tf.constant(is_weights, dtype=tf.float32),
            )
            # PER priority update
            if self.per_enabled:
                self.replay.update_priorities(indices, td_errors.numpy())

        # Anneal PER beta (step-based for smoother transition)
        if self.per_enabled and self.total_episodes and self.total_episodes > 0:
            # Estimate max steps (approx 288 per episode)
            max_steps = self.total_episodes * 288
            fraction  = min(self.total_steps / max_steps, 1.0)
            self.replay.anneal_beta(fraction)

        # Sync target network
        self.train_step_count += 1
        
        if self.use_soft_update:
            # Soft update every step
            self._sync_target()
        elif self.train_step_count % self.target_update_freq == 0:
            # Hard update periodically
            self._sync_target()

        self.loss_history.append(float(loss))
        return float(loss)

    @tf.function(reduce_retracing=True)
    def _update(
        self,
        states:      tf.Tensor,
        actions:     tf.Tensor,
        rewards:     tf.Tensor,
        next_states: tf.Tensor,
        dones:       tf.Tensor,
        next_masks:  tf.Tensor,
        is_weights:  tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute Double-IQN TD targets and apply quantile Huber loss (non-LSTM).

        Returns:
            (loss, per_sample_td_errors) for PER priority updates
        """
        B = tf.shape(states)[0]
        N  = self.num_quantiles_train
        N_ = self.num_quantiles_inference

        # ── Step 1: Compute target values ──────────────────────────────────────
        q_online_next, _ = self.online_network(
            tf.cast(next_states, tf.float32), num_quantiles=N_, training=False
        )
        q_mean_next = tf.reduce_mean(q_online_next, axis=1)

        mask_f   = tf.cast(next_masks, tf.float32)
        q_masked = q_mean_next * mask_f + (1.0 - mask_f) * (-1e9)
        a_star   = tf.argmax(q_masked, axis=1, output_type=tf.int32)

        z_target, _ = self.target_network(
            tf.cast(next_states, tf.float32), num_quantiles=N_, training=False
        )
        action_mask = tf.one_hot(a_star, depth=self.action_size, dtype=tf.float32)
        action_mask_exp = tf.expand_dims(action_mask, axis=1)
        z_target_astar = tf.reduce_sum(z_target * action_mask_exp, axis=2)

        rewards_exp = tf.expand_dims(rewards, 1)
        dones_exp   = tf.expand_dims(dones, 1)
        targets     = rewards_exp + self.gamma * (1.0 - dones_exp) * z_target_astar

        # ── Step 2: Compute online predictions ────────────────────────────────
        with tf.GradientTape() as tape:
            z_online, tau_online = self.online_network(
                tf.cast(states, tf.float32), num_quantiles=N, training=True
            )
            actions_mask = tf.one_hot(actions, depth=self.action_size, dtype=tf.float32)
            actions_mask_exp = tf.expand_dims(actions_mask, axis=1)
            z_online_a = tf.reduce_sum(z_online * actions_mask_exp, axis=2)

            # Quantile Huber loss
            targets_exp  = tf.expand_dims(targets, axis=1)
            z_online_exp = tf.expand_dims(z_online_a, axis=2)
            td_errors    = targets_exp - z_online_exp

            huber     = self._huber(td_errors)
            tau_i     = tf.expand_dims(tau_online, axis=2)
            indicator = tf.cast(td_errors < 0, tf.float32)
            rho       = tf.abs(tau_i - indicator) * huber

            # Per-sample loss (for PER), then weight by IS weights
            per_sample_loss = tf.reduce_mean(tf.reduce_sum(rho, axis=2), axis=1)  # (B,)
            weighted_loss = tf.reduce_mean(per_sample_loss * is_weights)

        gradients = tape.gradient(weighted_loss, self.online_network.trainable_variables)
        gradients = [tf.clip_by_norm(g, self.gradient_clip) for g in gradients]
        self.optimizer.apply_gradients(
            zip(gradients, self.online_network.trainable_variables)
        )

        # Per-sample TD errors for PER (mean absolute TD error)
        per_sample_td = tf.reduce_mean(tf.abs(td_errors), axis=[1, 2])  # (B,)

        return weighted_loss, per_sample_td

    @tf.function(reduce_retracing=True)
    def _update_lstm(
        self,
        seq_short:      tf.Tensor,   # (B, T_s, D)
        seq_medium:     tf.Tensor,   # (B, T_m, D)
        seq_long:       tf.Tensor,   # (B, T_l, D)
        actions:        tf.Tensor,
        rewards:        tf.Tensor,
        next_seq_short: tf.Tensor,   # (B, T_s, D)
        next_seq_medium: tf.Tensor,  # (B, T_m, D)
        next_seq_long:  tf.Tensor,   # (B, T_l, D)
        dones:          tf.Tensor,
        next_masks:     tf.Tensor,
        is_weights:     tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Double-IQN update with REAL LSTM sequences (not fake-tiled).

        Returns:
            (loss, per_sample_td_errors)
        """
        N  = self.num_quantiles_train
        N_ = self.num_quantiles_inference

        # ── Target values ──────────────────────────────────────────────────────
        q_online_next, _ = self.online_network(
            next_seq_short, next_seq_medium, next_seq_long,
            num_quantiles=N_, training=False
        )
        q_mean_next = tf.reduce_mean(q_online_next, axis=1)

        mask_f   = tf.cast(next_masks, tf.float32)
        q_masked = q_mean_next * mask_f + (1.0 - mask_f) * (-1e9)
        a_star   = tf.argmax(q_masked, axis=1, output_type=tf.int32)

        z_target, _ = self.target_network(
            next_seq_short, next_seq_medium, next_seq_long,
            num_quantiles=N_, training=False
        )
        action_mask = tf.one_hot(a_star, depth=self.action_size, dtype=tf.float32)
        action_mask_exp = tf.expand_dims(action_mask, axis=1)
        z_target_astar = tf.reduce_sum(z_target * action_mask_exp, axis=2)

        rewards_exp = tf.expand_dims(rewards, 1)
        dones_exp   = tf.expand_dims(dones, 1)
        targets     = rewards_exp + self.gamma * (1.0 - dones_exp) * z_target_astar

        # ── Online predictions ─────────────────────────────────────────────────
        with tf.GradientTape() as tape:
            z_online, tau_online = self.online_network(
                seq_short, seq_medium, seq_long,
                num_quantiles=N, training=True
            )
            actions_mask = tf.one_hot(actions, depth=self.action_size, dtype=tf.float32)
            actions_mask_exp = tf.expand_dims(actions_mask, axis=1)
            z_online_a = tf.reduce_sum(z_online * actions_mask_exp, axis=2)

            targets_exp  = tf.expand_dims(targets, axis=1)
            z_online_exp = tf.expand_dims(z_online_a, axis=2)
            td_errors    = targets_exp - z_online_exp

            huber     = self._huber(td_errors)
            tau_i     = tf.expand_dims(tau_online, axis=2)
            indicator = tf.cast(td_errors < 0, tf.float32)
            rho       = tf.abs(tau_i - indicator) * huber

            per_sample_loss = tf.reduce_mean(tf.reduce_sum(rho, axis=2), axis=1)
            weighted_loss = tf.reduce_mean(per_sample_loss * is_weights)

        gradients = tape.gradient(weighted_loss, self.online_network.trainable_variables)
        gradients = [tf.clip_by_norm(g, self.gradient_clip) for g in gradients]
        self.optimizer.apply_gradients(
            zip(gradients, self.online_network.trainable_variables)
        )

        per_sample_td = tf.reduce_mean(tf.abs(td_errors), axis=[1, 2])
        return weighted_loss, per_sample_td

    def _huber(self, u: tf.Tensor) -> tf.Tensor:
        """Element-wise Huber loss with threshold κ."""
        abs_u = tf.abs(u)
        return tf.where(
            abs_u <= self.huber_kappa,
            0.5 * tf.square(u),
            self.huber_kappa * (abs_u - 0.5 * self.huber_kappa),
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════════

    def _sync_target(self):
        """Sync online -> target network weights (hard or soft)."""
        if self.use_soft_update:
            # Polyak averaging: weight_target = tau * weight_online + (1-tau) * weight_target
            tau = self.target_update_tau
            online_weights = self.online_network.get_weights()
            target_weights = self.target_network.get_weights()
            
            new_weights = []
            for w_online, w_target in zip(online_weights, target_weights):
                new_w = tau * w_online + (1.0 - tau) * w_target
                new_weights.append(new_w)
            
            self.target_network.set_weights(new_weights)
        else:
            # Hard copy
            self.target_network.set_weights(self.online_network.get_weights())

    def decay_epsilon(self):
        """
        Call once per episode to decay ε.

        Supports linear and cosine schedules.
        """
        self._stage_episode_count += 1

        if self.epsilon_decay_type == "cosine" and self.epsilon_decay_eps:
            # Cosine annealing: slow early decay, fast middle, slow end
            start = getattr(self, '_stage_epsilon_start', self.epsilon)
            progress = min(self._stage_episode_count / self.epsilon_decay_eps, 1.0)
            import math
            self.epsilon = self.epsilon_min + 0.5 * (start - self.epsilon_min) * (
                1 + math.cos(math.pi * progress)
            )
        else:
            # Linear decay
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        self.episodes_done += 1

    def save(self, path: str):
        """Save online network weights."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.online_network.save_weights(str(p / "online.weights.h5"))
        self.target_network.save_weights(str(p / "target.weights.h5"))
        logger.info(f"Saved IQN weights → {p}")

    def load(self, path: str):
        """Load online network weights and sync to target."""
        p = Path(path)
        # Build weights by running a dummy pass first
        if self.lstm_enabled:
            dummy_short = tf.zeros((1, self.sequence_lengths[0], self.state_dim))
            dummy_medium = tf.zeros((1, self.sequence_lengths[1], self.state_dim))
            dummy_long = tf.zeros((1, self.sequence_lengths[2], self.state_dim))
            self.online_network(dummy_short, dummy_medium, dummy_long, training=False)
            self.target_network(dummy_short, dummy_medium, dummy_long, training=False)
        else:
            dummy = tf.zeros((1, self.state_dim))
            self.online_network(dummy, training=False)
            self.target_network(dummy, training=False)
        # Support both new format (online.weights.h5) and old v1 format (q_network.weights.h5)
        weights_file = p / "online.weights.h5"
        if not weights_file.exists():
            weights_file = p / "q_network.weights.h5"
            if not weights_file.exists():
                raise FileNotFoundError(
                    f"No weights found in {p}. "
                    f"Expected 'online.weights.h5' or 'q_network.weights.h5'"
                )
            logger.warning(f"Loading legacy v1 checkpoint format from {weights_file.name}")

        self.online_network.load_weights(str(weights_file))
        self._sync_target()
        logger.info(f"Loaded IQN weights ← {p}")

    def get_statistics(self) -> Dict:
        """Summary statistics for dashboard and logging."""
        loss_window = self.loss_history[-100:] if self.loss_history else [0.0]
        return {
            "epsilon":          self.epsilon,
            "total_steps":      self.total_steps,
            "train_step_count": self.train_step_count,
            "episodes_done":    self.episodes_done,
            "replay_size":      len(self.replay),
            "replay_ready":     self.replay.is_ready(self.min_replay_size),
            "loss_mean":        float(np.mean(loss_window)),
            "loss_last":        float(loss_window[-1]) if loss_window else 0.0,
        }

    def save_training_state(self) -> Dict:
        """Return agent training state as a serializable dict for checkpointing."""
        return {
            "epsilon":              self.epsilon,
            "total_steps":          self.total_steps,
            "train_step_count":     self.train_step_count,
            "episodes_done":        self.episodes_done,
            # Cosine decay state — without these, resume restarts the curve from 0
            # causing rapid collapse: start tracks self.epsilon each step instead of
            # being locked at the original value, creating a product recurrence.
            "stage_episode_count":  self._stage_episode_count,
            "stage_epsilon_start":  getattr(self, "_stage_epsilon_start", self.epsilon),
        }

    def load_training_state(self, state: Dict):
        """Restore agent training state from a checkpoint dict."""
        self.epsilon              = state.get("epsilon", self.epsilon)
        self.total_steps          = state.get("total_steps", self.total_steps)
        self.train_step_count     = state.get("train_step_count", self.train_step_count)
        self.episodes_done        = state.get("episodes_done", self.episodes_done)
        self._stage_episode_count = state.get("stage_episode_count", self._stage_episode_count)
        self._stage_epsilon_start = state.get("stage_epsilon_start", self.epsilon)
        logger.info(
            f"Restored agent state: ε={self.epsilon:.4f}, "
            f"steps={self.total_steps}, episodes={self.episodes_done}, "
            f"stage_count={self._stage_episode_count}, "
            f"ε_start={self._stage_epsilon_start:.4f}"
        )
