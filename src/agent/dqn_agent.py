"""
DQN Agent

Deep Q-Network agent implementation with:
- Double DQN (separate target network)
- Experience replay
- Epsilon-greedy exploration
- Gradient clipping

The agent learns to map states to optimal actions by minimizing
the temporal difference error between predicted and target Q-values.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Dict
from pathlib import Path
import yaml
import logging

from .network import create_q_network
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    Double DQN Agent for trading.

    Uses two networks:
    - Q-network (online): Updated every step
    - Target network: Periodically synced from Q-network

    This separation helps stabilize training by providing
    consistent target values during updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config_path: str = "config/agent.yaml",
    ):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            config_path: Path to agent configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config_path = Path(config_path)

        # Load configuration
        self.config = self._load_config()

        # Extract settings
        learning_cfg = self.config.get("learning", {})
        exploration_cfg = self.config.get("exploration", {})
        buffer_cfg = self.config.get("replay_buffer", {})

        # Learning parameters
        self.learning_rate = learning_cfg.get("learning_rate", 0.0001)
        self.gamma = learning_cfg.get("gamma", 0.99)
        self.batch_size = learning_cfg.get("batch_size", 64)
        self.target_update_freq = learning_cfg.get("target_update_freq", 1000)
        self.gradient_clip = learning_cfg.get("gradient_clip", 1.0)

        # Exploration parameters
        self.epsilon = exploration_cfg.get("epsilon_start", 1.0)
        self.epsilon_start = exploration_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = exploration_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay_steps = exploration_cfg.get("epsilon_decay_steps", 50000)
        self.exploration_type = exploration_cfg.get("exploration_type", "linear")

        # Create networks
        self.q_network = create_q_network(state_dim, action_dim, str(self.config_path))
        self.target_network = create_q_network(
            state_dim, action_dim, str(self.config_path)
        )

        # Initialize target network with same weights
        self.update_target_network()

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_cfg.get("capacity", 100000),
            state_dim=state_dim,
        )
        self.min_buffer_size = buffer_cfg.get("min_size", 1000)

        # Optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.gradient_clip,
        )

        # Loss function (Huber loss is more robust than MSE)
        self.loss_fn = keras.losses.Huber(delta=1.0)

        # Training state
        self.train_step_counter = 0
        self.episode_counter = 0

        # Metrics tracking
        self.training_losses = []
        self.episode_rewards = []
        
        # Q-value tracking for dashboard
        self.last_q_values = None
        self.last_avg_q = 0.0
        self.last_max_q = 0.0

        logger.info("DQNAgent initialized:")
        logger.info(f"  State dim: {state_dim}")
        logger.info(f"  Action dim: {action_dim}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Gamma: {self.gamma}")
        logger.info(f"  Epsilon: {self.epsilon_start} -> {self.epsilon_end}")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: If True, use exploration; if False, be greedy

        Returns:
            Selected action index
        """
        # Exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        # Exploitation
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.q_network(state_tensor, training=False)
        
        # Track Q-values for dashboard
        self.last_q_values = q_values.numpy()[0]
        self.last_avg_q = float(np.mean(self.last_q_values))
        self.last_max_q = float(np.max(self.last_q_values))
        
        return int(tf.argmax(q_values[0]).numpy())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute loss and update
        loss = self._update_q_network(states, actions, rewards, next_states, dones)

        # Update counters
        self.train_step_counter += 1

        # Update target network periodically
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()
            logger.debug(f"Target network updated at step {self.train_step_counter}")

        # Decay epsilon
        self._decay_epsilon()

        # Track loss
        self.training_losses.append(float(loss))

        return float(loss)

    @tf.function
    def _update_q_network(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor,
    ) -> tf.Tensor:
        """
        Update Q-network using Double DQN algorithm.

        Double DQN uses online network to select actions and
        target network to evaluate them, reducing overestimation.
        """
        # Compute target Q-values using Double DQN
        # 1. Use online network to select best actions for next states
        next_q_values_online = self.q_network(next_states, training=False)
        best_next_actions = tf.argmax(next_q_values_online, axis=1)

        # 2. Use target network to evaluate those actions
        next_q_values_target = self.target_network(next_states, training=False)
        batch_indices = tf.range(tf.shape(actions)[0])
        next_q_selected = tf.gather_nd(
            next_q_values_target,
            tf.stack([batch_indices, tf.cast(best_next_actions, tf.int32)], axis=1),
        )

        # 3. Compute TD targets: r + gamma * Q_target(s', argmax_a Q_online(s', a))
        targets = rewards + (1.0 - dones) * self.gamma * next_q_selected

        # Compute loss and gradients
        with tf.GradientTape() as tape:
            # Get Q-values for taken actions
            q_values = self.q_network(states, training=True)
            action_indices = tf.stack([batch_indices, actions], axis=1)
            q_selected = tf.gather_nd(q_values, action_indices)

            # Huber loss between predicted and target Q-values
            loss = self.loss_fn(targets, q_selected)

        # Apply gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )

        return loss

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.set_weights(self.q_network.get_weights())

    def _decay_epsilon(self):
        """Decay exploration rate."""
        if self.exploration_type == "linear":
            # Linear decay
            decay_rate = (
                self.epsilon_start - self.epsilon_end
            ) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_rate)
        else:
            # Exponential decay
            decay_rate = (self.epsilon_end / self.epsilon_start) ** (
                1 / self.epsilon_decay_steps
            )
            self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)

    def end_episode(self, episode_reward: float):
        """
        Called at the end of each episode.

        Args:
            episode_reward: Total reward for the episode
        """
        self.episode_counter += 1
        self.episode_rewards.append(episode_reward)

    def save(self, path: str):
        """
        Save agent state.

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save networks
        self.q_network.save_weights(str(path / "q_network.weights.h5"))
        self.target_network.save_weights(str(path / "target_network.weights.h5"))

        # Save training state
        state = {
            "epsilon": self.epsilon,
            "train_step_counter": self.train_step_counter,
            "episode_counter": self.episode_counter,
        }

        with open(path / "agent_state.yaml", "w") as f:
            yaml.dump(state, f)

        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """
        Load agent state.

        Args:
            path: Directory to load from
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load networks
        self.q_network.load_weights(str(path / "q_network.weights.h5"))
        self.target_network.load_weights(str(path / "target_network.weights.h5"))

        # Load training state
        state_file = path / "agent_state.yaml"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = yaml.safe_load(f)

            self.epsilon = state.get("epsilon", self.epsilon_end)
            self.train_step_counter = state.get("train_step_counter", 0)
            self.episode_counter = state.get("episode_counter", 0)

        logger.info(f"Agent loaded from {path}")

    def get_q_statistics(self) -> Dict:
        """Get Q-value statistics for dashboard."""
        return {
            "avg_q_value": self.last_avg_q,
            "max_q_value": self.last_max_q,
        }
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
        stats = {
            "epsilon": self.epsilon,
            "train_steps": self.train_step_counter,
            "episodes": self.episode_counter,
            "buffer_size": len(self.replay_buffer),
            "avg_q_value": self.last_avg_q,
            "max_q_value": self.last_max_q,
        }

        if self.training_losses:
            recent_losses = self.training_losses[-100:]
            stats["avg_loss"] = np.mean(recent_losses)

        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            stats["avg_reward"] = np.mean(recent_rewards)
            stats["max_reward"] = np.max(recent_rewards)

        return stats
