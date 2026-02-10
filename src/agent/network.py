"""
Neural Network Architecture

Defines the Q-Network for the DQN agent.
Takes state as input, outputs Q-values for each action.

Architecture:
- Fully connected layers with ReLU activation
- Optional dropout for regularization
- Output layer with linear activation (Q-values)
"""

import tensorflow as tf
from keras import layers, Model
from typing import List
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class QNetwork(Model):
    """
    Q-Network for Deep Q-Learning.

    Maps state observations to Q-values for each possible action.
    Uses a simple feedforward architecture with configurable layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [512, 256, 128],
        activation: str = "relu",
        dropout_rate: float = 0.2,
        use_batch_norm: bool = False,
        name: str = "q_network",
    ):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state input
            action_dim: Number of possible actions (output dimension)
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Model name
        """
        super().__init__(name=name)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Build layers
        self.hidden_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []

        for i, units in enumerate(hidden_layers):
            # Dense layer
            self.hidden_layers.append(
                layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer="he_normal",
                    name=f"hidden_{i}",
                )
            )

            # Optional batch normalization
            if use_batch_norm:
                self.batch_norm_layers.append(
                    layers.BatchNormalization(name=f"batch_norm_{i}")
                )

            # Dropout
            if dropout_rate > 0:
                self.dropout_layers.append(
                    layers.Dropout(dropout_rate, name=f"dropout_{i}")
                )

        # Output layer (Q-values for each action)
        self.output_layer = layers.Dense(
            action_dim,
            activation=None,  # Linear for Q-values
            kernel_initializer="he_normal",
            name="q_values",
        )

        logger.info(f"QNetwork created: {state_dim} -> {hidden_layers} -> {action_dim}")

    def call(self, state: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            training: Whether in training mode (affects dropout)

        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        x = state

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)

            if self.use_batch_norm and i < len(self.batch_norm_layers):
                x = self.batch_norm_layers[i](x, training=training)

            if self.dropout_rate > 0 and i < len(self.dropout_layers):
                x = self.dropout_layers[i](x, training=training)

        q_values = self.output_layer(x)

        return q_values

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_layers": self.hidden_layers_config,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }


def create_q_network(
    state_dim: int,
    action_dim: int,
    config_path: str = "config/agent.yaml",
) -> QNetwork:
    """
    Create Q-Network from configuration file.

    Args:
        state_dim: Dimension of state input
        action_dim: Number of possible actions
        config_path: Path to agent configuration

    Returns:
        QNetwork instance
    """
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        network_config = config.get("network", {})
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
        network_config = {}

    return QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=network_config.get("hidden_layers", [512, 256, 128]),
        activation=network_config.get("activation", "relu"),
        dropout_rate=network_config.get("dropout_rate", 0.2),
        use_batch_norm=network_config.get("use_batch_norm", False),
    )
