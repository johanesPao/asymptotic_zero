"""
Experience Replay Buffer

Stores transitions (state, action, reward, next_state, done) for training.
Enables breaking correlation between consecutive samples by random sampling.

Key Features:
- Circular buffer with fixed capacity
- Efficient numpy-based storage
- Random batch sampling
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single transition tuple."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores transitions and provides random sampling for training batches.
    Uses circular buffer to maintain fixed memory usage.
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 3907,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state vectors
        """
        self.capacity = capacity
        self.state_dim = state_dim

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        # Buffer state
        self.position = 0
        self.size = 0

        logger.info(f"ReplayBuffer created: capacity={capacity}, state_dim={state_dim}")

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Circular buffer
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} > buffer size {self.size}")

        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
            }

        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity * 100,
            "reward_mean": float(np.mean(self.rewards[: self.size])),
            "reward_std": float(np.std(self.rewards[: self.size])),
            "done_ratio": float(np.mean(self.dones[: self.size])),
        }
