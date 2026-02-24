"""
Experience Replay Buffer — v3

Changes from v2:
  - Sequence-aware sampling for LSTM: lookback within buffer to construct
    multi-scale sequences (short/medium/long) for training — no more fake tiling
  - Cross-episode lookback: sequences can span episode boundaries (user-requested)
  - Prioritized Experience Replay (PER) with SumTree for efficient sampling
  - Save/load for checkpoint persistence — buffer survives training restarts
  - state_dim defaults updated to match new 4023-dim state space
"""

import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SumTree for Prioritized Experience Replay
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """Binary tree where parent = sum of children. Enables O(log n) sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def update(self, idx: int, priority: float):
        """Update priority at data index `idx`."""
        tree_idx = idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta

    def get(self, value: float) -> int:
        """Sample a data index proportional to priority."""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def min_priority(self, size: int) -> float:
        """Get minimum priority among the first `size` entries."""
        leaves = self.tree[self.capacity - 1 : self.capacity - 1 + size]
        nonzero = leaves[leaves > 0]
        return float(np.min(nonzero)) if len(nonzero) > 0 else 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Circular experience replay buffer with PER and LSTM sequence support.

    Stores transitions:
      (state, action, reward, next_state, done, next_action_mask)

    When LSTM is enabled, `sample_with_sequences()` reconstructs multi-scale
    temporal sequences by looking back through the buffer. Cross-episode
    lookback is allowed — the LSTM sees transitions across day boundaries.

    PER (Prioritized Experience Replay):
      - New transitions get max priority
      - After training, priorities are updated with TD errors
      - Sampling is proportional to priority^alpha
      - Importance-sampling weights correct for bias (annealed with beta)
    """

    def __init__(
        self,
        capacity:    int  = 200_000,
        state_dim:   int  = 4023,
        action_size: int  = 143,
        # PER parameters
        per_enabled: bool  = True,
        per_alpha:   float = 0.6,   # prioritization exponent
        per_beta:    float = 0.4,   # IS correction start
        per_beta_end: float = 1.0,  # IS correction end
        per_epsilon: float = 1e-6,  # small constant to avoid zero priority
    ):
        self.capacity    = capacity
        self.state_dim   = state_dim
        self.action_size = action_size

        # Pre-allocate contiguous arrays for memory efficiency
        self.states      = np.zeros((capacity, state_dim),   dtype=np.float32)
        self.actions     = np.zeros(capacity,                dtype=np.int32)
        self.rewards     = np.zeros(capacity,                dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),   dtype=np.float32)
        self.dones       = np.zeros(capacity,                dtype=np.bool_)
        self.next_masks  = np.zeros((capacity, action_size), dtype=np.bool_)

        self.position = 0
        self.size     = 0

        # PER
        self.per_enabled = per_enabled
        self.per_alpha   = per_alpha
        self.per_beta       = per_beta
        self.per_beta_start = per_beta
        self.per_beta_end   = per_beta_end
        self.per_epsilon = per_epsilon
        self._max_priority = 1.0

        if per_enabled:
            self.sum_tree = SumTree(capacity)
        else:
            self.sum_tree = None

        logger.info(
            f"ReplayBuffer v3 | capacity={capacity:,} | "
            f"state_dim={state_dim} | action_size={action_size} | "
            f"PER={'ON' if per_enabled else 'OFF'}"
        )

    def push(
        self,
        state:       np.ndarray,
        action:      int,
        reward:      float,
        next_state:  np.ndarray,
        done:        bool,
        next_mask:   np.ndarray,
    ):
        """
        Store one transition.

        Args:
            state:      Current state (state_dim,)
            action:     Action taken (integer)
            reward:     Reward received (float)
            next_state: Next state (state_dim,)
            done:       Episode ended flag
            next_mask:  Valid actions in next state (action_size,) bool
        """
        i = self.position
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = done
        self.next_masks[i]  = next_mask

        # PER: new transitions get max priority
        if self.per_enabled:
            self.sum_tree.update(i, self._max_priority ** self.per_alpha)

        self.position = (self.position + 1) % self.capacity
        self.size     = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch (with PER if enabled).

        Returns:
            (states, actions, rewards, next_states, dones, next_masks,
             indices, is_weights)
            is_weights: importance-sampling weights (all 1.0 if PER disabled)
        """
        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} > buffer size {self.size}")

        if self.per_enabled:
            idx, is_weights = self._per_sample(batch_size)
        else:
            idx = np.random.choice(self.size, batch_size, replace=False)
            is_weights = np.ones(batch_size, dtype=np.float32)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx].astype(np.float32),
            self.next_masks[idx],
            idx,
            is_weights,
        )

    def sample_with_sequences(
        self,
        batch_size: int,
        sequence_lengths: list = None,
    ) -> Dict:
        """
        Sample a batch with multi-scale LSTM sequences.

        For each sampled index i, reconstructs sequences by looking back
        through the buffer. Cross-episode lookback is allowed (user-specified).
        Zero-padding is only used at the very start of the buffer when there
        aren't enough preceding transitions.

        Args:
            batch_size: Number of transitions to sample
            sequence_lengths: [short, medium, long] e.g. [10, 20, 30]

        Returns:
            dict with keys:
              'seq_short', 'seq_medium', 'seq_long':
                  np arrays of shape (B, T, state_dim) for current states
              'next_seq_short', 'next_seq_medium', 'next_seq_long':
                  np arrays of shape (B, T, state_dim) for next states
              'actions', 'rewards', 'dones', 'next_masks':
                  standard batch arrays
              'indices', 'is_weights':
                  for PER priority updates
        """
        if sequence_lengths is None:
            sequence_lengths = [10, 20, 30]

        if batch_size > self.size:
            raise ValueError(f"Batch size {batch_size} > buffer size {self.size}")

        # Sample indices
        if self.per_enabled:
            idx, is_weights = self._per_sample(batch_size)
        else:
            idx = np.random.choice(self.size, batch_size, replace=False)
            is_weights = np.ones(batch_size, dtype=np.float32)

        max_seq = max(sequence_lengths)

        # Build sequences for current and next states
        seq_results = {}
        for name, length in zip(['short', 'medium', 'long'], sequence_lengths):
            seqs = np.zeros((batch_size, length, self.state_dim), dtype=np.float32)
            next_seqs = np.zeros((batch_size, length, self.state_dim), dtype=np.float32)

            for b, i in enumerate(idx):
                # ── Current state sequence: text[i] is current step ──
                # Iterate backwards from current step i
                curr = i
                for t in range(length - 1, -1, -1):
                    # Store state at position t
                    seqs[b, t] = self.states[curr]
                    
                    # Move to previous step for next iteration
                    # Check boundary conditions
                    prev = (curr - 1)
                    if prev < 0:
                        prev = self.capacity - 1 if self.size >= self.capacity else -1
                    
                    # Stop if we hit start of buffer or episode boundary
                    # If dones[prev] is True, then state[curr] is start of new episode
                    if prev < 0 or (self.dones[prev] and curr != i): # curr!=i check? No.
                        # If prev was terminal, then curr is start of episode. 
                        # So we stop looking back. The remaining seqs[b, 0..t-1] are already 0.
                        break
                    curr = prev

                # ── Next state sequence: text[i+1] is next step ──
                # Last element is loop's next_state
                next_seqs[b, length - 1] = self.next_states[i]
                
                # Preceding elements are from current states, working backward from i
                curr = i
                for t in range(length - 2, -1, -1):
                    # Check if current state i was terminal? 
                    # If dones[i] is True, next_state is terminal. That's fine.
                    # We are looking for PREVIOUS episode boundaries.
                    
                    next_seqs[b, t] = self.states[curr]
                    
                    prev = (curr - 1)
                    if prev < 0:
                        prev = self.capacity - 1 if self.size >= self.capacity else -1

                    if prev < 0 or self.dones[prev]:
                        break
                    curr = prev

            seq_results[f'seq_{name}'] = seqs
            seq_results[f'next_seq_{name}'] = next_seqs

        seq_results['actions'] = self.actions[idx]
        seq_results['rewards'] = self.rewards[idx]
        seq_results['dones'] = self.dones[idx].astype(np.float32)
        seq_results['next_masks'] = self.next_masks[idx]
        seq_results['indices'] = idx
        seq_results['is_weights'] = is_weights

        return seq_results

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update PER priorities after computing TD errors."""
        if not self.per_enabled:
            return
        priorities = (np.abs(td_errors) + self.per_epsilon) ** self.per_alpha
        for idx, priority in zip(indices, priorities):
            self.sum_tree.update(int(idx), float(priority))
            self._max_priority = max(self._max_priority, float(priority) ** (1.0 / self.per_alpha))

    def anneal_beta(self, fraction: float):
        """Anneal PER beta from initial value toward 1.0."""
        self.per_beta = self.per_beta_start + fraction * (self.per_beta_end - self.per_beta_start)

    def _per_sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Proportional PER sampling with importance-sampling weights."""
        indices = np.zeros(batch_size, dtype=np.int32)
        is_weights = np.zeros(batch_size, dtype=np.float32)

        total = self.sum_tree.total
        segment = total / batch_size

        # Min probability for IS weight normalization
        min_prob = self.sum_tree.min_priority(self.size) / total
        max_weight = (self.size * min_prob) ** (-self.per_beta)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx = self.sum_tree.get(value)
            idx = np.clip(idx, 0, self.size - 1)
            indices[i] = idx

            # IS weight
            prob = self.sum_tree.tree[idx + self.sum_tree.capacity - 1] / total
            prob = max(prob, 1e-12)
            weight = (self.size * prob) ** (-self.per_beta)
            is_weights[i] = weight / max_weight  # normalize

        return indices, is_weights

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save buffer state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(p),
            states=self.states[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_states=self.next_states[:self.size],
            dones=self.dones[:self.size],
            next_masks=self.next_masks[:self.size],
            position=np.array([self.position]),
            size=np.array([self.size]),
        )
        logger.info(f"Saved replay buffer ({self.size:,} transitions) → {p}")

    def load(self, path: str) -> bool:
        """Load buffer state from disk. Returns True on success."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Replay buffer file not found: {p}")
            return False
        try:
            data = np.load(str(p))
            n = int(data['size'][0])
            if n > self.capacity:
                n = self.capacity
                logger.warning(f"Buffer file has {int(data['size'][0])} entries, "
                             f"truncating to capacity {self.capacity}")

            self.states[:n] = data['states'][:n]
            self.actions[:n] = data['actions'][:n]
            self.rewards[:n] = data['rewards'][:n]
            self.next_states[:n] = data['next_states'][:n]
            self.dones[:n] = data['dones'][:n]
            self.next_masks[:n] = data['next_masks'][:n]
            self.position = int(data['position'][0]) % self.capacity
            self.size = n

            # Rebuild PER tree with uniform priorities
            if self.per_enabled:
                for i in range(n):
                    self.sum_tree.update(i, self._max_priority ** self.per_alpha)

            logger.info(f"Loaded replay buffer ({n:,} transitions) ← {p}")
            return True
        except Exception as e:
            logger.error(f"Failed to load replay buffer: {e}")
            return False

    # ── Utilities ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def clear(self):
        self.position = 0
        self.size     = 0
        if self.per_enabled:
            self.sum_tree = SumTree(self.capacity)

    def get_statistics(self) -> dict:
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity, "utilization": 0.0}
        return {
            "size":         self.size,
            "capacity":     self.capacity,
            "utilization":  self.size / self.capacity * 100,
            "reward_mean":  float(np.mean(self.rewards[:self.size])),
            "reward_std":   float(np.std( self.rewards[:self.size])),
            "done_ratio":   float(np.mean(self.dones[:self.size])),
        }
