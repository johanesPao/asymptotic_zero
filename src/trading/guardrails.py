"""
Trading Guardrails
Implements safety constraints that were present during training:
- 5-step cooldown after opening position
- 3-step minimum hold before closing
- Max concurrent positions
"""

from typing import Dict, Optional


class GuardrailManager:
    """Manage trading guardrails to prevent overtrading and panic selling."""

    def __init__(
        self, cooldown_steps: int = 5, min_hold_steps: int = 3, max_positions: int = 3
    ):
        """
        Initialize guardrail manager.

        Args:
            cooldown_steps: Steps to wait after opening before next open
            min_hold_steps: Minimum to hold before allowing close
            max_positions: Maximum concurrent positions
        """
        self.cooldown_steps = cooldown_steps
        self.min_hold_steps = min_hold_steps
        self.max_positions = max_positions

        # Track position history
        # {symbol: {'opened_step}: int, 'last_action_step': int}}
        self.position_history: Dict[str, Dict] = {}

        # Track global state
        self.current_step = 0
        self.last_open_step = None  # Last step any position was opened

    def step(self):
        """Increment the step counter. Call this every 5-minute interval."""
        self.current_step += 1

    def can_open_position(
        self, symbol: str, current_positions: int
    ) -> tuple[bool, str]:
        """
        Check if allowed to open a position.

        Returns:
            (allowed, reason) tuple
        """
        # Check max positions
        if current_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Check cooldown
        if self.last_open_step is not None:
            steps_since_last_open = self.current_step - self.last_open_step
            if steps_since_last_open < self.cooldown_steps:
                remaining = self.cooldown_steps - steps_since_last_open
                return False, f"Cooldown active ({remaining} steps remaining)"

        return True, "OK"

    def can_close_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check if allowed to close a position.

        Returns:
            (allowed, reason) tuple
        """
        # Check if position exists in history
        if symbol not in self.position_history:
            return True, "Position not tracked (allow close)"

        # Check minimum hold time
        opened_step = self.position_history[symbol]["opened_step"]
        steps_held = self.current_step - opened_step

        if steps_held < self.min_hold_steps:
            remaining = self.min_hold_steps - steps_held
            return False, f"Min hold not met ({remaining} steps remaining)"

        return True, "OK"

    def record_position_opened(self, symbol: str):
        """Record that a position was opened."""
        self.position_history[symbol] = {
            "opened_step": self.current_step,
            "last_action_step": self.current_step,
        }
        self.last_open_step = self.current_step

    def record_position_closed(self, symbol: str):
        """Record that a position was closed."""
        if symbol in self.position_history:
            del self.position_history[symbol]

    def reset_daily(self):
        """Reset guardrails for new trading day."""
        self.current_step = 0
        self.last_open_step = None
        self.position_history.clear()

    def get_status(self) -> Dict:
        """Get current guardrail status."""
        cooldown_remaining = 0
        if self.last_open_step is not None:
            steps_since = self.current_step - self.last_open_step
            if steps_since < self.cooldown_steps:
                cooldown_remaining = self.cooldown_steps - steps_since

        return {
            "current_step": self.current_step,
            "cooldown_remaining": cooldown_remaining,
            "tracked_positions": len(self.position_history),
            "position_ages": {
                symbol: self.current_step - info["opened_step"]
                for symbol, info in self.position_history.items()
            },
        }
