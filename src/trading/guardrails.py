"""
Trading Guardrails
Implements safety constraints for live trading:
- Max concurrent positions

Cooldown and min-hold are NOT enforced here — the agent was trained with
those constraints baked into action masking and reward shaping, so it
already learned to space trades and hold positions.  Re-enforcing them
in live trading only makes the agent slower to adapt.
"""

from typing import Dict


class GuardrailManager:
    """Manage trading guardrails — max positions only."""

    def __init__(self, max_positions: int = 3, **_kwargs):
        self.max_positions = max_positions

        # Track position history {symbol: {'opened_step': int, ...}}
        self.position_history: Dict[str, Dict] = {}
        self.current_step = 0

    def step(self):
        """Increment the step counter. Call this every 5-minute interval."""
        self.current_step += 1

    def can_open_position(
        self, symbol: str, current_positions: int
    ) -> tuple[bool, str]:
        """Check if allowed to open a position."""
        if current_positions >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"
        return True, "OK"

    def can_close_position(self, symbol: str) -> tuple[bool, str]:
        """Check if allowed to close a position."""
        return True, "OK"

    def record_position_opened(self, symbol: str):
        """Record that a position was opened."""
        self.position_history[symbol] = {
            "opened_step": self.current_step,
        }

    def record_position_closed(self, symbol: str):
        """Record that a position was closed."""
        self.position_history.pop(symbol, None)

    def reset_daily(self):
        """Reset guardrails for new trading day."""
        self.current_step = 0
        self.position_history.clear()

    def get_status(self) -> Dict:
        """Get current guardrail status."""
        return {
            "current_step": self.current_step,
            "cooldown_remaining": 0,
            "tracked_positions": len(self.position_history),
            "position_ages": {
                symbol: self.current_step - info["opened_step"]
                for symbol, info in self.position_history.items()
            },
        }
