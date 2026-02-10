"""
Live Trading Dashboard

Terminal dashboard for monitoring Asymptotic Zero live trading.
Uses Rich library (same as training dashboard) for consistent UI.

Trade History panel supports keyboard scrolling:
  ↑  scroll up   (older trades)
  ↓  scroll down (newer trades / auto-follow latest)
"""

import sys
import tty
import termios
import select
import threading
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime
import time


class LiveTradingDashboard:
    """
    Real-time terminal dashboard for live trading.

    Sections:
    1. Header     - Bot status, session time, next step countdown
    2. Portfolio  - Balance, PnL, drawdown
    3. Positions  - Open positions with unrealized PnL
    4. Agent      - Last decision, Q-values, guardrail status
    5. History    - Scrollable trade log (↑/↓ arrow keys)
    """

    def __init__(self):
        self.console = Console()

        # Session info
        self.start_time = time.time()
        self.session_date = datetime.now().strftime("%Y-%m-%d")
        self.trading_mode = "testnet"

        # Portfolio state
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
        self.pnl_history = deque(maxlen=50)

        # Step tracking
        self.current_step = 0
        self.next_step_time: Optional[datetime] = None
        self.seconds_remaining = 0

        # Screening info
        self.top_gainer = ""
        self.top_gainer_pct = 0.0
        self.top_loser = ""
        self.top_loser_pct = 0.0
        self.screened_coins: List[str] = []

        # Positions
        self.open_positions: Dict[str, Dict] = {}

        # Agent state
        self.last_action = "—"
        self.last_action_num = -1
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        self.epsilon = 0.05

        # Guardrails
        self.cooldown_remaining = 0
        self.guardrail_step = 0

        # Daily stats
        self.daily_trades = 0
        self.max_daily_trades = 10
        self.winning_trades = 0
        self.losing_trades = 0

        # Trade history — store full day, not just last 10
        self.trade_history: deque = deque(maxlen=200)

        # Action history (last 20 steps)
        self.action_history: deque = deque(maxlen=20)

        # Error tracking
        self.last_error: str = ""
        self.error_count: int = 0

        # Scroll state for trade history
        # offset=0 means "show newest trades" (top of list)
        # offset=N means "scrolled N rows toward older trades"
        self.history_scroll_offset: int = 0
        self._history_visible_rows: int = 15     # updated dynamically each render
        self._scroll_lock = threading.Lock()

        # Keyboard listener thread
        self._stop_scroll = threading.Event()
        self._scroll_thread: Optional[threading.Thread] = None
        self._old_term_settings = None

    # ─── Keyboard Listener ─────────────────────────────────────────────

    def start_keyboard_listener(self):
        """Start background thread to read ↑/↓ arrow keys for history scrolling."""
        if self._scroll_thread and self._scroll_thread.is_alive():
            return
        self._stop_scroll.clear()
        self._scroll_thread = threading.Thread(
            target=self._keyboard_listener_thread,
            daemon=True,
            name="history-scroll"
        )
        self._scroll_thread.start()

    def stop_keyboard_listener(self):
        """Signal keyboard listener to stop and restore terminal settings."""
        self._stop_scroll.set()
        if self._scroll_thread:
            self._scroll_thread.join(timeout=1.0)

    def _keyboard_listener_thread(self):
        """
        Read stdin in cbreak mode.
        cbreak = single-keypress reads without buffering, but Ctrl+C still works.
        Arrow keys arrive as 3-byte escape sequences: ESC [ A/B
        """
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            # If we can't set cbreak (e.g. stdin is not a tty), silently skip
            return

        try:
            while not self._stop_scroll.is_set():
                # Non-blocking check: wait up to 0.1s for a keypress
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue

                ch = sys.stdin.read(1)

                # Arrow keys: ESC [ A (up) / ESC [ B (down)
                if ch == '\x1b':
                    # Read the next two bytes of the escape sequence
                    more, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not more:
                        continue
                    bracket = sys.stdin.read(1)
                    if bracket != '[':
                        continue
                    more2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not more2:
                        continue
                    direction = sys.stdin.read(1)

                    with self._scroll_lock:
                        total = len(self.trade_history)
                        max_offset = max(0, total - self._history_visible_rows)

                        if direction == 'A':   # ↑  — go toward older trades
                            self.history_scroll_offset = min(
                                max_offset,
                                self.history_scroll_offset + 1
                            )
                        elif direction == 'B': # ↓  — go toward newer trades
                            self.history_scroll_offset = max(
                                0,
                                self.history_scroll_offset - 1
                            )

                elif ch in ('q', 'Q'):
                    # q = scroll to top (newest) as a convenience
                    with self._scroll_lock:
                        self.history_scroll_offset = 0

        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    # ─── Update Methods ────────────────────────────────────────────────

    def update_portfolio(self, balance: float, initial_balance: float, daily_pnl: float):
        self.current_balance = balance
        self.initial_balance = initial_balance
        self.daily_pnl = daily_pnl
        self.pnl_history.append(daily_pnl)

    def update_step(self, step: int, next_step_time: datetime):
        self.current_step = step
        self.next_step_time = next_step_time

    def update_countdown(self, seconds_remaining: int):
        self.seconds_remaining = seconds_remaining

    def update_screening(self, coins: Dict):
        self.top_gainer = coins["gainers"][0] if coins["gainers"] else ""
        self.top_gainer_pct = coins["gainers_pct"][0] if coins["gainers_pct"] else 0.0
        self.top_loser = coins["losers"][0] if coins["losers"] else ""
        self.top_loser_pct = coins["losers_pct"][0] if coins["losers_pct"] else 0.0
        self.screened_coins = coins["gainers"] + coins["losers"]

    def update_positions(self, positions: Dict):
        self.open_positions = positions

    def update_agent(self, action: int, action_str: str, avg_q: float = 0.0, max_q: float = 0.0):
        self.last_action_num = action
        self.last_action = action_str
        self.last_avg_q = avg_q
        self.last_max_q = max_q
        self.action_history.append({
            "step": self.current_step,
            "action": action_str,
            "time": datetime.now().strftime("%H:%M:%S")
        })

    def update_guardrails(self, step: int, cooldown_remaining: int):
        self.guardrail_step = step
        self.cooldown_remaining = cooldown_remaining

    def update_config(self, mode: str, max_daily_loss: float, max_daily_trades: int, epsilon: float):
        self.trading_mode = mode
        self.max_daily_loss = max_daily_loss
        self.max_daily_trades = max_daily_trades
        self.epsilon = epsilon

    def record_trade(self, symbol: str, side: str, pnl: float, action: str):
        pnl_sign = "+" if pnl >= 0 else ""
        self.trade_history.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "pnl_str": f"{pnl_sign}{pnl:.2f}",
            "action": action,
        })
        self.daily_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        # If user is already at top (newest view), stay there (auto-follow)
        # If user has scrolled away, leave their position alone
        # — nothing to do here; offset stays as-is

    # ─── Layout Builder ────────────────────────────────────────────────

    def generate_dashboard(self) -> Layout:
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", size=22),   # fixed height for 3 top panels
            Layout(name="footer", ratio=1), # history gets all remaining height
        )

        layout["main"].split_row(
            Layout(name="portfolio", ratio=1),
            Layout(name="positions", ratio=1),
            Layout(name="agent", ratio=1),
        )

        layout["header"].update(self._create_header())
        layout["portfolio"].update(self._create_portfolio_panel())
        layout["positions"].update(self._create_positions_panel())
        layout["agent"].update(self._create_agent_panel())
        layout["footer"].update(self._create_history_panel())

        return layout

    # ─── Panel Builders ─────────────────────────────────────────────────

    def _create_header(self) -> Panel:
        elapsed = time.time() - self.start_time
        mode_style = "yellow bold" if self.trading_mode == "testnet" else "green bold"
        mode_label = "🧪 TESTNET" if self.trading_mode == "testnet" else "🔴 LIVE"

        mins, secs = divmod(self.seconds_remaining, 60)
        countdown = f"{mins:02d}:{secs:02d}"
        countdown_style = "red bold" if self.seconds_remaining < 30 else "yellow" if self.seconds_remaining < 60 else "green"

        pnl_style = "green bold" if self.daily_pnl >= 0 else "red bold"
        pnl_sign = "+" if self.daily_pnl >= 0 else ""

        text = Text()
        text.append("⚡ ASYMPTOTIC ZERO  ", style="bold cyan")
        text.append(f"[{mode_label}]  ", style=mode_style)
        text.append(f"📅 {self.session_date}  ", style="white")
        text.append(f"⏱ {self._fmt_time(elapsed)}  ", style="dim")
        text.append(f"  Step {self.current_step}  ", style="white")
        text.append("⏳ Next: ", style="white")
        text.append(countdown, style=countdown_style)
        text.append("  ", style="white")
        text.append("  Daily PnL: ", style="white")
        text.append(f"{pnl_sign}${self.daily_pnl:.2f}", style=pnl_style)

        return Panel(text, box=box.ROUNDED, border_style="cyan")

    def _create_portfolio_panel(self) -> Panel:
        table = Table(
            title="💰 Portfolio",
            title_style="bold green",
            box=box.SIMPLE,
            expand=True
        )
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right", width=18)

        balance_change = self.current_balance - self.initial_balance
        balance_style = "green" if balance_change >= 0 else "red"

        table.add_row("Balance", f"[white bold]${self.current_balance:,.2f}[/]")
        table.add_row(
            "Session PnL",
            f"[{balance_style} bold]{'+'if balance_change>=0 else ''}${balance_change:.2f}[/]"
        )

        pnl_pct = (self.daily_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
        pnl_style = "green bold" if self.daily_pnl >= 0 else "red bold"
        table.add_row(
            "Daily PnL",
            f"[{pnl_style}]{'+'if self.daily_pnl>=0 else ''}${self.daily_pnl:.2f} ({pnl_pct:+.2f}%)[/]"
        )

        loss_used_pct = abs(min(self.daily_pnl, 0)) / self.max_daily_loss * 100 if self.max_daily_loss > 0 else 0
        loss_bar = self._make_bar(loss_used_pct, 10, "red")
        table.add_row("Loss Limit", f"{loss_bar} {loss_used_pct:.0f}%")

        table.add_row("──────────────────", "──────────────────")

        win_rate = self.winning_trades / self.daily_trades * 100 if self.daily_trades > 0 else 0
        wr_style = "green" if win_rate >= 50 else "red"

        table.add_row("Trades Today", f"{self.daily_trades}/{self.max_daily_trades}")
        table.add_row(
            "Win Rate",
            f"[{wr_style}]{win_rate:.0f}%[/] ({self.winning_trades}W / {self.losing_trades}L)"
        )

        table.add_row("──────────────────", "──────────────────")
        table.add_row("Top Gainer", f"[green]{self.top_gainer} +{self.top_gainer_pct:.1f}%[/]")
        table.add_row("Top Loser",  f"[red]{self.top_loser} {self.top_loser_pct:.1f}%[/]")

        return Panel(table, box=box.ROUNDED, border_style="green")

    def _create_positions_panel(self) -> Panel:
        table = Table(
            title=f"📊 Positions ({len(self.open_positions)}/3)",
            title_style="bold yellow",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Symbol", style="cyan", width=14)
        table.add_column("Side", justify="center", width=6)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("PnL", justify="right", width=12)

        if self.open_positions:
            for symbol, pos in self.open_positions.items():
                side = pos.get("side", "?")
                entry = pos.get("entry_price", 0)
                unrealized_pnl = pos.get("unrealized_pnl", 0)
                side_style = "green" if side == "LONG" else "red"
                side_label = "▲ L" if side == "LONG" else "▼ S"
                pnl_style = "green bold" if unrealized_pnl >= 0 else "red bold"
                pnl_sign = "+" if unrealized_pnl >= 0 else ""
                table.add_row(
                    symbol,
                    f"[{side_style}]{side_label}[/]",
                    f"${entry:.4f}" if entry < 1 else f"${entry:.2f}",
                    f"[{pnl_style}]{pnl_sign}${unrealized_pnl:.2f}[/]",
                )
        else:
            table.add_row("[dim]No open positions[/]", "", "", "")

        table.add_row("──────────────────", "──────", "──────────", "────────────")

        cooldown_style = "red" if self.cooldown_remaining > 0 else "green"
        table.add_row("Guardrail Step", "", "", f"[white]{self.guardrail_step}[/]")
        table.add_row("Cooldown", "", "", f"[{cooldown_style}]{self.cooldown_remaining} steps[/]")

        return Panel(table, box=box.ROUNDED, border_style="yellow")

    def _create_agent_panel(self) -> Panel:
        table = Table(
            title="🤖 Agent",
            title_style="bold magenta",
            box=box.SIMPLE,
            expand=True,
        )
        table.add_column("Metric", style="cyan", width=16)
        table.add_column("Value", justify="right", width=20)

        action_color = self._get_action_color(self.last_action)
        table.add_row("Last Action", f"[{action_color} bold]{self.last_action}[/]")
        table.add_row("Action #", f"{self.last_action_num}")

        q_style = "green" if self.last_avg_q > 0 else "red"
        table.add_row("Avg Q-Value", f"[{q_style}]{self.last_avg_q:.3f}[/]")
        table.add_row("Max Q-Value", f"{self.last_max_q:.3f}")

        exp_pct = self.epsilon * 100
        exp_style = "yellow" if exp_pct > 10 else "green"
        table.add_row("Exploration", f"[{exp_style}]{exp_pct:.1f}%[/]")

        table.add_row("────────────────", "────────────────────")

        table.add_row("[dim]Recent Actions[/]", "")
        recent = list(self.action_history)[-5:]
        for entry in reversed(recent):
            color = self._get_action_color(entry["action"])
            short = entry["action"][:18] if len(entry["action"]) > 18 else entry["action"]
            table.add_row(f"  [dim]{entry['time']}[/]", f"[{color}]{short}[/]")

        if self.last_error:
            table.add_row("────────────────", "────────────────────")
            err_short = self.last_error[:36] if len(self.last_error) > 36 else self.last_error
            table.add_row(
                f"[red]⚠ Errors ({self.error_count})[/red]",
                f"[red dim]{err_short}[/red dim]"
            )

        return Panel(table, box=box.ROUNDED, border_style="magenta")

    def _create_history_panel(self) -> Panel:
        # How many rows fit? terminal height minus: header(4) + main(22) + borders(~4) + panel header(2)
        term_h = self.console.height or 40
        visible = max(5, term_h - 4 - 22 - 6)

        # Update the class attribute so the keyboard thread can use it
        with self._scroll_lock:
            self._history_visible_rows = visible
            total = len(self.trade_history)
            offset = self.history_scroll_offset
            # Clamp offset in case history shrank
            max_offset = max(0, total - visible)
            offset = min(offset, max_offset)
            self.history_scroll_offset = offset

        trades_list = list(self.trade_history)  # index 0 = newest

        # Slice the visible window
        window = trades_list[offset: offset + visible]

        # Build scroll indicator for the title
        if total == 0:
            scroll_hint = ""
        elif total <= visible:
            scroll_hint = f"  [dim]{total} trades[/dim]"
        else:
            top_row    = offset + 1
            bottom_row = min(offset + visible, total)
            at_top    = offset == 0
            at_bottom = offset >= max_offset

            up_arrow   = "[dim]↑[/dim]" if at_top    else "[bold white]↑[/bold white]"
            down_arrow = "[dim]↓[/dim]" if at_bottom else "[bold white]↓[/bold white]"
            scroll_hint = (
                f"  {up_arrow}[dim]/{down_arrow}[/dim]  "
                f"[dim]{top_row}–{bottom_row} of {total}[/dim]  "
                f"[dim italic]q=latest[/dim italic]"
            )

        table = Table(
            title=f"📋 Trade History{scroll_hint}",
            title_style="bold blue",
            box=box.SIMPLE,
            expand=True,
            show_header=True,
        )
        table.add_column("Time",   style="dim",    width=10)
        table.add_column("Symbol", style="cyan",   width=14)
        table.add_column("Action", width=22)
        table.add_column("Side",   justify="center", width=8)
        table.add_column("PnL",    justify="right",  width=14)

        if window:
            for trade in window:
                pnl_style  = "green bold" if trade["pnl"] >= 0 else "red bold"
                side_style = "green" if trade["side"] == "LONG" else "red"
                table.add_row(
                    trade["time"],
                    trade["symbol"],
                    trade["action"],
                    f"[{side_style}]{trade['side']}[/]",
                    f"[{pnl_style}]{trade['pnl_str']}[/]",
                )
        else:
            table.add_row("[dim]No trades yet...[/]", "", "", "", "")

        return Panel(table, box=box.ROUNDED, border_style="blue")

    # ─── Helpers ────────────────────────────────────────────────────────

    def _get_action_color(self, action: str) -> str:
        if not action or action == "—":
            return "dim"
        action_upper = action.upper()
        if "LONG"  in action_upper: return "green"
        if "SHORT" in action_upper: return "red"
        if "CLOSE" in action_upper: return "yellow"
        if "HOLD"  in action_upper: return "blue"
        return "white"

    def _make_bar(self, pct: float, width: int = 10, color: str = "green") -> str:
        filled = int(min(pct, 100) / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{color}]{bar}[/]"

    def _fmt_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
