"""
Enhanced Training Dashboard with RL Metrics

Adds meaningful performance indicators equivalent to "accuracy" in supervised learning.
"""

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.columns import Columns
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import time


class EnhancedDashboard:
    """
    Enhanced training dashboard with RL-specific metrics.
    
    Metrics Categories:
    1. Learning Progress - Is the model improving?
    2. Performance Quality - How good is the model?
    3. Action Intelligence - Is the model making smart decisions?
    4. Model Health - Any warning signs?
    """
    
    def __init__(self, total_episodes: int):
        self.total_episodes = total_episodes
        self.console = Console()
        
        # Current state
        self.current_episode = 0
        self.epsilon = 1.0
        self.buffer_size = 0
        self.train_steps = 0
        
        # Episode metrics
        self.last_reward = 0.0
        self.last_pnl = 0.0
        self.last_win_rate = 0.0
        self.last_trades = 0
        self.last_loss = 0.0
        self.last_avg_q = 0.0
        self.last_max_q = 0.0
        
        # History (last 100 episodes)
        self.rewards_history = deque(maxlen=100)
        self.pnl_history = deque(maxlen=100)
        self.win_rate_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        
        # Action tracking (last 100 episodes)
        self.action_counts = {}  # action_type -> count
        self.profitable_actions = {}  # action_type -> total_pnl
        
        # Best metrics
        self.best_reward = float("-inf")
        self.best_pnl = float("-inf")
        self.best_eval_reward = float("-inf")
        self.best_sharpe = float("-inf")
        
        # Timing
        self.start_time = time.time()
        
    def update_episode(self, stats: Dict):
        """Update dashboard with episode results."""
        self.current_episode = stats.get("episode", 0)
        self.last_reward = stats.get("reward", 0.0)
        self.last_pnl = stats.get("pnl", 0.0)
        self.last_win_rate = stats.get("win_rate", 0.0)
        self.last_trades = stats.get("trades", 0)
        self.last_loss = stats.get("loss", 0.0)
        self.last_avg_q = stats.get("avg_q_value", 0.0)
        self.last_max_q = stats.get("max_q_value", 0.0)
        self.epsilon = stats.get("epsilon", 1.0)
        self.buffer_size = stats.get("buffer_size", 0)
        self.train_steps = stats.get("train_steps", 0)
        
        # Update histories
        self.rewards_history.append(self.last_reward)
        self.pnl_history.append(self.last_pnl)
        self.win_rate_history.append(self.last_win_rate)
        self.loss_history.append(self.last_loss)
        self.q_value_history.append(self.last_avg_q)
        
        # Update action tracking
        action_distribution = stats.get("action_distribution", {})
        for action, count in action_distribution.items():
            self.action_counts[action] = self.action_counts.get(action, 0) + count
            
        action_pnl = stats.get("action_pnl", {})
        for action, pnl in action_pnl.items():
            self.profitable_actions[action] = self.profitable_actions.get(action, 0) + pnl
        
        # Update bests
        if self.last_reward > self.best_reward:
            self.best_reward = self.last_reward
        if self.last_pnl > self.best_pnl:
            self.best_pnl = self.last_pnl
            
    def update_eval(self, eval_stats: Dict):
        """Update with evaluation results."""
        eval_reward = eval_stats.get("avg_reward", 0.0)
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            
        sharpe = eval_stats.get("sharpe_ratio", 0.0)
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
    
    def generate_dashboard(self) -> Layout:
        """Generate enhanced dashboard layout."""
        layout = Layout()
        
        # Main structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=7),
        )
        
        layout["main"].split_row(
            Layout(name="performance", ratio=1),
            Layout(name="learning", ratio=1),
            Layout(name="health", ratio=1),
        )
        
        # Header - Progress bar
        layout["header"].update(self._create_header())
        
        # Left - Performance Quality
        layout["performance"].update(self._create_performance_panel())
        
        # Center - Learning Progress  
        layout["learning"].update(self._create_learning_panel())
        
        # Right - Model Health
        layout["health"].update(self._create_health_panel())
        
        # Footer - Action Intelligence
        layout["footer"].update(self._create_action_panel())
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create header with progress."""
        progress_pct = (self.current_episode / self.total_episodes) * 100
        elapsed = time.time() - self.start_time
        eps_per_sec = self.current_episode / elapsed if elapsed > 0 else 0
        eta = (self.total_episodes - self.current_episode) / eps_per_sec if eps_per_sec > 0 else 0
        
        text = Text()
        text.append("🚀 ASYMPTOTIC ZERO ", style="bold cyan")
        text.append(f"Episode {self.current_episode}/{self.total_episodes} ", style="white")
        text.append(f"[{progress_pct:.1f}%] ", style="yellow")
        text.append(f"⏱ {self._format_time(elapsed)} ", style="green")
        text.append(f"ETA: {self._format_time(eta)}", style="dim")
        
        return Panel(text, box=box.ROUNDED)
    
    def _create_performance_panel(self) -> Panel:
        """Create performance quality panel."""
        table = Table(
            title="💰 Performance Quality",
            title_style="bold green",
            box=box.SIMPLE,
            expand=True
        )
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current", justify="right", width=12)
        table.add_column("Avg (100)", justify="right", width=12)
        table.add_column("Trend", justify="center", width=6)
        
        # Calculate averages and trends
        avg_reward = np.mean(self.rewards_history) if self.rewards_history else 0
        avg_pnl = np.mean(self.pnl_history) if self.pnl_history else 0
        avg_win_rate = np.mean(self.win_rate_history) if self.win_rate_history else 0
        
        # Profit factor (gross profit / gross loss)
        profit_factor = self._calculate_profit_factor()
        
        # Sharpe ratio (simplified)
        sharpe = self._calculate_sharpe()
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown()
        
        # Add rows with highlighting
        reward_style = self._get_value_style(self.last_reward)
        pnl_style = self._get_value_style(self.last_pnl)
        
        table.add_row(
            "💵 Reward",
            f"[{reward_style}]{self.last_reward:+.2f}[/]",
            f"[{self._get_value_style(avg_reward)}]{avg_reward:+.2f}[/]",
            self._get_trend_arrow(self.rewards_history)
        )
        
        table.add_row(
            "💰 PnL",
            f"[{pnl_style}]${self.last_pnl:+.2f}[/]",
            f"[{self._get_value_style(avg_pnl)}]${avg_pnl:+.2f}[/]",
            self._get_trend_arrow(self.pnl_history)
        )
        
        table.add_row(
            "🎯 Win Rate",
            f"{self.last_win_rate:.1f}%",
            f"{avg_win_rate:.1f}%",
            self._get_trend_arrow(self.win_rate_history)
        )
        
        table.add_row("─" * 20, "─" * 12, "─" * 12, "─" * 6)
        
        # Quality metrics
        pf_style = "green bold" if profit_factor > 1.5 else "yellow" if profit_factor > 1.0 else "red"
        sharpe_style = "green bold" if sharpe > 1.0 else "yellow" if sharpe > 0.5 else "red"
        dd_style = "green" if max_dd > -10 else "yellow" if max_dd > -20 else "red"
        
        table.add_row(
            "📊 Profit Factor",
            f"[{pf_style}]{profit_factor:.2f}[/]",
            self._get_quality_label(profit_factor, [1.5, 1.0]),
            ""
        )
        
        table.add_row(
            "📈 Sharpe Ratio",
            f"[{sharpe_style}]{sharpe:.2f}[/]",
            self._get_quality_label(sharpe, [1.0, 0.5]),
            ""
        )
        
        table.add_row(
            "📉 Max Drawdown",
            f"[{dd_style}]{max_dd:.1f}%[/]",
            self._get_quality_label(-max_dd, [10, 20], reverse=True),
            ""
        )
        
        return Panel(table, box=box.ROUNDED, border_style="green")
    
    def _create_learning_panel(self) -> Panel:
        """Create learning progress panel."""
        table = Table(
            title="🧠 Learning Progress",
            title_style="bold blue",
            box=box.SIMPLE,
            expand=True
        )
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right", width=12)
        table.add_column("Status", justify="center", width=12)
        
        # Q-value metrics
        avg_q = np.mean(self.q_value_history) if self.q_value_history else 0
        q_stability = np.std(self.q_value_history) if len(self.q_value_history) > 1 else 0
        
        # Loss metrics
        avg_loss = np.mean(self.loss_history) if self.loss_history else 0
        loss_trend = self._get_trend_direction(self.loss_history)
        
        # Exploration ratio
        exploration_ratio = self.epsilon * 100
        
        # Learning rate (episodes with positive reward)
        learning_rate = sum(1 for r in self.rewards_history if r > 0) / len(self.rewards_history) * 100 if self.rewards_history else 0
        
        # Add rows
        q_style = "green" if avg_q > 0 else "yellow" if avg_q > -50 else "red"
        table.add_row(
            "🎯 Avg Q-Value",
            f"[{q_style}]{avg_q:.2f}[/]",
            self._get_q_status(avg_q)
        )
        
        table.add_row(
            "📊 Q Stability",
            f"{q_stability:.2f}",
            self._get_stability_status(q_stability)
        )
        
        table.add_row(
            "📉 Training Loss",
            f"{self.last_loss:.4f}",
            self._get_loss_status(loss_trend)
        )
        
        table.add_row("─" * 20, "─" * 12, "─" * 12)
        
        exp_style = "yellow" if exploration_ratio > 10 else "green"
        table.add_row(
            "🔍 Exploration",
            f"[{exp_style}]{exploration_ratio:.1f}%[/]",
            self._get_exploration_status(exploration_ratio)
        )
        
        lr_style = "green bold" if learning_rate > 60 else "yellow" if learning_rate > 40 else "red"
        table.add_row(
            "✅ Success Rate",
            f"[{lr_style}]{learning_rate:.1f}%[/]",
            self._get_learning_status(learning_rate)
        )
        
        table.add_row(
            "📚 Buffer Size",
            f"{self.buffer_size:,}",
            "✓ Full" if self.buffer_size > 5000 else "⚠ Filling"
        )
        
        return Panel(table, box=box.ROUNDED, border_style="blue")
    
    def _create_health_panel(self) -> Panel:
        """Create model health panel."""
        table = Table(
            title="🏥 Model Health",
            title_style="bold magenta",
            box=box.SIMPLE,
            expand=True
        )
        table.add_column("Check", style="cyan", width=20)
        table.add_column("Status", justify="center", width=20)
        
        # Health checks
        checks = []
        
        # 1. Reward stability
        if len(self.rewards_history) > 10:
            recent_std = np.std(list(self.rewards_history)[-10:])
            overall_std = np.std(self.rewards_history)
            if recent_std < overall_std * 0.8:
                checks.append(("Reward Stable", "✅ Converging", "green"))
            elif recent_std > overall_std * 1.5:
                checks.append(("Reward Stable", "⚠️ Volatile", "yellow"))
            else:
                checks.append(("Reward Stable", "→ Learning", "blue"))
        else:
            checks.append(("Reward Stable", "⏳ Warming up", "dim"))
        
        # 2. Q-value health
        if self.last_avg_q < -1000:
            checks.append(("Q-Values", "❌ Exploded", "red bold"))
        elif self.last_avg_q > 1000:
            checks.append(("Q-Values", "❌ Exploded", "red bold"))
        elif abs(self.last_avg_q) < 0.01:
            checks.append(("Q-Values", "⚠️ Too small", "yellow"))
        else:
            checks.append(("Q-Values", "✅ Healthy", "green"))
        
        # 3. Loss trend
        loss_trend = self._get_trend_direction(self.loss_history)
        if loss_trend == "down":
            checks.append(("Loss Trend", "✅ Decreasing", "green"))
        elif loss_trend == "up":
            checks.append(("Loss Trend", "⚠️ Increasing", "yellow"))
        else:
            checks.append(("Loss Trend", "→ Stable", "blue"))
        
        # 4. Performance trend
        perf_trend = self._get_trend_direction(self.pnl_history)
        if perf_trend == "up":
            checks.append(("PnL Trend", "✅ Improving", "green bold"))
        elif perf_trend == "down":
            checks.append(("PnL Trend", "⚠️ Declining", "yellow"))
        else:
            checks.append(("PnL Trend", "→ Plateau", "blue"))
        
        # 5. Exploration balance
        if self.epsilon < 0.01:
            checks.append(("Exploration", "✅ Exploiting", "green"))
        elif self.epsilon > 0.5:
            checks.append(("Exploration", "🔍 Exploring", "yellow"))
        else:
            checks.append(("Exploration", "⚖️ Balanced", "blue"))
        
        # 6. Training activity
        if self.train_steps > 0:
            checks.append(("Training Active", "✅ Learning", "green"))
        else:
            checks.append(("Training Active", "⏸️ Buffering", "yellow"))
        
        # Add all checks
        for check_name, status, style in checks:
            table.add_row(check_name, f"[{style}]{status}[/]")
        
        return Panel(table, box=box.ROUNDED, border_style="magenta")
    
    def _create_action_panel(self) -> Panel:
        """Create action intelligence panel."""
        table = Table(
            title="🎮 Action Intelligence (Last 100 Episodes)",
            title_style="bold yellow",
            box=box.SIMPLE,
            expand=True,
            show_header=True
        )
        table.add_column("Action", style="cyan", width=15)
        table.add_column("Count", justify="right", width=10)
        table.add_column("Frequency", justify="right", width=12)
        table.add_column("Avg PnL", justify="right", width=12)
        table.add_column("Quality", justify="center", width=10)
        
        if self.action_counts:
            total_actions = sum(self.action_counts.values())
            
            # Sort by frequency
            sorted_actions = sorted(
                self.action_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:8]  # Top 8 actions
            
            for action, count in sorted_actions:
                freq_pct = (count / total_actions * 100) if total_actions > 0 else 0
                avg_pnl = self.profitable_actions.get(action, 0) / count if count > 0 else 0
                
                # Style based on profitability
                pnl_style = "green" if avg_pnl > 0 else "red"
                quality = "🟢" if avg_pnl > 10 else "🟡" if avg_pnl > 0 else "🔴"
                
                table.add_row(
                    action,
                    str(count),
                    f"{freq_pct:.1f}%",
                    f"[{pnl_style}]${avg_pnl:+.2f}[/]",
                    quality
                )
        else:
            table.add_row("No data yet...", "", "", "", "")
        
        return Panel(table, box=box.ROUNDED, border_style="yellow")
    
    # Helper methods
    def _get_value_style(self, value: float) -> str:
        """Get color style based on value."""
        if value > 10:
            return "green bold"
        elif value > 0:
            return "green"
        elif value > -10:
            return "red"
        else:
            return "red bold"
    
    def _get_trend_arrow(self, history: deque) -> str:
        """Get trend arrow based on recent history."""
        if len(history) < 10:
            return "⏳"
        
        recent = list(history)[-10:]
        older = list(history)[-20:-10] if len(history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return "[green]↗[/]"
        elif recent_avg < older_avg * 0.9:
            return "[red]↘[/]"
        else:
            return "[yellow]→[/]"
    
    def _get_trend_direction(self, history: deque) -> str:
        """Get trend direction: up, down, or stable."""
        if len(history) < 10:
            return "stable"
        
        recent = list(history)[-10:]
        older = list(history)[-20:-10] if len(history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.05:
            return "up"
        elif recent_avg < older_avg * 0.95:
            return "down"
        else:
            return "stable"
    
    def _get_quality_label(self, value: float, thresholds: List[float], reverse: bool = False) -> str:
        """Get quality label based on thresholds."""
        if reverse:
            if value < thresholds[0]:
                return "[green bold]Excellent[/]"
            elif value < thresholds[1]:
                return "[green]Good[/]"
            else:
                return "[yellow]Fair[/]"
        else:
            if value > thresholds[0]:
                return "[green bold]Excellent[/]"
            elif value > thresholds[1]:
                return "[green]Good[/]"
            else:
                return "[yellow]Fair[/]"
    
    def _get_q_status(self, avg_q: float) -> str:
        """Get Q-value status."""
        if avg_q > 100:
            return "✅ Confident"
        elif avg_q > 0:
            return "→ Learning"
        elif avg_q > -100:
            return "⚠️ Uncertain"
        else:
            return "❌ Confused"
    
    def _get_stability_status(self, std: float) -> str:
        """Get stability status."""
        if std < 10:
            return "✅ Very stable"
        elif std < 50:
            return "→ Stable"
        elif std < 100:
            return "⚠️ Volatile"
        else:
            return "❌ Unstable"
    
    def _get_loss_status(self, trend: str) -> str:
        """Get loss status."""
        if trend == "down":
            return "✅ Improving"
        elif trend == "up":
            return "⚠️ Worsening"
        else:
            return "→ Stable"
    
    def _get_exploration_status(self, pct: float) -> str:
        """Get exploration status."""
        if pct > 50:
            return "🔍 Exploring"
        elif pct > 10:
            return "⚖️ Balanced"
        else:
            return "✅ Exploiting"
    
    def _get_learning_status(self, rate: float) -> str:
        """Get learning status."""
        if rate > 70:
            return "✅ Excellent"
        elif rate > 50:
            return "→ Good"
        elif rate > 30:
            return "⚠️ Struggling"
        else:
            return "❌ Not learning"
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        if not self.pnl_history:
            return 0.0
        
        profits = [p for p in self.pnl_history if p > 0]
        losses = [abs(p) for p in self.pnl_history if p < 0]
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 1  # Avoid division by zero
        
        return total_profit / total_loss if total_loss > 0 else 0.0
    
    def _calculate_sharpe(self) -> float:
        """Calculate simplified Sharpe ratio."""
        if len(self.pnl_history) < 2:
            return 0.0
        
        returns = np.array(self.pnl_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage."""
        if not self.pnl_history:
            return 0.0
        
        cumulative = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return np.min(drawdown) / 100 if len(cumulative) > 0 else 0.0  # Convert to percentage
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
