"""
DQN Trainer

Handles the training loop for the DQN agent:
- Episode management
- Training coordination
- Evaluation
- Checkpointing
- Logging
- Live dashboard visualization
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import yaml
import logging
import json
import time

from .dqn_agent import DQNAgent
from ..trading import TradingEnvironment
from .enhanced_dashboard import EnhancedDashboard

logger = logging.getLogger(__name__)

# Try to import Rich for dashboard
try:
    from rich.console import Console
    from rich.live import Live

    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    logger.warning("Rich not installed. Install with: pip install rich")


# OLD TrainingDashboard removed - using EnhancedDashboard now
# Skip to DQNTrainer class below


# Old TrainingDashboard class removed - now using EnhancedDashboard


class DQNTrainer:
    """
    Trainer for DQN agent.

    Manages the training loop, evaluation, and checkpointing.
    """

    def __init__(
        self,
        env: TradingEnvironment,
        agent: DQNAgent,
        config_path: str = "config/agent.yaml",
    ):
        """
        Initialize trainer.

        Args:
            env: Trading environment
            agent: DQN agent
            config_path: Path to agent configuration
        """
        self.env = env
        self.agent = agent
        self.config_path = Path(config_path)

        # Load configuration
        self.config = self._load_config()

        # Training settings
        training_cfg = self.config.get("training", {})
        self.total_episodes = training_cfg.get("total_episodes", 10000)
        self.max_steps_per_episode = training_cfg.get("max_steps_per_episode", 100)
        self.train_freq = training_cfg.get("train_freq", 4)
        self.save_freq = training_cfg.get("save_freq", 100)
        self.eval_freq = training_cfg.get("eval_freq", 50)
        self.eval_episodes = training_cfg.get("eval_episodes", 10)

        # Checkpointing
        checkpoint_cfg = self.config.get("checkpointing", {})
        self.save_dir = Path(checkpoint_cfg.get("save_dir", "checkpoints"))
        self.save_best = checkpoint_cfg.get("save_best", True)
        self.keep_last_n = checkpoint_cfg.get("keep_last_n", 5)

        # Logging
        logging_cfg = self.config.get("logging", {})
        self.log_dir = Path(logging_cfg.get("log_dir", "logs"))
        self.log_freq = logging_cfg.get("log_freq", 10)
        self.verbose = logging_cfg.get("verbose", 1)

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_eval_reward = float("-inf")
        self.training_history: List[Dict] = []
        self.eval_history: List[Dict] = []

        # Step counter
        self.global_step = 0

        # Dashboard
        self.dashboard: Optional[EnhancedDashboard] = None

        logger.info("Trainer initialized:")
        logger.info(f"  Total episodes: {self.total_episodes}")
        logger.info(f"  Train freq: {self.train_freq}")
        logger.info(f"  Save dir: {self.save_dir}")

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def train(
        self, resume_from: Optional[str] = None, use_dashboard: bool = True
    ) -> Dict:
        """
        Run the full training loop.

        Args:
            resume_from: Path to checkpoint to resume from
            use_dashboard: Whether to use live dashboard (requires Rich)

        Returns:
            Training statistics
        """
        start_episode = 0

        # Resume if checkpoint provided
        if resume_from:
            self.agent.load(resume_from)
            start_episode = self.agent.episode_counter
            logger.info(f"Resumed from episode {start_episode}")

        logger.info(f"Starting training from episode {start_episode}")

        # Use dashboard if available
        use_dashboard = use_dashboard and HAS_RICH and self.verbose >= 1

        if use_dashboard:
            self.dashboard = EnhancedDashboard(self.total_episodes)
            self._train_with_dashboard(start_episode)
        else:
            self._train_without_dashboard(start_episode)

        # Save final model
        self.agent.save(str(self.save_dir / "final"))

        # Save training history
        self._save_history()

        logger.info("Training complete!")

        return self._get_training_summary()

    def _train_with_dashboard(self, start_episode: int):
        """Training loop with live dashboard."""
        console = Console()

        with Live(
            self.dashboard.generate_dashboard(), console=console, refresh_per_second=2
        ) as live:
            for episode in range(start_episode, self.total_episodes):
                # Run one episode
                episode_stats = self._run_episode(training=True)
                
                # Get Q-value statistics from agent
                q_stats = self.agent.get_q_statistics()

                # Update dashboard with complete data
                self.dashboard.update_episode({
                    "episode": episode + 1,
                    "reward": episode_stats["reward"],
                    "pnl": episode_stats["total_pnl"],
                    "win_rate": episode_stats["win_rate"],
                    "trades": episode_stats["total_trades"],
                    "loss": episode_stats["avg_loss"],
                    "epsilon": self.agent.epsilon,
                    "buffer_size": len(self.agent.replay_buffer),
                    "train_steps": self.agent.train_step_counter,
                    "avg_q_value": q_stats["avg_q_value"],
                    "max_q_value": q_stats["max_q_value"],
                    "action_distribution": episode_stats.get("action_distribution", {}),
                    "action_pnl": episode_stats.get("action_pnl", {}),
                })

                # Evaluate periodically
                if (episode + 1) % self.eval_freq == 0:
                    eval_stats = self.evaluate()
                    self.dashboard.update_eval(eval_stats)

                    # Save best model
                    if (
                        self.save_best
                        and eval_stats["avg_reward"] > self.best_eval_reward
                    ):
                        self.best_eval_reward = eval_stats["avg_reward"]
                        self.agent.save(str(self.save_dir / "best"))

                # Periodic checkpoint
                if (episode + 1) % self.save_freq == 0:
                    checkpoint_path = self.save_dir / f"checkpoint_{episode + 1}"
                    self.agent.save(str(checkpoint_path))
                    self._cleanup_old_checkpoints()

                # Update display
                live.update(self.dashboard.generate_dashboard())

    def _train_without_dashboard(self, start_episode: int):
        """Training loop without dashboard (fallback)."""
        for episode in range(start_episode, self.total_episodes):
            # Run one episode
            episode_stats = self._run_episode(training=True)

            # Log progress
            if (episode + 1) % self.log_freq == 0:
                self._log_progress(episode, episode_stats)

            # Evaluate
            if (episode + 1) % self.eval_freq == 0:
                eval_stats = self.evaluate()
                self._log_evaluation(episode, eval_stats)

                # Save best model
                if self.save_best and eval_stats["avg_reward"] > self.best_eval_reward:
                    self.best_eval_reward = eval_stats["avg_reward"]
                    self.agent.save(str(self.save_dir / "best"))
                    logger.info(
                        f"New best model saved! Reward: {self.best_eval_reward:.2f}"
                    )

            # Periodic checkpoint
            if (episode + 1) % self.save_freq == 0:
                checkpoint_path = self.save_dir / f"checkpoint_{episode + 1}"
                self.agent.save(str(checkpoint_path))
                self._cleanup_old_checkpoints()

    def _run_episode(self, training: bool = True) -> Dict:
        """
        Run a single episode.

        Args:
            training: Whether to train during episode

        Returns:
            Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        
        # Track actions for dashboard
        action_counts = {}
        action_pnls = {}

        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state, training=training)

            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Track action and PnL
            action_name = self._action_to_name(action, info.get('action_type', 'unknown'))
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            if info.get('pnl', 0) != 0:
                action_pnls[action_name] = action_pnls.get(action_name, 0) + info['pnl']

            # Store transition
            if training:
                self.agent.store_transition(state, action, reward, next_state, done)

                # Train
                if self.global_step % self.train_freq == 0:
                    loss = self.agent.train_step()
                    if loss is not None:
                        episode_losses.append(loss)

                self.global_step += 1

            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # End episode
        if training:
            self.agent.end_episode(episode_reward)

        # Collect statistics
        stats = {
            "reward": episode_reward,
            "steps": episode_steps,
            "epsilon": self.agent.epsilon,
            "avg_loss": np.mean(episode_losses) if episode_losses else 0.0,
            "action_distribution": action_counts,
            "action_pnl": action_pnls,
        }

        # Add trading statistics
        trading_stats = self.env.get_episode_statistics()
        stats.update(
            {
                "total_trades": trading_stats["total_trades"],
                "win_rate": trading_stats["win_rate"],
                "total_pnl": trading_stats["total_pnl"],
                "return_pct": trading_stats["return_pct"],
            }
        )

        self.training_history.append(stats)

        return stats
    
    def _action_to_name(self, action: int, action_type: str) -> str:
        """Convert action index to readable name."""
        n = 10  # coins per side
        
        if action == 0:
            return "HOLD"
        elif 1 <= action <= n:
            return "LONG_gainer"
        elif n+1 <= action <= 2*n:
            return "SHORT_gainer"
        elif 2*n+1 <= action <= 3*n:
            return "CLOSE_gainer"
        elif 3*n+1 <= action <= 4*n:
            return "LONG_loser"
        elif 4*n+1 <= action <= 5*n:
            return "SHORT_loser"
        elif 5*n+1 <= action <= 6*n:
            return "CLOSE_loser"
        elif action == 61:
            return "CLOSE_ALL"
        elif action == 62:
            return "CLOSE_WORST"
        else:
            return "INVALID"

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict:
        """
        Evaluate agent performance.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation statistics
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        rewards = []
        pnls = []
        win_rates = []

        for _ in range(num_episodes):
            stats = self._run_episode(training=False)
            rewards.append(stats["reward"])
            pnls.append(stats["total_pnl"])
            win_rates.append(stats["win_rate"])

        eval_stats = {
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_pnl": np.mean(pnls),
            "avg_win_rate": np.mean(win_rates),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
        }

        self.eval_history.append(eval_stats)

        return eval_stats

    def _log_progress(self, episode: int, stats: Dict):
        """Log training progress."""
        if self.verbose == 0:
            return

        agent_stats = self.agent.get_statistics()

        msg = (
            f"Episode {episode + 1}/{self.total_episodes} | "
            f"Reward: {stats['reward']:.2f} | "
            f"PnL: ${stats['total_pnl']:.2f} | "
            f"WinRate: {stats['win_rate']:.1f}% | "
            f"Epsilon: {agent_stats['epsilon']:.3f} | "
            f"Loss: {stats['avg_loss']:.4f}"
        )

        if self.verbose >= 1:
            print(msg)

        logger.info(msg)

    def _log_evaluation(self, episode: int, stats: Dict):
        """Log evaluation results."""
        msg = (
            f"[EVAL] Episode {episode + 1} | "
            f"Avg Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f} | "
            f"Avg PnL: ${stats['avg_pnl']:.2f} | "
            f"Win Rate: {stats['avg_win_rate']:.1f}%"
        )

        if self.verbose >= 1:
            print(msg)

        logger.info(msg)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1])
        )

        while len(checkpoints) > self.keep_last_n:
            oldest = checkpoints.pop(0)
            import shutil

            shutil.rmtree(oldest)
            logger.debug(f"Removed old checkpoint: {oldest}")

    def _save_history(self):
        """Save training history to file."""
        history = {
            "training": self.training_history,
            "evaluation": self.eval_history,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(
            f"Training history saved to {self.log_dir / 'training_history.json'}"
        )

    def _get_training_summary(self) -> Dict:
        """Get summary of training."""
        if not self.training_history:
            return {}

        recent = self.training_history[-100:]

        return {
            "total_episodes": len(self.training_history),
            "avg_reward": np.mean([s["reward"] for s in recent]),
            "avg_pnl": np.mean([s["total_pnl"] for s in recent]),
            "avg_win_rate": np.mean([s["win_rate"] for s in recent]),
            "best_eval_reward": self.best_eval_reward,
            "final_epsilon": self.agent.epsilon,
        }
