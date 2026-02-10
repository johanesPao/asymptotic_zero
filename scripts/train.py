"""
Training Script for Asymptotic Zero

Trains the DQN agent on cryptocurrency trading data.

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 5000
    python scripts/train.py --resume checkpoints/checkpoint_1000
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading import make_env
from src.agent import DQNAgent, DQNTrainer


def setup_logging(log_dir: Path, verbose: int):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    level = logging.DEBUG if verbose >= 2 else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for cryptocurrency trading"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (overrides config)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/agent.yaml",
        help="Path to agent configuration",
    )

    parser.add_argument(
        "--trading-config",
        type=str,
        default="config/trading.yaml",
        help="Path to trading configuration",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=silent, 1=progress, 2=debug)",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires --resume)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    log_dir = Path("logs")
    logger = setup_logging(log_dir, args.verbose)

    logger.info("=" * 60)
    logger.info("ASYMPTOTIC ZERO - DQN TRAINING")
    logger.info("=" * 60)

    # Create environment
    logger.info("Creating trading environment...")
    env = make_env(
        config_path=args.trading_config,
        data_directory="data/volatility",
    )
    logger.info(f"  Action space: {env.get_action_space_size()}")
    logger.info(f"  State space: {env.get_state_space_size()}")
    logger.info(f"  Available dates: {len(env.available_dates)}")

    # Create agent
    logger.info("Creating DQN agent...")
    agent = DQNAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
        config_path=args.config,
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        config_path=args.config,
    )

    # Override episodes if specified
    if args.episodes is not None:
        trainer.total_episodes = args.episodes
        logger.info(f"Overriding total episodes to {args.episodes}")

    # Evaluation only mode
    if args.eval_only:
        if args.resume is None:
            logger.error("--eval-only requires --resume checkpoint")
            return 1

        logger.info(f"Loading checkpoint from {args.resume}")
        agent.load(args.resume)

        logger.info("Running evaluation...")
        eval_stats = trainer.evaluate(num_episodes=20)

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(
            f"  Avg Reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}"
        )
        logger.info(f"  Avg PnL: ${eval_stats['avg_pnl']:.2f}")
        logger.info(f"  Avg Win Rate: {eval_stats['avg_win_rate']:.1f}%")
        logger.info(f"  Max Reward: {eval_stats['max_reward']:.2f}")
        logger.info(f"  Min Reward: {eval_stats['min_reward']:.2f}")

        return 0

    # Train
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    try:
        summary = trainer.train(resume_from=args.resume)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Total Episodes: {summary.get('total_episodes', 0)}")
        logger.info(f"  Avg Reward (last 100): {summary.get('avg_reward', 0):.2f}")
        logger.info(f"  Avg PnL (last 100): ${summary.get('avg_pnl', 0):.2f}")
        logger.info(f"  Avg Win Rate (last 100): {summary.get('avg_win_rate', 0):.1f}%")
        logger.info(f"  Best Eval Reward: {summary.get('best_eval_reward', 0):.2f}")
        logger.info(f"  Final Epsilon: {summary.get('final_epsilon', 0):.4f}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        agent.save(str(Path("checkpoints") / "interrupted"))
        logger.info("Checkpoint saved to checkpoints/interrupted")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
