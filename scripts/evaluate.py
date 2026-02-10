"""
Evaluation Script for Asymptotic Zero

Evaluates a trained DQN agent on trading data.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best
    python scripts/evaluate.py --checkpoint checkpoints/best --episodes 50
    python scripts/evaluate.py --checkpoint checkpoints/best --render
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading import make_env
from src.agent import DQNAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DQN agent for cryptocurrency trading"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )

    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=None,
        help="Specific dates to evaluate on (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level",
    )

    return parser.parse_args()


def evaluate_episode(env, agent, date=None, render=False):
    """Run a single evaluation episode."""
    state = env.reset(date=date, random_date=(date is None))

    total_reward = 0
    steps = 0
    actions_taken = []

    while True:
        # Select action (greedy, no exploration)
        action = agent.select_action(state, training=False)
        actions_taken.append(action)

        # Execute action
        next_state, reward, done, info = env.step(action)

        # Render if requested
        if render:
            env.render()

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    # Get episode statistics
    stats = env.get_episode_statistics()

    return {
        "date": env.current_date,
        "total_reward": total_reward,
        "steps": steps,
        "total_trades": stats["total_trades"],
        "winning_trades": stats["winning_trades"],
        "losing_trades": stats["losing_trades"],
        "win_rate": stats["win_rate"],
        "total_pnl": stats["total_pnl"],
        "return_pct": stats["return_pct"],
        "final_portfolio_value": stats["final_portfolio_value"],
        "total_fees": stats["total_fees"],
        "action_distribution": {
            "hold": actions_taken.count(0),
            "open": sum(1 for a in actions_taken if 1 <= a <= 60),
            "close_all": actions_taken.count(61),
            "close_worst": actions_taken.count(62),
        },
    }


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ASYMPTOTIC ZERO - EVALUATION")
    logger.info("=" * 60)

    # Create environment
    logger.info("Creating environment...")
    env = make_env()

    # Create agent
    logger.info("Creating agent...")
    agent = DQNAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    agent.load(args.checkpoint)
    logger.info(f"  Epsilon: {agent.epsilon}")
    logger.info(f"  Train steps: {agent.train_step_counter}")

    # Run evaluation
    logger.info(f"\nRunning {args.episodes} evaluation episodes...")

    results = []
    dates_to_eval = args.dates if args.dates else [None] * args.episodes

    for i, date in enumerate(dates_to_eval):
        episode_result = evaluate_episode(env, agent, date=date, render=args.render)
        results.append(episode_result)

        if args.verbose >= 1:
            logger.info(
                f"  Episode {i+1}/{len(dates_to_eval)}: "
                f"Date={episode_result['date']}, "
                f"PnL=${episode_result['total_pnl']:.2f}, "
                f"WinRate={episode_result['win_rate']:.1f}%, "
                f"Trades={episode_result['total_trades']}"
            )

    # Calculate aggregate statistics
    import numpy as np

    summary = {
        "num_episodes": len(results),
        "checkpoint": args.checkpoint,
        "avg_reward": np.mean([r["total_reward"] for r in results]),
        "std_reward": np.std([r["total_reward"] for r in results]),
        "avg_pnl": np.mean([r["total_pnl"] for r in results]),
        "std_pnl": np.std([r["total_pnl"] for r in results]),
        "total_pnl": sum([r["total_pnl"] for r in results]),
        "avg_win_rate": np.mean([r["win_rate"] for r in results]),
        "avg_trades": np.mean([r["total_trades"] for r in results]),
        "avg_return_pct": np.mean([r["return_pct"] for r in results]),
        "profitable_episodes": sum(1 for r in results if r["total_pnl"] > 0),
        "best_pnl": max([r["total_pnl"] for r in results]),
        "worst_pnl": min([r["total_pnl"] for r in results]),
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Episodes: {summary['num_episodes']}")
    logger.info(
        f"  Avg Reward: {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f}"
    )
    logger.info(f"  Avg PnL: ${summary['avg_pnl']:.2f} ± {summary['std_pnl']:.2f}")
    logger.info(f"  Total PnL: ${summary['total_pnl']:.2f}")
    logger.info(f"  Avg Win Rate: {summary['avg_win_rate']:.1f}%")
    logger.info(f"  Avg Trades/Episode: {summary['avg_trades']:.1f}")
    logger.info(f"  Avg Return: {summary['avg_return_pct']:.2f}%")
    logger.info(
        f"  Profitable Episodes: {summary['profitable_episodes']}/{summary['num_episodes']}"
    )
    logger.info(f"  Best PnL: ${summary['best_pnl']:.2f}")
    logger.info(f"  Worst PnL: ${summary['worst_pnl']:.2f}")

    # Save results if requested
    if args.output:
        output_data = {
            "summary": summary,
            "episodes": results,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
