"""
Visualization Utilities

Functions for plotting training progress and evaluation results.
"""

import json
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Visualization functions disabled.")


def plot_training_history(
    history_file: str,
    output_file: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training history from JSON file.

    Args:
        history_file: Path to training_history.json
        output_file: Path to save figure (optional)
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return

    # Load history
    with open(history_file, "r") as f:
        history = json.load(f)

    training = history.get("training", [])
    evaluation = history.get("evaluation", [])

    if not training:
        logger.warning("No training history to plot")
        return

    # Extract data
    episodes = list(range(1, len(training) + 1))
    rewards = [t["reward"] for t in training]
    pnls = [t["total_pnl"] for t in training]
    win_rates = [t["win_rate"] for t in training]
    epsilons = [t["epsilon"] for t in training]
    losses = [t["avg_loss"] for t in training]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("DQN Training Progress", fontsize=14, fontweight="bold")

    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="blue")
    # Moving average
    window = min(50, len(rewards) // 5) if len(rewards) > 10 else 1
    if window > 1:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window, len(rewards) + 1),
            ma,
            color="blue",
            linewidth=2,
            label=f"MA({window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: PnL
    ax = axes[0, 1]
    ax.plot(episodes, pnls, alpha=0.3, color="green")
    if window > 1:
        ma = np.convolve(pnls, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window, len(pnls) + 1),
            ma,
            color="green",
            linewidth=2,
            label=f"MA({window})",
        )
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("PnL ($)")
    ax.set_title("Episode PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Win Rate
    ax = axes[0, 2]
    ax.plot(episodes, win_rates, alpha=0.3, color="purple")
    if window > 1:
        ma = np.convolve(win_rates, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window, len(win_rates) + 1),
            ma,
            color="purple",
            linewidth=2,
            label=f"MA({window})",
        )
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Epsilon
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color="orange", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate")
    ax.grid(True, alpha=0.3)

    # Plot 5: Loss
    ax = axes[1, 1]
    valid_losses = [(i, l) for i, l in enumerate(losses, 1) if l > 0]
    if valid_losses:
        loss_episodes, loss_values = zip(*valid_losses)
        ax.plot(loss_episodes, loss_values, alpha=0.3, color="red")
        if len(loss_values) > window:
            ma = np.convolve(loss_values, np.ones(window) / window, mode="valid")
            ax.plot(
                range(loss_episodes[0] + window - 1, loss_episodes[-1] + 1),
                ma,
                color="red",
                linewidth=2,
                label=f"MA({window})",
            )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Evaluation Results
    ax = axes[1, 2]
    if evaluation:
        eval_episodes = list(range(1, len(evaluation) + 1))
        eval_rewards = [e["avg_reward"] for e in evaluation]
        eval_stds = [e["std_reward"] for e in evaluation]

        ax.errorbar(
            eval_episodes,
            eval_rewards,
            yerr=eval_stds,
            fmt="o-",
            color="darkblue",
            capsize=3,
            label="Eval Reward",
        )
        ax.set_xlabel("Evaluation #")
        ax.set_ylabel("Avg Reward")
        ax.set_title("Evaluation Results")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No evaluation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Evaluation Results")

    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_evaluation_results(
    results_file: str,
    output_file: Optional[str] = None,
    show: bool = True,
):
    """
    Plot evaluation results from JSON file.

    Args:
        results_file: Path to evaluation results JSON
        output_file: Path to save figure (optional)
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return

    # Load results
    with open(results_file, "r") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    summary = data.get("summary", {})

    if not episodes:
        logger.warning("No evaluation episodes to plot")
        return

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Evaluation Results", fontsize=14, fontweight="bold")

    # Plot 1: PnL Distribution
    ax = axes[0, 0]
    pnls = [e["total_pnl"] for e in episodes]
    ax.hist(pnls, bins=20, color="green", alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", label="Break-even")
    ax.axvline(
        x=summary["avg_pnl"],
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean: ${summary['avg_pnl']:.2f}",
    )
    ax.set_xlabel("PnL ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("PnL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Win Rate Distribution
    ax = axes[0, 1]
    win_rates = [e["win_rate"] for e in episodes]
    ax.hist(win_rates, bins=20, color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(x=50, color="red", linestyle="--", label="50%")
    ax.axvline(
        x=summary["avg_win_rate"],
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {summary['avg_win_rate']:.1f}%",
    )
    ax.set_xlabel("Win Rate (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Win Rate Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: PnL by Episode
    ax = axes[1, 0]
    episode_nums = list(range(1, len(episodes) + 1))
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax.bar(episode_nums, pnls, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("PnL ($)")
    ax.set_title("PnL by Episode")
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary Stats
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = f"""
    EVALUATION SUMMARY
    ─────────────────────────────
    Episodes:            {summary['num_episodes']}
    
    Avg PnL:             ${summary['avg_pnl']:.2f} ± ${summary['std_pnl']:.2f}
    Total PnL:           ${summary['total_pnl']:.2f}
    Best PnL:            ${summary['best_pnl']:.2f}
    Worst PnL:           ${summary['worst_pnl']:.2f}
    
    Avg Win Rate:        {summary['avg_win_rate']:.1f}%
    Avg Trades/Episode:  {summary['avg_trades']:.1f}
    Avg Return:          {summary['avg_return_pct']:.2f}%
    
    Profitable Episodes: {summary['profitable_episodes']}/{summary['num_episodes']}
                         ({100*summary['profitable_episodes']/summary['num_episodes']:.1f}%)
    """

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")

    # Show if requested
    if show:
        plt.show()

    plt.close()
