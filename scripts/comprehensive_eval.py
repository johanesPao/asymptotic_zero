#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Selection

Tests ALL checkpoints to find the TRULY best one based on:
- Risk-adjusted returns (Sharpe)
- Consistency (std dev)
- Profit factor
- Maximum drawdown
- Win rate

NOT just highest single reward!
"""

import sys
sys.path.insert(0, '/home/jpao/projects/asymptotic_zero')

import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from dataclasses import dataclass, asdict
import tensorflow as tf

from src.trading import make_env
from src.agent import DQNAgent


@dataclass
class ModelEvaluation:
    """Complete evaluation metrics for a model."""
    checkpoint_name: str
    
    # Performance metrics
    avg_pnl: float
    median_pnl: float
    std_pnl: float
    total_pnl: float
    
    # Risk metrics  
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Consistency metrics
    positive_episodes: int
    negative_episodes: int
    max_loss_streak: int
    max_win_streak: int
    
    # Composite score
    composite_score: float
    
    # Episode details
    num_episodes: int
    pnl_list: List[float] = None  # For detailed analysis


def evaluate_checkpoint(
    checkpoint_path: str,
    env: 'TradingEnvironment',
    num_episodes: int = 100,
    verbose: bool = True
) -> ModelEvaluation:
    """
    Comprehensively evaluate a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        env: Trading environment
        num_episodes: Number of episodes to test
        verbose: Print progress
        
    Returns:
        ModelEvaluation with all metrics
    """
    if verbose:
        print(f"\nEvaluating {Path(checkpoint_path).name}...")
    
    # Load agent
    agent = DQNAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
        config_path="config/agent.yaml"
    )
    
    # CRITICAL: Build the networks first by passing dummy data
    # This establishes the input shape so weights can be loaded
    dummy_input = np.zeros((1, env.get_state_space_size()), dtype=np.float32)
    _ = agent.q_network(dummy_input, training=False)
    _ = agent.target_network(dummy_input, training=False)
    
    # Now load the checkpoint
    agent.load(checkpoint_path)
    
    # Run episodes
    pnls = []
    wins = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Get episode stats
        stats = env.get_episode_statistics()
        pnls.append(stats['total_pnl'])
        wins.append(1 if stats['total_pnl'] > 0 else 0)
        
        if verbose and (ep + 1) % 20 == 0:
            print(f"  {ep + 1}/{num_episodes} episodes completed...")
    
    # Calculate metrics
    pnls = np.array(pnls)
    
    # Performance
    avg_pnl = np.mean(pnls)
    median_pnl = np.median(pnls)
    std_pnl = np.std(pnls)
    total_pnl = np.sum(pnls)
    
    # Risk
    sharpe_ratio = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    max_drawdown_pct = (max_drawdown / 10000) * 100  # Assuming $10k initial
    
    # Trading
    win_rate = np.mean(wins) * 100
    winning_pnls = pnls[pnls > 0]
    losing_pnls = pnls[pnls < 0]
    profit_factor = (np.sum(winning_pnls) / np.abs(np.sum(losing_pnls)) 
                     if len(losing_pnls) > 0 else np.inf)
    avg_win = np.mean(winning_pnls) if len(winning_pnls) > 0 else 0
    avg_loss = np.mean(losing_pnls) if len(losing_pnls) > 0 else 0
    
    # Consistency
    positive_episodes = int(np.sum(pnls > 0))
    negative_episodes = int(np.sum(pnls < 0))
    
    # Loss/win streaks
    current_loss_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    max_win_streak = 0
    
    for pnl in pnls:
        if pnl < 0:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
            current_win_streak = 0
        else:
            current_win_streak += 1
            max_win_streak = max(max_win_streak, current_win_streak)
            current_loss_streak = 0
    
    # Composite score (higher is better)
    # Normalize each metric to 0-100 scale, then weight
    norm_pnl = min(avg_pnl / 100 * 100, 100)  # Cap at $100
    norm_sharpe = min(sharpe_ratio / 5 * 100, 100)  # Cap at 5.0
    norm_pf = min(profit_factor / 3 * 100, 100)  # Cap at 3.0
    norm_wr = win_rate  # Already 0-100
    norm_dd = max(0, 100 + max_drawdown_pct * 2)  # Penalty for drawdown
    
    composite_score = (
        0.30 * norm_pnl +        # 30% weight on avg PnL
        0.25 * norm_sharpe +     # 25% weight on Sharpe
        0.20 * norm_pf +         # 20% weight on profit factor
        0.15 * norm_wr +         # 15% weight on win rate
        0.10 * norm_dd           # 10% weight on drawdown (less bad = better)
    )
    
    return ModelEvaluation(
        checkpoint_name=Path(checkpoint_path).name,
        avg_pnl=avg_pnl,
        median_pnl=median_pnl,
        std_pnl=std_pnl,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        positive_episodes=positive_episodes,
        negative_episodes=negative_episodes,
        max_loss_streak=max_loss_streak,
        max_win_streak=max_win_streak,
        composite_score=composite_score,
        num_episodes=num_episodes,
        pnl_list=pnls.tolist()
    )


def compare_models(evaluations: List[ModelEvaluation]) -> None:
    """Print comparison table of all models."""
    print("\n" + "="*120)
    print("MODEL COMPARISON - COMPREHENSIVE EVALUATION")
    print("="*120)
    
    # Sort by composite score
    evaluations.sort(key=lambda x: x.composite_score, reverse=True)
    
    # Print header
    print(f"\n{'Rank':<6} {'Model':<20} {'Avg PnL':<12} {'Sharpe':<10} {'PF':<8} "
          f"{'Win%':<8} {'MaxDD%':<10} {'Score':<10}")
    print("-" * 120)
    
    # Print each model
    for i, ev in enumerate(evaluations, 1):
        rank_symbol = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        
        print(f"{rank_symbol} {i:<4} {ev.checkpoint_name:<20} "
              f"${ev.avg_pnl:>7.2f}     "
              f"{ev.sharpe_ratio:>6.2f}    "
              f"{ev.profit_factor:>5.2f}   "
              f"{ev.win_rate:>5.1f}%   "
              f"{ev.max_drawdown_pct:>7.1f}%    "
              f"{ev.composite_score:>7.1f}")
    
    print("="*120)
    
    # Detailed best model
    best = evaluations[0]
    print(f"\n🏆 BEST MODEL: {best.checkpoint_name}")
    print("-" * 120)
    print(f"\n  Performance:")
    print(f"    Avg PnL:          ${best.avg_pnl:.2f} per episode")
    print(f"    Median PnL:       ${best.median_pnl:.2f}")
    print(f"    Total PnL:        ${best.total_pnl:.2f} over {best.num_episodes} episodes")
    print(f"    Std Deviation:    ${best.std_pnl:.2f}")
    
    print(f"\n  Risk Metrics:")
    print(f"    Sharpe Ratio:     {best.sharpe_ratio:.2f} (higher is better, >1.0 is good)")
    print(f"    Max Drawdown:     ${best.max_drawdown:.2f} ({best.max_drawdown_pct:.1f}%)")
    
    print(f"\n  Trading Metrics:")
    print(f"    Win Rate:         {best.win_rate:.1f}%")
    print(f"    Profit Factor:    {best.profit_factor:.2f} (win $1 for every ${1/best.profit_factor:.2f} lost)")
    print(f"    Avg Win:          ${best.avg_win:.2f}")
    print(f"    Avg Loss:         ${best.avg_loss:.2f}")
    
    print(f"\n  Consistency:")
    print(f"    Positive Episodes: {best.positive_episodes}/{best.num_episodes} ({best.win_rate:.1f}%)")
    print(f"    Max Win Streak:    {best.max_win_streak} episodes")
    print(f"    Max Loss Streak:   {best.max_loss_streak} episodes")
    
    print(f"\n  Composite Score: {best.composite_score:.1f}/100")
    print("="*120)


def main():
    """Main evaluation script."""
    print("="*120)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*120)
    print("\nThis will evaluate ALL checkpoints using 100 episodes each.")
    print("Metrics include: PnL, Sharpe, Profit Factor, Win Rate, Drawdown, and more!")
    print("\nNOTE: This will take time - each checkpoint needs 100 full episodes.")
    print("="*120)
    
    # Find all checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoints = []
    
    # Check for specific checkpoints
    if (checkpoint_dir / "best").exists():
        checkpoints.append(str(checkpoint_dir / "best"))
    if (checkpoint_dir / "final").exists():
        checkpoints.append(str(checkpoint_dir / "final"))
    
    # Add numbered checkpoints
    for cp in sorted(checkpoint_dir.glob("checkpoint_*")):
        if cp.is_dir():
            checkpoints.append(str(cp))
    
    if not checkpoints:
        print("❌ No checkpoints found in checkpoints/")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints to evaluate:")
    for cp in checkpoints:
        print(f"  - {Path(cp).name}")
    
    # Create environment
    print("\nInitializing environment...")
    env = make_env()
    
    # Evaluate each checkpoint
    evaluations = []
    for cp in checkpoints:
        try:
            evaluation = evaluate_checkpoint(
                cp, 
                env, 
                num_episodes=100,
                verbose=True
            )
            evaluations.append(evaluation)
        except Exception as e:
            print(f"❌ Failed to evaluate {Path(cp).name}: {e}")
    
    # Compare all models
    if evaluations:
        compare_models(evaluations)
        
        # Save results
        results_file = "model_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(ev) for ev in evaluations], f, indent=2)
        print(f"\n✅ Detailed results saved to: {results_file}")
        
        # Recommendation
        best = evaluations[0]
        print(f"\n{'='*120}")
        print("RECOMMENDATION")
        print("="*120)
        print(f"\n✅ Use checkpoint: checkpoints/{best.checkpoint_name}")
        print(f"\n   This model has the highest composite score ({best.composite_score:.1f}/100)")
        print(f"   balancing profitability, risk, and consistency.")
        print("\n" + "="*120)
    else:
        print("❌ No models were successfully evaluated.")


if __name__ == "__main__":
    main()
