"""
Reward-PnL Correlation Analyzer

Run this after training to verify your reward system is aligned with PnL.

Usage:
    python scripts/analyze_correlation.py --log logs/training_latest.log
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_training_log(log_file: Path):
    """Extract episode rewards and PnL from training log."""
    rewards = []
    pnls = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Look for episode summary lines
            # Example: "Episode 100 | Reward: 45.23 | PnL: $132.50"
            
            # Extract reward
            reward_match = re.search(r'Reward:\s*([-\d.]+)', line)
            pnl_match = re.search(r'PnL:\s*\$?([-\d.]+)', line)
            
            if reward_match and pnl_match:
                rewards.append(float(reward_match.group(1)))
                pnls.append(float(pnl_match.group(1)))
    
    return np.array(rewards), np.array(pnls)


def analyze_correlation(rewards, pnls):
    """Analyze and visualize reward-PnL correlation."""
    
    if len(rewards) == 0:
        print("❌ No data found in log file!")
        return
    
    print(f"📊 Analyzing {len(rewards)} episodes...")
    print()
    
    # Calculate correlation
    correlation = np.corrcoef(rewards, pnls)[0, 1]
    
    print("="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    print(f"Reward-PnL Correlation: {correlation:.4f}")
    print()
    
    # Interpret correlation
    if correlation > 0.95:
        print("✅ EXCELLENT: Rewards strongly aligned with PnL!")
        print("   Your agent is learning to maximize profit.")
    elif correlation > 0.80:
        print("⚠️  GOOD: Reasonable alignment, but could be better.")
        print("   Consider simplifying your reward function.")
    elif correlation > 0.50:
        print("⚠️  FAIR: Weak alignment detected.")
        print("   Agent may not be learning optimal profit behavior.")
    else:
        print("❌ POOR: Rewards not aligned with PnL!")
        print("   Agent is learning the WRONG objective.")
        print("   Redesign your reward function immediately.")
    
    print()
    
    # Sign agreement analysis
    same_sign = np.sum((rewards > 0) == (pnls > 0))
    sign_agreement = same_sign / len(rewards) * 100
    
    print(f"Sign Agreement: {sign_agreement:.1f}%")
    print(f"  (Reward and PnL have same sign)")
    print()
    
    if sign_agreement < 80:
        print("❌ WARNING: Reward and PnL often have opposite signs!")
        print("   Agent being punished for profits or rewarded for losses.")
    
    # Top episodes analysis
    top_10_reward_idx = np.argsort(rewards)[-10:]
    top_10_pnl_idx = np.argsort(pnls)[-10:]
    
    overlap = len(set(top_10_reward_idx) & set(top_10_pnl_idx))
    
    print(f"Top 10 Episodes Overlap: {overlap}/10")
    print(f"  (Episodes with highest rewards vs highest PnL)")
    print()
    
    if overlap < 7:
        print("⚠️  WARNING: Top reward episodes ≠ top PnL episodes")
        print("   Agent optimizing for something other than profit.")
    
    # Statistics
    print("="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Reward  - Mean: {np.mean(rewards):8.2f}  Std: {np.std(rewards):8.2f}")
    print(f"PnL     - Mean: ${np.mean(pnls):7.2f}  Std: ${np.std(pnls):7.2f}")
    print()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot
    axes[0, 0].scatter(pnls, rewards, alpha=0.5)
    axes[0, 0].plot([pnls.min(), pnls.max()], 
                     [pnls.min(), pnls.max()], 
                     'r--', label='Perfect Alignment')
    axes[0, 0].set_xlabel('PnL ($)')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title(f'Reward vs PnL (r={correlation:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series
    episodes = np.arange(len(rewards))
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    
    ax1.plot(episodes, rewards, 'b-', alpha=0.7, label='Reward')
    ax2.plot(episodes, pnls, 'g-', alpha=0.7, label='PnL')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color='b')
    ax2.set_ylabel('PnL ($)', color='g')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    axes[0, 1].set_title('Reward and PnL Over Time')
    
    # 3. Distribution comparison
    axes[1, 0].hist(rewards, bins=30, alpha=0.5, label='Reward', density=True)
    axes[1, 0].hist(pnls, bins=30, alpha=0.5, label='PnL', density=True)
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals
    # Fit linear relationship
    slope, intercept = np.polyfit(pnls, rewards, 1)
    predicted_rewards = slope * pnls + intercept
    residuals = rewards - predicted_rewards
    
    axes[1, 1].scatter(pnls, residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('PnL ($)')
    axes[1, 1].set_ylabel('Residuals (Reward - Predicted)')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path('correlation_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📈 Visualization saved to: {output_file}")
    print()
    
    # Final recommendation
    print("="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if correlation > 0.95 and sign_agreement > 90:
        print("✅ Your reward system is working perfectly!")
        print("   Continue training with confidence.")
    elif correlation > 0.80:
        print("⚠️  Your reward system is decent but improvable.")
        print("   Consider simplifying to: reward = pnl * scale")
    else:
        print("❌ Your reward system needs fixing!")
        print("   Implement the simplified reward calculator:")
        print("   reward = (pnl / initial_balance) * 100")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze reward-PnL correlation from training logs'
    )
    parser.add_argument(
        '--log',
        type=str,
        required=True,
        help='Path to training log file'
    )
    
    args = parser.parse_args()
    
    log_file = Path(args.log)
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return 1
    
    rewards, pnls = parse_training_log(log_file)
    analyze_correlation(rewards, pnls)
    
    return 0


if __name__ == '__main__':
    exit(main())
