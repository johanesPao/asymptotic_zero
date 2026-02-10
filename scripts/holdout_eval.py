#!/usr/bin/env python3
"""
Holdout Evaluation + Strategy Analysis

CRITICAL: Training used ALL 2192 dates randomly. There was no train/test split.
This script reserves the LAST 200 dates as holdout — dates the agent likely
saw very few times (or rarely) during training — and tests performance on them.

Also analyzes WHAT the agent actually learned:
- Does it prefer LONG or SHORT?
- Gainers or losers?
- How long does it hold?
- Which coins does it trade most?
"""

import sys
sys.path.insert(0, '/home/jpao/projects/asymptotic_zero')

import numpy as np
from pathlib import Path
from collections import defaultdict
import json

from src.trading import make_env
from src.agent import DQNAgent


def build_agent(env):
    """Build agent with network initialized before loading weights."""
    agent = DQNAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
        config_path="config/agent.yaml"
    )
    # Build networks before loading weights
    dummy_input = np.zeros((1, env.get_state_space_size()), dtype=np.float32)
    _ = agent.q_network(dummy_input, training=False)
    _ = agent.target_network(dummy_input, training=False)
    return agent


def run_holdout_eval(env, agent, holdout_dates, verbose=True):
    """
    Run evaluation ONLY on holdout dates.
    
    Returns list of episode result dicts.
    """
    results = []
    
    for i, date in enumerate(holdout_dates):
        state = env.reset(date=date)
        done = False
        episode_trades = []  # Track each trade
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            # Record non-hold actions
            if info['action_type'] != 'hold':
                episode_trades.append({
                    'step': info['step'],
                    'action': info['action'],
                    'action_type': info['action_type'],
                    'success': info['action_success'],
                    'pnl': info['pnl'],
                    'message': info['action_message'],
                })
            
            state = next_state
        
        stats = env.get_episode_statistics()
        results.append({
            'date': date,
            'total_pnl': stats['total_pnl'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'win_rate': stats['win_rate'],
            'trades': episode_trades,
        })
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(holdout_dates)} dates completed...")
    
    return results


def analyze_strategy(results):
    """Analyze what strategy the agent learned from trade data."""
    
    # Aggregate all trades
    all_trades = []
    for ep in results:
        for trade in ep['trades']:
            trade['date'] = ep['date']
            all_trades.append(trade)
    
    # --- Action type breakdown ---
    action_counts = defaultdict(int)
    action_pnls = defaultdict(list)
    
    for trade in all_trades:
        atype = trade['action_type']
        action_counts[atype] += 1
        if trade['pnl'] != 0:
            action_pnls[atype].append(trade['pnl'])
    
    # --- Long vs Short analysis ---
    # Action mapping from trading.yaml:
    # 1-10: LONG gainer, 11-20: SHORT gainer
    # 31-40: LONG loser,  41-50: SHORT loser
    long_trades = [t for t in all_trades if t['action_type'] == 'open' and 
                   (1 <= t['action'] <= 10 or 31 <= t['action'] <= 40)]
    short_trades = [t for t in all_trades if t['action_type'] == 'open' and 
                    (11 <= t['action'] <= 20 or 41 <= t['action'] <= 50)]
    
    # --- Gainer vs Loser analysis ---
    gainer_trades = [t for t in all_trades if t['action_type'] == 'open' and 
                     (1 <= t['action'] <= 30)]
    loser_trades = [t for t in all_trades if t['action_type'] == 'open' and 
                    (31 <= t['action'] <= 60)]
    
    # --- Per-coin analysis ---
    # Which coin index is traded most?
    coin_trade_counts = defaultdict(int)
    for trade in all_trades:
        if trade['action_type'] == 'open':
            action = trade['action']
            n = 10  # num_coins_per_side
            if 1 <= action <= n:
                coin_idx = action - 1  # gainer 0-9
            elif n+1 <= action <= 2*n:
                coin_idx = action - n - 1  # gainer 0-9 (short)
            elif 3*n+1 <= action <= 4*n:
                coin_idx = n + (action - 3*n - 1)  # loser 10-19
            elif 4*n+1 <= action <= 5*n:
                coin_idx = n + (action - 4*n - 1)  # loser 10-19 (short)
            else:
                continue
            coin_trade_counts[coin_idx] += 1
    
    # --- Hold analysis (how often does agent just hold?) ---
    total_steps = sum(len(ep['trades']) for ep in results)
    # Each episode has 88 tradeable steps, total hold = 88*episodes - non-hold actions
    total_possible_steps = 88 * len(results)
    non_hold_actions = len([t for t in all_trades if t['action_type'] != 'hold'])
    hold_count = total_possible_steps - non_hold_actions
    
    # --- Close action analysis ---
    close_actions = [t for t in all_trades if t['action_type'] in 
                     ['close', 'close_all', 'close_worst', 'force_close']]
    
    return {
        'action_counts': dict(action_counts),
        'action_pnls': {k: v for k, v in action_pnls.items()},
        'long_count': len(long_trades),
        'short_count': len(short_trades),
        'gainer_count': len(gainer_trades),
        'loser_count': len(loser_trades),
        'coin_trade_counts': dict(coin_trade_counts),
        'hold_rate': hold_count / total_possible_steps * 100 if total_possible_steps > 0 else 0,
        'close_counts': {
            'close': len([t for t in close_actions if t['action_type'] == 'close']),
            'close_all': len([t for t in close_actions if t['action_type'] == 'close_all']),
            'close_worst': len([t for t in close_actions if t['action_type'] == 'close_worst']),
        },
        'total_open_trades': len(long_trades) + len(short_trades),
    }


def print_holdout_results(results, holdout_dates):
    """Print holdout evaluation results."""
    pnls = np.array([r['total_pnl'] for r in results])
    win_rates = np.array([r['win_rate'] for r in results])
    
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    median_pnl = np.median(pnls)
    
    # Sharpe
    sharpe = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
    
    # Profit factor
    winning = pnls[pnls > 0]
    losing = pnls[pnls < 0]
    profit_factor = np.sum(winning) / np.abs(np.sum(losing)) if len(losing) > 0 and np.sum(losing) != 0 else float('inf')
    
    # Drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    
    # Win/loss streaks
    max_loss_streak = 0
    current_streak = 0
    for p in pnls:
        if p < 0:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0
    
    profitable_episodes = int(np.sum(pnls > 0))
    
    print("\n" + "="*70)
    print("  HOLDOUT EVALUATION RESULTS")
    print("  (Dates the agent had minimal exposure to during training)")
    print("="*70)
    
    print(f"\n  Dates tested:          {len(holdout_dates)}")
    print(f"  Date range:            {holdout_dates[0]} → {holdout_dates[-1]}")
    
    print(f"\n  --- Performance ---")
    print(f"  Avg PnL:               ${avg_pnl:.2f} ± ${std_pnl:.2f}")
    print(f"  Median PnL:            ${median_pnl:.2f}")
    print(f"  Total PnL:             ${np.sum(pnls):.2f}")
    print(f"  Best Day:              ${np.max(pnls):.2f}")
    print(f"  Worst Day:             ${np.min(pnls):.2f}")
    
    print(f"\n  --- Risk ---")
    print(f"  Sharpe Ratio:          {sharpe:.2f}")
    print(f"  Max Drawdown:          ${max_drawdown:.2f}")
    print(f"  Max Loss Streak:       {max_loss_streak} days")
    
    print(f"\n  --- Consistency ---")
    print(f"  Profitable Days:       {profitable_episodes}/{len(results)} ({profitable_episodes/len(results)*100:.1f}%)")
    print(f"  Profit Factor:         {profit_factor:.2f}")
    print(f"  Avg Win Rate/Day:      {np.mean(win_rates):.1f}%")
    
    # Verdict
    print(f"\n  --- VERDICT ---")
    if avg_pnl > 30 and profit_factor > 1.5 and sharpe > 1.0:
        print(f"  ✅ PASS - Agent generalizes well to unseen dates!")
        print(f"     Safe to proceed to paper trading.")
    elif avg_pnl > 0 and profit_factor > 1.0:
        print(f"  ⚠️  MARGINAL - Agent is profitable but weaker on unseen data.")
        print(f"     Consider retraining with train/test split before live trading.")
    else:
        print(f"  ❌ FAIL - Agent does NOT generalize to unseen dates.")
        print(f"     DO NOT deploy. Need retraining with proper holdout split.")
    
    print("="*70)
    
    return {
        'avg_pnl': avg_pnl,
        'std_pnl': std_pnl,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'profitable_pct': profitable_episodes / len(results) * 100,
    }


def print_strategy_analysis(strategy):
    """Print strategy analysis."""
    print("\n" + "="*70)
    print("  STRATEGY ANALYSIS - What Did the Agent Learn?")
    print("="*70)
    
    total_opens = strategy['total_open_trades']
    
    # Long vs Short
    long_pct = strategy['long_count'] / total_opens * 100 if total_opens > 0 else 0
    short_pct = strategy['short_count'] / total_opens * 100 if total_opens > 0 else 0
    
    print(f"\n  --- Direction Preference ---")
    print(f"  LONG trades:           {strategy['long_count']} ({long_pct:.1f}%)")
    print(f"  SHORT trades:          {strategy['short_count']} ({short_pct:.1f}%)")
    if long_pct > short_pct + 10:
        print(f"  → Agent prefers LONG (bullish bias)")
    elif short_pct > long_pct + 10:
        print(f"  → Agent prefers SHORT (bearish bias)")
    else:
        print(f"  → Agent is balanced (no strong directional bias)")
    
    # Gainer vs Loser
    gainer_pct = strategy['gainer_count'] / total_opens * 100 if total_opens > 0 else 0
    loser_pct = strategy['loser_count'] / total_opens * 100 if total_opens > 0 else 0
    
    print(f"\n  --- Target Preference ---")
    print(f"  Trades on GAINERS:     {strategy['gainer_count']} ({gainer_pct:.1f}%)")
    print(f"  Trades on LOSERS:      {strategy['loser_count']} ({loser_pct:.1f}%)")
    if gainer_pct > loser_pct + 10:
        print(f"  → Agent targets GAINERS (momentum following)")
    elif loser_pct > gainer_pct + 10:
        print(f"  → Agent targets LOSERS (mean reversion)")
    else:
        print(f"  → Agent trades both equally")
    
    # Hold rate
    print(f"\n  --- Activity ---")
    print(f"  Hold rate:             {strategy['hold_rate']:.1f}% of steps")
    print(f"  Total opens:           {total_opens}")
    print(f"  Close methods:")
    print(f"    Individual close:    {strategy['close_counts']['close']}")
    print(f"    Close all:           {strategy['close_counts']['close_all']}")
    print(f"    Close worst:         {strategy['close_counts']['close_worst']}")
    
    if strategy['close_counts']['close_all'] > strategy['close_counts']['close'] * 0.3:
        print(f"  → Agent uses 'close all' frequently (risk-averse exits)")
    
    # Per-coin popularity
    print(f"\n  --- Coin Popularity (by index) ---")
    print(f"  Index 0-9  = Gainers ranked #1-#10 by daily % change")
    print(f"  Index 10-19 = Losers ranked #1-#10 by daily % change")
    
    sorted_coins = sorted(strategy['coin_trade_counts'].items(), key=lambda x: x[1], reverse=True)
    for coin_idx, count in sorted_coins[:10]:
        side = "Gainer" if coin_idx < 10 else "Loser"
        rank = coin_idx + 1 if coin_idx < 10 else coin_idx - 9
        print(f"    #{rank} {side} (idx {coin_idx}): {count} opens")
    
    print("="*70)


def main():
    print("="*70)
    print("  HOLDOUT EVALUATION + STRATEGY ANALYSIS")
    print("="*70)
    print("\n  This tests your agent on dates it had MINIMAL exposure")
    print("  to during training. This is the TRUE test of generalization.")
    
    # Initialize environment
    print("\n  Initializing environment...")
    env = make_env()
    
    all_dates = sorted(env.available_dates)
    print(f"  Total available dates: {len(all_dates)}")
    print(f"  Date range: {all_dates[0]} → {all_dates[-1]}")
    
    # Reserve last 200 dates as holdout
    # These are the most RECENT dates — least likely to have been heavily
    # sampled during 10k episodes of random selection
    HOLDOUT_SIZE = 200
    holdout_dates = all_dates[-HOLDOUT_SIZE:]
    
    print(f"\n  Holdout dates: {len(holdout_dates)}")
    print(f"  Holdout range: {holdout_dates[0]} → {holdout_dates[-1]}")
    print(f"  (Last {HOLDOUT_SIZE} dates in the dataset)")
    
    # Load best model
    checkpoint = "checkpoints/best"
    print(f"\n  Loading checkpoint: {checkpoint}")
    agent = build_agent(env)
    agent.load(checkpoint)
    # Force greedy (no exploration)
    agent.epsilon = 0.0
    
    # --- HOLDOUT EVALUATION ---
    print(f"\n  Running holdout evaluation ({HOLDOUT_SIZE} episodes)...")
    holdout_results = run_holdout_eval(env, agent, holdout_dates, verbose=True)
    
    # Print holdout results
    holdout_metrics = print_holdout_results(holdout_results, holdout_dates)
    
    # --- STRATEGY ANALYSIS ---
    print(f"\n  Analyzing learned strategy...")
    strategy = analyze_strategy(holdout_results)
    print_strategy_analysis(strategy)
    
    # --- Save results ---
    output = {
        'holdout_metrics': {k: float(v) for k, v in holdout_metrics.items()},
        'strategy_summary': {
            'long_count': strategy['long_count'],
            'short_count': strategy['short_count'],
            'gainer_count': strategy['gainer_count'],
            'loser_count': strategy['loser_count'],
            'hold_rate': strategy['hold_rate'],
            'total_opens': strategy['total_open_trades'],
        },
        'per_episode': [
            {'date': r['date'], 'pnl': r['total_pnl'], 'trades': r['total_trades'], 'win_rate': r['win_rate']}
            for r in holdout_results
        ]
    }
    
    with open('holdout_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  ✅ Full results saved to: holdout_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
