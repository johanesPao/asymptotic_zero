#!/usr/bin/env python3
"""
Paper Trading Simulation with Guardrails

The trained agent has strong performance but learned some bad habits:
1. Never holds (0% hold rate) — trades every single step
2. Spams close actions on coins it doesn't hold
3. Concentrates 93% of trades on just 2 coin ranks

These guardrails wrap the agent's decisions and enforce better behavior
WITHOUT retraining. They simulate realistic constraints the agent would
face in a live environment anyway.

Guardrails:
- Cooldown: After opening, must wait N steps before next open
- Action validation: Block close actions on coins without positions
- Concentration limit: Cap trades per coin rank per episode
- Min hold period: Must hold a position for at least N steps
"""

import sys
sys.path.insert(0, '/home/jpao/projects/asymptotic_zero')

import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

from src.trading import make_env
from src.agent import DQNAgent


# ─────────────────────────────────────────────────────────────────────
# GUARDRAIL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

GUARDRAILS = {
    # After any open, wait this many steps before next open
    # Prevents constant churn. 88 steps/day, max 3 positions
    # So 5 steps cooldown = at most ~17 opens/day (realistic)
    "open_cooldown_steps": 5,
    
    # Must hold a position for at least this many steps before closing
    # Prevents open-close-open-close spam
    # 3 steps = 15 minutes minimum hold
    "min_hold_steps": 3,
    
    # Maximum opens on the same coin rank per episode
    # Prevents the agent from hammering rank #2 gainer 30 times/day
    "max_opens_per_rank": 3,
}


class GuardrailWrapper:
    """
    Wraps agent actions with trading guardrails.
    
    The agent still decides what to do, but this wrapper:
    1. Blocks invalid actions (close on empty position)
    2. Enforces cooldowns between opens
    3. Enforces minimum hold periods
    4. Caps concentration per coin rank
    
    When an action is blocked, falls back to HOLD (action 0).
    """
    
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or GUARDRAILS
        
        # State tracking
        self.steps_since_last_open = 999  # No cooldown at start
        self.position_open_steps = {}     # coin_idx -> step when opened
        self.opens_per_rank = defaultdict(int)  # coin_idx -> open count this episode
        self.current_positions = set()    # Set of coin indices with open positions
        
        # Stats
        self.blocked_actions = defaultdict(int)
        self.total_actions = 0
    
    def reset(self):
        """Reset guardrail state for new episode."""
        self.steps_since_last_open = 999
        self.position_open_steps = {}
        self.opens_per_rank = defaultdict(int)
        self.current_positions = set()
        self.blocked_actions = defaultdict(int)
        self.total_actions = 0
    
    def _parse_action(self, action):
        """Parse action into (type, coin_idx, side). Mirrors environment logic."""
        n = 10  # num_coins_per_side
        
        if action == 0:
            return "hold", -1
        if 1 <= action <= n:
            return "open", action - 1          # long gainer
        if n+1 <= action <= 2*n:
            return "open", action - n - 1      # short gainer  
        if 2*n+1 <= action <= 3*n:
            return "close", action - 2*n - 1   # close gainer
        if 3*n+1 <= action <= 4*n:
            return "open", n + (action - 3*n - 1)   # long loser
        if 4*n+1 <= action <= 5*n:
            return "open", n + (action - 4*n - 1)   # short loser
        if 5*n+1 <= action <= 6*n:
            return "close", n + (action - 5*n - 1)  # close loser
        if action == 61:
            return "close_all", -1
        if action == 62:
            return "close_worst", -1
        
        return "invalid", -1
    
    def validate(self, action, current_step):
        """
        Check if action passes all guardrails.
        
        Returns: (allowed, reason)
        """
        action_type, coin_idx = self._parse_action(action)
        
        # --- HOLD always allowed ---
        if action_type == "hold":
            return True, "hold"
        
        # --- Block close on coins without positions ---
        if action_type == "close":
            if coin_idx not in self.current_positions:
                return False, f"close_empty (coin {coin_idx} not held)"
        
        # --- Block opens during cooldown ---
        if action_type == "open":
            if self.steps_since_last_open < self.config["open_cooldown_steps"]:
                return False, f"cooldown ({self.config['open_cooldown_steps'] - self.steps_since_last_open} steps left)"
            
            # Block if already holding this coin
            if coin_idx in self.current_positions:
                return False, f"already_holding (coin {coin_idx})"
            
            # Block if at max positions (3)
            if len(self.current_positions) >= 3:
                return False, "max_positions (3)"
            
            # Block if this rank has been opened too many times
            if self.opens_per_rank[coin_idx] >= self.config["max_opens_per_rank"]:
                return False, f"concentration_limit (coin {coin_idx}, max {self.config['max_opens_per_rank']})"
        
        # --- Block early closes (min hold period) ---
        if action_type == "close" and coin_idx in self.position_open_steps:
            steps_held = current_step - self.position_open_steps[coin_idx]
            if steps_held < self.config["min_hold_steps"]:
                return False, f"min_hold ({self.config['min_hold_steps'] - steps_held} steps left)"
        
        # --- close_all / close_worst: always allowed if positions exist ---
        if action_type in ("close_all", "close_worst"):
            if not self.current_positions:
                return False, "close_portfolio_empty"
        
        return True, action_type
    
    def apply(self, action, current_step):
        """
        Apply guardrails. Returns the (possibly modified) action.
        
        If original action is blocked, returns HOLD (0).
        """
        self.total_actions += 1
        
        allowed, reason = self.validate(action, current_step)
        
        if not allowed:
            self.blocked_actions[reason.split(" ")[0]] += 1
            return 0  # Fall back to HOLD
        
        return action
    
    def record_outcome(self, action, info):
        """Update internal state based on what actually happened."""
        action_type, coin_idx = self._parse_action(action)
        
        if action_type == "open" and info["action_success"]:
            self.current_positions.add(coin_idx)
            self.position_open_steps[coin_idx] = info["step"] - 1  # step already incremented
            self.opens_per_rank[coin_idx] += 1
            self.steps_since_last_open = 0
        
        elif action_type == "close" and info["action_success"]:
            self.current_positions.discard(coin_idx)
            self.position_open_steps.pop(coin_idx, None)
        
        elif action_type == "close_all" and info["action_success"]:
            self.current_positions.clear()
            self.position_open_steps.clear()
        
        # Increment cooldown counter
        self.steps_since_last_open += 1


def build_agent(env):
    """Build agent with network initialized before loading weights."""
    agent = DQNAgent(
        state_dim=env.get_state_space_size(),
        action_dim=env.get_action_space_size(),
        config_path="config/agent.yaml"
    )
    dummy_input = np.zeros((1, env.get_state_space_size()), dtype=np.float32)
    _ = agent.q_network(dummy_input, training=False)
    _ = agent.target_network(dummy_input, training=False)
    return agent


def run_paper_trading(env, agent, dates, guardrails, verbose=True):
    """
    Run paper trading simulation with optional guardrails.
    
    Args:
        guardrails: Dict of guardrail settings, or None for no guardrails
    
    Returns per-episode results.
    """
    # Only create wrapper if guardrails provided
    wrapper = GuardrailWrapper(env, guardrails) if guardrails else None
    results = []
    
    for i, date in enumerate(dates):
        state = env.reset(date=date)
        if wrapper:
            wrapper.reset()
        done = False
        step = 200  # warmup_candles
        
        while not done:
            # Agent decides
            raw_action = agent.select_action(state, training=False)
            
            # Guardrails filter (if enabled)
            if wrapper:
                action = wrapper.apply(raw_action, step)
            else:
                action = raw_action
            
            # Execute
            next_state, reward, done, info = env.step(action)
            
            # Update guardrail state (if enabled)
            if wrapper:
                wrapper.record_outcome(action, info)
            
            state = next_state
            step += 1
        
        stats = env.get_episode_statistics()
        result = {
            'date': date,
            'total_pnl': stats['total_pnl'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'win_rate': stats['win_rate'],
        }
        
        if wrapper:
            result['blocked'] = dict(wrapper.blocked_actions)
        
        results.append(result)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(dates)} episodes completed...")
    
    return results


def print_comparison(raw_results, guarded_results, dates):
    """Print side-by-side comparison: raw agent vs agent + guardrails."""
    
    raw_pnls = np.array([r['total_pnl'] for r in raw_results])
    guard_pnls = np.array([r['total_pnl'] for r in guarded_results])
    
    def calc_metrics(pnls):
        avg = np.mean(pnls)
        std = np.std(pnls)
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        winning = pnls[pnls > 0]
        losing = pnls[pnls < 0]
        pf = np.sum(winning) / np.abs(np.sum(losing)) if len(losing) > 0 and np.sum(losing) != 0 else float('inf')
        profitable = int(np.sum(pnls > 0))
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = np.min(cumulative - running_max)
        
        # Max loss streak
        streak = 0
        max_streak = 0
        for p in pnls:
            if p < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        
        return {
            'avg_pnl': avg, 'std_pnl': std, 'sharpe': sharpe,
            'profit_factor': pf, 'profitable_pct': profitable / len(pnls) * 100,
            'max_drawdown': max_dd, 'max_loss_streak': max_streak,
            'total_pnl': np.sum(pnls),
            'avg_trades': np.mean([r['total_trades'] for r in (raw_results if pnls is raw_pnls else guarded_results)]),
        }
    
    raw_m = calc_metrics(raw_pnls)
    guard_m = calc_metrics(guard_pnls)
    
    # Blocked action summary (only for guarded)
    total_blocked = defaultdict(int)
    for r in guarded_results:
        if 'blocked' in r:  # Only guarded results have this
            for reason, count in r['blocked'].items():
                total_blocked[reason] += count
    
    print("\n" + "="*70)
    print("  PAPER TRADING: RAW AGENT vs AGENT + GUARDRAILS")
    print("="*70)
    
    print(f"\n  {'Metric':<25} {'Raw Agent':>15} {'+ Guardrails':>15} {'Better?':>10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")
    
    metrics = [
        ("Avg PnL", f"${raw_m['avg_pnl']:.2f}", f"${guard_m['avg_pnl']:.2f}", 
         "✅" if guard_m['avg_pnl'] > raw_m['avg_pnl'] else "—"),
        ("Sharpe Ratio", f"{raw_m['sharpe']:.2f}", f"{guard_m['sharpe']:.2f}",
         "✅" if guard_m['sharpe'] > raw_m['sharpe'] else "—"),
        ("Profit Factor", f"{guard_m['profit_factor'] if guard_m['profit_factor'] != float('inf') else '∞'}", 
         f"{guard_m['profit_factor'] if guard_m['profit_factor'] != float('inf') else '∞'}",
         "—"),
        ("Profitable Days", f"{raw_m['profitable_pct']:.1f}%", f"{guard_m['profitable_pct']:.1f}%",
         "✅" if guard_m['profitable_pct'] > raw_m['profitable_pct'] else "—"),
        ("Max Drawdown", f"${raw_m['max_drawdown']:.2f}", f"${guard_m['max_drawdown']:.2f}",
         "✅" if guard_m['max_drawdown'] > raw_m['max_drawdown'] else "—"),
        ("Max Loss Streak", f"{raw_m['max_loss_streak']} days", f"{guard_m['max_loss_streak']} days",
         "✅" if guard_m['max_loss_streak'] < raw_m['max_loss_streak'] else "—"),
        ("Avg Trades/Day", f"{raw_m['avg_trades']:.1f}", f"{guard_m['avg_trades']:.1f}",
         "✅ fewer" if guard_m['avg_trades'] < raw_m['avg_trades'] else "—"),
        ("Total PnL (200d)", f"${raw_m['total_pnl']:.2f}", f"${guard_m['total_pnl']:.2f}",
         "✅" if guard_m['total_pnl'] > raw_m['total_pnl'] else "—"),
    ]
    
    for name, raw_val, guard_val, better in metrics:
        print(f"  {name:<25} {raw_val:>15} {guard_val:>15} {better:>10}")
    
    # Blocked actions
    print(f"\n  --- Actions Blocked by Guardrails ---")
    for reason, count in sorted(total_blocked.items(), key=lambda x: x[1], reverse=True):
        print(f"    {reason:<30} {count:>6} times")
    
    print(f"\n  --- VERDICT ---")
    if guard_m['avg_pnl'] > 30 and guard_m['sharpe'] > 1.0:
        print(f"  ✅ Guardrails maintain strong performance!")
        print(f"     Agent + guardrails is ready for paper trading with real capital.")
    elif guard_m['avg_pnl'] > 0:
        print(f"  ⚠️  Guardrails reduced performance but kept it profitable.")
        print(f"     Consider adjusting guardrail parameters.")
    else:
        print(f"  ❌ Guardrails killed performance. Agent relies on the bad habits.")
        print(f"     Need to retrain with these constraints built into the environment.")
    
    print("="*70)
    
    return raw_m, guard_m


def main():
    print("="*70)
    print("  PAPER TRADING SIMULATION WITH GUARDRAILS")
    print("="*70)
    print(f"\n  Guardrails:")
    print(f"    Open cooldown:       {GUARDRAILS['open_cooldown_steps']} steps (25 min)")
    print(f"    Min hold period:     {GUARDRAILS['min_hold_steps']} steps (15 min)")
    print(f"    Max opens per rank:  {GUARDRAILS['max_opens_per_rank']} per day")
    
    # Force CPU to avoid GPU memory issues
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("\n  NOTE: Using CPU to avoid GPU memory conflicts (slower but stable)")
    
    # Use same holdout dates as previous eval for fair comparison
    print("\n  Initializing environment...")
    env = make_env()
    all_dates = sorted(env.available_dates)
    HOLDOUT_SIZE = 200
    holdout_dates = all_dates[-HOLDOUT_SIZE:]
    
    # Load agent ONCE
    print("  Loading checkpoint: checkpoints/best")
    agent = build_agent(env)
    agent.load("checkpoints/best")
    agent.epsilon = 0.0  # Fully greedy
    
    # --- Run RAW agent (no guardrails) ---
    print(f"\n  [1/2] Running RAW agent (no guardrails)...")
    print("  NOTE: Agent can spam actions freely, no constraints")
    print("  (This will take ~10-15 minutes on CPU)")
    raw_results = run_paper_trading(env, agent, holdout_dates, guardrails=None, verbose=True)
    
    # --- Run agent WITH guardrails (reuse same agent) ---
    print(f"\n  [2/2] Running agent WITH guardrails...")
    guarded_results = run_paper_trading(env, agent, holdout_dates, guardrails=GUARDRAILS, verbose=True)
    
    # --- Compare ---
    raw_m, guard_m = print_comparison(raw_results, guarded_results, holdout_dates)
    
    # Save results
    output = {
        'guardrails': GUARDRAILS,
        'raw': {
            'avg_pnl': float(raw_m['avg_pnl']),
            'sharpe': float(raw_m['sharpe']),
            'profit_factor': float(raw_m['profit_factor']) if raw_m['profit_factor'] != float('inf') else None,
        },
        'guarded': {
            'avg_pnl': float(guard_m['avg_pnl']),
            'sharpe': float(guard_m['sharpe']),
            'profit_factor': float(guard_m['profit_factor']) if guard_m['profit_factor'] != float('inf') else None,
        },
        'raw_episodes': [
            {'date': r['date'], 'pnl': r['total_pnl'], 'trades': r['total_trades']}
            for r in raw_results
        ],
        'guarded_episodes': [
            {'date': r['date'], 'pnl': r['total_pnl'], 'trades': r['total_trades'], 
             'blocked': r.get('blocked', {})}
            for r in guarded_results
        ],
    }
    
    with open('paper_trading_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  ✅ Results saved to: paper_trading_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
