"""
Training Script — v2 (IQN + Curriculum + Walk-Forward)

Usage:
    python scripts/train.py
    python scripts/train.py --episodes 16000
    python scripts/train.py --resume checkpoints/best
    python scripts/train.py --stage trending          # single curriculum stage
    python scripts/train.py --walk-forward            # evaluation mode only
    python scripts/train.py --no-curriculum           # skip stages, full random

Curriculum stages (auto-progresses unless --stage is specified):
    1. trending  — high-ADX days, clearest momentum signals
    2. ranging   — low-ADX days, teaches patience
    3. volatile  — high-volatility days, teaches risk management
    4. full      — all days, final generalization

Walk-forward evaluation:
    Splits data into rolling 12-month train / 1-month test windows.
    Reports per-fold Sharpe, win rate, profit factor, and invalid action rate.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import tensorflow as tf
import yaml

# ── Graceful shutdown flag ────────────────────────────────────────────────────
_shutdown_requested = False


def _handle_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT by setting a flag — the training loop saves and exits."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    print(
        f"\n[{sig_name}] Shutdown requested — saving checkpoint after current episode..."
    )
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading import make_env
from src.agent.iqn_agent import IQNAgent

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Live metrics writer — read by monitor.py
# ══════════════════════════════════════════════════════════════════════════════


def write_live_metrics(
    log_dir: Path,
    training_start: str,
    current_stage: str,
    global_episode: int,
    total_episodes: int,
    epsilon: float,
    recent_episodes: list,
    stage_history: dict,
    action_counts: dict,
    walk_forward_folds: list,
    model_config: dict = None,
    elapsed_seconds: float = 0.0,
    episode_times: list = None,
    replay_stats: dict = None,
) -> None:
    """
    Write logs/training/live_metrics.json so monitor.py can display
    real-time training progress in the browser.

    Called every 10 episodes from run_curriculum().
    This is a best-effort write — if it fails, training continues.
    """
    # Calculate ETA from recent episode timing
    ep_times = episode_times or []
    avg_ep_time = float(np.mean(ep_times[-50:])) if ep_times else 0.0
    episodes_remaining = max(0, total_episodes - global_episode)
    eta_seconds = episodes_remaining * avg_ep_time

    payload = {
        "training_started": training_start,
        "current_stage": current_stage,
        "global_episode": global_episode,
        "total_episodes": total_episodes,
        "epsilon": round(epsilon, 4),
        "recent_episodes": recent_episodes[-200:],  # keep last 200 for charts
        "stage_history": stage_history,
        "action_counts": action_counts,
        "walk_forward_folds": walk_forward_folds,
        "last_updated": datetime.now().isoformat(),
        # ── Enhanced fields ──────────────────────────────────────────────
        "model_config": model_config or {},
        "elapsed_seconds": round(elapsed_seconds, 1),
        "avg_episode_time": round(avg_ep_time, 3),
        "eta_seconds": round(eta_seconds, 1),
        "episodes_per_sec": round(1.0 / avg_ep_time, 2) if avg_ep_time > 0 else 0,
        "replay_stats": replay_stats or {},
    }
    try:
        out = log_dir / "live_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(out)  # atomic replace — monitor never reads partial file
    except Exception as e:
        logger.debug(f"write_live_metrics failed (non-fatal): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Setup
# ══════════════════════════════════════════════════════════════════════════════


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{ts}.log"
    level = logging.DEBUG if verbose else logging.INFO
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
    p = argparse.ArgumentParser(description="Train IQN agent — Asymptotic Zero v2")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["trending", "ranging", "volatile", "full"],
        help="Run a single curriculum stage only",
    )
    p.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Skip curriculum, train on all dates immediately",
    )
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward evaluation on saved checkpoint",
    )
    p.add_argument("--config", type=str, default="config/trading.yaml")
    p.add_argument("--data-dir", type=str, default="data/volatility")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-dir", type=str, default="logs/training")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-dashboard", action="store_true")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Regime labelling — classifies dates for curriculum
# ══════════════════════════════════════════════════════════════════════════════


def label_regimes(
    top_movers_file: Path,
    data_dir: Path,
    config: dict,
    cache_path: Path,
) -> Dict[str, List[str]]:
    """
    Assign each available date one or more regime labels: trending / ranging / volatile.

    Uses percentile-based cutoffs relative to the dataset distribution.
    A day CAN have multiple labels (e.g., both trending AND volatile).

    Strategy:
      - Load all 5m data for ALL available coins per day (not just top 2)
      - Compute average ADX and realized volatility per day
      - Label based on percentile cutoffs from config

    Returns:
        date_str → list of regime labels  (e.g. {"2024-01-15": ["trending", "volatile"]})
    """
    if cache_path.exists():
        df = pl.read_parquet(cache_path)
        labels = {}
        for row in df.iter_rows(named=True):
            date_str = str(row["date"])
            regime = row["regime"]
            # Support multi-label: stored as comma-separated string
            labels[date_str] = regime.split(",") if "," in regime else [regime]
        logger.info(f"Loaded regime cache: {len(labels)} dates from {cache_path}")
        return labels

    logger.info("Computing regime labels (first run — cached after this)...")

    cur_cfg = config.get("curriculum", {})
    stages = {s["name"]: s for s in cur_cfg.get("stages", [])}

    # Percentile thresholds from config
    trending_adx_pct_min = stages.get("trending", {}).get("adx_percentile_min", 67)
    ranging_adx_pct_max = stages.get("ranging", {}).get("adx_percentile_max", 33)
    volatile_vol_pct_min = stages.get("volatile", {}).get("vol_percentile_min", 67)

    top_movers = pl.read_parquet(top_movers_file)
    dates = [str(d) for d in top_movers["date"].unique().sort().to_list()]

    day_stats = []
    for date in dates:
        row = top_movers.filter(
            pl.col("date") == datetime.strptime(date, "%Y-%m-%d").date()
        )
        if len(row) == 0:
            continue
        row_data = row.row(0, named=True)
        # Use ALL available coins (not just top 2+2)
        symbols = list(row_data["gainers"]) + list(row_data["losers"])

        adx_vals = []
        vol_vals = []
        for sym in symbols:
            fpath = data_dir / "5m" / sym / f"{date}.parquet"
            if not fpath.exists():
                continue
            df = pl.read_parquet(fpath)
            if "adx_14" in df.columns:
                adx_clean = df["adx_14"].drop_nulls().to_numpy()
                if len(adx_clean) > 0:
                    adx_vals.append(float(np.nanmean(adx_clean)))
            closes = df["close"].to_numpy().astype(np.float64)
            if len(closes) > 1:
                rets = np.diff(np.log(closes + 1e-8))
                vol_vals.append(float(np.std(rets)))

        avg_adx = float(np.mean(adx_vals)) if adx_vals else 20.0
        avg_vol = float(np.mean(vol_vals)) if vol_vals else 0.003
        day_stats.append({"date": date, "adx": avg_adx, "vol": avg_vol})

    # Compute percentile ranks for ADX and volatility
    adx_values = np.array([d["adx"] for d in day_stats])
    vol_values = np.array([d["vol"] for d in day_stats])

    adx_pct = {
        d["date"]: float(np.sum(adx_values <= d["adx"]) / len(adx_values) * 100)
        for d in day_stats
    }
    vol_pct = {
        d["date"]: float(np.sum(vol_values <= d["vol"]) / len(vol_values) * 100)
        for d in day_stats
    }

    # Multi-label assignment using percentiles
    labels: Dict[str, List[str]] = {}
    for d in day_stats:
        date = d["date"]
        day_labels = []

        if adx_pct[date] >= trending_adx_pct_min:
            day_labels.append("trending")
        if adx_pct[date] <= ranging_adx_pct_max:
            day_labels.append("ranging")
        if vol_pct[date] >= volatile_vol_pct_min:
            day_labels.append("volatile")

        # Days with no specific label only appear in "full" stage
        if not day_labels:
            day_labels.append("unlabeled")

        labels[date] = day_labels

    # Cache (store multi-labels as comma-separated string)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": list(labels.keys()),
            "regime": [",".join(l) for l in labels.values()],
        }
    ).write_parquet(cache_path)

    # Count stats
    trending_ct = sum(1 for v in labels.values() if "trending" in v)
    ranging_ct = sum(1 for v in labels.values() if "ranging" in v)
    volatile_ct = sum(1 for v in labels.values() if "volatile" in v)
    multi_ct = sum(1 for v in labels.values() if len(v) > 1)
    logger.info(
        f"Regime labels computed and cached → {cache_path} | "
        f"trending={trending_ct} | ranging={ranging_ct} | "
        f"volatile={volatile_ct} | multi-label={multi_ct} | "
        f"total={len(labels)}"
    )
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ══════════════════════════════════════════════════════════════════════════════


def run_episode(
    env: "TradingEnvironment",
    agent: IQNAgent,
    train: bool = True,
) -> Dict:
    """
    Run one full episode.

    Args:
        env:   Trading environment
        agent: IQN agent
        train: If True, store transitions and call agent.train()

    Returns:
        Episode statistics dict
    """
    state, mask = env.reset()
    done = False

    episode_reward = 0.0
    invalid_actions = 0
    action_counts = {}
    steps = 0
    train_losses = []

    while not done:
        # Get state (sequences if LSTM, array if not)
        if agent.lstm_enabled:
            agent_state = env.get_state_sequences()
        else:
            agent_state = state

        action = agent.select_action(agent_state, mask, deterministic=(not train))

        next_state, reward, done, next_mask, info = env.step(action)

        if train:
            # Store transitions in replay buffer
            # Note: For LSTM, we store flat states. The replay buffer's
            # sample_with_sequences() reconstructs multi-scale sequences
            # by looking back through the buffer history.
            agent.remember(state, action, reward, next_state, done, next_mask)
            loss = agent.train()
            if loss is not None:
                train_losses.append(loss)

        episode_reward += reward
        if info.get("masked", False) or not info.get("action_success", True):
            invalid_actions += 1

        action_type = info.get("action_type", "unknown")
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

        state, mask = next_state, next_mask
        steps += 1

    # Collect episode stats
    ep_stats = env.get_episode_statistics()
    agent_stats = agent.get_statistics()

    return {
        "total_reward": episode_reward,
        "episode_pnl": ep_stats.get("total_pnl", 0.0),
        "win_rate": ep_stats.get("win_rate", 0.0),
        "trade_count": ep_stats.get("total_trades", 0),
        "invalid_actions": invalid_actions,
        "steps": steps,
        "loss_mean": float(np.mean(train_losses)) if train_losses else 0.0,
        "epsilon": agent.epsilon,
        "action_counts": action_counts,
        "date": ep_stats.get("date", ""),
        "reward_stats": ep_stats.get("reward_stats", {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum training
# ══════════════════════════════════════════════════════════════════════════════


def _run_periodic_eval(
    env: "TradingEnvironment",
    agent: IQNAgent,
    num_episodes: int = 30,
) -> Dict:
    """
    Run deterministic evaluation episodes on random dates.
    Returns aggregated eval metrics.
    """
    eval_pnls = []
    eval_wrs = []
    eval_trades = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # force greedy

    for _ in range(num_episodes):
        stats = run_episode(env, agent, train=False)
        eval_pnls.append(stats["episode_pnl"])
        eval_wrs.append(stats["win_rate"])
        eval_trades.append(stats["trade_count"])

    agent.epsilon = old_epsilon  # restore

    pnls_arr = np.array(eval_pnls)
    sharpe = (
        float(np.mean(pnls_arr) / (np.std(pnls_arr) + 1e-8))
        if len(pnls_arr) > 1
        else 0.0
    )
    return {
        "pnl": float(np.mean(pnls_arr)),
        "win_rate": float(np.mean(eval_wrs)),
        "sharpe": sharpe,
        "trade_count": float(np.mean(eval_trades)),
        "num_episodes": num_episodes,
    }


def warmup_state_normalization(
    env: "TradingEnvironment",
    agent: "IQNAgent",  # Type hint string to avoid circular import if needed
    num_steps: int = 2000,
):
    """
    Run random steps to initialize state normalization statistics.
    Crucial for z-score normalization to work from step 1.
    """
    logger.info(f"Warming up state normalization stats ({num_steps} random steps)...")

    # Store old epsilon to restore later
    old_epsilon = agent.epsilon
    agent.epsilon = 1.0  # Force random exploration

    states = []
    state, mask = env.reset(random_date=True)

    # Handle LSTM vs Flat state
    if isinstance(state, dict):
        # For LSTM, we collect the 'short' sequence tail as a proxy for state distribution
        # or we just collect the raw normalized states?
        # StateBuilder.update_statistics expects a list of flat arrays (if configured that way).
        # Actually StateBuilder normalizes internally. We need to feed it *raw* states?
        # No, update_statistics takes ALREADY BUILT states and computes stats on them.
        # But build_state calls _normalize.

        # We need to bypass normalization to collect raw stats?
        # StateBuilder.update_statistics computes means/stds from the provided states.
        # If provided states are already normalized (clipped), it defeats the purpose.
        # However, _normalize only normalizes if means/stds are set.
        # Initially they are None, so _normalize just clips.
        # So the states we get here are (unnormalized) clipped states.
        # We collect them, then compute stats.
        pass

    for _ in range(num_steps):
        action = agent.select_action(state, mask)
        next_state, _, done, next_mask, _ = env.step(action)

        # Collect flat state for statistics
        if isinstance(next_state, dict):
            # For LSTM, getting the "current" flat state is tricky since it returns sequences.
            # But StateBuilder.update_statistics likely expects flat vectors.
            # Let's check environment.py -> _build_state
            # It builds flat state, normalizes it, THEN appends to buffer.
            # The state returned by env.step() is the SEQUENCE dictionary (if LSTM).
            # We can't easily reverse-engineer the flat state from the sequence dict here without access to internal buffer.
            #
            # ALTERNATIVE: Access environment's state_builder directly.
            pass
        else:
            states.append(next_state)

        if done:
            state, mask = env.reset(random_date=True)
        else:
            state, mask = next_state, next_mask

    # Restore epsilon
    agent.epsilon = old_epsilon

    # If using LSTM, the states we collected from step() are dictionaries, which won't work.
    # We need a way to capture the FLAT state before sequence construction.
    # Use a hook or strict "flat" mode during warmup?
    #
    # Actually, if LSTM is enabled, StateBuilder still builds a flat state first.
    # But env.step returns the sequence dict.
    #
    # Better approach: Just use the fact that StateBuilder.update_statistics works on flat arrays.
    # If we are in LSTM mode, we can't easily get the flat arrays from env.step().
    #
    # Workaround: Temporarily disable LSTM in env? No.
    #
    # Solution: We should rely on a method in Environment that exposes the flat state,
    # OR, we just initialize with zeros/ones if we can't run the environment? No.
    #
    # Let's trust that we can extract the latest state from the sequence?
    # batch['seq_short'] -> (10, dim). The last element is the current state.
    if states:
        env.state_builder.update_statistics(states)
        logger.info(
            f"State normalization initialized (mean={np.mean(env.state_builder._feature_means):.4f})"
        )
    elif env.lstm_enabled:
        logger.warning(
            "Skipping explicit state warmup for LSTM (complexity). Stats will adapt if online update enabled."
        )


def run_curriculum(
    env: "TradingEnvironment",
    agent: IQNAgent,
    config: dict,
    date_labels: Dict[str, List[str]],
    stage_name: Optional[str],
    total_episodes: int,
    checkpoint_dir: Path,
    log_dir: Path,
    dashboard: bool,
    resume_state: Optional[Dict] = None,
) -> List[Dict]:
    """
    Run full curriculum training or a single specified stage.

    Each stage gets its own full epsilon decay cycle and guardrail settings.
    Advances to next stage when rolling win_rate > threshold for N episodes.
    """
    cur_cfg = config.get("curriculum", {})
    stages_cfg = cur_cfg.get("stages", [])
    advance_wr = cur_cfg.get("stage_advance_win_rate", 0.52)
    advance_win = cur_cfg.get("stage_advance_window", 200)

    # Monitoring config
    mon_cfg = config.get("monitoring", {})
    eval_freq = mon_cfg.get("eval_freq_episodes", 500)
    eval_episodes = mon_cfg.get("eval_episodes", 30)
    hold_warn_pct = mon_cfg.get("hold_warning_pct", 0.80)

    # Default guardrails from config (used if stage doesn't override)
    default_guardrails = config.get("guardrails", {})

    # ── TensorBoard writer ────────────────────────────────────────────────────
    tb_dir = log_dir.parent / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = tf.summary.create_file_writer(str(tb_dir))
    logger.info(f"TensorBoard logs → {tb_dir}")
    logger.info(f"  View: tensorboard --logdir {tb_dir.parent}")

    # Live monitoring state (written to live_metrics.json every 10 episodes)
    training_start = datetime.now().isoformat()
    training_start_time = time.time()
    stage_history: Dict[str, dict] = {}
    live_action_counts: Dict[str, int] = {}
    all_recent_eps: List[Dict] = []
    walk_forward_folds_cache: List[Dict] = []
    episode_times: List[float] = []
    # Per-episode action tracking for distribution monitoring
    rolling_action_counts: Dict[str, int] = {}

    # Build model config summary for dashboard display
    lstm_cfg = config.get("lstm", {})
    net_cfg = config.get("network", {})
    agent_cfg = config.get("agent", {})
    model_config = {
        "architecture": "LSTM-IQN" if lstm_cfg.get("enabled", False) else "IQN",
        "state_dim": agent.state_dim,
        "action_size": agent.action_size,
        "lstm": {
            "enabled": lstm_cfg.get("enabled", False),
            "sequence_lengths": lstm_cfg.get("sequence_lengths", []),
            "hidden_sizes": lstm_cfg.get("hidden_sizes", []),
            "dropout_rate": lstm_cfg.get("dropout_rate", 0),
        },
        "network": {
            "hidden_layers": net_cfg.get("hidden_layers", []),
            "embedding_dim": net_cfg.get("embedding_dim", 64),
            "num_quantiles_train": net_cfg.get("num_quantiles_train", 8),
            "num_quantiles_inference": net_cfg.get("num_quantiles_inference", 32),
            "risk_distortion": net_cfg.get("risk_distortion", "neutral"),
            "use_layer_norm": net_cfg.get("use_layer_norm", False),
        },
        "agent": {
            "learning_rate": agent_cfg.get("learning_rate", 0),
            "gamma": agent_cfg.get("gamma", 0.99),
            "batch_size": agent_cfg.get("batch_size", 64),
            "replay_capacity": agent_cfg.get("replay_capacity", 200000),
            "target_update_freq": agent_cfg.get("target_update_freq", 100),
            "train_freq": agent_cfg.get("train_freq", 4),
            "gradient_clip": agent_cfg.get("gradient_clip", 1.0),
            "epsilon_start": agent_cfg.get("epsilon_start", 1.0),
            "epsilon_end": agent_cfg.get("epsilon_end", 0.05),
            "epsilon_decay_pct": agent_cfg.get("epsilon_decay_percentage", 0.85),
        },
    }

    # Build stage list
    if stage_name:
        stages_to_run = [s for s in stages_cfg if s["name"] == stage_name]
    elif config.get("no_curriculum"):
        stages_to_run = [{"name": "full", "episodes": total_episodes}]
    else:
        stages_to_run = stages_cfg

    if not stages_to_run:
        logger.warning("No curriculum stages configured — running full random")
        stages_to_run = [{"name": "full", "episodes": total_episodes}]

    agent.set_total_episodes(total_episodes)

    all_history = []
    global_episode = 0
    shutdown_clean = False

    # ── Resume state: skip completed stages, restore position ─────────────
    resume_stage_name = None
    resume_stage_ep = 0
    completed_stages = set()
    if resume_state:
        global_episode = resume_state.get("global_episode", 0)
        resume_stage_name = resume_state.get("current_stage")
        resume_stage_ep = resume_state.get("stage_episode", 0)
        completed_stages = set(resume_state.get("completed_stages", []))
        logger.info(
            f"Resuming training: global_ep={global_episode}, "
            f"stage={resume_stage_name}, stage_ep={resume_stage_ep}, "
            f"completed={completed_stages}"
        )

    for stage_idx, stage in enumerate(stages_to_run):
        sname = stage["name"]
        s_eps = stage.get("episodes", total_episodes)

        # Skip stages already completed in a previous run
        if sname in completed_stages:
            logger.info(f"Skipping completed stage: {sname}")
            stage_history[sname] = {
                "episodes": s_eps,
                "avg_pnl": None,
                "avg_win_rate": None,
                "completed": True,
            }
            continue

        # Filter dates for this stage (multi-label: match if label list contains stage name)
        if sname == "full":
            stage_dates = env.available_dates
        else:
            stage_dates = [d for d, lbls in date_labels.items() if sname in lbls]
            if len(stage_dates) < 10:
                logger.warning(
                    f"Stage '{sname}' has only {len(stage_dates)} dates — "
                    "merging with full set"
                )
                stage_dates = env.available_dates

        env.set_curriculum_dates(stage_dates)

        # ── Per-stage epsilon decay ───────────────────────────────────────────
        stage_epsilon_start = stage.get("epsilon_start", None)
        if not (resume_state and sname == resume_stage_name and resume_stage_ep > 0):
            # Fresh stage start — reset epsilon for full exploration cycle
            agent.reset_epsilon(s_eps, epsilon_start=stage_epsilon_start)
        else:
            # Resuming mid-stage — keep current epsilon from checkpoint
            logger.info(f"Resuming mid-stage, keeping ε={agent.epsilon:.4f}")

        # ── Per-stage guardrails ──────────────────────────────────────────────
        stage_guardrails = stage.get("guardrails", default_guardrails)
        if stage_guardrails:
            env.set_guardrails(stage_guardrails)

        # Determine starting episode (resume within stage if applicable)
        start_ep = 1
        if resume_state and sname == resume_stage_name and resume_stage_ep > 0:
            start_ep = resume_stage_ep + 1
            logger.info(f"Resuming stage '{sname}' from episode {start_ep}")

        logger.info(
            f"\n{'═' * 60}\n"
            f"  STAGE: {sname.upper()} | {len(stage_dates)} dates | "
            f"ep {start_ep}→{s_eps} | ε={agent.epsilon:.2f}\n"
            f"{'═' * 60}"
        )

        # Rolling win rate window for stage progression
        win_rate_window = deque(maxlen=advance_win)
        stage_episodes = []  # episode dicts for THIS stage only
        best_win_rate = 0.0

        # Initialize stage_history entry for monitor
        stage_history[sname] = {
            "episodes": 0,
            "avg_pnl": None,
            "avg_win_rate": None,
            "completed": False,
        }

        for ep in range(start_ep, s_eps + 1):
            ep_start_time = time.time()
            stats = run_episode(env, agent, train=True)
            agent.decay_epsilon()
            episode_times.append(time.time() - ep_start_time)

            win_rate_window.append(stats["win_rate"])
            stage_episodes.append(stats)
            all_history.append(stats)
            global_episode += 1

            # ── Accumulate live monitoring data ──────────────────────────────
            all_recent_eps.append(
                {
                    "ep": global_episode,
                    "pnl": round(stats["episode_pnl"], 4),
                    "win_rate": round(stats["win_rate"], 4),
                    "loss": round(stats["loss_mean"], 6),
                    "invalid": stats["invalid_actions"],
                    "date": stats.get("date", ""),
                    "trades": stats.get("trade_count", 0),
                    "reward": round(stats.get("total_reward", 0), 4),
                    "steps": stats.get("steps", 0),
                    "ep_time": round(episode_times[-1], 3) if episode_times else 0,
                }
            )
            for atype, cnt in stats.get("action_counts", {}).items():
                live_action_counts[atype] = live_action_counts.get(atype, 0) + cnt
                rolling_action_counts[atype] = rolling_action_counts.get(atype, 0) + cnt

            stage_history[sname]["episodes"] = ep
            stage_history[sname]["avg_pnl"] = round(
                float(np.mean([s["episode_pnl"] for s in stage_episodes[-100:]])), 4
            )
            stage_history[sname]["avg_win_rate"] = round(
                float(np.mean([s["win_rate"] for s in stage_episodes[-100:]])), 4
            )

            # ── TensorBoard: per-episode scalars ──────────────────────────────
            with tb_writer.as_default(step=global_episode):
                tf.summary.scalar("train/loss", stats["loss_mean"])
                tf.summary.scalar("train/pnl", stats["episode_pnl"])
                tf.summary.scalar("train/reward", stats.get("total_reward", 0))
                tf.summary.scalar("train/win_rate", stats["win_rate"])
                tf.summary.scalar("train/epsilon", stats["epsilon"])
                tf.summary.scalar("train/trades", stats.get("trade_count", 0))
                tf.summary.scalar("train/invalid", stats["invalid_actions"])

            # ── TensorBoard: rolling metrics + action distribution (every 50 eps)
            if ep % 50 == 0 and len(stage_episodes) >= 2:
                recent_50 = stage_episodes[-50:]
                r_pnls = np.array([s["episode_pnl"] for s in recent_50])
                r_wrs = np.array([s["win_rate"] for s in recent_50])
                r_sharpe = float(np.mean(r_pnls) / (np.std(r_pnls) + 1e-8))

                with tb_writer.as_default(step=global_episode):
                    tf.summary.scalar("rolling/pnl_50ep", float(np.mean(r_pnls)))
                    tf.summary.scalar("rolling/win_rate_50ep", float(np.mean(r_wrs)))
                    tf.summary.scalar("rolling/sharpe_50ep", r_sharpe)

                    # Action distribution
                    total_acts = max(sum(rolling_action_counts.values()), 1)
                    hold_cnt = rolling_action_counts.get("hold", 0)
                    open_long_cnt = sum(
                        v
                        for k, v in rolling_action_counts.items()
                        if k.startswith("open_long")
                    )
                    open_short_cnt = sum(
                        v
                        for k, v in rolling_action_counts.items()
                        if k.startswith("open_short")
                    )
                    close_long_cnt = sum(
                        v
                        for k, v in rolling_action_counts.items()
                        if k.startswith("close") and k.endswith("long")
                    )
                    close_short_cnt = sum(
                        v
                        for k, v in rolling_action_counts.items()
                        if k.startswith("close") and k.endswith("short")
                    )
                    close_all_cnt = rolling_action_counts.get("close_all", 0)
                    close_cnt = close_long_cnt + close_short_cnt + close_all_cnt

                    hold_pct = hold_cnt / total_acts
                    tf.summary.scalar("actions/hold_pct", hold_pct)
                    tf.summary.scalar(
                        "actions/open_long_pct", open_long_cnt / total_acts
                    )
                    tf.summary.scalar(
                        "actions/open_short_pct", open_short_cnt / total_acts
                    )
                    tf.summary.scalar(
                        "actions/close_long_pct", close_long_cnt / total_acts
                    )
                    tf.summary.scalar(
                        "actions/close_short_pct", close_short_cnt / total_acts
                    )
                    tf.summary.scalar("actions/close_pct", close_cnt / total_acts)

                # HOLD-dominance warning
                if hold_pct > hold_warn_pct:
                    logger.warning(
                        f"⚠ HOLD dominance: {hold_pct:.1%} of actions are HOLD "
                        f"(threshold: {hold_warn_pct:.0%}) — agent may need reward tuning"
                    )

                # Reset rolling action counts for next window
                rolling_action_counts.clear()

            # ── TensorBoard: periodic evaluation (every eval_freq eps) ────────
            if ep % eval_freq == 0 and not _shutdown_requested:
                logger.info(
                    f"Running periodic eval ({eval_episodes} episodes, deterministic)..."
                )
                eval_results = _run_periodic_eval(
                    env, agent, num_episodes=eval_episodes
                )
                with tb_writer.as_default(step=global_episode):
                    tf.summary.scalar("eval/pnl", eval_results["pnl"])
                    tf.summary.scalar("eval/win_rate", eval_results["win_rate"])
                    tf.summary.scalar("eval/sharpe", eval_results["sharpe"])
                    tf.summary.scalar("eval/trade_count", eval_results["trade_count"])
                logger.info(
                    f"  Eval: PnL=${eval_results['pnl']:+.2f} | "
                    f"WR={eval_results['win_rate']:.1%} | "
                    f"Sharpe={eval_results['sharpe']:.2f} | "
                    f"trades={eval_results['trade_count']:.1f}"
                )
                # Restore curriculum dates after eval (eval may have changed them)
                env.set_curriculum_dates(stage_dates)

            # ── Flush TensorBoard periodically ────────────────────────────────
            if ep % 10 == 0:
                tb_writer.flush()
                write_live_metrics(
                    log_dir=log_dir,
                    training_start=training_start,
                    current_stage=sname,
                    global_episode=global_episode,
                    total_episodes=total_episodes,
                    epsilon=agent.epsilon,
                    recent_episodes=all_recent_eps,
                    stage_history=stage_history,
                    action_counts=live_action_counts,
                    walk_forward_folds=walk_forward_folds_cache,
                    model_config=model_config,
                    elapsed_seconds=time.time() - training_start_time,
                    episode_times=episode_times,
                    replay_stats=agent.replay.get_statistics(),
                )
            # ─────────────────────────────────────────────────────────────────

            # Logging
            if ep % 50 == 0 or ep == start_ep:
                rolling_wr = np.mean(win_rate_window) if win_rate_window else 0.0
                rolling_pnl = np.mean([s["episode_pnl"] for s in stage_episodes[-50:]])
                logger.info(
                    f"[{sname}] ep={ep:>5}/{s_eps} | "
                    f"ε={stats['epsilon']:.3f} | "
                    f"PnL=${rolling_pnl:>+7.2f} | "
                    f"WR={rolling_wr:.1%} | "
                    f"loss={stats['loss_mean']:.4f} | "
                    f"invalid={stats['invalid_actions']}"
                )

            # Save checkpoint every 500 episodes
            if ep % 500 == 0:
                ckpt_path = checkpoint_dir / f"stage_{sname}_ep{global_episode}"
                agent.save(str(ckpt_path))

            # Save best checkpoint
            if len(win_rate_window) >= advance_win:
                rolling_wr = float(np.mean(win_rate_window))
                if rolling_wr > best_win_rate:
                    best_win_rate = rolling_wr
                    agent.save(str(checkpoint_dir / f"best_{sname}"))

            # ── Graceful shutdown: save full state and exit ────────────────
            if _shutdown_requested:
                logger.info("Shutdown requested — saving checkpoint...")
                _save_training_checkpoint(
                    agent=agent,
                    checkpoint_dir=checkpoint_dir,
                    global_episode=global_episode,
                    current_stage=sname,
                    stage_episode=ep,
                    completed_stages=[
                        s for s, h in stage_history.items() if h.get("completed")
                    ],
                )
                shutdown_clean = True
                break

            # Auto-advance check (skip for forced single stage or final stage)
            if (
                stage_name is None  # not forced to one stage
                and sname != "full"  # not already on final stage
                and len(win_rate_window) >= advance_win
            ):
                rolling_wr = float(np.mean(win_rate_window))
                if rolling_wr >= advance_wr:
                    logger.info(
                        f"Stage '{sname}' complete: "
                        f"rolling WR={rolling_wr:.1%} ≥ {advance_wr:.1%} "
                        f"over {advance_win} episodes. Advancing."
                    )
                    stage_history[sname]["completed"] = True
                    break

        # If shutdown was requested, break out of stage loop too
        if shutdown_clean:
            logger.info(
                "Training interrupted — checkpoint saved to checkpoints/latest/"
            )
            return all_history

        # Mark stage completed if it ran to full episode count
        stage_history[sname]["completed"] = True
        env.clear_curriculum_dates()

    # Final best checkpoint across all stages (only on normal completion)
    agent.save(str(checkpoint_dir / "best"))
    # Clean up checkpoints/latest/ — training is done, no need to resume
    latest_dir = checkpoint_dir / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
        logger.info("Cleaned up checkpoints/latest/ (training complete)")

    # Save training history to JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = log_dir / f"training_history_{ts}.json"
    with open(history_file, "w") as f:
        json.dump(all_history, f, indent=2)
    logger.info(f"Training history saved → {history_file}")

    logger.info(f"Training complete. Best checkpoint → {checkpoint_dir / 'best'}")
    return all_history


def _save_training_checkpoint(
    agent: IQNAgent,
    checkpoint_dir: Path,
    global_episode: int,
    current_stage: str,
    stage_episode: int,
    completed_stages: List[str],
) -> None:
    """Save weights + training state + replay buffer to checkpoints/latest/ for seamless resume."""
    latest_dir = checkpoint_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    # Save network weights
    agent.save(str(latest_dir))

    # Save replay buffer
    agent.replay.save(str(latest_dir / "replay_buffer.npz"))

    # Save training state
    training_state = {
        "global_episode": global_episode,
        "current_stage": current_stage,
        "stage_episode": stage_episode,
        "completed_stages": completed_stages,
        "agent_state": agent.save_training_state(),
        "saved_at": datetime.now().isoformat(),
    }
    state_file = latest_dir / "training_state.json"
    with open(state_file, "w") as f:
        json.dump(training_state, f, indent=2)

    logger.info(
        f"Checkpoint saved → {latest_dir} "
        f"(ep={global_episode}, stage={current_stage}, stage_ep={stage_episode})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward evaluation
# ══════════════════════════════════════════════════════════════════════════════


def run_walk_forward(
    env: "TradingEnvironment",
    agent: IQNAgent,
    config: dict,
    all_dates: List[str],
) -> List[Dict]:
    """
    Walk-forward validation:
      Train on 12 months → evaluate on next 1 month → slide forward.

    Evaluation is read-only (no weight updates, deterministic actions).
    Uses agent weights as they are — call this after full training.
    """
    wf_cfg = config.get("evaluation", {}).get("walk_forward", {})
    train_months = wf_cfg.get("train_window_months", 12)
    test_months = wf_cfg.get("test_window_months", 1)
    step_months = wf_cfg.get("step_months", 1)
    min_folds = wf_cfg.get("min_folds", 6)

    # Sort dates and group by month
    sorted_dates = sorted(all_dates)
    months: Dict[str, List[str]] = {}
    for d in sorted_dates:
        ym = d[:7]  # "YYYY-MM"
        months.setdefault(ym, []).append(d)
    month_keys = sorted(months.keys())

    if len(month_keys) < train_months + test_months:
        logger.warning(
            f"Not enough months for walk-forward: "
            f"have {len(month_keys)}, need {train_months + test_months}"
        )
        return []

    folds = []
    fold_idx = 0
    start_idx = 0

    while start_idx + train_months + test_months <= len(month_keys):
        # ── Check for shutdown ──────────────────────────────────────────────
        if _shutdown_requested:
            logger.info(
                "Walk-forward interrupted by shutdown — returning partial results"
            )
            break
        test_start = start_idx + train_months
        test_end = test_start + test_months

        test_month_keys = month_keys[test_start:test_end]
        test_dates = []
        for mk in test_month_keys:
            test_dates.extend(months[mk])

        if not test_dates:
            start_idx += step_months
            continue

        fold_idx += 1
        logger.info(
            f"Walk-Forward Fold {fold_idx}: "
            f"test months = {test_month_keys[0]} → {test_month_keys[-1]} "
            f"({len(test_dates)} dates)"
        )

        env.set_curriculum_dates(test_dates)

        fold_pnls = []
        fold_trades = []
        fold_wins = []
        fold_invalids = []

        for date in test_dates:
            if _shutdown_requested:
                break
            stats = run_episode(env, agent, train=False)
            fold_pnls.append(stats["episode_pnl"])
            fold_trades.append(stats["trade_count"])
            fold_wins.append(stats["win_rate"])
            fold_invalids.append(stats["invalid_actions"])

        pnls_arr = np.array(fold_pnls)
        sharpe = (
            float(np.mean(pnls_arr) / (np.std(pnls_arr) + 1e-8))
            if len(pnls_arr) > 1
            else 0.0
        )
        wins_above = float(np.sum(pnls_arr > 0))
        losses_below = float(np.sum(pnls_arr <= 0))
        pf = float(np.sum(pnls_arr[pnls_arr > 0])) / (
            abs(float(np.sum(pnls_arr[pnls_arr < 0]))) + 1e-8
        )

        fold_result = {
            "fold": fold_idx,
            "test_months": test_month_keys,
            "num_days": len(test_dates),
            "avg_daily_pnl": float(np.mean(pnls_arr)),
            "sharpe_ratio": sharpe,
            "profit_factor": pf,
            "win_rate": float(np.mean(fold_wins)),
            "profitable_days": int(wins_above),
            "losing_days": int(losses_below),
            "avg_daily_trades": float(np.mean(fold_trades)),
            "avg_invalid_actions": float(np.mean(fold_invalids)),
            "max_daily_loss": float(np.min(pnls_arr)),
            "max_daily_profit": float(np.max(pnls_arr)),
        }
        folds.append(fold_result)

        logger.info(
            f"  Fold {fold_idx} results: "
            f"Sharpe={sharpe:.2f} | "
            f"PF={pf:.2f} | "
            f"WR={fold_result['win_rate']:.1%} | "
            f"avg PnL=${fold_result['avg_daily_pnl']:+.2f} | "
            f"invalid={fold_result['avg_invalid_actions']:.1f}"
        )

        start_idx += step_months

    env.clear_curriculum_dates()

    if folds:
        avg_sharpe = float(np.mean([f["sharpe_ratio"] for f in folds]))
        avg_pf = float(np.mean([f["profit_factor"] for f in folds]))
        avg_wr = float(np.mean([f["win_rate"] for f in folds]))
        avg_pnl = float(np.mean([f["avg_daily_pnl"] for f in folds]))
        avg_inv = float(np.mean([f["avg_invalid_actions"] for f in folds]))
        logger.info(
            f"\n{'═' * 60}\n"
            f"  WALK-FORWARD SUMMARY ({len(folds)} folds)\n"
            f"  Avg Sharpe:      {avg_sharpe:.2f}\n"
            f"  Avg Profit Factor: {avg_pf:.2f}\n"
            f"  Avg Win Rate:    {avg_wr:.1%}\n"
            f"  Avg Daily PnL:   ${avg_pnl:+.2f}\n"
            f"  Avg Invalid Acts: {avg_inv:.1f} per day\n"
            f"{'═' * 60}"
        )

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    args = parse_args()

    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger_main = setup_logging(log_dir, args.verbose)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine total episodes
    cur_cfg = config.get("curriculum", {})
    default_total = sum(s.get("episodes", 2000) for s in cur_cfg.get("stages", []))
    total_episodes = args.episodes or default_total

    logger.info(
        f"\n{'═' * 60}\n"
        f"  ASYMPTOTIC ZERO — IQN Training v2\n"
        f"  Config:     {args.config}\n"
        f"  Data dir:   {args.data_dir}\n"
        f"  Episodes:   {total_episodes}\n"
        f"  Curriculum: {'disabled' if args.no_curriculum else 'enabled'}\n"
        f"  Walk-fwd:   {'yes' if args.walk_forward else 'no'}\n"
        f"{'═' * 60}"
    )

    # Build environment
    env = make_env(config_path=args.config, data_directory=args.data_dir)

    # Build agent
    agent = IQNAgent(
        state_dim=env.get_state_space_size(),
        action_size=env.get_action_space_size(),
        config_path=args.config,
    )

    # Resume from checkpoint if specified
    resume_state = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            agent.load(str(resume_path))
            logger.info(f"Loaded weights from {resume_path}")

            # Load replay buffer if available
            buffer_file = resume_path / "replay_buffer.npz"
            if buffer_file.exists():
                agent.replay.load(str(buffer_file))
            else:
                logger.info("No replay buffer found — starting with empty buffer")

            # Load full training state if available (for true resume)
            state_file = resume_path / "training_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    resume_state = json.load(f)
                # Restore agent internal state (epsilon, steps, etc.)
                agent_state = resume_state.get("agent_state", {})
                if agent_state:
                    agent.load_training_state(agent_state)
                logger.info(
                    f"Resuming from ep {resume_state.get('global_episode', '?')}, "
                    f"stage={resume_state.get('current_stage', '?')}, "
                    f"ε={agent.epsilon:.4f}"
                )
            else:
                logger.info(
                    "No training_state.json found — loading weights only (fresh curriculum)"
                )
        else:
            logger.warning(f"Resume path not found: {resume_path}")

    # Compute regime labels (cached after first run)
    regime_cache = Path(
        config.get("data", {}).get("regime_cache", "data/regime_labels.parquet")
    )
    date_labels = label_regimes(
        top_movers_file=Path(args.data_dir) / "top_movers.parquet",
        data_dir=Path(args.data_dir),
        config=config,
        cache_path=regime_cache,
    )

    if args.no_curriculum:
        config["no_curriculum"] = True

    # ── Walk-forward evaluation only ──────────────────────────────────────────
    if args.walk_forward:
        if not args.resume:
            logger.warning(
                "--walk-forward without --resume uses untrained weights. "
                "Pass --resume checkpoints/best for meaningful results."
            )
        folds = run_walk_forward(
            env=env,
            agent=agent,
            config=config,
            all_dates=env.available_dates,
        )
        out = log_dir / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w") as f:
            json.dump(folds, f, indent=2)
        logger.info(f"Walk-forward results saved → {out}")
        return

    # ── Full training ─────────────────────────────────────────────────────────
    start_time = time.time()
    history = run_curriculum(
        env=env,
        agent=agent,
        config=config,
        date_labels=date_labels,
        stage_name=args.stage,
        total_episodes=total_episodes,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        dashboard=not args.no_dashboard,
        resume_state=resume_state,
    )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 3600:.1f}h | {len(history)} episodes")

    # Save training history
    hist_out = (
        log_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(hist_out, "w") as f:
        # Convert deques / numpy types to native Python for JSON
        serializable = [
            {
                k: (v if isinstance(v, (int, float, str, dict)) else float(v))
                for k, v in ep.items()
            }
            for ep in history
        ]
        json.dump(serializable, f, indent=2)
    logger.info(f"Training history → {hist_out}")

    # ── Walk-forward after training (opt-in only) ───────────────────────────────
    if _shutdown_requested:
        logger.info("Skipping walk-forward (shutdown requested)")
    elif not args.walk_forward:
        logger.info("Walk-forward skipped (pass --walk-forward to run after training)")
    else:
        logger.info("Running post-training walk-forward evaluation...")
        folds = run_walk_forward(
            env=env,
            agent=agent,
            config=config,
            all_dates=env.available_dates,
        )
        wf_out = (
            log_dir
            / f"walk_forward_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(wf_out, "w") as f:
            json.dump(folds, f, indent=2)
        logger.info(f"Walk-forward results → {wf_out}")

        # Patch walk-forward folds into live_metrics.json so monitor shows them
        try:
            metrics_path = log_dir / "live_metrics.json"
            if metrics_path.exists() and folds:
                with open(metrics_path) as f:
                    existing = json.load(f)
                existing["walk_forward_folds"] = folds
                existing["last_updated"] = datetime.now().isoformat()
                tmp = metrics_path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(existing, f)
                tmp.replace(metrics_path)
                logger.info("Walk-forward folds patched into live_metrics.json")
        except Exception as e:
            logger.debug(f"walk-forward patch to live_metrics failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
