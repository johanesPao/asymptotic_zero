"""
Trading Logger — fire-and-forget database persistence for Asymptotic Zero.

Design principles:
  1. Never crash the trading bot.  All writes are async (queue → background thread).
  2. Survive JSONB segfaults.  Every dict is round-tripped through json.dumps/loads
     before touching SQLAlchemy, converting custom objects to plain Python dicts.
  3. Fail gracefully.  If a write fails, it is logged to file and dropped —
     the bot keeps trading.
  4. Zero blocking.  The trading loop calls logger.* methods and returns instantly.
     The background thread does all the actual SQL work.

Usage (in trading_bot.py):
    logger = TradingLogger(db_url)
    session_id = logger.open_session(mode, initial_balance, screened_coins)
    trade_id   = logger.log_trade_open(session_id, symbol, side, entry_price, ...)
    logger.log_trade_close(trade_id, exit_price, pnl)
    logger.close_session(session_id, final_balance, ...)
    logger.shutdown()   # call on bot exit — waits up to 5s to flush queue
"""

import json
import logging
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from src.database.models import TradingSession, Trade, SystemLog, Base
from src.utils.log_setup import get_logger

# File logger for DB errors — daily rotation, 30-day retention
Path("logs").mkdir(exist_ok=True)
_file_log = get_logger("db")


def _safe_json(data: Any) -> Optional[Any]:
    """
    Round-trip data through json.dumps/loads.

    This converts any custom Python objects (numpy types, Polars types, etc.)
    into plain dicts/lists/scalars that psycopg2 can safely serialize to JSONB.
    Returns None if serialization fails completely.
    """
    if data is None:
        return None
    try:
        return json.loads(json.dumps(data, default=str))
    except Exception as e:
        _file_log.error(f"_safe_json serialization failed: {e} | data type: {type(data)}")
        return None


class TradingLogger:
    """
    Async database logger for live trading activity.

    All public methods are non-blocking — they put work items on an internal
    queue and return immediately.  A single background daemon thread drains
    the queue and executes SQL.
    """

    def __init__(self, db_url: str):
        self._db_url = db_url
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._engine = None
        self._Session = None

        # Pre-warm libpq/OpenSSL in the MAIN thread before any background
        # threads or Binance HTTPS calls start.  libpq links against OpenSSL
        # and must finish its one-time initialisation single-threadedly.
        # If the background writer thread does this first connect() at the
        # same time as the main thread opens an HTTPS socket, both sides race
        # to initialise OpenSSL → heap corruption → free(): invalid pointer.
        self._warmup_connection()

        # Start background writer thread
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="db-logger",
            daemon=True,
        )
        self._thread.start()

    # ─── Public API (non-blocking) ─────────────────────────────────────

    def open_session(
        self,
        mode: str,
        initial_balance: float,
        screened_coins: Dict,
    ) -> str:
        """
        Create a new TradingSession row.

        Returns a client-side session_key (datetime string) that can be
        passed to subsequent calls.  The actual DB id is looked up by the
        background thread when it processes the event.
        """
        session_key = datetime.utcnow().isoformat()
        self._enqueue("open_session", {
            "session_key": session_key,
            "mode": mode,
            "initial_balance": initial_balance,
            "screened_coins": screened_coins,
            "screening_time": datetime.utcnow(),
        })
        return session_key

    def log_trade_open(
        self,
        session_key: str,
        symbol: str,
        side: str,
        action_id: int,
        entry_price: float,
        quantity: float,
        portfolio_value: float,
        open_positions: int,
        avg_q: float = 0.0,
        coin_rank: int = 0,
        is_gainer: bool = True,
    ) -> str:
        """
        Insert a Trade row for a newly opened position.

        Returns a trade_key (timestamp string) for later update on close.
        """
        trade_key = f"{symbol}_{datetime.utcnow().isoformat()}"
        self._enqueue("trade_open", {
            "trade_key": trade_key,
            "session_key": session_key,
            "symbol": symbol,
            "side": side,
            "action_id": action_id,
            "entry_price": entry_price,
            "quantity": quantity,
            "portfolio_value": portfolio_value,
            "open_positions": open_positions,
            "avg_q": avg_q,
            "coin_rank": coin_rank,
            "is_gainer": is_gainer,
            "timestamp": datetime.utcnow(),
        })
        return trade_key

    def log_trade_close(
        self,
        trade_key: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float = 0.0,
    ):
        """Update an existing Trade row with close details."""
        self._enqueue("trade_close", {
            "trade_key": trade_key,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "closed_at": datetime.utcnow(),
        })

    def close_session(
        self,
        session_key: str,
        final_balance: float,
        total_pnl: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        daily_limit_hit: bool = False,
        stop_reason: str = "",
    ):
        """Update TradingSession with end-of-day summary."""
        self._enqueue("close_session", {
            "session_key": session_key,
            "final_balance": final_balance,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "daily_limit_hit": daily_limit_hit,
            "stop_reason": stop_reason,
            "trading_end_time": datetime.utcnow(),
        })

    def log_event(
        self,
        level: str,
        category: str,
        message: str,
        session_key: str = "",
        exception: str = "",
        extra_data: Dict = None,
    ):
        """Insert a SystemLog row (INFO / WARNING / ERROR / CRITICAL)."""
        self._enqueue("system_log", {
            "level": level,
            "category": category,
            "message": message,
            "session_key": session_key,
            "exception": exception,
            "extra_data": extra_data or {},
            "timestamp": datetime.utcnow(),
        })

    def shutdown(self, timeout: float = 5.0):
        """
        Graceful shutdown — wait up to `timeout` seconds for the queue to drain,
        then stop the background thread.
        """
        self._stop_event.set()
        self._thread.join(timeout=timeout)

    # ─── Internal ──────────────────────────────────────────────────────

    def _enqueue(self, event_type: str, payload: Dict):
        """Put a work item on the queue (never blocks — queue is unbounded)."""
        self._queue.put_nowait((event_type, payload))

    def _warmup_connection(self):
        """Force libpq/OpenSSL to initialise in the main thread.

        Makes one real TCP connection to PostgreSQL and immediately closes it.
        This is called from __init__ (main thread) before the background writer
        thread starts and before any Binance HTTPS calls happen.  After this,
        OpenSSL's one-time global initialisation is complete and thread-safe.
        """
        try:
            engine = create_engine(
                self._db_url,
                poolclass=NullPool,
                echo=False,
                connect_args={"connect_timeout": 10},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            _file_log.info("DB warmup connection OK — libpq/OpenSSL initialised")
        except Exception as e:
            # Warmup failure is non-fatal: the background thread will retry.
            # We still get the single-threaded init benefit for subsequent calls.
            _file_log.warning(f"DB warmup failed (non-fatal): {e}")

    def _get_session(self):
        """Lazily create the SQLAlchemy engine + session on first use in the thread.

        NullPool is critical here.  SQLAlchemy's default QueuePool calls
        psycopg2.connect() at unpredictable times (health pings, replenishment)
        from the background thread.  libpq uses OpenSSL internally, and if that
        first connect() races with the main thread's active Binance HTTPS socket
        (also OpenSSL), both sides try to initialise OpenSSL simultaneously,
        corrupting the allocator → free(): invalid pointer.

        The real fix is _warmup_connection() called in __init__ from the MAIN
        thread before any background thread or HTTPS call starts.  That forces
        libpq/OpenSSL to fully initialise once, single-threaded.  After that,
        modern OpenSSL is thread-safe.  NullPool is kept as a secondary defence:
        it ensures no surprise connect() calls from the background thread.
        """
        if self._Session is None:
            self._engine = create_engine(
                self._db_url,
                poolclass=NullPool,   # no pool → no surprise connect() calls
                echo=False,
                connect_args={"connect_timeout": 10},
            )
            self._Session = sessionmaker(bind=self._engine)
        return self._Session()

    def _writer_loop(self):
        """
        Background thread: drain the queue and write to PostgreSQL.
        Runs until _stop_event is set AND the queue is empty.
        """
        # In-memory maps from client-side keys → DB integer ids
        # (session_key str → TradingSession.id)
        # (trade_key str   → Trade.id)
        session_id_map: Dict[str, int] = {}
        trade_id_map:   Dict[str, int] = {}

        while True:
            try:
                event_type, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            try:
                db = self._get_session()
                try:
                    if event_type == "open_session":
                        sid = self._write_open_session(db, payload)
                        if sid:
                            session_id_map[payload["session_key"]] = sid

                    elif event_type == "trade_open":
                        sid = session_id_map.get(payload["session_key"])
                        tid = self._write_trade_open(db, payload, sid)
                        if tid:
                            trade_id_map[payload["trade_key"]] = tid

                    elif event_type == "trade_close":
                        tid = trade_id_map.get(payload["trade_key"])
                        if tid:
                            self._write_trade_close(db, payload, tid)

                    elif event_type == "close_session":
                        sid = session_id_map.get(payload["session_key"])
                        if sid:
                            self._write_close_session(db, payload, sid)

                    elif event_type == "system_log":
                        sid = session_id_map.get(payload.get("session_key", ""))
                        self._write_system_log(db, payload, sid)

                except Exception as e:
                    db.rollback()
                    _file_log.error(
                        f"DB write failed [{event_type}]: {e}\n{traceback.format_exc()}"
                    )
                finally:
                    db.close()

            except Exception as e:
                _file_log.error(f"DB writer loop error: {e}\n{traceback.format_exc()}")

            finally:
                self._queue.task_done()

    # ─── SQL Writers (run inside background thread) ────────────────────

    def _write_open_session(self, db, p: Dict) -> Optional[int]:
        """INSERT into trading_sessions. Returns the new row id."""
        session = TradingSession(
            date=p["screening_time"],
            mode=p["mode"],
            initial_balance=p["initial_balance"],
            screening_time=p["screening_time"],
            trading_start_time=p["screening_time"],
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            emergency_stop=False,
            daily_limit_hit=False,
        )

        # JSONB write — isolated so failure here doesn't lose the whole row
        try:
            session.screened_coins = _safe_json(p["screened_coins"])
        except Exception as e:
            _file_log.error(f"screened_coins JSONB write failed (skipping): {e}")
            session.screened_coins = None

        db.add(session)
        db.commit()
        db.refresh(session)
        return session.id

    def _write_trade_open(self, db, p: Dict, session_id: Optional[int]) -> Optional[int]:
        """INSERT into trades for a newly opened position. Returns the new row id."""
        trade = Trade(
            session_id=session_id,
            timestamp=p["timestamp"],
            symbol=p["symbol"],
            coin_rank=p["coin_rank"],
            is_gainer=p["is_gainer"],
            action_type="open",
            action_id=p["action_id"],
            side=p["side"],
            entry_price=p["entry_price"],
            quantity=p["quantity"],
            agent_q_value=p["avg_q"],
            portfolio_value=p["portfolio_value"],
            open_positions=p["open_positions"],
            pnl=0.0,
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return trade.id

    def _write_trade_close(self, db, p: Dict, trade_id: int):
        """UPDATE trades row with close/PnL data."""
        trade = db.query(Trade).filter(Trade.id == trade_id).first()
        if trade is None:
            _file_log.error(f"trade_close: Trade id={trade_id} not found")
            return
        trade.exit_price = p["exit_price"]
        trade.pnl = p["pnl"]
        trade.pnl_percent = p["pnl_percent"]
        trade.action_type = "close"
        db.commit()

    def _write_close_session(self, db, p: Dict, session_id: int):
        """UPDATE trading_sessions with end-of-day summary."""
        session = db.query(TradingSession).filter(TradingSession.id == session_id).first()
        if session is None:
            _file_log.error(f"close_session: Session id={session_id} not found")
            return
        session.final_balance = p["final_balance"]
        session.total_pnl = p["total_pnl"]
        session.total_trades = p["total_trades"]
        session.winning_trades = p["winning_trades"]
        session.losing_trades = p["losing_trades"]
        session.trading_end_time = p["trading_end_time"]
        session.daily_limit_hit = p["daily_limit_hit"]
        session.stop_reason = p["stop_reason"]
        db.commit()

    def _write_system_log(self, db, p: Dict, session_id: Optional[int]):
        """INSERT into system_logs."""
        log = SystemLog(
            timestamp=p["timestamp"],
            level=p["level"],
            category=p["category"],
            message=p["message"],
            session_id=session_id,
            exception=p.get("exception", ""),
        )
        # JSONB write — isolated
        try:
            log.extra_data = _safe_json(p.get("extra_data")) or {}
        except Exception as e:
            _file_log.error(f"extra_data JSONB write failed (skipping): {e}")
            log.extra_data = None

        db.add(log)
        db.commit()
