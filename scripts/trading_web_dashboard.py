"""
Asymptotic Zero - Trading Web Dashboard

Web-based dashboard for live trading bot monitoring.
Serves real-time trading data via web interface for Tailscale network access.

Usage:
    # Start trading bot (in terminal 1)
    python scripts/trading_bot.py
    
    # Start web dashboard (in terminal 2)  
    python scripts/trading_web_dashboard.py
    # Access via: http://localhost:8766 or Tailscale IP

Dependencies:
    pip install fastapi uvicorn websockets
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Live Binance poller
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.config.secrets import get_secret
from binance.client import Client as BinanceClient

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_PORT     = 8585

app = FastAPI(title="Asymptotic Zero Trading Dashboard", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Live Binance State ────────────────────────────────────────────────────────
_binance_client: BinanceClient | None = None
_live_state: dict = {
    "positions": [],
    "balance_usdt": 0.0,
    "last_poll": None,
    "poll_error": None,
}

# ── Database (session history) ────────────────────────────────────────────────
_db_engine = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        """Broadcast data to all connected clients."""
        if self.active_connections:
            message = json.dumps(data)
            # Send to all connections and remove failed ones
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    disconnected.append(connection)
            
            # Remove failed connections
            for conn in disconnected:
                self.disconnect(conn)

manager = ConnectionManager()

# ── Live Binance Poller ───────────────────────────────────────────────────────

async def _fetch_binance_state() -> None:
    """Fetch positions and balance from Binance and update _live_state."""
    global _live_state
    if _binance_client is None:
        return
    try:
        raw_positions, account = await asyncio.gather(
            asyncio.to_thread(_binance_client.futures_position_information),
            asyncio.to_thread(_binance_client.futures_account),
        )
        positions = []
        for p in raw_positions:
            amt = float(p.get("positionAmt", 0))
            if abs(amt) == 0:
                continue
            positions.append({
                "symbol": p["symbol"],
                "side": "LONG" if amt > 0 else "SHORT",
                "size": abs(amt),
                "entry_price": float(p.get("entryPrice", 0)),
                "mark_price": float(p.get("markPrice", 0)),
                "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
                "leverage": int(p.get("leverage", 1)),
            })
        # totalMarginBalance = wallet + unrealized (true equity)
        balance_usdt = float(account.get("totalMarginBalance", 0) or
                             account.get("totalWalletBalance", 0))
        _live_state = {
            "positions": positions,
            "balance_usdt": balance_usdt,
            "last_poll": datetime.now().isoformat(),
            "poll_error": None,
        }
    except Exception as e:
        _live_state["poll_error"] = str(e)
        _live_state["last_poll"] = datetime.now().isoformat()


async def _poll_binance_loop() -> None:
    """Background task: poll Binance every 2 seconds."""
    while True:
        await _fetch_binance_state()
        await asyncio.sleep(2)


@app.on_event("startup")
async def _startup() -> None:
    """Initialize Binance client and start background poller."""
    global _binance_client
    try:
        import time
        api_key = get_secret("BINANCE_API_KEY")
        api_secret = get_secret("BINANCE_SECRET")
        trading_mode = get_secret("TRADING_MODE", "testnet")
        _binance_client = BinanceClient(
            api_key, api_secret, testnet=(trading_mode == "testnet")
        )
        server_time = await asyncio.to_thread(_binance_client.get_server_time)
        local_time = int(time.time() * 1000)
        _binance_client.timestamp_offset = server_time["serverTime"] - local_time
        asyncio.create_task(_poll_binance_loop())
        print("✅ Binance poller started (2s interval)", flush=True)
    except Exception as e:
        print(f"⚠️  Binance poller failed to start: {e}", flush=True)

    global _db_engine
    try:
        from sqlalchemy import create_engine
        db_url = get_secret("DATABASE_URL")
        _db_engine = create_engine(db_url, pool_pre_ping=True, pool_size=2, max_overflow=2)
        # Verify connection
        with _db_engine.connect():
            pass
        print("✅ DB connected for session history", flush=True)
    except Exception as e:
        print(f"⚠️  DB connection failed (session history unavailable): {e}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_session_pnl() -> list:
    """Query DB for historical per-session PnL."""
    if _db_engine is None:
        return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT date, total_pnl FROM sessions "
                "WHERE total_pnl IS NOT NULL ORDER BY date ASC"
            )).fetchall()
        return [
            {
                "date": (row[0].strftime("%Y-%m-%d")
                         if hasattr(row[0], "strftime") else str(row[0])[:10]),
                "pnl": float(row[1]),
                "type": "realized",
            }
            for row in rows
        ]
    except Exception:
        return []


def _load_heartbeat() -> Optional[dict]:
    """Load bot_heartbeat row from DB. Returns None if unavailable."""
    if _db_engine is None:
        return None
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            row = conn.execute(text(
                "SELECT status, session_started, last_updated, agent_status, "
                "epsilon, current_step, last_action, avg_q, "
                "current_balance, initial_balance, trade_count, "
                "winning_trades, losing_trades, guardrail_status, "
                "coin_table, screened_coins, trading_mode, session_id "
                "FROM bot_heartbeat WHERE bot_id = 'main'"
            )).fetchone()
        if row is None:
            return None
        return {
            "status": row[0],
            "session_started": row[1].isoformat() if row[1] else None,
            "last_updated": row[2].isoformat() if row[2] else None,
            "agent_status": row[3],
            "epsilon": row[4],
            "current_step": row[5],
            "last_action": row[6],
            "avg_q": row[7],
            "current_balance": row[8],
            "initial_balance": row[9],
            "trade_count": row[10],
            "winning_trades": row[11] or 0,
            "losing_trades": row[12] or 0,
            "guardrail_status": row[13] or {},
            "coin_table": row[14] or [],
            "screened_coins": row[15],
            "trading_mode": row[16],
            "session_id": row[17],
        }
    except Exception:
        return None


def _load_activity_log(session_id: int) -> list:
    """Load recent activity log entries for a session."""
    if _db_engine is None or session_id is None:
        return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT timestamp, message FROM activity_log "
                "WHERE session_id = :sid ORDER BY timestamp DESC LIMIT 50"
            ), {"sid": session_id}).fetchall()
        return [
            {
                "time": row[0].strftime("%H:%M:%S") if row[0] else "",
                "message": row[1],
            }
            for row in rows
        ]
    except Exception:
        return []


def _load_error_log(session_id: int) -> list:
    """Load recent system log entries (WARNING/ERROR) for a session."""
    if _db_engine is None or session_id is None:
        return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT timestamp, level, source, message FROM system_logs "
                "WHERE session_id = :sid ORDER BY timestamp DESC LIMIT 50"
            ), {"sid": session_id}).fetchall()
        return [
            {
                "time": row[0].strftime("%H:%M:%S") if row[0] else "",
                "level": row[1],
                "source": row[2],
                "message": row[3],
            }
            for row in rows
        ]
    except Exception:
        return []


def _load_pnl_history(session_id: int) -> list:
    """Load PnL snapshots for a session (seeds the PnL chart on page load)."""
    if _db_engine is None or session_id is None:
        return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT timestamp, pnl FROM pnl_snapshots "
                "WHERE session_id = :sid ORDER BY timestamp ASC"
            ), {"sid": session_id}).fetchall()
        return [
            {"t": row[0].isoformat() if row[0] else "", "pnl": float(row[1])}
            for row in rows
        ]
    except Exception:
        return []


def _load_recent_trades(session_id: int) -> list:
    """Load recent closed trades for a session."""
    if _db_engine is None or session_id is None:
        return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT timestamp, symbol, side, entry_price, exit_price, pnl, pnl_percent "
                "FROM trades WHERE session_id = :sid AND action_type = 'close' "
                "ORDER BY timestamp DESC LIMIT 20"
            ), {"sid": session_id}).fetchall()
        return [
            {
                "timestamp": row[0].isoformat() if row[0] else "",
                "symbol": row[1],
                "side": row[2],
                "entry_price": round(float(row[3] or 0), 6),
                "exit_price": round(float(row[4] or 0), 6),
                "pnl": round(float(row[5] or 0), 4),
                "pnl_pct": round(float(row[6] or 0), 2),
            }
            for row in rows
        ]
    except Exception:
        return []


def _load_trading_metrics() -> dict:
    """Load trading metrics from bot_heartbeat + activity/error logs, overlay live Binance data."""
    hb = _load_heartbeat()

    if hb is None or hb.get("status") == "offline" or hb.get("session_started") is None:
        data = {
            "status": hb["status"] if hb else "waiting",
            "message": "Waiting for trading bot to start...",
            "session_started": hb.get("session_started") if hb else None,
            "current_balance": 0,
            "initial_balance": 0,
            "total_pnl": 0,
            "today_pnl": 0,
            "win_rate": 0,
            "trade_count": 0,
            "current_positions": [],
            "recent_trades": [],
            "agent_status": hb.get("agent_status", "offline") if hb else "offline",
            "guardrail_status": {},
            "last_updated": hb.get("last_updated") if hb else None,
            "activity_log": [],
            "error_log": [],
            "pnl_history": [],
            "coin_table": [],
            "screened_coins": None,
            "trading_mode": hb.get("trading_mode") if hb else None,
        }
    else:
        session_id = hb.get("session_id")
        winning = hb.get("winning_trades", 0)
        losing = hb.get("losing_trades", 0)
        total = winning + losing

        data = {
            "status": hb["status"],
            "session_started": hb["session_started"],
            "current_balance": hb.get("current_balance", 0),
            "initial_balance": hb.get("initial_balance", 0),
            "total_pnl": (hb.get("current_balance") or 0) - (hb.get("initial_balance") or 0),
            "today_pnl": 0,
            "win_rate": winning / total if total > 0 else 0.0,
            "trade_count": hb.get("trade_count", 0),
            "current_positions": [],
            "recent_trades": _load_recent_trades(session_id),
            "agent_status": hb.get("agent_status", "active"),
            "epsilon": hb.get("epsilon", 0),
            "current_step": hb.get("current_step", 0),
            "last_action": hb.get("last_action", ""),
            "avg_q": hb.get("avg_q", 0),
            "screened_coins": hb.get("screened_coins"),
            "guardrail_status": hb.get("guardrail_status", {}),
            "activity_log": _load_activity_log(session_id),
            "error_log": _load_error_log(session_id),
            "pnl_history": _load_pnl_history(session_id),
            "coin_table": hb.get("coin_table", []),
            "trading_mode": hb.get("trading_mode"),
            "last_updated": hb.get("last_updated"),
        }

    # Overlay live Binance data if available
    if _live_state["last_poll"] is not None:
        data["current_positions"] = _live_state["positions"]
        live_balance = _live_state["balance_usdt"]

        data["current_balance"] = live_balance
        if data.get("session_started"):
            initial = data.get("initial_balance") or 0
            if initial > 0:
                pnl = live_balance - initial
                data["total_pnl"] = pnl
                data["today_pnl"] = pnl  # same as session PnL for now
                if "guardrail_status" in data and isinstance(data["guardrail_status"], dict):
                    data["guardrail_status"]["daily_pnl"] = pnl

        data["last_updated"] = _live_state["last_poll"]
        if _live_state["poll_error"]:
            data["poll_error"] = _live_state["poll_error"]

    return data

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML page."""
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Asymptotic Zero - Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg: #0a0c10;
            --surface: #13161e;
            --surface2: #1a1e2a;
            --border: #232838;
            --text: #d4dae8;
            --muted: #616b82;
            --accent: #4f8ef7;
            --green: #3ecf8e;
            --red: #f05252;
            --yellow: #f0b429;
            --radius: 10px;
            --shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            padding: 14px 24px;
            border-bottom: 1px solid var(--border);
            background: var(--surface);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 18px;
            font-weight: 600;
            color: var(--accent);
        }
        .header h1 .title-asymptotic { color: #ffffff; }
        .header h1 .title-zero       { color: var(--accent); }
        .header h1 .title-dash       { color: #ffffff; }
        .header h1 .title-mode-testnet { color: #f59e0b; }  /* amber */
        .header h1 .title-mode-live    { color: #22c55e; }  /* green */
        
        .status {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--muted);
        }
        
        .status-indicator.running { background: var(--green); }
        .status-indicator.error { background: var(--red); }
        .status-indicator.waiting { background: var(--yellow); }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            padding: 16px;
        }
        
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            box-shadow: var(--shadow);
            min-width: 0;        /* prevent grid blowout on narrow screens */
        }
        
        .card h2 {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text);
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
        }
        
        .metric {
            background: var(--surface2);
            padding: 12px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
        }
        
        .metric-label {
            font-size: 12px;
            color: var(--muted);
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 18px;
            font-weight: 600;
            color: var(--text);
        }
        
        .metric-value.positive { color: var(--green); }
        .metric-value.negative { color: var(--red); }
        .pct-badge {
            font-size: 12px;
            font-weight: 500;
            opacity: 0.85;
        }
        
        .positions-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .position-item {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr auto;
            gap: 8px;
            align-items: center;
            padding: 10px 12px;
            background: var(--surface2);
            border-radius: var(--radius-sm);
            margin-bottom: 6px;
            border: 1px solid var(--border);
        }

        .position-symbol {
            font-weight: 600;
            color: var(--text);
        }

        .position-side { font-size: 12px; }
        .position-side.long { color: var(--green); }
        .position-side.short { color: var(--red); }

        .position-prices {
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.6;
        }
        .position-prices span { color: var(--text); font-weight: 500; }

        .position-amounts {
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.6;
            text-align: right;
        }
        .position-amounts span { color: var(--text); font-weight: 500; }

        .position-pnl {
            font-weight: 600;
            font-size: 13px;
            text-align: right;
            min-width: 60px;
        }

        .position-pnl.positive { color: var(--green); }
        .position-pnl.negative { color: var(--red); }
        
        .trades-list {
            max-height: 250px;
            overflow-y: auto;
        }
        
        .trade-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: var(--surface2);
            border-radius: var(--radius-sm);
            margin-bottom: 6px;
            border: 1px solid var(--border);
        }
        
        .trade-time {
            font-size: 11px;
            color: var(--muted);
        }
        
        .trade-symbol {
            font-weight: 600;
            color: var(--text);
        }
        
        .trade-pnl {
            font-weight: 600;
            font-size: 12px;
        }
        
        .chart-container {
            height: 300px;
            position: relative;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100px;
            color: var(--muted);
        }
        
        .error {
            color: var(--red);
            text-align: center;
            padding: 20px;
        }
        
        .timestamp {
            font-size: 11px;
            color: var(--muted);
            text-align: right;
            margin-top: 12px;
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
                padding: 10px;
                gap: 10px;
            }
            .card { padding: 14px; }

            /* Header: wrap title + status on small screens */
            .header {
                flex-wrap: wrap;
                gap: 6px;
                padding: 10px 14px;
            }
            .header h1 { font-size: 15px; }
            .status { font-size: 12px; flex-wrap: wrap; gap: 8px; }

            /* Coin table: allow horizontal scroll */
            .coin-table-wrapper {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }

            /* Position card: 2×2 grid — symbol/side top-left, PnL top-right,
               prices bottom-left, amounts bottom-right */
            .position-item {
                grid-template-columns: 1fr auto;
                grid-template-rows: auto auto;
                gap: 6px 10px;
            }
            /* DOM order: 1=symbol, 2=prices, 3=amounts, 4=pnl */
            .position-item > div:nth-child(1) { grid-column: 1; grid-row: 1; }
            .position-item > div:nth-child(2) { grid-column: 1; grid-row: 2; }
            .position-item > div:nth-child(3) { grid-column: 2; grid-row: 2; text-align: right; }
            .position-item > div:nth-child(4) { grid-column: 2; grid-row: 1; text-align: right; }
            .position-prices, .position-amounts { font-size: 10px; }
            .position-pnl { font-size: 12px; }

            /* Charts: shorter on mobile */
            .chart-container { height: 160px !important; }
        }

        /* ── Screened Coins Table ─────────────────────────────────────── */
        .coin-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            white-space: nowrap;
        }
        .coin-table th {
            color: var(--muted);
            text-align: right;
            padding: 5px 10px;
            border-bottom: 1px solid var(--border);
            font-weight: 500;
            font-size: 11px;
            background: var(--surface);
            position: sticky;
            top: 0;
            z-index: 2;
        }
        /* ── Frozen columns: Coin | 24h% | Close ─────────────────────────── */
        .coin-table th:nth-child(1) {
            text-align: left;
            position: sticky;
            left: 0;
            min-width: 72px;
            z-index: 3;
            background: var(--surface);
        }
        .coin-table th:nth-child(2) {
            position: sticky;
            left: 72px;
            min-width: 62px;
            z-index: 3;
            background: var(--surface);
        }
        .coin-table th:nth-child(3) {
            position: sticky;
            left: 134px;
            min-width: 72px;
            z-index: 3;
            background: var(--surface);
            border-right: 1px solid var(--border);
        }
        .coin-table td:nth-child(1) {
            text-align: left;
            position: sticky;
            left: 0;
            z-index: 1;
            background: var(--surface);
        }
        .coin-table td:nth-child(2) {
            position: sticky;
            left: 72px;
            z-index: 1;
            background: var(--surface);
        }
        .coin-table td:nth-child(3) {
            position: sticky;
            left: 134px;
            z-index: 1;
            background: var(--surface);
            border-right: 1px solid var(--border);
        }
        /* Slightly different bg on hover rows so sticky cells match */
        .coin-table tr:hover td:nth-child(1),
        .coin-table tr:hover td:nth-child(2),
        .coin-table tr:hover td:nth-child(3) { background: var(--surface2); }
        .coin-table td {
            padding: 5px 10px;
            text-align: right;
            border-bottom: 1px solid rgba(35,40,56,0.5);
        }

        .coin-table tr:hover td { background: var(--surface2); }
        .coin-table .text-left { text-align: left; }
        .coin-table .text-green { color: var(--green); }
        .coin-table .text-red { color: var(--red); }
        .coin-table .text-yellow { color: var(--yellow); }
        .coin-table .q-best { color: var(--accent); font-weight: 700; }
        .coin-table .dim { color: var(--muted); }
        @keyframes blink-up {
            0%   { background: rgba(62,207,142,0.45); }
            100% { background: transparent; }
        }
        @keyframes blink-down {
            0%   { background: rgba(240,82,82,0.45); }
            100% { background: transparent; }
        }
        .coin-table .blink-up   { animation: blink-up   1.2s ease-out forwards; }
        .coin-table .blink-down { animation: blink-down 1.2s ease-out forwards; }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span class="title-asymptotic">Asymptotic</span>
            <span class="title-zero"> Zero</span>
            <span class="title-dash"> — </span>
            <span id="title-mode" class="title-mode-testnet">Testnet Trading</span>
        </h1>
        <div class="status">
            <span id="last-updated" style="font-size:11px;color:var(--muted);margin-right:12px;">-</span>
            <span id="status-text">Connecting...</span>
            <div id="status-indicator" class="status-indicator"></div>
        </div>
    </div>
    
    <div class="dashboard">

        <!-- Screened Coins Table -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Screened Coins &mdash; Live TA &amp; Agent Q-Values</h2>
            <div class="coin-table-wrapper" style="overflow-x:auto;max-height:420px;overflow-y:auto;">
                <table class="coin-table">
                    <thead>
                        <tr>
                            <th style="text-align:left;">Coin</th>
                            <th>24h%</th>
                            <th>Close</th>
                            <th>RSI</th>
                            <th>MACD H</th>
                            <th>BB%B</th>
                            <th>Vol&times;</th>
                            <th>ATR</th>
                            <th>ADX</th>
                            <th>Q-HOLD</th>
                            <th>Q-LONG</th>
                            <th>Q-SHORT</th>
                            <th>Q-CLOSE</th>
                        </tr>
                    </thead>
                    <tbody id="coin-table-body">
                        <tr><td colspan="13" style="text-align:center;color:var(--muted);padding:20px;">Waiting for data...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Portfolio Overview -->
        <div class="card">
            <h2>Portfolio</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Balance</div>
                    <div id="balance" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Initial</div>
                    <div id="initial-balance" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Session PnL</div>
                    <div id="pnl" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Today's PnL</div>
                    <div id="today-pnl" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div id="win-rate" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Trades</div>
                    <div id="trade-count" class="metric-value">-</div>
                </div>
            </div>
        </div>
        
        <!-- PnL Chart -->
        <div class="card">
            <h2>Today's PnL</h2>
            <div class="chart-container">
                <canvas id="pnl-chart"></canvas>
            </div>
            <div id="chart-timestamp" class="timestamp">-</div>
        </div>

        <!-- Session PnL History -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Daily PnL History</h2>
            <div style="height:220px;position:relative;">
                <canvas id="session-pnl-chart"></canvas>
            </div>
            <div style="margin-top:8px;font-size:11px;color:var(--muted);">
                Solid bars = realized (past days) &nbsp;·&nbsp; Faded bar = today (realized + unrealized, live)
            </div>
        </div>

        <!-- Current Positions -->
        <div class="card">
            <h2>Positions</h2>
            <div id="positions-list" class="positions-list">
                <div class="loading">Loading positions...</div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="card">
            <h2>Recent Trades</h2>
            <div id="trades-list" class="trades-list">
                <div class="loading">Loading trades...</div>
            </div>
        </div>
        
        <!-- Agent Status -->
        <div class="card">
            <h2>Agent</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Status</div>
                    <div id="agent-status" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Epsilon</div>
                    <div id="epsilon" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Step</div>
                    <div id="current-step" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Q</div>
                    <div id="avg-q" class="metric-value">-</div>
                </div>
            </div>
            <div style="margin-top:12px;">
                <div class="metric-label">Last Decision</div>
                <div id="last-action" style="margin-top:4px;font-size:15px;font-weight:600;color:var(--accent);">-</div>
            </div>
            <div style="margin-top:10px;">
                <div class="metric-label">Screened Coins</div>
                <div id="screened-coins" style="margin-top:4px;font-size:12px;color:var(--muted);line-height:1.8;">-</div>
            </div>
        </div>
        
        <!-- Guardrails -->
        <div class="card">
            <h2>Guardrails</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Cooldown</div>
                    <div id="cooldown" class="metric-value">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Daily Trades</div>
                    <div id="daily-trades" class="metric-value">-</div>
                </div>
            </div>
        </div>

        <!-- Activity Log -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Activity Log</h2>
            <div id="activity-log" style="max-height:180px;overflow-y:auto;font-family:monospace;font-size:12px;">
                <div class="loading">Waiting for bot activity...</div>
            </div>
        </div>

        <!-- System Log (WARNING/ERROR from all components) -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2 style="color:var(--red, #f05252);">System Log</h2>
            <div id="error-log" style="max-height:200px;overflow-y:auto;font-family:monospace;font-size:11px;">
                <div class="loading">No warnings or errors</div>
            </div>
        </div>

    </div>
    
    <script>
        // WebSocket connection for real-time updates
        let ws;
        let pnlChart;
        let sessionPnlChart;
        let sessionStartMs = 0;

        function computeSessionStart() {
            const now = new Date();
            const today7AM = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 7, 0, 0, 0);
            sessionStartMs = now < today7AM
                ? today7AM.getTime() - 86400000
                : today7AM.getTime();
        }

        function sessionHoursToLabel(h) {
            const ms = sessionStartMs + h * 3600000;
            const d = new Date(ms);
            return String(d.getHours()).padStart(2, '0') + ':00';
        }
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                updateStatus('connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                updateStatus('disconnected');
                // Reconnect after 3 seconds
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateStatus('error');
            };
        }
        
        function updateStatus(status) {
            const indicator = document.getElementById('status-indicator');
            const text = document.getElementById('status-text');
            
            indicator.className = 'status-indicator';
            
            switch(status) {
                case 'connected':
                    indicator.classList.add('running');
                    text.textContent = 'Connected';
                    break;
                case 'disconnected':
                    text.textContent = 'Reconnecting...';
                    break;
                case 'error':
                    indicator.classList.add('error');
                    text.textContent = 'Connection Error';
                    break;
                default:
                    text.textContent = 'Connecting...';
            }
        }
        
        function updateDashboard(data) {
            // Update timestamp FIRST so it always shows even if later code errors
            const luEl = document.getElementById('last-updated');
            if (luEl) luEl.textContent = 'Updated: ' + new Date().toLocaleTimeString();

            try {
            // Update portfolio metrics
            document.getElementById('balance').textContent = data.current_balance ? `$${data.current_balance.toFixed(2)}` : '-';
            document.getElementById('initial-balance').textContent = data.initial_balance ? `$${data.initial_balance.toFixed(2)}` : '-';

            const initBal = data.initial_balance || 0;
            const pnl = initBal > 0 ? (data.current_balance || 0) - initBal : (data.total_pnl || 0);
            const pnlPct = initBal > 0 ? (pnl / initBal * 100) : 0;
            const pnlElement = document.getElementById('pnl');
            const pnlSign = pnlPct >= 0 ? '+' : '';
            pnlElement.innerHTML = `$${pnl.toFixed(2)}<br><span class="pct-badge">${pnlSign}${pnlPct.toFixed(2)}%</span>`;
            pnlElement.className = 'metric-value ' + (pnl >= 0 ? 'positive' : 'negative');

            const todayPnl = data.today_pnl ?? 0;
            const todayPct = initBal > 0 ? (todayPnl / initBal * 100) : 0;
            const todayPnlEl = document.getElementById('today-pnl');
            const todaySign = todayPct >= 0 ? '+' : '';
            todayPnlEl.innerHTML = `$${todayPnl.toFixed(2)}<br><span class="pct-badge">${todaySign}${todayPct.toFixed(2)}%</span>`;
            todayPnlEl.className = 'metric-value ' + (todayPnl >= 0 ? 'positive' : 'negative');

            // Update header title mode badge
            const modeEl = document.getElementById('title-mode');
            if (modeEl && data.trading_mode) {
                const isTestnet = data.trading_mode === 'testnet';
                modeEl.textContent = isTestnet ? 'Testnet Trading' : 'Live Trading';
                modeEl.className = isTestnet ? 'title-mode-testnet' : 'title-mode-live';
            }

            document.getElementById('win-rate').textContent = data.win_rate ? `${(data.win_rate * 100).toFixed(1)}%` : '-';
            document.getElementById('trade-count').textContent = data.trade_count || '0';

            // Update agent status
            document.getElementById('agent-status').textContent = data.agent_status || '-';
            document.getElementById('epsilon').textContent = data.epsilon ? data.epsilon.toFixed(3) : '-';
            document.getElementById('current-step').textContent = data.current_step || '0';
            document.getElementById('avg-q').textContent = data.avg_q !== undefined ? data.avg_q.toFixed(3) : '-';
            document.getElementById('last-action').textContent = data.last_action || '-';

            if (data.screened_coins) {
                const g = (data.screened_coins.gainers || []).slice(0, 10).join(' ');
                const l = (data.screened_coins.losers || []).slice(0, 10).join(' ');
                document.getElementById('screened-coins').innerHTML =
                    '<span style="color:var(--green)">▲</span> ' + g +
                    '<br><span style="color:var(--red)">▼</span> ' + l;
            }

            updateActivityLog(data.activity_log || []);
            updateErrorLog(data.error_log || []);

            // Update guardrails
            const guardrails = data.guardrail_status || {};
            document.getElementById('cooldown').textContent = guardrails.cooldown_remaining || '0';
            document.getElementById('daily-trades').textContent = guardrails.daily_trades || '0';

            // Update positions
            updatePositions(data.current_positions || []);

            // Update trades
            updateTrades(data.recent_trades || []);

            // Update PnL chart
            updatePnLChart(data);

            // Update screened coins table
            updateCoinTable(data.coin_table || []);

            // Live-patch today's bar in the daily PnL history chart (every 2s)
            if (sessionPnlChart && data.today_pnl !== undefined) {
                const todayStr = new Date().toISOString().slice(5, 10); // MM-DD
                const idx = sessionPnlChart.data.labels.indexOf(todayStr);
                if (idx >= 0) {
                    sessionPnlChart.data.datasets[0].data[idx] = data.today_pnl;
                    const pos = data.today_pnl >= 0;
                    sessionPnlChart.data.datasets[0].backgroundColor[idx] =
                        pos ? 'rgba(62,207,142,0.5)' : 'rgba(240,82,82,0.5)';
                    sessionPnlChart.update('none');
                }
            }

            } catch (e) { console.error('updateDashboard error:', e); }
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions-list');
            
            if (positions.length === 0) {
                container.innerHTML = '<div class="loading">No open positions</div>';
                return;
            }
            
            container.innerHTML = positions.map(pos => {
                const side      = pos.side || 'UNKNOWN';
                const size      = pos.size || 0;
                const entry     = pos.entry_price || 0;
                const mark      = pos.mark_price  || entry;
                const leverage  = pos.leverage || 1;
                const entryAmt  = size * entry;
                const markAmt   = size * mark;
                const margin    = entryAmt / leverage;          // initial margin used
                const upnl      = pos.unrealized_pnl || 0;
                const roe       = margin > 0 ? (upnl / margin * 100) : 0;
                const pnlSign   = upnl >= 0 ? '+' : '';
                const pnlClass  = upnl >= 0 ? 'positive' : 'negative';
                const fmt = (v, d=4) => v != null ? v.toFixed(d) : '—';
                return `
                <div class="position-item">
                    <div>
                        <div class="position-symbol">${pos.symbol}</div>
                        <div class="position-side ${side.toLowerCase()}">${side} ${leverage}x</div>
                    </div>
                    <div class="position-prices">
                        Entry&nbsp;<span>${fmt(pos.entry_price)}</span><br>
                        Mark&nbsp;&nbsp;<span>${fmt(pos.mark_price)}</span>
                    </div>
                    <div class="position-amounts">
                        Entry&nbsp;<span>$${entryAmt.toFixed(2)}</span><br>
                        Now&nbsp;&nbsp;&nbsp;<span>$${markAmt.toFixed(2)}</span>
                    </div>
                    <div class="position-pnl ${pnlClass}">
                        ${pnlSign}${upnl.toFixed(2)}<br>
                        <span style="font-size:11px;font-weight:600">${pnlSign}${roe.toFixed(2)}%</span><br>
                        <span style="font-size:10px;font-weight:400;color:var(--text-muted)">${fmt(size,4)} qty</span>
                    </div>
                </div>`;
            }).join('');
        }
        
        function updateTrades(trades) {
            const container = document.getElementById('trades-list');
            
            if (trades.length === 0) {
                container.innerHTML = '<div class="loading">No recent trades</div>';
                return;
            }
            
            container.innerHTML = trades.map(trade => `
                <div class="trade-item">
                    <div>
                        <div class="trade-time">${new Date(trade.timestamp).toLocaleTimeString()}</div>
                        <div class="trade-symbol">${trade.symbol}</div>
                        <div class="trade-side">${trade.side}</div>
                    </div>
                    <div class="trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">
                        ${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                    </div>
                </div>
            `).join('');
        }
        
        function updatePnLChart(data) {
            if (!pnlChart || !data.session_started) return;

            const now = Date.now();
            const elapsed = (now - sessionStartMs) / 3600000;
            // Snap to the next hourly tick (02:35 → 03:00)
            const tick = Math.ceil(elapsed);

            let pts = pnlChart.data.datasets[0].data;

            // Seed from server history — bucket into hourly ticks
            if (pts.length === 0 && data.pnl_history && data.pnl_history.length > 0) {
                const buckets = {};
                for (const p of data.pnl_history) {
                    const h = (new Date(p.t).getTime() - sessionStartMs) / 3600000;
                    const t = Math.ceil(h);
                    buckets[t] = p.pnl; // last value in each bucket wins
                }
                pnlChart.data.datasets[0].data = Object.entries(buckets)
                    .sort(([a], [b]) => Number(a) - Number(b))
                    .map(([x, y]) => ({ x: Number(x), y }));
                pts = pnlChart.data.datasets[0].data;
            }

            const initB = data.initial_balance || 0;
            const livePnl = initB > 0 ? ((data.current_balance || 0) - initB) : (data.total_pnl ?? 0);

            // Update the point at the current hourly tick in-place;
            // push a new point only when we cross into a new hour.
            if (pts.length > 0 && pts[pts.length - 1].x === tick) {
                pts[pts.length - 1].y = livePnl;
            } else {
                pts.push({ x: tick, y: livePnl });
                if (pts.length > 100) pts.shift();
            }

            // X-axis: start from first data point, end 1 tick ahead of now
            if (pts.length > 0) {
                pnlChart.options.scales.x.min = Math.max(0, pts[0].x - 1);
            }
            pnlChart.options.scales.x.max = tick + 1;

            // Adaptive x-axis tick spacing
            const span = (tick + 1) - (pnlChart.options.scales.x.min || 0);
            pnlChart.options.scales.x.ticks.stepSize = span > 12 ? 3 : span > 6 ? 2 : 1;

            // Auto-fit Y-axis to data range with 15% padding
            if (pts.length > 0) {
                const ys = pts.map(p => p.y);
                let yMin = Math.min(...ys);
                let yMax = Math.max(...ys);
                const range = yMax - yMin || Math.abs(yMax) * 0.1 || 1;
                const pad = range * 0.15;
                pnlChart.options.scales.y.min = yMin - pad;
                pnlChart.options.scales.y.max = yMax + pad;
            }

            pnlChart.update('none');
        }
        
        function initSessionChart() {
            const ctx = document.getElementById('session-pnl-chart').getContext('2d');
            sessionPnlChart = new Chart(ctx, {
                type: 'bar',
                data: { labels: [], datasets: [{ label: 'PnL', data: [], backgroundColor: [] }] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false },
                        tooltip: { callbacks: { label: (c) => ' $' + c.parsed.y.toFixed(2) } }
                    },
                    scales: {
                        x: { ticks: { color: '#616b82' }, grid: { color: '#232838' } },
                        y: {
                            ticks: { color: '#616b82', callback: (v) => '$' + v.toFixed(0) },
                            grid: {
                                color: (ctx) => ctx.tick.value === 0 ? 'rgba(255,255,255,0.55)' : '#232838',
                                lineWidth: (ctx) => ctx.tick.value === 0 ? 2 : 1,
                            },
                            border: { dash: [4, 4] },
                        }
                    }
                }
            });
        }

        function updateSessionChart(sessions) {
            if (!sessionPnlChart || !sessions || sessions.length === 0) return;
            sessionPnlChart.data.labels = sessions.map(s => s.date.slice(5)); // MM-DD
            sessionPnlChart.data.datasets[0].data = sessions.map(s => s.pnl);
            sessionPnlChart.data.datasets[0].backgroundColor = sessions.map(s => {
                if (s.type === 'future') return 'rgba(97,107,130,0.15)';  // dim placeholder
                if (s.type === 'live')   return s.pnl >= 0 ? 'rgba(62,207,142,0.5)' : 'rgba(240,82,82,0.5)';
                return s.pnl >= 0 ? '#3ecf8e' : '#f05252';
            });
            sessionPnlChart.update('none');
        }

        async function fetchAndUpdateSessionChart() {
            try {
                const resp = await fetch('/api/session_pnl');
                const sessions = await resp.json();
                updateSessionChart(sessions);
            } catch (e) { /* silent */ }
        }

        function updateActivityLog(log) {
            const container = document.getElementById('activity-log');
            if (!log || log.length === 0) {
                container.innerHTML = '<div class="loading">Waiting for bot activity...</div>';
                return;
            }
            container.innerHTML = log.map(entry =>
                '<div style="padding:3px 0;border-bottom:1px solid var(--border);">' +
                '<span style="color:var(--muted);">' + entry.time + '</span>' +
                '<span style="margin-left:10px;">' + entry.message + '</span>' +
                '</div>'
            ).join('');
        }

        function updateErrorLog(log) {
            const container = document.getElementById('error-log');
            if (!container) return;
            if (!log || log.length === 0) {
                container.innerHTML = '<div class="loading">No warnings or errors</div>';
                return;
            }
            container.innerHTML = log.map(entry => {
                const isErr = entry.level === 'ERROR';
                const badge = isErr
                    ? '<span style="color:#f05252;font-weight:700;margin:0 6px;">[ERR]</span>'
                    : '<span style="color:#e6a23c;font-weight:700;margin:0 6px;">[WARN]</span>';
                return '<div style="padding:3px 0;border-bottom:1px solid var(--border);">' +
                    '<span style="color:var(--muted);">' + entry.time + '</span>' +
                    badge +
                    '<span style="color:var(--muted);font-size:10px;">' + (entry.source || '') + '</span> ' +
                    '<span>' + entry.message + '</span>' +
                    '</div>';
            }).join('');
        }

        // ── Screened Coins Table ──────────────────────────────────────────────
        let prevCoinData = {};

        function rsiCls(v) {
            if (v === null || v === undefined) return '';
            if (v > 70) return 'text-red';
            if (v < 30) return 'text-green';
            return '';
        }
        function macdCls(v) {
            if (v === null || v === undefined) return '';
            return v > 0 ? 'text-green' : 'text-red';
        }
        function bbCls(v) {
            if (v === null || v === undefined) return '';
            if (v > 1.0) return 'text-red';
            if (v < 0.0) return 'text-green';
            return '';
        }
        function volCls(v) {
            if (v === null || v === undefined) return '';
            return v > 1.5 ? 'text-yellow' : '';
        }
        function adxCls(v) {
            if (v === null || v === undefined) return '';
            return v > 25 ? 'text-yellow' : 'dim';
        }
        function fmtClose(v) {
            if (v === null || v === undefined) return '-';
            if (v < 0.001) return v.toFixed(6);
            if (v < 1)     return v.toFixed(4);
            if (v < 100)   return v.toFixed(3);
            return v.toFixed(2);
        }

        function updateCoinTable(coinTable) {
            const tbody = document.getElementById('coin-table-body');
            if (!tbody) return;
            if (!coinTable || coinTable.length === 0) {
                tbody.innerHTML = '<tr><td colspan="14" style="text-align:center;color:var(--muted);padding:20px;">Waiting for data...</td></tr>';
                return;
            }

            let html = '';
            for (const coin of coinTable) {
                const prev = prevCoinData[coin.symbol] || {};
                const ta   = coin.ta || {};

                const rsi   = ta.rsi_14;
                const macdH = ta.macd_histogram;
                const bbpb  = ta.bb_pctb_20;
                const volR  = ta.volume_ratio;
                const atr   = ta.atr_14;
                const adx   = ta.adx_14;
                const cls   = ta.close;

                const rsiS  = rsi   != null ? rsi.toFixed(1)   : '-';
                const macdS = macdH != null ? macdH.toFixed(4) : '-';
                const bbS   = bbpb  != null ? bbpb.toFixed(2)  : '-';
                const volS  = volR  != null ? volR.toFixed(2)  : '-';
                const atrS  = atr   != null ? atr.toFixed(4)   : '-';
                const adxS  = adx   != null ? adx.toFixed(1)   : '-';

                // Directional blink: compare numeric values, flash green=up, red=down
                const blinkCls = (val, key) => {
                    const p = prev[key];
                    if (p === undefined || val == null || p == null) return '';
                    if (val === p) return '';
                    return val > p ? ' blink-up' : ' blink-down';
                };

                // Q-values: highlight the best valid one (including HOLD)
                const qs = [coin.q_hold, coin.q_long, coin.q_short, coin.q_close].filter(q => q != null);
                const maxQ = qs.length ? Math.max(...qs) : null;
                const qCls = (q) => (q != null && maxQ != null && q === maxQ) ? 'q-best' : '';
                const fmtQ = (q) => q != null ? q.toFixed(3) : '-';

                const typeCls  = coin.type === 'gainer' ? 'text-green' : 'text-red';
                const typeIcon = coin.type === 'gainer' ? '▲' : '▼';
                const chg      = coin.change_24h;
                const chgS     = chg != null ? (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%' : '-';
                const chgCls   = chg >= 0 ? 'text-green' : 'text-red';
                const symShort = coin.symbol.replace('USDT', '');

                html += `<tr>
<td class="text-left ${typeCls}" style="font-weight:600;letter-spacing:0.3px;">${symShort}</td>
<td class="${chgCls}">${chgS}</td>
<td class="${blinkCls(cls,'close')}">${fmtClose(cls)}</td>
<td class="${rsiCls(rsi)}${blinkCls(rsi,'rsi')}">${rsiS}</td>
<td class="${macdCls(macdH)}${blinkCls(macdH,'macd')}">${macdS}</td>
<td class="${bbCls(bbpb)}${blinkCls(bbpb,'bb')}">${bbS}</td>
<td class="${volCls(volR)}${blinkCls(volR,'vol')}">${volS}</td>
<td class="dim${blinkCls(atr,'atr')}">${atrS}</td>
<td class="${adxCls(adx)}${blinkCls(adx,'adx')}">${adxS}</td>
<td class="${qCls(coin.q_hold)}${blinkCls(coin.q_hold,'q_hold')}">${fmtQ(coin.q_hold)}</td>
<td class="${qCls(coin.q_long)}${blinkCls(coin.q_long,'q_long')}">${fmtQ(coin.q_long)}</td>
<td class="${qCls(coin.q_short)}${blinkCls(coin.q_short,'q_short')}">${fmtQ(coin.q_short)}</td>
<td class="${qCls(coin.q_close)}${blinkCls(coin.q_close,'q_close')}">${fmtQ(coin.q_close)}</td>
</tr>`;

                // Store raw numeric values for next-update direction comparison
                prevCoinData[coin.symbol] = {
                    close: cls, rsi, macd: macdH, bb: bbpb, vol: volR, atr, adx,
                    q_hold: coin.q_hold, q_long: coin.q_long,
                    q_short: coin.q_short, q_close: coin.q_close,
                };
            }
            tbody.innerHTML = html;
        }

        // Initialize chart
        function initChart() {
            computeSessionStart();
            const ctx = document.getElementById('pnl-chart').getContext('2d');

            // Plugin: dashed $0 reference line (only when $0 is in view)
            const zeroLine = {
                id: 'zeroLine',
                afterDraw(chart) {
                    if (chart.canvas.id !== 'pnl-chart') return;
                    const yScale = chart.scales.y;
                    if (yScale.min > 0 || yScale.max < 0) return;
                    const y = yScale.getPixelForValue(0);
                    const c = chart.ctx;
                    const ca = chart.chartArea;
                    c.save();
                    c.setLineDash([6, 4]);
                    c.strokeStyle = 'rgba(255,255,255,0.35)';
                    c.lineWidth = 1.5;
                    c.beginPath();
                    c.moveTo(ca.left, y);
                    c.lineTo(ca.right, y);
                    c.stroke();
                    c.restore();
                }
            };

            // Plugin: glowing dot + value label at the current (rightmost) point
            const currentDot = {
                id: 'currentDot',
                afterDraw(chart) {
                    if (chart.canvas.id !== 'pnl-chart') return;
                    const ds = chart.data.datasets[0];
                    if (!ds || !ds.data || ds.data.length === 0) return;
                    const meta = chart.getDatasetMeta(0);
                    const el = meta.data[meta.data.length - 1];
                    if (!el) return;
                    const val = ds.data[ds.data.length - 1].y;
                    const color = val >= 0 ? '#3ecf8e' : '#f05252';
                    const glow  = val >= 0 ? 'rgba(62,207,142,0.25)' : 'rgba(240,82,82,0.25)';
                    const px = el.x, py = el.y;
                    const c = chart.ctx;
                    const ca = chart.chartArea;
                    c.save();
                    // Outer glow
                    c.beginPath(); c.arc(px, py, 8, 0, Math.PI * 2);
                    c.fillStyle = glow; c.fill();
                    // Inner dot
                    c.beginPath(); c.arc(px, py, 4, 0, Math.PI * 2);
                    c.fillStyle = color; c.fill();
                    // Value label
                    const sign = val >= 0 ? '+' : '';
                    const label = sign + '$' + val.toFixed(2);
                    c.font = 'bold 11px monospace';
                    const tw = c.measureText(label).width;
                    const pad = 4;
                    const lx = (px + tw + 20 > ca.right) ? px - tw - 16 : px + 10;
                    const ly = Math.max(ca.top + 14, Math.min(py, ca.bottom - 6));
                    // Background pill
                    const rx = lx - pad, ry = ly - 11, rw = tw + pad * 2, rh = 16, cr = 3;
                    c.fillStyle = 'rgba(20,24,36,0.85)';
                    c.strokeStyle = color; c.lineWidth = 1;
                    c.beginPath();
                    c.moveTo(rx + cr, ry);
                    c.arcTo(rx + rw, ry, rx + rw, ry + rh, cr);
                    c.arcTo(rx + rw, ry + rh, rx, ry + rh, cr);
                    c.arcTo(rx, ry + rh, rx, ry, cr);
                    c.arcTo(rx, ry, rx + rw, ry, cr);
                    c.fill(); c.stroke();
                    // Text
                    c.fillStyle = color;
                    c.fillText(label, lx, ly);
                    c.restore();
                }
            };

            pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Session PnL',
                        data: [],
                        borderColor: '#3ecf8e',
                        borderWidth: 2.5,
                        tension: 0.4,
                        cubicInterpolationMode: 'monotone',
                        fill: {
                            target: 'origin',
                            above: 'rgba(62, 207, 142, 0.15)',
                            below: 'rgba(240, 82, 82, 0.15)',
                        },
                        segment: {
                            borderColor: seg => {
                                const y0 = seg.p0.parsed.y, y1 = seg.p1.parsed.y;
                                if (y0 >= 0 && y1 >= 0) return '#3ecf8e';
                                if (y0 <= 0 && y1 <= 0) return '#f05252';
                                return '#aaaaaa';
                            },
                        },
                        pointRadius: 0,
                        pointHoverRadius: 4,
                    }]
                },
                plugins: [zeroLine, currentDot],
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                title: (items) => {
                                    const h = items[0].parsed.x;
                                    const ms = sessionStartMs + h * 3600000;
                                    const d = new Date(ms);
                                    return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
                                },
                                label: (item) => `PnL: $${item.parsed.y.toFixed(2)}`,
                            }
                        }
                    },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            min: 0,
                            max: 1,
                            ticks: {
                                stepSize: 1,
                                color: '#616b82',
                                callback: (v) => {
                                    if (!Number.isInteger(v)) return '';
                                    const tick = Math.ceil((Date.now() - sessionStartMs) / 3600000);
                                    return v <= tick ? sessionHoursToLabel(v) : '';
                                },
                                autoSkip: false,
                                maxRotation: 0,
                            },
                            grid: { color: '#232838' },
                        },
                        y: {
                            ticks: {
                                color: '#616b82',
                                callback: (v) => '$' + v.toFixed(2),
                            },
                            grid: { color: '#232838' },
                        }
                    }
                }
            });
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            initSessionChart();
            fetchAndUpdateSessionChart();
            setInterval(fetchAndUpdateSessionChart, 60000); // full refresh every 60s (today's bar patches via WS)
            connectWebSocket();
            
            // Also poll for updates every 5 seconds as backup
            setInterval(async function() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    try {
                        const response = await fetch('/api/metrics');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Failed to fetch metrics:', error);
                    }
                }
            }, 5000);
        });
    </script>
</body>
</html>
    '''
    return HTMLResponse(html_content)

@app.get("/api/metrics")
async def get_metrics():
    """Return current trading metrics as JSON."""
    return JSONResponse(_load_trading_metrics())

@app.get("/api/session_pnl")
async def get_session_pnl():
    """Return daily PnL bars anchored to the first session day, minimum 7 bars."""
    from datetime import date as _date, timedelta as _td

    db_sessions = await asyncio.to_thread(_load_session_pnl)
    metrics = _load_trading_metrics()

    # Aggregate DB sessions by calendar day
    day_pnl: dict[str, float] = {}
    for s in db_sessions:
        d = s["date"]
        day_pnl[d] = day_pnl.get(d, 0.0) + float(s.get("pnl", 0))

    today = _date.today()

    # Determine the earliest anchor date
    session_started = metrics.get("session_started")
    if session_started:
        anchor = _date.fromisoformat(session_started[:10])
    elif day_pnl:
        anchor = _date.fromisoformat(min(day_pnl))
    else:
        anchor = today

    # End date: show at least 7 bars forward from anchor (expands as days accumulate)
    end = max(today, anchor + _td(days=6))

    # Build list of daily bars from anchor → end (anchor on left, newest on right)
    result = []
    d = anchor
    while d <= end:
        d_str = d.isoformat()
        if d == today:
            pnl = float(metrics.get("today_pnl") or day_pnl.get(d_str, 0.0))
            typ = "live" if metrics.get("session_started") else "historical"
        elif d > today:
            # Future dates: empty placeholder bars
            pnl = 0.0
            typ = "future"
        else:
            pnl = day_pnl.get(d_str, 0.0)
            typ = "historical"
        result.append({"date": d_str, "pnl": pnl, "type": typ})
        d += _td(days=1)

    return JSONResponse(result)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "time": datetime.now().isoformat(),
        "status": "running"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send current data to new connection
            data = _load_trading_metrics()
            await websocket.send_text(json.dumps(data))
            
            # Wait for updates or check every 2 seconds
            await asyncio.sleep(2)
            
            # Send updated data
            new_data = _load_trading_metrics()
            if new_data != data:
                await websocket.send_text(json.dumps(new_data))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Asymptotic Zero Trading Web Dashboard")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run on")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"Starting Trading Web Dashboard")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Dashboard: http://{args.host}:{args.port}")
    print(f"   Tailscale: http://<tailscale-ip>:{args.port}")
    print(f"   API: http://{args.host}:{args.port}/api/metrics")
    print(f"   WebSocket: ws://{args.host}:{args.port}/ws")
    print("\n📊 Dashboard ready for Tailscale network access!")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
