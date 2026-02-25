"""
Asymptotic Zero - Trading Web Dashboard (Websocket Edition)

Web-based dashboard for live trading bot monitoring.
Uses Binance Websockets for real-time market and account data to avoid REST API bans.

v2.6: Final Robustness & UI Fidelity Fix.
"""

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
import pytz
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# project imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.config.secrets import get_secret
from binance import BinanceSocketManager, AsyncClient as BinanceAsyncClient

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_PORT = 8585
WIB = pytz.timezone('Asia/Jakarta')

app = FastAPI(title="Asymptotic Zero Trading Dashboard", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Live Binance State ────────────────────────────────────────────────────────
_live_state: dict = {
    "positions": {},      # symbol -> pos_info
    "balance_usdt": 0.0,
    "prices": {},         # symbol -> {price: float, change_24h: float}
    "last_update": None,
    "ws_status": "disconnected",
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
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        if not self.active_connections: return
        message = json.dumps(data)
        disconnected = []
        for connection in self.active_connections:
            try: await connection.send_text(message)
            except: disconnected.append(connection)
        for conn in disconnected: self.disconnect(conn)

manager = ConnectionManager()

# ── Binance Websocket Logic ───────────────────────────────────────────────────

async def _binance_ws_manager():
    """Background task: Maintain Binance WebSocket connections."""
    global _live_state
    
    api_key = get_secret("BINANCE_API_KEY")
    api_secret = get_secret("BINANCE_SECRET")
    trading_mode = get_secret("TRADING_MODE", "testnet")
    testnet = (trading_mode == "testnet")

    client = await BinanceAsyncClient.create(api_key, api_secret, testnet=testnet)
    if not hasattr(client, 'https_proxy'): client.https_proxy = None
    if not hasattr(client, 'http_proxy'): client.http_proxy = None
        
    bsm = BinanceSocketManager(client)

    async def handle_user_data():
        _log_info("Starting Binance Futures User Data Stream...")
        async with bsm.futures_user_socket() as stream:
            while True:
                msg = await stream.recv()
                if isinstance(msg, str):
                    try: msg = json.loads(msg)
                    except: continue
                if not msg or not isinstance(msg, dict): continue
                
                if msg.get('e') == 'ACCOUNT_UPDATE':
                    for b in msg.get('a', {}).get('B', []):
                        if b.get('a') == 'USDT':
                            _live_state["balance_usdt"] = float(b.get('wb', 0))
                    for p in msg.get('a', {}).get('P', []):
                        sym = p.get('s')
                        amt = float(p.get('pa', 0))
                        if abs(amt) > 0:
                            prev = _live_state["positions"].get(sym, {})
                            _live_state["positions"][sym] = {
                                "symbol": sym,
                                "side": "LONG" if amt > 0 else "SHORT",
                                "size": abs(amt),
                                "entry_price": float(p.get('ep', 0)),
                                "unrealized_pnl": float(p.get('up', 0)),
                                "mark_price": _live_state["prices"].get(sym, {}).get("price", 0.0),
                                "leverage": prev.get("leverage", 1),
                            }
                        else:
                            _live_state["positions"].pop(sym, None)
                    _live_state["last_update"] = datetime.now(timezone.utc).isoformat()

    async def handle_ticker_updates():
        _log_info("Starting Binance Futures Ticker Stream...")
        async with bsm.futures_ticker_socket() as stream:
            while True:
                msg = await stream.recv()
                if isinstance(msg, str):
                    try: msg = json.loads(msg)
                    except: continue
                
                # Robustly handle list, wrapped dict, or single dict
                tickers = []
                if isinstance(msg, list): tickers = msg
                elif isinstance(msg, dict):
                    data = msg.get('data')
                    if data: tickers = data if isinstance(data, list) else [data]
                    else: tickers = [msg]
                else: continue
                
                for t in tickers:
                    if not isinstance(t, dict): continue
                    sym = t.get('s')
                    if not sym: continue
                    price = float(t.get('c', 0))
                    change = float(t.get('P', 0))
                    _live_state["prices"][sym] = {"price": price, "change_24h": change}
                    
                    if sym in _live_state["positions"]:
                        pos = _live_state["positions"][sym]
                        pos["mark_price"] = price
                        entry, size = pos["entry_price"], pos["size"]
                        if pos["side"] == "LONG": pos["unrealized_pnl"] = (price - entry) * size
                        else: pos["unrealized_pnl"] = (entry - price) * size
                
                _live_state["last_update"] = datetime.now(timezone.utc).isoformat()

    # Initial Snapshot
    try:
        _log_info("Fetching initial snapshot...")
        acc = await client.futures_account()
        _live_state["balance_usdt"] = float(acc.get("totalMarginBalance", 0))
        pos_info = await client.futures_position_information()
        for p in pos_info:
            amt = float(p.get("positionAmt", 0))
            if abs(amt) > 0:
                _live_state["positions"][p["symbol"]] = {
                    "symbol": p["symbol"], "side": "LONG" if amt > 0 else "SHORT",
                    "size": abs(amt), "entry_price": float(p.get("entryPrice", 0)),
                    "mark_price": float(p.get("markPrice", 0)), "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
                    "leverage": int(p.get("leverage", 1)),
                }
        _live_state["ws_status"] = "connected"
    except Exception as e: _log_error(f"Snapshot failed: {e}")

    while True:
        try: await asyncio.gather(handle_user_data(), handle_ticker_updates())
        except Exception as e:
            _live_state["ws_status"] = "error"
            _log_error(f"Binance WS error: {e}")
            await asyncio.sleep(10)

def _log_info(msg):
    ts = datetime.now(WIB).strftime('%H:%M:%S')
    print(f"[{ts}] INFO: {msg}", flush=True)

def _log_error(msg):
    import traceback
    ts = datetime.now(WIB).strftime('%H:%M:%S')
    print(f"[{ts}] ERROR: {msg}\n{traceback.format_exc()}", flush=True, file=sys.stderr)

@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_binance_ws_manager())
    global _db_engine
    try:
        from sqlalchemy import create_engine
        _db_engine = create_engine(get_secret("DATABASE_URL"), pool_pre_ping=True, pool_size=2, max_overflow=2)
        print("✅ DB connected", flush=True)
    except Exception as e: print(f"⚠️ DB connection failed: {e}", flush=True)

# ── Database Helpers ──────────────────────────────────────────────────────────

def _load_session_pnl() -> list:
    if _db_engine is None: return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            rows = conn.execute(text("SELECT date, total_pnl FROM sessions WHERE total_pnl IS NOT NULL ORDER BY date ASC")).fetchall()
        return [{"date": str(row[0])[:10], "pnl": float(row[1]), "type": "realized"} for row in rows]
    except: return []

def _load_heartbeat() -> Optional[dict]:
    if _db_engine is None: return None
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            row = conn.execute(text("SELECT status, session_started, last_updated, agent_status, epsilon, current_step, last_action, avg_q, current_balance, initial_balance, trade_count, winning_trades, losing_trades, guardrail_status, coin_table, screened_coins, trading_mode, session_id FROM bot_heartbeat WHERE bot_id = 'main'")).fetchone()
        if not row: return None
        return {
            "status": row[0], "session_started": row[1].isoformat() if row[1] else None,
            "last_updated": row[2].isoformat() if row[2] else None, "agent_status": row[3],
            "epsilon": row[4], "current_step": row[5], "last_action": row[6], "avg_q": row[7],
            "current_balance": row[8], "initial_balance": row[9], "trade_count": row[10],
            "winning_trades": row[11] or 0, "losing_trades": row[12] or 0, "guardrail_status": row[13] or {},
            "coin_table": row[14] or [], "screened_coins": row[15], "trading_mode": row[16], "session_id": row[17],
        }
    except: return None

def _get_today_session_ids(conn, current_sid) -> list:
    """Return session IDs for today, always including current_sid.

    Sessions store date as UTC (screening_time.date()), so we check both
    today's UTC date and WIB date to handle the 7 AM WIB = 00:00 UTC edge.
    """
    from sqlalchemy import text
    today_utc = datetime.now(timezone.utc).date()
    today_wib = datetime.now(WIB).date()
    dates = list({today_utc, today_wib})
    rows = conn.execute(text("SELECT id FROM sessions WHERE date = ANY(:dates)"), {"dates": dates}).fetchall()
    sids = [row[0] for row in rows]
    if current_sid and current_sid not in sids:
        sids.append(current_sid)
    return sids

def _load_activity_log(session_id: int) -> list:
    if _db_engine is None or session_id is None: return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            sids = _get_today_session_ids(conn, session_id)
            rows = conn.execute(text("SELECT timestamp, message FROM activity_log WHERE session_id = ANY(:sids) ORDER BY timestamp DESC LIMIT 50"), {"sids": sids}).fetchall()
        return [{"time": row[0].replace(tzinfo=timezone.utc).astimezone(WIB).strftime("%H:%M:%S") if row[0] else "", "message": row[1]} for row in rows]
    except: return []

def _load_error_log(session_id: int) -> list:
    if _db_engine is None or session_id is None: return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            sids = _get_today_session_ids(conn, session_id)
            rows = conn.execute(text("SELECT timestamp, level, source, message FROM system_logs WHERE session_id = ANY(:sids) ORDER BY timestamp DESC LIMIT 50"), {"sids": sids}).fetchall()
        return [{"time": row[0].replace(tzinfo=timezone.utc).astimezone(WIB).strftime("%H:%M:%S") if row[0] else "", "level": row[1], "source": row[2], "message": row[3]} for row in rows]
    except: return []

def _load_pnl_history(session_id: int) -> list:
    if _db_engine is None or session_id is None: return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            sids = _get_today_session_ids(conn, session_id)
            rows = conn.execute(text("SELECT timestamp, pnl FROM pnl_snapshots WHERE session_id = ANY(:sids) ORDER BY timestamp ASC"), {"sids": sids}).fetchall()
        return [{"t": row[0].isoformat() if row[0] else "", "pnl": float(row[1])} for row in rows]
    except: return []

def _load_recent_trades(session_id: int) -> list:
    if _db_engine is None: return []
    try:
        from sqlalchemy import text
        with _db_engine.connect() as conn:
            sids = _get_today_session_ids(conn, session_id) if session_id else []
            if not sids:
                row = conn.execute(text("SELECT id FROM sessions ORDER BY id DESC LIMIT 1")).fetchone()
                if row: sids = [row[0]]
            if not sids: return []
            rows = conn.execute(text(
                "SELECT timestamp, symbol, side, entry_price, exit_price, pnl, pnl_percent, action_type "
                "FROM trades WHERE session_id = ANY(:sids) "
                "ORDER BY timestamp DESC LIMIT 20"
            ), {"sids": sids}).fetchall()
        return [{"timestamp": row[0].isoformat() if row[0] else "", "symbol": row[1], "side": row[2], "entry_price": float(row[3] or 0), "exit_price": float(row[4] or 0), "pnl": float(row[5] or 0), "pnl_pct": float(row[6] or 0), "type": row[7]} for row in rows]
    except: return []

def _load_trading_metrics() -> dict:
    hb = _load_heartbeat()
    if not hb: return {"status": "waiting", "ws_status": _live_state["ws_status"]}
    sid = hb.get("session_id")
    data = {
        **hb, "activity_log": _load_activity_log(sid), "error_log": _load_error_log(sid),
        "pnl_history": _load_pnl_history(sid), "recent_trades": _load_recent_trades(sid),
        "current_positions": list(_live_state["positions"].values()), "ws_status": _live_state["ws_status"],
    }
    if _live_state["balance_usdt"] > 0:
        data["current_balance"] = _live_state["balance_usdt"]
        if hb["initial_balance"] > 0:
            data["total_pnl"] = _live_state["balance_usdt"] - hb["initial_balance"]
            data["today_pnl"] = data["total_pnl"]
    for coin in data.get("coin_table", []):
        sym = coin.get("symbol")
        if sym in _live_state["prices"]:
            ps = _live_state["prices"][sym]
            if "ta" not in coin: coin["ta"] = {}
            coin["ta"]["close"] = ps["price"]
            coin["change_24h"] = ps["change_24h"]
    return data

# ── Dashboard HTML ────────────────────────────────────────────────────────────

HTML_CONTENT = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Asymptotic Zero - Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root { --bg: #0a0c10; --surface: #13161e; --surface2: #1a1e2a; --border: #232838; --text: #d4dae8; --muted: #616b82; --accent: #4f8ef7; --green: #3ecf8e; --red: #f05252; --yellow: #f0b429; --radius: 10px; --shadow: 0 2px 8px rgba(0,0,0,0.3); }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; font-size: 14px; line-height: 1.5; min-height: 100vh; overflow-x: hidden; }
        .header { padding: 14px 24px; border-bottom: 1px solid var(--border); background: var(--surface); display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 18px; font-weight: 600; color: var(--accent); }
        .header h1 .title-asymptotic { color: #ffffff; }
        .header h1 .title-zero { color: var(--accent); }
        .header h1 .title-dash { color: #ffffff; }
        .header h1 .title-mode-testnet { color: #f59e0b; }
        .header h1 .title-mode-live { color: #22c55e; }
        .status { display: flex; align-items: center; gap: 12px; }
        .status-indicator { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
        .status-indicator.running { background: var(--green); }
        .status-indicator.error { background: var(--red); }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow); min-width: 0; }
        .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; }
        .metric { background: var(--surface2); padding: 12px; border-radius: 6px; border: 1px solid var(--border); }
        .metric-label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
        .metric-value { font-size: 18px; font-weight: 600; color: var(--text); }
        .metric-value.positive { color: var(--green); }
        .metric-value.negative { color: var(--red); }
        .pct-badge { font-size: 12px; font-weight: 500; opacity: 0.85; }
        .positions-list { max-height: 200px; overflow-y: auto; }
        .position-item { display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 8px; align-items: center; padding: 10px 12px; background: var(--surface2); border-radius: 6px; margin-bottom: 6px; border: 1px solid var(--border); }
        .position-symbol { font-weight: 600; color: var(--text); }
        .position-side.long { color: var(--green); }
        .position-side.short { color: var(--red); }
        .position-pnl { font-weight: 600; font-size: 13px; text-align: right; min-width: 70px; }
        .position-pnl.positive { color: var(--green); }
        .position-pnl.negative { color: var(--red); }
        .position-pnl-pct { font-size: 11px; font-weight: 500; text-align: right; }
        .position-pnl-pct.positive { color: var(--green); }
        .position-pnl-pct.negative { color: var(--red); }
        .pnl-arrow { font-size: 11px; margin-right: 2px; opacity: 0.9; }
        .trades-list { max-height: 250px; overflow-y: auto; }
        .trade-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: var(--surface2); border-radius: 6px; margin-bottom: 6px; border: 1px solid var(--border); }
        .trade-symbol { font-weight: 600; color: var(--text); }
        .coin-table { width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }
        .coin-table th { color: var(--muted); text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--border); font-weight: 500; font-size: 11px; background: var(--surface); position: sticky; top: 0; z-index: 2; }
        .coin-table th:nth-child(1) { text-align: left; position: sticky; left: 0; min-width: 72px; z-index: 3; background: var(--surface); }
        .coin-table th:nth-child(2) { position: sticky; left: 72px; min-width: 62px; z-index: 3; background: var(--surface); }
        .coin-table th:nth-child(3) { position: sticky; left: 134px; min-width: 72px; z-index: 3; background: var(--surface); border-right: 1px solid var(--border); }
        .coin-table td:nth-child(1) { text-align: left; position: sticky; left: 0; z-index: 1; background: var(--surface); font-weight: 600; }
        .coin-table td:nth-child(2) { position: sticky; left: 72px; z-index: 1; background: var(--surface); }
        .coin-table td:nth-child(3) { position: sticky; left: 134px; z-index: 1; background: var(--surface); border-right: 1px solid var(--border); }
        .coin-table td { padding: 5px 10px; text-align: right; border-bottom: 1px solid rgba(35,40,56,0.5); }
        .coin-table tr:hover td { background: var(--surface2) !important; }
        .text-green { color: var(--green); }
        .text-red { color: var(--red); }
        .text-yellow { color: var(--yellow); }
        .q-best { color: var(--accent); font-weight: 700; }
        .dim { color: var(--muted); }
        #activity-log > div, #error-log > div { padding:3px 0; border-bottom: 1px solid var(--border); }
        @keyframes blink-up { 0% { background: rgba(62,207,142,0.45); } 100% { background: transparent; } }
        @keyframes blink-down { 0% { background: rgba(240,82,82,0.45); } 100% { background: transparent; } }
        .blink-up { animation: blink-up 1.2s ease-out forwards; }
        .blink-down { animation: blink-down 1.2s ease-out forwards; }
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="title-asymptotic">Asymptotic</span><span class="title-zero"> Zero</span><span class="title-dash"> — </span><span id="title-mode" class="title-mode-testnet">Live Dashboard</span></h1>
        <div class="status"><span id="last-updated" style="font-size:11px;color:var(--muted);margin-right:12px;">-</span><span id="status-text">Connecting...</span><div id="status-indicator" class="status-indicator"></div></div>
    </div>
    <div class="dashboard">
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Screened Coins &mdash; Live TA &amp; Agent Q-Values</h2>
            <div class="coin-table-wrapper" style="overflow-x:auto;max-height:420px;overflow-y:auto;">
                <table class="coin-table">
                    <thead><tr><th style="text-align:left;">Coin</th><th>24h%</th><th>Close</th><th>RSI</th><th>MACD H</th><th>BB%B</th><th>Vol&times;</th><th>ATR</th><th>ADX</th><th>Q-HOLD</th><th>Q-LONG</th><th>Q-SHORT</th><th>Q-CLOSE</th></tr></thead>
                    <tbody id="coin-table-body"><tr><td colspan="13" style="text-align:center;color:var(--muted);padding:20px;">Waiting for data...</td></tr></tbody>
                </table>
            </div>
        </div>
        <div class="card">
            <h2>Portfolio</h2>
            <div class="metrics-grid">
                <div class="metric"><div class="metric-label">Balance</div><div id="balance" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Initial Balance</div><div id="initial-balance" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Session PnL</div><div id="pnl" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Today's PnL</div><div id="today-pnl" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Win Rate</div><div id="win-rate" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Trade Count</div><div id="trade-count" class="metric-value">-</div></div>
            </div>
        </div>
        <div class="card">
            <h2>Today's PnL</h2>
            <div class="chart-container"><canvas id="pnl-chart"></canvas></div>
            <div id="chart-timestamp" class="timestamp" style="font-size:11px;color:var(--muted);text-align:right;margin-top:12px;">-</div>
        </div>
        <div class="card" style="grid-column: 1 / -1;">
            <h2>Daily PnL History</h2>
            <div style="height:220px;position:relative;"><canvas id="session-pnl-chart"></canvas></div>
        </div>
        <div class="card">
            <h2>Active Positions</h2>
            <div id="positions-list" class="positions-list"><div class="loading">Loading positions...</div></div>
        </div>
        <div class="card">
            <h2>Recent Trades</h2>
            <div id="trades-list" class="trades-list"><div class="loading">Loading trades...</div></div>
        </div>
        <div class="card">
            <h2>Agent Status</h2>
            <div class="metrics-grid">
                <div class="metric"><div class="metric-label">Status</div><div id="agent-status" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Epsilon</div><div id="epsilon" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Step</div><div id="current-step" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Avg Q</div><div id="avg-q" class="metric-value">-</div></div>
            </div>
            <div style="margin-top:15px;"><div class="metric-label">Last Decision</div><div id="last-action" style="font-size:18px;font-weight:bold;color:var(--accent)">-</div></div>
            <div style="margin-top:10px;"><div class="metric-label">Screened Coins</div><div id="screened-coins" style="margin-top:4px;font-size:12px;color:var(--muted);line-height:1.8;">-</div></div>
        </div>
        <div class="card">
            <h2>Guardrails</h2>
            <div class="metrics-grid">
                <div class="metric"><div class="metric-label">Cooldown</div><div id="cooldown" class="metric-value">-</div></div>
                <div class="metric"><div class="metric-label">Daily Trades</div><div id="daily-trades" class="metric-value">-</div></div>
            </div>
        </div>
        <div class="card" style="grid-column: 1 / -1;"><h2>Activity Log</h2><div id="activity-log" style="max-height:180px;overflow-y:auto;font-family:monospace;font-size:12px;"></div></div>
        <div class="card" style="grid-column: 1 / -1;"><h2 style="color:var(--red);">System Log</h2><div id="error-log" style="max-height:200px;overflow-y:auto;font-family:monospace;font-size:11px;"></div></div>
    </div>
    <script>
        let ws, pnlChart, sessionPnlChart, sessionStartMs = 0, prevCoinData = {};
        function computeSessionStart() { const now = new Date(); const today7AM = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 7, 0, 0, 0); sessionStartMs = now < today7AM ? today7AM.getTime() - 86400000 : today7AM.getTime(); }
        function sessionHoursToLabel(h) { const ms = sessionStartMs + h * 3600000; const d = new Date(ms); return String(d.getHours()).padStart(2, '0') + ':00'; }
        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => { document.getElementById('status-text').textContent = 'Connected'; document.getElementById('status-indicator').className = 'status-indicator running'; };
            ws.onmessage = (e) => updateDashboard(JSON.parse(e.data));
            ws.onclose = () => { document.getElementById('status-text').textContent = 'Disconnected'; document.getElementById('status-indicator').className = 'status-indicator error'; setTimeout(connect, 3000); };
        }
        function rsiCls(v) { return v > 70 ? 'text-red' : (v < 30 ? 'text-green' : ''); }
        function macdCls(v) { return v > 0 ? 'text-green' : 'text-red'; }
        function bbCls(v) { return v > 1.0 ? 'text-red' : (v < 0.0 ? 'text-green' : ''); }
        function volCls(v) { return v > 1.5 ? 'text-yellow' : ''; }
        function adxCls(v) { return v > 25 ? 'text-yellow' : 'dim'; }
        function fmtClose(v) { if (v == null) return '-'; if (v < 0.001) return v.toFixed(6); if (v < 1) return v.toFixed(4); if (v < 100) return v.toFixed(3); return v.toFixed(2); }
        function updateDashboard(data) {
            document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
            try {
                const initBal = data.initial_balance || 0;
                document.getElementById('balance').textContent = `$${data.current_balance?.toFixed(2) || '-'}`;
                document.getElementById('initial-balance').textContent = `$${initBal.toFixed(2) || '-'}`;
                const pnl = initBal > 0 ? (data.current_balance || 0) - initBal : (data.total_pnl || 0);
                const pnlPct = initBal > 0 ? (pnl / initBal * 100) : 0;
                const todayPnl = data.today_pnl ?? 0;
                const todayPct = initBal > 0 ? (todayPnl / initBal * 100) : 0;
                document.getElementById('pnl').innerHTML = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}<br><span class="pct-badge">${pnlPct.toFixed(2)}%</span>`;
                document.getElementById('pnl').className = 'metric-value ' + (pnl >= 0 ? 'positive' : 'negative');
                document.getElementById('today-pnl').innerHTML = `${todayPnl >= 0 ? '+' : ''}$${todayPnl.toFixed(2)}<br><span class="pct-badge">${todayPct.toFixed(2)}%</span>`;
                document.getElementById('today-pnl').className = 'metric-value ' + (todayPnl >= 0 ? 'positive' : 'negative');
                const modeEl = document.getElementById('title-mode');
                if (modeEl && data.trading_mode) { modeEl.textContent = data.trading_mode === 'testnet' ? 'Testnet Trading' : 'Live Trading'; modeEl.className = data.trading_mode === 'testnet' ? 'title-mode-testnet' : 'title-mode-live'; }
                document.getElementById('win-rate').textContent = data.win_rate ? `${(data.win_rate * 100).toFixed(1)}%` : '-';
                document.getElementById('trade-count').textContent = data.trade_count || '0';
                document.getElementById('agent-status').textContent = data.agent_status || '-';
                document.getElementById('epsilon').textContent = data.epsilon ? data.epsilon.toFixed(3) : '-';
                document.getElementById('current-step').textContent = data.current_step || '0';
                document.getElementById('avg-q').textContent = data.avg_q !== undefined ? data.avg_q.toFixed(3) : '-';
                document.getElementById('last-action').textContent = data.last_action || '-';
                if (data.screened_coins) {
                    const g = (data.screened_coins.gainers || []).slice(0, 10).join(' '), l = (data.screened_coins.losers || []).slice(0, 10).join(' ');
                    document.getElementById('screened-coins').innerHTML = '<span style="color:var(--green)">▲</span> ' + g + '<br><span style="color:var(--red)">▼</span> ' + l;
                }
                document.getElementById('activity-log').innerHTML = data.activity_log?.map(entry => `<div><span style="color:var(--muted);">${entry.time}</span><span style="margin-left:10px;">${entry.message}</span></div>`).join('') || '';
                document.getElementById('error-log').innerHTML = data.error_log?.map(entry => `<div><span style="color:var(--muted);">${entry.time}</span><span style="color:${entry.level==='ERROR'?'#f05252':'#e6a23c'};font-weight:700;margin:0 6px;">[${entry.level==='ERROR'?'ERR':'WARN'}]</span><span>${entry.message}</span></div>`).join('') || '';
                document.getElementById('cooldown').textContent = data.guardrail_status?.cooldown_remaining || '0';
                document.getElementById('daily-trades').textContent = data.guardrail_status?.daily_trades || '0';
                updatePositions(data.current_positions || []);
                updateTrades(data.recent_trades || []);
                updatePnLChart(data);
                updateCoinTable(data.coin_table || []);
                if (sessionPnlChart && data.today_pnl !== undefined) {
                    const todayStr = new Date().toISOString().slice(5, 10);
                    const idx = sessionPnlChart.data.labels.indexOf(todayStr);
                    if (idx >= 0) { sessionPnlChart.data.datasets[0].data[idx] = data.today_pnl; sessionPnlChart.data.datasets[0].backgroundColor[idx] = data.today_pnl >= 0 ? 'rgba(62,207,142,0.5)' : 'rgba(240,82,82,0.5)'; sessionPnlChart.update('none'); }
                }
            } catch (e) { console.error('updateDashboard error:', e); }
        }
        const _prevPnl = {};
        function updatePositions(positions) {
            const container = document.getElementById('positions-list');
            if (positions.length === 0) { container.innerHTML = '<div class="loading">No open positions</div>'; return; }
            const activeSym = new Set();
            container.innerHTML = positions.map(pos => {
                const sym = pos.symbol, side = pos.side || 'UNKNOWN', size = pos.size || 0, entry = pos.entry_price || 0, mark = pos.mark_price || entry, leverage = pos.leverage || 1, upnl = pos.unrealized_pnl || 0, pnlClass = upnl >= 0 ? 'positive' : 'negative', fmt = (v, d=4) => v != null ? v.toFixed(d) : '—';
                const margin = size * entry / leverage;
                const pnlPct = margin > 0 ? (upnl / margin * 100) : 0;
                const prev = _prevPnl[sym];
                let arrow = '';
                if (prev !== undefined && upnl !== prev) { const up = upnl > prev; arrow = `<span class="pnl-arrow" style="color:var(--${up ? 'green' : 'red'})">${up ? '\u25B2' : '\u25BC'}</span>`; }
                _prevPnl[sym] = upnl;
                activeSym.add(sym);
                return `<div class="position-item"><div><div class="position-symbol">${sym}</div><div class="position-side ${side.toLowerCase()}">${side} ${leverage}x</div></div><div class="position-prices">Entry&nbsp;<span>${fmt(pos.entry_price)}</span><br>Mark&nbsp;&nbsp;<span>${fmt(pos.mark_price)}</span></div><div class="position-amounts">Entry&nbsp;<span>$${(size * entry).toFixed(2)}</span><br>Now&nbsp;&nbsp;&nbsp;<span>$${(size * mark).toFixed(2)}</span></div><div>${arrow}<span class="position-pnl ${pnlClass}">${upnl >= 0 ? '+' : ''}${upnl.toFixed(2)}</span><div class="position-pnl-pct ${pnlClass}">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</div></div></div>`;
            }).join('');
            for (const k of Object.keys(_prevPnl)) { if (!activeSym.has(k)) delete _prevPnl[k]; }
        }
        function updateTrades(trades) {
            const container = document.getElementById('trades-list');
            if (trades.length === 0) { container.innerHTML = '<div class="loading">No recent trades</div>'; return; }
            container.innerHTML = trades.map(trade => `<div class="trade-item"><div><div style="font-size:11px;color:var(--muted);">${new Date(trade.timestamp).toLocaleTimeString()}</div><div class="trade-symbol">${trade.symbol} (${trade.side})</div><div style="font-size:10px;color:var(--muted);">${trade.type.toUpperCase()}</div></div><div class="trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}</div></div>`).join('');
        }
        function updatePnLChart(data) {
            if (!pnlChart || !data.session_started) return;
            const now = Date.now(), elapsed = (now - sessionStartMs) / 3600000, tick = Math.ceil(elapsed);
            let pts = pnlChart.data.datasets[0].data;
            if (pts.length === 0 && data.pnl_history && data.pnl_history.length > 0) {
                const buckets = {};
                for (const p of data.pnl_history) { const h = (new Date(p.t).getTime() - sessionStartMs) / 3600000; buckets[Math.ceil(h)] = p.pnl; }
                pnlChart.data.datasets[0].data = Object.entries(buckets).sort(([a],[b]) => Number(a)-Number(b)).map(([x,y]) => ({ x: Number(x), y }));
                pts = pnlChart.data.datasets[0].data;
            }
            const livePnl = (data.current_balance || 0) - (data.initial_balance || 0);
            if (pts.length > 0 && pts[pts.length - 1].x === tick) { pts[pts.length - 1].y = livePnl; }
            else { pts.push({ x: tick, y: livePnl }); if (pts.length > 100) pts.shift(); }
            pnlChart.options.scales.x.min = pts.length > 0 ? Math.max(0, pts[0].x - 1) : 0;
            pnlChart.options.scales.x.max = tick + 1;
            pnlChart.data.datasets[0].backgroundColor = livePnl >= 0 ? 'rgba(62, 207, 142, 0.1)' : 'rgba(240, 82, 82, 0.1)';
            pnlChart.update('none');
        }
        function updateCoinTable(coinTable) {
            const tbody = document.getElementById('coin-table-body');
            if (!tbody || coinTable.length === 0) return;
            tbody.innerHTML = coinTable.map(coin => {
                const prev = prevCoinData[coin.symbol] || {}, ta = coin.ta || {}, cls = ta.close;
                const blinkCls = (val, key) => (prev[key] !== undefined && val != null && val !== prev[key]) ? (val > prev[key] ? ' blink-up' : ' blink-down') : '';
                const qs = [coin.q_hold, coin.q_long, coin.q_short, coin.q_close].filter(q => q != null), maxQ = qs.length ? Math.max(...qs) : null, qCls = (q) => (q != null && maxQ != null && q === maxQ) ? 'q-best' : '', fmt = (v, d=2) => v != null ? v.toFixed(d) : '-';
                const html = `<tr><td>${coin.symbol.replace('USDT','')}</td><td class="${coin.change_24h>=0?'text-green':'text-red'}">${fmt(coin.change_24h)}%</td><td class="${blinkCls(cls,'close')}">${fmtClose(cls)}</td><td class="${rsiCls(ta.rsi_14)}">${fmt(ta.rsi_14,1)}</td><td class="${macdCls(ta.macd_histogram)}">${fmt(ta.macd_histogram,4)}</td><td class="${bbCls(ta.bb_pctb_20)}">${fmt(ta.bb_pctb_20,2)}</td><td class="${volCls(ta.volume_ratio)}">${fmt(ta.volume_ratio,2)}</td><td class="dim">${fmt(ta.atr_14,4)}</td><td class="${adxCls(ta.adx_14)}">${fmt(ta.adx_14,1)}</td><td class="${qCls(coin.q_hold)}">${fmt(coin.q_hold,3)}</td><td class="${qCls(coin.q_long)}">${fmt(coin.q_long,3)}</td><td class="${qCls(coin.q_short)}">${fmt(coin.q_short,3)}</td><td class="${qCls(coin.q_close)}">${fmt(coin.q_close,3)}</td></tr>`;
                prevCoinData[coin.symbol] = { close: cls }; return html;
            }).join('');
        }
        function initCharts() {
            computeSessionStart();
            const ctx = document.getElementById('pnl-chart').getContext('2d'), zeroLine = { id: 'zeroLine', afterDraw(chart) { const yScale = chart.scales.y; if (yScale.min > 0 || yScale.max < 0) return; const y = yScale.getPixelForValue(0), c = chart.ctx; c.save(); c.setLineDash([6, 4]); c.strokeStyle = 'rgba(255,255,255,0.35)'; c.lineWidth = 1.5; c.beginPath(); c.moveTo(chart.chartArea.left, y); c.lineTo(chart.chartArea.right, y); c.stroke(); c.restore(); } }, currentDot = { id: 'currentDot', afterDraw(chart) { const ds = chart.data.datasets[0]; if (!ds.data.length) return; const meta = chart.getDatasetMeta(0), el = meta.data[meta.data.length - 1], val = ds.data[ds.data.length - 1].y, color = val >= 0 ? '#3ecf8e' : '#f05252', glow = val >= 0 ? 'rgba(62,207,142,0.25)' : 'rgba(240,82,82,0.25)', c = chart.ctx; c.save(); c.beginPath(); c.arc(el.x, el.y, 8, 0, Math.PI * 2); c.fillStyle = glow; c.fill(); c.beginPath(); c.arc(el.x, el.y, 4, 0, Math.PI * 2); c.fillStyle = color; c.fill(); const label = (val >= 0 ? '+' : '') + '$' + val.toFixed(2), tw = c.measureText(label).width, lx = (el.x + tw + 20 > chart.chartArea.right) ? el.x - tw - 16 : el.x + 10; c.fillStyle = 'rgba(20,24,36,0.85)'; c.strokeStyle = color; c.lineWidth = 1; c.beginPath(); c.roundRect(lx - 4, el.y - 11, tw + 8, 16, 3); c.fill(); c.stroke(); c.fillStyle = color; c.fillText(label, lx, el.y); c.restore(); } };
            pnlChart = new Chart(ctx, { type: 'line', data: { datasets: [{ label: 'PnL', data: [], borderColor: '#3ecf8e', borderWidth: 2.5, tension: 0.4, fill: true, backgroundColor: 'rgba(62, 207, 142, 0.1)', pointRadius: 0, segment: { borderColor: seg => { const y0 = seg.p0.parsed.y, y1 = seg.p1.parsed.y; return (y0 >= 0 && y1 >= 0) ? '#3ecf8e' : ((y0 < 0 && y1 < 0) ? '#f05252' : '#aaaaaa'); } } }] }, plugins: [zeroLine, currentDot], options: { responsive: true, maintainAspectRatio: false, animation: false, scales: { x: { type: 'linear', ticks: { color: '#616b82', stepSize: 1, callback: v => Number.isInteger(v) ? sessionHoursToLabel(v) : '' }, grid: { color: '#232838' } }, y: { ticks: { color: '#616b82', callback: v => '$' + v.toFixed(2) }, grid: { color: '#232838' } } }, plugins: { legend: { display: false } } } });
            const ctx2 = document.getElementById('session-pnl-chart').getContext('2d');
            sessionPnlChart = new Chart(ctx2, { type: 'bar', data: { labels: [], datasets: [{ label: 'Daily PnL', data: [], backgroundColor: [] }] }, options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: '#616b82' }, grid: { color: '#232838' } }, y: { ticks: { color: '#616b82', callback: v => '$' + v.toFixed(0) }, grid: { color: ctx => ctx.tick.value === 0 ? 'rgba(255,255,255,0.55)' : '#232838', lineWidth: ctx => ctx.tick.value === 0 ? 2 : 1 } } }, plugins: { legend: { display: false } } } });
        }
        async function fetchAndUpdateSessionChart() { try { const resp = await fetch('/api/session_pnl'), d = await resp.json(); sessionPnlChart.data.labels = d.map(s => s.date.slice(5)); sessionPnlChart.data.datasets[0].data = d.map(s => s.pnl); sessionPnlChart.data.datasets[0].backgroundColor = d.map(s => s.type === 'future' ? 'rgba(97,107,130,0.15)' : (s.type === 'live' ? (s.pnl >= 0 ? 'rgba(62,207,142,0.5)' : 'rgba(240,82,82,0.5)') : (s.pnl >= 0 ? '#3ecf8e' : '#f05252'))); sessionPnlChart.update('none'); } catch (e) {} }
        initCharts(); connect(); setInterval(fetchAndUpdateSessionChart, 60000); fetchAndUpdateSessionChart();
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def dashboard(): return HTMLResponse(HTML_CONTENT)

@app.get("/api/metrics")
async def get_metrics(): return JSONResponse(_load_trading_metrics())

@app.get("/api/session_pnl")
async def get_session_pnl():
    from datetime import date as _date, timedelta as _td
    db_sessions = await asyncio.to_thread(_load_session_pnl)
    metrics = _load_trading_metrics()
    now_wib = datetime.now(WIB)
    trading_today = (now_wib if now_wib.hour >= 7 else now_wib - timedelta(days=1)).date()
    day_pnl = {s["date"]: float(s.get("pnl", 0)) for s in db_sessions}
    session_started = metrics.get("session_started")
    anchor = _date.fromisoformat(session_started[:10]) if session_started else trading_today
    end = max(trading_today, anchor + _td(days=6))
    result = []
    d = anchor
    while d <= end:
        d_str = d.isoformat()
        if d == trading_today:
            pnl = float(metrics.get("today_pnl") or day_pnl.get(d_str, 0.0))
            typ = "live" if metrics.get("session_started") else "historical"
        elif d > trading_today: pnl, typ = 0.0, "future"
        else: pnl, typ = day_pnl.get(d_str, 0.0), "historical"
        result.append({"date": d_str, "pnl": pnl, "type": typ})
        d += _td(days=1)
    return JSONResponse(result)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.send_text(json.dumps(_load_trading_metrics()))
            await asyncio.sleep(2)
    except WebSocketDisconnect: manager.disconnect(websocket)
    except: manager.disconnect(websocket)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    _log_info(f"Dashboard starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
