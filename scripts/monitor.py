"""
Asymptotic Zero — Training Monitor
====================================
Live web dashboard that reads logs written by train.py.

Usage:
    # Terminal 1 — start training
    python scripts/train.py

    # Terminal 2 — start monitor
    python scripts/monitor.py
    # Then open: http://localhost:8765

    # Or with custom port / log directory:
    python scripts/monitor.py --port 9000 --log-dir logs/training

Dependencies (add to requirements if not present):
    pip install fastapi uvicorn

The monitor is READ-ONLY. It never touches the training process.
It simply reads the live_metrics.json file that train.py writes every
10 episodes and serves it to the browser.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_PORT     = 8765
DEFAULT_LOG_DIR  = "logs/training"
METRICS_FILENAME = "live_metrics.json"

app = FastAPI(title="Asymptotic Zero Monitor", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Will be set at startup from CLI args
LOG_DIR: Path = Path(DEFAULT_LOG_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_metrics_file() -> Path | None:
    """
    Returns the path to live_metrics.json if it exists.
    Tries LOG_DIR/live_metrics.json first, then any subdirectory.
    """
    direct = LOG_DIR / METRICS_FILENAME
    if direct.exists():
        return direct

    # Fallback: find newest live_metrics.json anywhere under LOG_DIR
    candidates = sorted(LOG_DIR.rglob(METRICS_FILENAME), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def _load_metrics() -> dict:
    """Load and return live_metrics.json, or a sensible 'not started' placeholder."""
    path = _find_metrics_file()
    if path is None:
        return {
            "status": "waiting",
            "message": f"Waiting for training to start. Make sure train.py is running and writing to {LOG_DIR / METRICS_FILENAME}",
            "training_started": None,
            "current_stage": None,
            "global_episode": 0,
            "total_episodes": 0,
            "epsilon": 1.0,
            "recent_episodes": [],
            "stage_history": {},
            "action_counts": {},
            "walk_forward_folds": [],
            "last_updated": None,
        }

    try:
        with open(path) as f:
            data = json.load(f)
        data["status"] = "running"
        data["metrics_file"] = str(path)
        return data
    except (json.JSONDecodeError, OSError) as e:
        return {
            "status": "error",
            "message": f"Could not read metrics file: {e}",
            "metrics_file": str(path),
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics():
    """Return the current training metrics as JSON. Polled by the frontend."""
    return JSONResponse(_load_metrics())


@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the monitor HTML page."""
    html_path = Path(__file__).parent / "monitor.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>monitor.html not found</h1>"
            "<p>Make sure monitor.html is in the same directory as monitor.py</p>",
            status_code=500,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Asymptotic Zero Training Monitor")
    p.add_argument("--port",    type=int, default=DEFAULT_PORT)
    p.add_argument("--host",    type=str, default="0.0.0.0")
    p.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    LOG_DIR = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*55}")
    print(f"  Asymptotic Zero — Training Monitor")
    print(f"{'═'*55}")
    print(f"  Dashboard : http://localhost:{args.port}")
    print(f"  Log dir   : {LOG_DIR.resolve()}")
    print(f"  Metrics   : {LOG_DIR.resolve() / METRICS_FILENAME}")
    print(f"{'═'*55}\n")

    uvicorn.run(
        "monitor:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="warning",   # suppress per-request noise
        app_dir=str(Path(__file__).parent),
    )
