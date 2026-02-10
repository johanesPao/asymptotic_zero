#!/usr/bin/env bash
# =============================================================================
# prune.sh
#
# Deletes files that don't belong in a production deployment.
# Does NOT touch: az_env/, data/, .env (go to .gitignore instead).
#
# Two categories deleted:
#  A. Never needed in production (superseded/test/verify scripts, old docs)
#  B. Runtime artifacts (compiled bytecode, old logs, stale JSON results)
#
# Review the DRY_RUN output first:
#   bash scripts/prune.sh --dry-run
# Then actually delete:
#   bash scripts/prune.sh
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

deleted=0
skipped=0

remove() {
    local path="$ROOT/$1"
    if [ ! -e "$path" ]; then
        echo "  (not found, skip) $1"
        ((skipped++)) || true
        return
    fi

    if $DRY_RUN; then
        echo "  [DRY-RUN] would delete: $1"
    else
        rm -rf "$path"
        echo "  ✅ deleted: $1"
    fi
    ((deleted++)) || true
}

echo ""
echo "════════════════════════════════════════════════════════════════"
$DRY_RUN && echo "  DRY-RUN mode — nothing will be deleted" || echo "  PRUNE — deleting production-irrelevant files"
echo "════════════════════════════════════════════════════════════════"

# ── A1. Superseded / never-used scripts ──────────────────────────────────────
echo ""
echo "── A1. Superseded scripts"
remove "scripts/live_trading_bot.py"  # old pre-dashboard prototype
remove "scripts/monitor.py"           # reads old plain-text logs; DB replaces this

# ── A2. One-shot test / verify scripts (job done) ────────────────────────────
echo ""
echo "── A2. Test & verify scripts"
remove "scripts/test_infisical.py"
remove "scripts/test_infisical_db.py"
remove "scripts/test_load_agent.py"
remove "scripts/test_stack.py"
remove "scripts/verify_db_logging.py"
remove "scripts/verify_log_rotation.py"
remove "scripts/scan_prune_candidates.sh"   # this script itself, job done after prune
remove "prune_candidates.txt"

# ── A3. Old documentation (replaced by Docker + GHA approach) ────────────────
echo ""
echo "── A3. Stale markdown docs"
remove "VPS_DEPLOYMENT_SUMMARY.md"
remove "QUICKSTART_VPS.md"
remove "QUICK_START.md"
remove "DEPLOYMENT_GUIDE.md"

# ── A4. Old deployment folder (systemd approach, replaced by Docker) ─────────
echo ""
echo "── A4. Old deployment folder"
remove "deployment"

# ── A5. Redundant requirements ───────────────────────────────────────────────
echo ""
echo "── A5. Redundant files"
remove "requirements_vps.txt"

# ── B1. Stale JSON result snapshots at project root ──────────────────────────
echo ""
echo "── B1. Stale JSON result snapshots"
remove "holdout_results.json"
remove "model_evaluation_results.json"
remove "paper_trading_results.json"

# ── B2. All training logs ─────────────────────────────────────────────────────
echo ""
echo "── B2. Training logs"
find "$ROOT/logs" -name "training_*.log" | sort | while read -r f; do
    rel="${f#$ROOT/}"
    if $DRY_RUN; then
        echo "  [DRY-RUN] would delete: $rel"
    else
        rm -f "$f"
        echo "  ✅ deleted: $rel"
    fi
    ((deleted++)) || true
done

# ── B3. __pycache__ dirs and bytecode ────────────────────────────────────────
echo ""
echo "── B3. Python bytecode / __pycache__"
find "$ROOT" \
    -not -path "$ROOT/.git/*" \
    -not -path "$ROOT/az_env/*" \
    \( -type d -name "__pycache__" \) | sort | while read -r d; do
    rel="${d#$ROOT/}"
    if $DRY_RUN; then
        echo "  [DRY-RUN] would delete: $rel/"
    else
        rm -rf "$d"
        echo "  ✅ deleted: $rel/"
    fi
    ((deleted++)) || true
done

find "$ROOT" \
    -not -path "$ROOT/.git/*" \
    -not -path "$ROOT/az_env/*" \
    \( -name "*.pyc" -o -name "*.pyo" \) | sort | while read -r f; do
    rel="${f#$ROOT/}"
    if $DRY_RUN; then
        echo "  [DRY-RUN] would delete: $rel"
    else
        rm -f "$f"
        echo "  ✅ deleted: $rel"
    fi
    ((deleted++)) || true
done

# ── B4. IDE config ───────────────────────────────────────────────────────────
echo ""
echo "── B4. IDE config"
remove ".vscode"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
if $DRY_RUN; then
    echo "  DRY-RUN complete. $deleted items would be deleted, $skipped not found."
    echo "  Run without --dry-run to actually delete."
else
    echo "  Prune complete. $deleted items deleted, $skipped not found."
fi
echo ""
echo "  NOT touched (add to .gitignore instead):"
echo "    az_env/   data/   .env   logs/   checkpoints/"
echo "════════════════════════════════════════════════════════════════"
echo ""
