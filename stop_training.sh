#!/bin/bash
# Asymptotic Zero — Stop Training + Monitor
# Usage: ./stop_training.sh
#
# Sends SIGTERM to training process, which triggers a graceful
# checkpoint save to checkpoints/latest/ before exiting.
# Waits indefinitely — shows real-time elapsed seconds.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs/training"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Asymptotic Zero processes...${NC}"

# ── Stop training (graceful: SIGTERM → wait until checkpoint save completes) ──
if [ -f "$LOG_DIR/.train_pid" ]; then
    TRAIN_PID=$(cat "$LOG_DIR/.train_pid")
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo -e "  Sending SIGTERM to training (PID: ${TRAIN_PID})..."
        echo -e "  ${BLUE}Waiting for checkpoint save to complete...${NC}"
        kill $TRAIN_PID 2>/dev/null || true

        # Wait indefinitely — show elapsed seconds in real-time
        WAITED=0
        while ps -p $TRAIN_PID > /dev/null 2>&1; do
            sleep 1
            WAITED=$((WAITED + 1))
            # Overwrite the same line with elapsed time
            printf "\r  ⏳ Saving checkpoint... %ds elapsed" $WAITED
        done
        echo ""  # newline after the timer

        echo -e "  ${GREEN}✓ Training stopped gracefully (${WAITED}s)${NC}"

        # Check if checkpoint was saved
        if [ -f "checkpoints/latest/training_state.json" ]; then
            echo -e "  ${GREEN}✓ Checkpoint saved to checkpoints/latest/${NC}"
            # Show resume info
            SAVED_EP=$(python3 -c "import json; d=json.load(open('checkpoints/latest/training_state.json')); print(f\"ep={d['global_episode']}, stage={d['current_stage']}, stage_ep={d['stage_episode']}\")" 2>/dev/null || echo "unknown")
            echo -e "  ${BLUE}  Resume point: ${SAVED_EP}${NC}"
        else
            echo -e "  ${RED}⚠ No training_state.json found — checkpoint may be incomplete${NC}"
        fi
    else
        echo -e "  Training process (PID: ${TRAIN_PID}) already exited"
    fi
    rm -f "$LOG_DIR/.train_pid"
else
    echo -e "  No training PID file found"
fi

# ── Stop monitor ──────────────────────────────────────────────────────────────
if [ -f "$LOG_DIR/.monitor_pid" ]; then
    MONITOR_PID=$(cat "$LOG_DIR/.monitor_pid")
    if ps -p $MONITOR_PID > /dev/null 2>&1; then
        echo -e "  Stopping monitor (PID: ${MONITOR_PID})..."
        kill $MONITOR_PID 2>/dev/null || true
        echo -e "  ${GREEN}✓ Monitor stopped${NC}"
    fi
    rm -f "$LOG_DIR/.monitor_pid"
fi

# ── Stop TensorBoard ─────────────────────────────────────────────────────────
if [ -f "$LOG_DIR/.tensorboard_pid" ]; then
    TB_PID=$(cat "$LOG_DIR/.tensorboard_pid")
    if ps -p $TB_PID > /dev/null 2>&1; then
        echo -e "  Stopping TensorBoard (PID: ${TB_PID})..."
        kill $TB_PID 2>/dev/null || true
        echo -e "  ${GREEN}✓ TensorBoard stopped${NC}"
    fi
    rm -f "$LOG_DIR/.tensorboard_pid"
fi

echo ""
echo -e "${GREEN}All processes stopped${NC}"

# Show restart hint
if [ -f "checkpoints/latest/training_state.json" ]; then
    echo -e ""
    echo -e "${BLUE}To resume training from where you left off:${NC}"
    echo -e "  ${GREEN}./start_training.sh${NC}    (auto-resumes from checkpoints/latest/)"
fi
