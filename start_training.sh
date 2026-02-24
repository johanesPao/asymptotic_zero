#!/bin/bash
# Asymptotic Zero — Start Training + Monitor + TensorBoard
# Usage:
#   ./start_training.sh                     # 10000 episodes (default), prompts to resume
#   ./start_training.sh --episodes 16000    # override episode count
#   ./start_training.sh --resume <path>     # resume from specific checkpoint
#   ./start_training.sh --no-resume         # force fresh start (skip prompt)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── Activate virtual environment ─────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/az_env/bin/activate" ]; then
    source "$PROJECT_ROOT/az_env/bin/activate"
elif [ -n "$VIRTUAL_ENV" ]; then
    : # Already in a venv
else
    echo -e "${RED}ERROR: az_env not found. Run: python -m venv az_env && source az_env/bin/activate && pip install -r requirements.txt${NC}"
    exit 1
fi

# ── Default training config ──────────────────────────────────────────────────
DEFAULT_EPISODES=10000

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/training"
MONITOR_PORT=8765
TB_PORT=6006

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Asymptotic Zero — Training + Monitor${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Create log directory
mkdir -p "$LOG_DIR"

# Kill any existing monitor, tensorboard, or training processes (cleanup)
echo -e "${YELLOW}Checking for existing processes...${NC}"
pkill -f "python scripts/monitor.py" 2>/dev/null || true
pkill -f "tensorboard" 2>/dev/null || true
pkill -f "python scripts/train.py" 2>/dev/null || true
sleep 1

# ── Parse our flags from args ────────────────────────────────────────────────
TRAIN_ARGS=()
HAS_RESUME=false
NO_RESUME=false
EPISODES=$DEFAULT_EPISODES
SKIP_NEXT=false

for i in "$@"; do
    if [ "$SKIP_NEXT" = true ]; then SKIP_NEXT=false; continue; fi
    if [ "$i" = "--resume" ]; then HAS_RESUME=true; fi
    if [ "$i" = "--no-resume" ]; then NO_RESUME=true; fi
done

# Extract --episodes value (consume it so we control placement)
ARGS_WITHOUT_EPISODES=()
SKIP_NEXT=false
for i in "$@"; do
    if [ "$SKIP_NEXT" = true ]; then EPISODES="$i"; SKIP_NEXT=false; continue; fi
    if [ "$i" = "--episodes" ]; then SKIP_NEXT=true; continue; fi
    ARGS_WITHOUT_EPISODES+=("$i")
done

LATEST_STATE="checkpoints/latest/training_state.json"

echo -e "${BLUE}  Episodes: ${GREEN}${EPISODES}${NC}"
echo ""

if [ "$HAS_RESUME" = false ] && [ "$NO_RESUME" = false ] && [ -f "$LATEST_STATE" ]; then
    # Found a saved checkpoint — ask the user what to do
    SAVED_EP=$(python3 -c "import json; d=json.load(open('$LATEST_STATE')); print(d['global_episode'])" 2>/dev/null || echo "?")
    SAVED_STAGE=$(python3 -c "import json; d=json.load(open('$LATEST_STATE')); print(d['current_stage'])" 2>/dev/null || echo "?")
    SAVED_EPS=$(python3 -c "import json; d=json.load(open('$LATEST_STATE')); print(f\"{d['agent_state']['epsilon']:.4f}\")" 2>/dev/null || echo "?")
    SAVED_AT=$(python3 -c "import json; d=json.load(open('$LATEST_STATE')); print(d.get('saved_at','?')[:19])" 2>/dev/null || echo "?")

    echo -e "${BLUE}Found saved training checkpoint:${NC}"
    echo -e "  Episode:  ${GREEN}${SAVED_EP}${NC}"
    echo -e "  Stage:    ${GREEN}${SAVED_STAGE}${NC}"
    echo -e "  Epsilon:  ${GREEN}${SAVED_EPS}${NC}"
    echo -e "  Saved at: ${SAVED_AT}"
    echo ""
    echo -e "${YELLOW}What would you like to do?${NC}"
    echo -e "  ${GREEN}[R]${NC}esume from checkpoint"
    echo -e "  ${RED}[N]${NC}ew training (discard checkpoint)"
    echo ""
    read -r -p "Choice [R/n]: " choice

    case "$choice" in
        [nN])
            echo -e "${YELLOW}Starting fresh — clearing checkpoints/latest/...${NC}"
            rm -rf checkpoints/latest/*
            for arg in "${ARGS_WITHOUT_EPISODES[@]}"; do
                TRAIN_ARGS+=("$arg")
            done
            ;;
        *)
            echo -e "${GREEN}Resuming from checkpoint...${NC}"
            TRAIN_ARGS=("--resume" "checkpoints/latest")
            for arg in "${ARGS_WITHOUT_EPISODES[@]}"; do
                TRAIN_ARGS+=("$arg")
            done
            ;;
    esac
else
    # No checkpoint found, or user explicitly passed --resume / --no-resume
    if [ "$NO_RESUME" = true ]; then
        echo -e "${YELLOW}Fresh start requested — clearing checkpoints/latest/...${NC}"
        rm -rf checkpoints/latest/*
    fi
    for arg in "${ARGS_WITHOUT_EPISODES[@]}"; do
        TRAIN_ARGS+=("$arg")
    done
fi

# Filter out --no-resume (not a real train.py argument) and always inject --episodes
FILTERED_ARGS=("--episodes" "$EPISODES")
for arg in "${TRAIN_ARGS[@]}"; do
    if [ "$arg" != "--no-resume" ]; then
        FILTERED_ARGS+=("$arg")
    fi
done

# ── Start monitor ────────────────────────────────────────────────────────────
echo -e "${GREEN}[1/3] Starting web monitor on port ${MONITOR_PORT}...${NC}"
nohup python scripts/monitor.py \
    --port "$MONITOR_PORT" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/monitor_${TIMESTAMP}.log" 2>&1 &
MONITOR_PID=$!

sleep 2

if ps -p $MONITOR_PID > /dev/null; then
    echo -e "${GREEN}      ✓ Monitor running (PID: ${MONITOR_PID})${NC}"
    echo -e "${GREEN}      → http://localhost:${MONITOR_PORT}${NC}"
else
    echo -e "${YELLOW}      ⚠ Monitor failed to start (check logs)${NC}"
fi

# ── Start TensorBoard ────────────────────────────────────────────────────────
echo -e "${GREEN}[2/3] Starting TensorBoard on port ${TB_PORT}...${NC}"
TB_LOGDIR="logs/tensorboard"
mkdir -p "$TB_LOGDIR"
nohup tensorboard --logdir "$TB_LOGDIR" --port "$TB_PORT" --bind_all \
    > "$LOG_DIR/tensorboard_${TIMESTAMP}.log" 2>&1 &
TB_PID=$!

sleep 2

if ps -p $TB_PID > /dev/null; then
    echo -e "${GREEN}      ✓ TensorBoard running (PID: ${TB_PID})${NC}"
    echo -e "${GREEN}      → http://localhost:${TB_PORT}${NC}"
else
    echo -e "${YELLOW}      ⚠ TensorBoard failed to start (check logs)${NC}"
fi

# ── Start training ───────────────────────────────────────────────────────────
echo -e "${GREEN}[3/3] Starting training...${NC}"
nohup python scripts/train.py "${FILTERED_ARGS[@]}" \
    > "$LOG_DIR/train_${TIMESTAMP}.log" 2>&1 &
TRAIN_PID=$!

sleep 2

if ps -p $TRAIN_PID > /dev/null; then
    echo -e "${GREEN}      ✓ Training started (PID: ${TRAIN_PID})${NC}"
else
    echo -e "${YELLOW}      ⚠ Training failed to start (check logs)${NC}"
    echo -e "${YELLOW}      Check: tail -20 ${LOG_DIR}/train_${TIMESTAMP}.log${NC}"
    exit 1
fi

# Save PIDs for later cleanup
echo "$MONITOR_PID" > "$LOG_DIR/.monitor_pid"
echo "$TRAIN_PID" > "$LOG_DIR/.train_pid"
echo "$TB_PID" > "$LOG_DIR/.tensorboard_pid"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}All processes running in background${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Monitor PID:     ${MONITOR_PID}"
echo -e "  TensorBoard PID: ${TB_PID}"
echo -e "  Training PID:    ${TRAIN_PID}"
echo ""
echo -e "  Dashboard:    ${GREEN}http://localhost:${MONITOR_PORT}${NC}"
echo -e "  TensorBoard:  ${GREEN}http://localhost:${TB_PORT}${NC}"
echo -e "  Training log: ${LOG_DIR}/train_${TIMESTAMP}.log"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo -e "  View training:  tail -f ${LOG_DIR}/train_${TIMESTAMP}.log"
echo -e "  Stop all:       ./stop_training.sh"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
