#!/bin/bash
# Asymptotic Zero — Check Training Status
# Usage: ./status.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="logs/training"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Asymptotic Zero — Status${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Check monitor
MONITOR_RUNNING=$(ps aux | grep "python scripts/monitor.py" | grep -v grep | wc -l)
if [ "$MONITOR_RUNNING" -gt 0 ]; then
    MONITOR_PID=$(ps aux | grep "python scripts/monitor.py" | grep -v grep | awk '{print $2}' | head -1)
    echo -e "${GREEN}✓ Monitor:  RUNNING (PID: ${MONITOR_PID})${NC}"
    echo -e "             http://localhost:8765"
else
    echo -e "${RED}✗ Monitor:  NOT RUNNING${NC}"
fi

# Check training
TRAIN_RUNNING=$(ps aux | grep "python scripts/train.py" | grep -v grep | wc -l)
if [ "$TRAIN_RUNNING" -gt 0 ]; then
    TRAIN_PID=$(ps aux | grep "python scripts/train.py" | grep -v grep | awk '{print $2}' | head -1)
    echo -e "${GREEN}✓ Training: RUNNING (PID: ${TRAIN_PID})${NC}"
    
    # Show latest training log snippet
    LATEST_LOG=$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1)
    if [ -f "$LATEST_LOG" ]; then
        echo ""
        echo -e "${YELLOW}Latest log output:${NC}"
        tail -5 "$LATEST_LOG" | sed 's/^/  /'
    fi
else
    echo -e "${RED}✗ Training: NOT RUNNING${NC}"
fi

# Check live metrics file
if [ -f "$LOG_DIR/live_metrics.json" ]; then
    echo ""
    echo -e "${YELLOW}Training progress:${NC}"
    EPISODE=$(grep -o '"global_episode":[0-9]*' "$LOG_DIR/live_metrics.json" | grep -o '[0-9]*')
    TOTAL=$(grep -o '"total_episodes":[0-9]*' "$LOG_DIR/live_metrics.json" | grep -o '[0-9]*')
    STAGE=$(grep -o '"current_stage":"[^"]*"' "$LOG_DIR/live_metrics.json" | cut -d'"' -f4)
    
    if [ -n "$EPISODE" ] && [ -n "$TOTAL" ]; then
        PCT=$(echo "scale=1; $EPISODE * 100 / $TOTAL" | bc)
        echo -e "  Episode:  ${EPISODE} / ${TOTAL} (${PCT}%)"
        echo -e "  Stage:    ${STAGE}"
    fi
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Commands:${NC}"
echo -e "  Follow log:  tail -f $LOG_DIR/train_*.log"
echo -e "  Stop all:    ./stop_training.sh"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
