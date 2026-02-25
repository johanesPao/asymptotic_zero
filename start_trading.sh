#!/bin/bash
# Asymptotic Zero - Trading Bot & Dashboard Launcher
# Starts both trading bot and web dashboard with process management

set -e  # Exit on any error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/az_env"
LOG_DIR="$PROJECT_ROOT/logs/trading"
TRADING_BOT_PORT="${TRADING_BOT_PORT:-9997}"
WEB_DASHBOARD_PORT="${WEB_DASHBOARD_PORT:-8585}"
HOST="${HOST:-0.0.0.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        return 0  # Port is in use
    elif ss -tlnp 2>/dev/null | grep -q ":$port "; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to find available port
find_available_port() {
    local base_port=$1
    local port=$base_port
    
    while check_port $port; do
        port=$((port + 1))
        if [ $port -gt $((base_port + 10)) ]; then
            log_error "Could not find available port in range $base_port-$((base_port + 10))"
            exit 1
        fi
    done
    
    echo $port
}

# Function to check if process is running
is_process_running() {
    local pidfile=$1
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # Process is running
        else
            rm -f "$pidfile"  # Clean up stale PID file
            return 1  # Process is not running
        fi
    fi
    return 1  # PID file doesn't exist
}

# Function to stop a process
stop_process() {
    local name=$1
    local pidfile=$2
    
    if is_process_running "$pidfile"; then
        local pid=$(cat "$pidfile")
        log_info "Stopping $name (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        
        # Wait for process to stop
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
            sleep 0.5
            count=$((count + 1))
        done
        
        if kill -0 "$pid" 2>/dev/null; then
            log_warning "$name didn't stop gracefully, forcing..."
            kill -KILL "$pid" 2>/dev/null || true
        fi
        
        rm -f "$pidfile"
        log_success "$name stopped"
    fi
}

# Function to start trading bot
start_trading_bot() {
    log_info "Starting trading bot..."
    
    if is_process_running "$LOG_DIR/trading_bot.pid"; then
        log_warning "Trading bot is already running"
        return 0
    fi
    
    # Find available port
    TRADING_BOT_PORT=$(find_available_port $TRADING_BOT_PORT)
    log_info "Trading bot will use port: $TRADING_BOT_PORT"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Start trading bot fully detached:
    #   setsid  — new session, no controlling terminal (prevents Rich from opening /dev/tty)
    #   </dev/null — no stdin
    #   disown  — remove from shell job table so closing this terminal won't SIGHUP it
    cd "$PROJECT_ROOT"
    setsid nohup python scripts/trading_bot.py \
        < /dev/null > "$LOG_DIR/trading_bot.log" 2>&1 &
    local bot_pid=$!
    disown $bot_pid

    # Save PID
    echo $bot_pid > "$LOG_DIR/trading_bot.pid"

    # Brief wait to confirm launch
    sleep 3

    if kill -0 $bot_pid 2>/dev/null; then
        log_success "Trading bot started (PID: $bot_pid)"
        echo "   Log: $LOG_DIR/trading_bot.log"
    else
        log_error "Failed to start trading bot"
        rm -f "$LOG_DIR/trading_bot.pid"
        return 1
    fi
}

# Function to start web dashboard
start_web_dashboard() {
    log_info "Starting web dashboard..."
    
    if is_process_running "$LOG_DIR/web_dashboard.pid"; then
        log_warning "Web dashboard is already running"
        return 0
    fi
    
    # Find available port
    WEB_DASHBOARD_PORT=$(find_available_port $WEB_DASHBOARD_PORT)
    log_info "Web dashboard will use port: $WEB_DASHBOARD_PORT"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Start web dashboard fully detached
    cd "$PROJECT_ROOT"
    setsid nohup python scripts/trading_web_dashboard.py \
        --port $WEB_DASHBOARD_PORT \
        --host $HOST \
        < /dev/null > "$LOG_DIR/web_dashboard.log" 2>&1 &
    local dashboard_pid=$!
    disown $dashboard_pid

    # Save PID
    echo $dashboard_pid > "$LOG_DIR/web_dashboard.pid"

    # Brief wait to confirm launch
    sleep 2

    if kill -0 $dashboard_pid 2>/dev/null; then
        log_success "Web dashboard started (PID: $dashboard_pid)"
        echo "   Dashboard: http://$HOST:$WEB_DASHBOARD_PORT"
        echo "   Tailscale: http://<tailscale-ip>:$WEB_DASHBOARD_PORT"
        echo "   Log: $LOG_DIR/web_dashboard.log"
    else
        log_error "Failed to start web dashboard"
        rm -f "$LOG_DIR/web_dashboard.pid"
        return 1
    fi
}

# Function to show status
show_status() {
    echo -e "\n${BLUE}📊 TRADING SYSTEM STATUS${NC}"
    echo "========================"
    
    # Trading bot status
    if is_process_running "$LOG_DIR/trading_bot.pid"; then
        local bot_pid=$(cat "$LOG_DIR/trading_bot.pid")
        log_success "Trading Bot: RUNNING (PID: $bot_pid)"
    else
        log_error "Trading Bot: STOPPED"
    fi
    
    # Web dashboard status
    if is_process_running "$LOG_DIR/web_dashboard.pid"; then
        local dashboard_pid=$(cat "$LOG_DIR/web_dashboard.pid")
        log_success "Web Dashboard: RUNNING (PID: $dashboard_pid)"
        echo "   URL: http://$HOST:$WEB_DASHBOARD_PORT"
        echo "   Tailscale: http://<tailscale-ip>:$WEB_DASHBOARD_PORT"
    else
        log_error "Web Dashboard: STOPPED"
    fi
    
    echo ""
}

# Function to stop all services
stop_all() {
    log_info "Stopping all services..."
    
    stop_process "Trading Bot" "$LOG_DIR/trading_bot.pid"
    stop_process "Web Dashboard" "$LOG_DIR/web_dashboard.pid"
    
    log_success "All services stopped"
}

# Function to show logs
tail_logs() {
    local service=$1
    local logfile=""
    
    case $service in
        "bot"|"trading"|"trading_bot")
            logfile="$LOG_DIR/trading_bot.log"
            ;;
        "dashboard"|"web"|"web_dashboard")
            logfile="$LOG_DIR/web_dashboard.log"
            ;;
        "all")
            echo -e "\n${BLUE}📋 TRADING SYSTEM LOGS${NC}"
            echo "====================="
            echo -e "\n${YELLOW}Trading Bot Log:${NC}"
            tail -f "$LOG_DIR/trading_bot.log" &
            echo -e "\n${YELLOW}Web Dashboard Log:${NC}"
            tail -f "$LOG_DIR/web_dashboard.log" &
            wait
            return
            ;;
        *)
            log_error "Unknown service: $service"
            echo "Available: bot, dashboard, all"
            return 1
            ;;
    esac
    
    if [ -f "$logfile" ]; then
        echo -e "\n${BLUE}📋 $service Log:${NC}"
        tail -f "$logfile"
    else
        log_error "Log file not found: $logfile"
    fi
}

# Function to show help
show_help() {
    echo "Asymptotic Zero - Trading Bot & Dashboard Launcher"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start     Start both trading bot and web dashboard"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show status of all services"
    echo "  logs      Show logs for services"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  TRADING_BOT_PORT     Port for trading bot (default: 9997)"
    echo "  WEB_DASHBOARD_PORT   Port for web dashboard (default: 9998)"
    echo "  HOST                 Host to bind to (default: 0.0.0.0)"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all services"
    echo "  $0 stop                     # Stop all services"
    echo "  $0 status                   # Check status"
    echo "  $0 logs bot                 # Show trading bot logs"
    echo "  $0 logs dashboard           # Show web dashboard logs"
    echo "  $0 logs all                 # Show all logs"
    echo ""
}

# Main function
main() {
    local command=${1:-start}
    
    case $command in
        "start")
            log_info "Starting Asymptotic Zero Trading System..."
            start_trading_bot
            start_web_dashboard
            show_status
            log_success "Trading system started successfully!"
            echo ""
            echo "📊 Access your dashboard at: http://$HOST:$WEB_DASHBOARD_PORT"
            echo "🔗 Tailscale: http://<tailscale-ip>:$WEB_DASHBOARD_PORT"
            echo ""
            echo "To view logs: $0 logs all"
            echo "To stop: $0 stop"
            ;;
        "stop")
            pkill -f "tail.*$LOG_DIR" 2>/dev/null || true
            stop_all
            ;;
        "restart")
            log_info "Restarting Asymptotic Zero Trading System..."
            # Kill any stray tail processes watching our log files
            pkill -f "tail.*$LOG_DIR" 2>/dev/null || true
            stop_all
            sleep 2
            start_trading_bot
            start_web_dashboard
            show_status
            log_success "Trading system restarted successfully!"
            ;;
        "status")
            show_status
            ;;
        "logs")
            if [ -z "$2" ]; then
                log_error "Please specify which logs to view: bot, dashboard, or all"
                exit 1
            fi
            # Kill any background tail jobs before exiting so they don't leak to the terminal
            trap 'kill $(jobs -p) 2>/dev/null; exit 0' INT TERM
            tail_logs "$2"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
