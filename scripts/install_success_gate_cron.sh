#!/bin/bash
"""
Install Success Gate Cron Job - Tasks.md Step 1D
Sets up nightly A/B test evaluation as a cron job

Usage:
  ./install_success_gate_cron.sh install   - Install the cron job
  ./install_success_gate_cron.sh uninstall - Remove the cron job
  ./install_success_gate_cron.sh status    - Check cron job status
"""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CRON_SCRIPT="$SCRIPT_DIR/success_gate_cron.sh"

# Cron job configuration
CRON_TIME="0 2 * * *"  # Daily at 2 AM
CRON_USER=$(whoami)
CRON_COMMENT="# Montage Success Gate - Tasks.md Step 1D"
CRON_LINE="$CRON_TIME $CRON_SCRIPT"

install_cron() {
    echo "Installing Success Gate cron job..."
    
    # Make cron script executable
    chmod +x "$CRON_SCRIPT" || {
        echo "ERROR: Could not make cron script executable: $CRON_SCRIPT"
        exit 1
    }
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "success_gate_cron.sh"; then
        echo "WARNING: Success Gate cron job already exists"
        echo "Current cron jobs for $CRON_USER:"
        crontab -l 2>/dev/null | grep "success_gate"
        echo ""
        read -p "Replace existing cron job? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled"
            exit 0
        fi
        
        # Remove existing cron job
        echo "Removing existing cron job..."
        (crontab -l 2>/dev/null | grep -v "success_gate_cron.sh") | crontab -
    fi
    
    # Add new cron job
    echo "Adding new cron job: $CRON_LINE"
    (
        crontab -l 2>/dev/null
        echo "$CRON_COMMENT"
        echo "$CRON_LINE"
    ) | crontab -
    
    if [ $? -eq 0 ]; then
        echo "✅ Success Gate cron job installed successfully"
        echo "   Schedule: Daily at 2:00 AM"
        echo "   Script: $CRON_SCRIPT"
        echo "   User: $CRON_USER"
        echo ""
        echo "Logs will be written to: /tmp/montage_success_gate_YYYYMMDD.log"
        echo ""
        echo "To test manually, run:"
        echo "   $CRON_SCRIPT"
    else
        echo "ERROR: Failed to install cron job"
        exit 1
    fi
}

uninstall_cron() {
    echo "Uninstalling Success Gate cron job..."
    
    # Check if cron job exists
    if ! crontab -l 2>/dev/null | grep -q "success_gate_cron.sh"; then
        echo "No Success Gate cron job found for user $CRON_USER"
        exit 0
    fi
    
    # Remove cron job
    (crontab -l 2>/dev/null | grep -v "success_gate") | crontab -
    
    if [ $? -eq 0 ]; then
        echo "✅ Success Gate cron job uninstalled successfully"
    else
        echo "ERROR: Failed to uninstall cron job"
        exit 1
    fi
}

check_status() {
    echo "=== Success Gate Cron Job Status ==="
    echo "User: $CRON_USER"
    echo "Project Root: $PROJECT_ROOT"
    echo "Cron Script: $CRON_SCRIPT"
    echo ""
    
    # Check if cron script exists and is executable
    if [ -f "$CRON_SCRIPT" ]; then
        if [ -x "$CRON_SCRIPT" ]; then
            echo "✅ Cron script exists and is executable"
        else
            echo "❌ Cron script exists but is not executable"
        fi
    else
        echo "❌ Cron script not found"
        exit 1
    fi
    
    # Check if cron job is installed
    if crontab -l 2>/dev/null | grep -q "success_gate_cron.sh"; then
        echo "✅ Cron job is installed"
        echo ""
        echo "Current cron job:"
        crontab -l 2>/dev/null | grep -A1 -B1 "success_gate"
    else
        echo "❌ Cron job is not installed"
    fi
    
    echo ""
    echo "=== Environment Check ==="
    
    # Check Python
    if command -v python3 >/dev/null 2>&1; then
        echo "✅ Python3 available: $(python3 --version)"
    else
        echo "❌ Python3 not found"
    fi
    
    # Check Redis
    if python3 -c "import redis" 2>/dev/null; then
        echo "✅ Redis Python module available"
    else
        echo "❌ Redis Python module not found"
    fi
    
    # Check database module
    if python3 -c "import psycopg2" 2>/dev/null; then
        echo "✅ psycopg2 Python module available"
    else
        echo "❌ psycopg2 Python module not found"
    fi
    
    # Check recent logs
    echo ""
    echo "=== Recent Logs ==="
    RECENT_LOG=$(ls -t /tmp/montage_success_gate_*.log 2>/dev/null | head -1)
    if [ -n "$RECENT_LOG" ]; then
        echo "Most recent log: $RECENT_LOG"
        echo "Last 10 lines:"
        tail -10 "$RECENT_LOG"
    else
        echo "No log files found in /tmp/montage_success_gate_*.log"
    fi
    
    # Check cron service
    echo ""
    echo "=== Cron Service Status ==="
    if command -v systemctl >/dev/null 2>&1; then
        systemctl status cron 2>/dev/null || systemctl status crond 2>/dev/null || echo "Could not check cron service status"
    elif command -v service >/dev/null 2>&1; then
        service cron status 2>/dev/null || service crond status 2>/dev/null || echo "Could not check cron service status"
    else
        echo "Could not determine how to check cron service status"
    fi
}

test_run() {
    echo "=== Testing Success Gate Manually ==="
    echo "Running Success Gate outside of cron for testing..."
    echo ""
    
    cd "$PROJECT_ROOT" || {
        echo "ERROR: Could not change to project directory"
        exit 1
    }
    
    # Run with output
    "$CRON_SCRIPT"
    EXIT_CODE=$?
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Test run completed successfully"
    else
        echo "❌ Test run failed with exit code $EXIT_CODE"
    fi
    
    return $EXIT_CODE
}

show_help() {
    echo "Success Gate Cron Job Manager - Tasks.md Step 1D"
    echo ""
    echo "Usage: $0 {install|uninstall|status|test|help}"
    echo ""
    echo "Commands:"
    echo "  install    - Install the nightly cron job (runs daily at 2 AM)"
    echo "  uninstall  - Remove the cron job"
    echo "  status     - Show cron job status and environment check"
    echo "  test       - Run Success Gate manually for testing"
    echo "  help       - Show this help message"
    echo ""
    echo "The Success Gate evaluates A/B test results and promotes successful experiments."
    echo "Logs are written to /tmp/montage_success_gate_YYYYMMDD.log"
}

# Main script logic
case "${1:-help}" in
    install)
        install_cron
        ;;
    uninstall)
        uninstall_cron
        ;;
    status)
        check_status
        ;;
    test)
        test_run
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac