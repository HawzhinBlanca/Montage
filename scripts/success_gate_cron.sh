#!/bin/bash
"""
Success Gate Cron Job Script - Tasks.md Step 1D
Runs nightly A/B test evaluation and promotion

This script is designed to be executed by cron daily at 2 AM:
0 2 * * * /path/to/montage/scripts/success_gate_cron.sh
"""

# Set script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/montage_success_gate_$(date +%Y%m%d).log"

# Environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export MONTAGE_ENV="${MONTAGE_ENV:-production}"

# Redirect all output to log file with timestamps
exec 1> >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line"; done | tee -a "$LOG_FILE")
exec 2>&1

echo "=== Starting Montage Success Gate Cron Job ==="
echo "Project Root: $PROJECT_ROOT"
echo "Python Path: $PYTHONPATH"
echo "Environment: $MONTAGE_ENV"
echo "Log File: $LOG_FILE"

# Change to project directory
cd "$PROJECT_ROOT" || {
    echo "ERROR: Could not change to project directory: $PROJECT_ROOT"
    exit 1
}

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate || {
        echo "WARNING: Could not activate virtual environment"
    }
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate || {
        echo "WARNING: Could not activate virtual environment"
    }
else
    echo "No virtual environment found, using system Python"
fi

# Check Python and dependencies
python3 --version || {
    echo "ERROR: Python3 not available"
    exit 1
}

# Check if required modules are available
python3 -c "import redis, psycopg2" || {
    echo "ERROR: Required Python modules not available"
    echo "Please install: pip install redis psycopg2-binary"
    exit 1
}

# Set Redis URL if not already set
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

# Check Redis connectivity
echo "Checking Redis connectivity..."
python3 -c "
import redis
try:
    client = redis.Redis.from_url('$REDIS_URL')
    client.ping()
    print('Redis connection: OK')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
" || {
    echo "ERROR: Redis not available"
    exit 1
}

# Check database connectivity if DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "Checking database connectivity..."
    python3 -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    conn.close()
    print('Database connection: OK')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" || {
    echo "ERROR: Database not available"
    exit 1
}
else
    echo "No DATABASE_URL set, using default database configuration"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the Success Gate evaluation
echo "Starting Success Gate evaluation..."
START_TIME=$(date +%s)

python3 -m src.core.success_gate || {
    EXIT_CODE=$?
    echo "ERROR: Success Gate failed with exit code $EXIT_CODE"
    
    # Send alert email if configured
    if [ -n "$ALERT_EMAIL" ]; then
        echo "Success Gate failed at $(date)" | mail -s "Montage Success Gate Failed" "$ALERT_EMAIL"
    fi
    
    exit $EXIT_CODE
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Success Gate completed successfully in ${DURATION} seconds"

# Cleanup old log files (keep last 7 days)
find /tmp -name "montage_success_gate_*.log" -mtime +7 -delete 2>/dev/null || true

# Optional: Send success notification
if [ -n "$ALERT_EMAIL" ] && [ "$MONTAGE_ENV" = "production" ]; then
    echo "Success Gate completed successfully at $(date). Runtime: ${DURATION}s" | \
        mail -s "Montage Success Gate Success" "$ALERT_EMAIL"
fi

echo "=== Success Gate Cron Job Completed ==="