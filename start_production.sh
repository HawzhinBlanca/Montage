#!/bin/bash
# Start production Montage services

set -e

echo "🚀 Starting Montage Production Services..."

# Check dependencies
if ! command -v redis-cli &> /dev/null; then
    echo "❌ Redis not found. Please install Redis or run docker-compose up -d redis"
    exit 1
fi

if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL client not found. Please install PostgreSQL"
    exit 1
fi

# Check Redis connection
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Starting Redis..."
    docker-compose up -d redis
    sleep 2
fi

# Check PostgreSQL connection
if ! PGPASSWORD=pass psql -h localhost -U postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running. Starting PostgreSQL..."
    docker-compose up -d postgres
    sleep 5
fi

# Run migrations
echo "📦 Running database migrations..."
PGPASSWORD=pass psql -h localhost -U postgres -d postgres < schema.sql
if [ -f migrations/003_add_web_api_jobs_table.sql ]; then
    PGPASSWORD=pass psql -h localhost -U postgres -d postgres < migrations/003_add_web_api_jobs_table.sql
fi

# Create required directories
mkdir -p uploads output logs

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "celery worker" || true
pkill -f "celery beat" || true
pkill -f "uvicorn" || true
sleep 2

# Start Celery worker in background
echo "🔧 Starting Celery worker..."
celery -A src.api.celery_app worker --loglevel=info --logfile=logs/celery_worker.log &
WORKER_PID=$!

# Start Celery beat for periodic tasks
echo "⏰ Starting Celery beat..."
celery -A src.api.celery_app beat --loglevel=info --logfile=logs/celery_beat.log &
BEAT_PID=$!

# Start FastAPI server
echo "🌐 Starting FastAPI server..."
uvicorn src.api.web_server:app --host 0.0.0.0 --port 8000 --reload --log-config=logging.json &
API_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    kill $WORKER_PID $BEAT_PID $API_PID 2>/dev/null || true
    exit 0
}

trap cleanup EXIT INT TERM

echo "✅ All services started!"
echo ""
echo "📍 API Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait