#!/bin/bash
# Deploy monitoring stack for AI Video Processing Pipeline

set -e

echo "🚀 Deploying Video Pipeline Monitoring Stack"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p prometheus/alerts
mkdir -p grafana/dashboards
mkdir -p grafana/datasources
mkdir -p alertmanager
mkdir -p process-exporter

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Grafana credentials
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# Database connection (adjust as needed)
DB_USER=video_user
DB_PASS=video_password
DB_HOST=host.docker.internal
DB_NAME=video_pipeline

# Redis connection
REDIS_URL=redis://host.docker.internal:6379

# Webhook token for alerts
WEBHOOK_TOKEN=secure_webhook_token_here

# Metrics ports
METRICS_PORT=8000
ALERTS_PORT=8001
EOF
    echo "⚠️  Please edit .env file with your actual configuration"
fi

# Validate configuration files
echo "✅ Validating configuration files..."

if [ ! -f "prometheus/prometheus.yml" ]; then
    echo "❌ prometheus/prometheus.yml not found"
    exit 1
fi

if [ ! -f "grafana/dashboards/video_pipeline_dashboard.json" ]; then
    echo "❌ grafana/dashboards/video_pipeline_dashboard.json not found"
    exit 1
fi

# Pull latest images
echo "🐳 Pulling Docker images..."
docker-compose pull

# Start the stack
echo "🚀 Starting monitoring stack..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🏥 Checking service health..."

check_service() {
    local service=$1
    local port=$2
    local url=$3
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}${url}" | grep -q "200\|302"; then
        echo "✅ ${service} is running on port ${port}"
    else
        echo "❌ ${service} failed to start on port ${port}"
    fi
}

check_service "Prometheus" 9090 "/-/healthy"
check_service "Grafana" 3000 "/api/health"
check_service "Alertmanager" 9093 "/#/alerts"

echo ""
echo "📊 Monitoring stack deployed!"
echo ""
echo "Access points:"
echo "  📈 Grafana:      http://localhost:3000 (${GRAFANA_USER}/${GRAFANA_PASSWORD})"
echo "  🔍 Prometheus:   http://localhost:9090"
echo "  🚨 Alertmanager: http://localhost:9093"
echo ""
echo "Next steps:"
echo "  1. Update .env with your actual configuration"
echo "  2. Start your application with metrics enabled"
echo "  3. Configure alert notifications in Alertmanager"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop:      docker-compose down"
echo "To remove:    docker-compose down -v"