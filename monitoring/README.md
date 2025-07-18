# Video Pipeline Monitoring Setup

This directory contains the complete monitoring stack for the AI Video Processing Pipeline.

## Components

- **Prometheus**: Time-series database for metrics
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notifications
- **Node Exporter**: System metrics
- **PostgreSQL Exporter**: Database metrics
- **Redis Exporter**: Cache metrics
- **Process Exporter**: FFmpeg process monitoring

## Quick Start

1. **Set environment variables**:
   ```bash
   # Create .env file in monitoring directory
   cat > .env << EOF
   GRAFANA_USER=admin
   GRAFANA_PASSWORD=secure_password
   DB_USER=video_user
   DB_PASS=video_password
   DB_HOST=host.docker.internal
   DB_NAME=video_pipeline
   REDIS_URL=redis://host.docker.internal:6379
   WEBHOOK_TOKEN=your_webhook_token
   EOF
   ```

2. **Start the monitoring stack**:
   ```bash
   docker-compose up -d
   ```

3. **Access services**:
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Alertmanager: http://localhost:9093

## Dashboard Overview

The main dashboard includes:

### Performance Metrics
- Job processing rate (success/failure)
- Processing time vs source duration ratio
- Queue sizes and backlogs
- API call rates and latencies

### Cost Tracking
- Current job cost gauge (with $5 limit)
- API cost rate by provider
- Budget exceeded counter
- Cost breakdown by API

### Quality Metrics
- Audio loudness spread (LU)
- Video rejection reasons
- HDR input rejection rate
- Face detection coverage

### System Health
- Database connection pool status
- Redis memory usage
- FFmpeg process count and crashes
- API rate limiting events

## Alert Rules

### Critical Alerts
- **BudgetExceeded**: Job cost > $5
- **HighErrorRate**: >10% job failure rate
- **FFmpegCrashes**: Frequent FFmpeg crashes
- **DatabaseConnectionPoolExhausted**: No available connections

### Warning Alerts
- **SlowProcessing**: Processing time > 1.2x source
- **QueueBacklog**: >50 pending jobs
- **HighCostRate**: API costs > $0.50/sec
- **AudioLoudnessSpreadHigh**: >1.5 LU spread

## Custom Metrics Integration

Your application should expose metrics at `/metrics` endpoint:

```python
from prometheus_client import start_http_server
from metrics import registry

# Start metrics server
start_http_server(8000, registry=registry)
```

## Alert Webhooks

Configure your application to receive alerts:

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook/critical', methods=['POST'])
def handle_critical_alert():
    alert = request.json
    # Handle critical alerts (e.g., pause processing)
    return "OK"

@app.route('/webhook/budget', methods=['POST'])
def handle_budget_alert():
    alert = request.json
    # Handle budget alerts (e.g., stop expensive operations)
    return "OK"
```

## Scaling Considerations

1. **Prometheus Storage**: Configure retention and downsampling
   ```yaml
   - '--storage.tsdb.retention.time=30d'
   - '--storage.tsdb.retention.size=50GB'
   ```

2. **High Availability**: Run multiple Prometheus instances
3. **Long-term Storage**: Consider Thanos or Cortex
4. **Dashboard Backups**: Export and version control dashboards

## Troubleshooting

### No metrics appearing
1. Check application is exposing metrics: `curl localhost:8000/metrics`
2. Verify Prometheus can reach application
3. Check Prometheus targets: http://localhost:9090/targets

### Alerts not firing
1. Check alert rules syntax: `promtool check rules alerts.yml`
2. Verify metrics exist in Prometheus
3. Check Alertmanager logs: `docker-compose logs alertmanager`

### Dashboard errors
1. Verify datasource configuration
2. Check Grafana logs: `docker-compose logs grafana`
3. Ensure required plugins are installed

## Production Deployment

For production:

1. Use external PostgreSQL and Redis instances
2. Configure proper authentication
3. Set up TLS for all endpoints
4. Use persistent volumes for data
5. Configure backup strategies
6. Set up proper log aggregation

## Maintenance

### Update dashboards
```bash
# Export dashboard from Grafana UI
# Save to monitoring/grafana/dashboards/
docker-compose restart grafana
```

### Update alert rules
```bash
# Edit monitoring/prometheus/alerts/video_pipeline_alerts.yml
docker-compose exec prometheus kill -HUP 1
```

### Backup Prometheus data
```bash
docker-compose exec prometheus promtool tsdb snapshot /prometheus
docker cp prometheus:/prometheus/snapshots/<snapshot> ./backups/
```