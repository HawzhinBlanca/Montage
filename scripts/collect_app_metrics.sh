#!/bin/bash
# Phase 4-0: Collect baseline metrics for API merge decision

set -euo pipefail

# Args
DURATION="${1:-48h}"
OUTPUT_FILE="${2:-app_metrics_premerge.json}"

# Prometheus URL (local or staging)
PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"

echo "Collecting app metrics for duration: $DURATION"
echo "Output file: $OUTPUT_FILE"

# Helper function to query Prometheus
query_prometheus() {
    local query="$1"
    local result=$(curl -s -G \
        --data-urlencode "query=$query" \
        "${PROM_URL}/api/v1/query" | \
        jq -r '.data.result')
    echo "$result"
}

# Helper to get range data
query_range() {
    local query="$1"
    local duration="$2"
    local end_time=$(date +%s)
    local start_time=$((end_time - $(echo $duration | sed 's/h/*3600/' | bc)))
    
    curl -s -G \
        --data-urlencode "query=$query" \
        --data-urlencode "start=$start_time" \
        --data-urlencode "end=$end_time" \
        --data-urlencode "step=300" \
        "${PROM_URL}/api/v1/query_range" | \
        jq -r '.data.result'
}

# Collect current metrics
echo "Querying Prometheus..."

# Generate metrics JSON
cat > "$OUTPUT_FILE" <<EOF
{
  "collection_time": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
  "duration": "$DURATION",
  "prometheus_url": "$PROM_URL",
  "apps": {
    "public": {
      "job_name": "montage-public",
      "metrics": {
        "req_total": $(query_prometheus 'app:req_total{job="montage-public"}' | jq -r '.[0].value[1] // 0'),
        "latency_p50_ms": $(query_prometheus 'app:latency_p50_ms{job="montage-public"}' | jq -r '.[0].value[1] // 0'),
        "latency_p90_ms": $(query_prometheus 'app:latency_p90_ms{job="montage-public"}' | jq -r '.[0].value[1] // 0'),
        "latency_p95_ms": $(query_prometheus 'app:latency_p95_ms{job="montage-public"}' | jq -r '.[0].value[1] // 0'),
        "latency_p99_ms": $(query_prometheus 'app:latency_p99_ms{job="montage-public"}' | jq -r '.[0].value[1] // 0'),
        "error_rate": $(query_prometheus 'app:error_rate{job="montage-public"}' | jq -r '.[0].value[1] // 0'),
        "memory_usage_mb": $(query_prometheus 'avg(app:memory_usage_mb{job="montage-public"})' | jq -r '.[0].value[1] // 0'),
        "cpu_usage_cores": $(query_prometheus 'avg(app:cpu_usage_cores{job="montage-public"})' | jq -r '.[0].value[1] // 0'),
        "active_connections": $(query_prometheus 'app:active_connections{job="montage-public"}' | jq -r '.[0].value[1] // 0')
      }
    },
    "admin": {
      "job_name": "montage-admin",
      "metrics": {
        "req_total": $(query_prometheus 'app:req_total{job="montage-admin"}' | jq -r '.[0].value[1] // 0'),
        "latency_p50_ms": $(query_prometheus 'app:latency_p50_ms{job="montage-admin"}' | jq -r '.[0].value[1] // 0'),
        "latency_p90_ms": $(query_prometheus 'app:latency_p90_ms{job="montage-admin"}' | jq -r '.[0].value[1] // 0'),
        "latency_p95_ms": $(query_prometheus 'app:latency_p95_ms{job="montage-admin"}' | jq -r '.[0].value[1] // 0'),
        "latency_p99_ms": $(query_prometheus 'app:latency_p99_ms{job="montage-admin"}' | jq -r '.[0].value[1] // 0'),
        "error_rate": $(query_prometheus 'app:error_rate{job="montage-admin"}' | jq -r '.[0].value[1] // 0'),
        "memory_usage_mb": $(query_prometheus 'avg(app:memory_usage_mb{job="montage-admin"})' | jq -r '.[0].value[1] // 0'),
        "cpu_usage_cores": $(query_prometheus 'avg(app:cpu_usage_cores{job="montage-admin"})' | jq -r '.[0].value[1] // 0'),
        "active_connections": $(query_prometheus 'app:active_connections{job="montage-admin"}' | jq -r '.[0].value[1] // 0')
      }
    }
  },
  "aggregated": {
    "total_req_rate": $(query_prometheus 'sum(app:req_total)' | jq -r '.[0].value[1] // 0'),
    "combined_memory_mb": $(query_prometheus 'sum(app:memory_usage_mb)' | jq -r '.[0].value[1] // 0'),
    "combined_cpu_cores": $(query_prometheus 'sum(app:cpu_usage_cores)' | jq -r '.[0].value[1] // 0'),
    "deployment_footprint": {
      "public_replicas": $(query_prometheus 'count(up{job="montage-public"})' | jq -r '.[0].value[1] // 0'),
      "admin_replicas": $(query_prometheus 'count(up{job="montage-admin"})' | jq -r '.[0].value[1] // 0'),
      "total_pods": $(query_prometheus 'count(up{job=~"montage-(public|admin)"})' | jq -r '.[0].value[1] // 0')
    }
  },
  "traffic_ratio": {
    "public_percentage": $(query_prometheus 'app:req_total{job="montage-public"} / ignoring(job) sum(app:req_total) * 100' | jq -r '.[0].value[1] // 0'),
    "admin_percentage": $(query_prometheus 'app:req_total{job="montage-admin"} / ignoring(job) sum(app:req_total) * 100' | jq -r '.[0].value[1] // 0')
  }
}
EOF

echo "Metrics collected and saved to $OUTPUT_FILE"

# Pretty print summary
echo ""
echo "=== SUMMARY ==="
jq -r '
  "Public API: \(.apps.public.metrics.req_total) req/s, P95: \(.apps.public.metrics.latency_p95_ms)ms",
  "Admin API: \(.apps.admin.metrics.req_total) req/s, P95: \(.apps.admin.metrics.latency_p95_ms)ms",
  "Total Memory: \(.aggregated.combined_memory_mb)MB across \(.aggregated.deployment_footprint.total_pods) pods",
  "Traffic Split: Public=\(.traffic_ratio.public_percentage)%, Admin=\(.traffic_ratio.admin_percentage)%"
' "$OUTPUT_FILE"