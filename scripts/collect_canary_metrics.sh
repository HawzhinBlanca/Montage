#!/usr/bin/env bash
set -euo pipefail
DUR=${1:-2h}
OUT=${2:-canary_metrics.json}
PROM="${PROM_PROM_URL:-}"
TOKEN="${PROM_TOKEN:-}"

if [[ -z "$PROM" || -z "$TOKEN" ]]; then
  echo "PROM_PROM_URL and PROM_TOKEN environment variables required" >&2
  exit 1
fi

q() {
  # Calculate start time (macOS compatible)
  if command -v gdate >/dev/null 2>&1; then
    START_TIME=$(gdate -d "-$DUR" +%s)
  else
    # Fallback for macOS - convert 2h to seconds and subtract
    SECONDS_AGO=$(echo "$DUR" | sed 's/h/* 3600/' | sed 's/m/* 60/' | bc -l 2>/dev/null || echo "7200")
    START_TIME=$(($(date +%s) - ${SECONDS_AGO%.*}))
  fi

  curl -sG -H "Authorization: Bearer $TOKEN" \
       --data-urlencode "query=$1" \
       --data-urlencode "start=$START_TIME" \
       --data-urlencode "end=$(date +%s)" \
       "$PROM/api/v1/query_range"
}

jq -n --arg dur "$DUR" '{
  duration:$dur,
  p99_latency: (input|fromjson),
  error_rate:  (input|fromjson),
  import_errors:(input|fromjson)
}' \
  < <(q 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job="montage"}[5m])) by (le))') \
  < <(q 'sum(rate(http_requests_total{job="montage",status=~"5..|4.."}[5m])) / sum(rate(http_requests_total{job="montage"}[5m]))') \
  < <(q 'increase(ImportError_total[5m])') \
  > "$OUT"

echo "Canary metrics stored in $OUT"
