#!/bin/bash
# Phase 5: Collect database performance metrics

set -euo pipefail

# Args
DURATION="${1:-2h}"
OUTPUT_FILE="${2:-db_base.json}"

# Database URL from environment or default
DATABASE_URL="${DATABASE_URL:-postgresql://localhost/montage}"

echo "Collecting DB metrics for duration: $DURATION"
echo "Output file: $OUTPUT_FILE"

# Helper function to run psql query
run_query() {
    local query="$1"
    psql "$DATABASE_URL" -t -A -c "$query" 2>/dev/null || echo "0"
}

# Get connection info
DB_HOST=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:/]*\).*/\1/p')
DB_NAME=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')

# Collect current metrics
echo "Querying database metrics..."

# Connection stats
TOTAL_CONNECTIONS=$(run_query "SELECT count(*) FROM pg_stat_activity;")
ACTIVE_CONNECTIONS=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
IDLE_CONNECTIONS=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';")
MAX_CONNECTIONS=$(run_query "SHOW max_connections;" | grep -o '[0-9]*')

# Database size
DB_SIZE=$(run_query "SELECT pg_database_size('$DB_NAME');")
DB_SIZE_HUMAN=$(run_query "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));")

# Table stats
TABLE_COUNT=$(run_query "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';")
INDEX_COUNT=$(run_query "SELECT count(*) FROM pg_indexes WHERE schemaname = 'public';")

# Query performance
QUERIES_PER_SEC=$(run_query "SELECT round(sum(calls)/extract(epoch from now() - stats_reset), 2) FROM pg_stat_user_tables;" || echo "0")
CACHE_HIT_RATIO=$(run_query "SELECT round(100.0 * sum(heap_blks_hit) / nullif(sum(heap_blks_hit) + sum(heap_blks_read),0), 2) FROM pg_statio_user_tables;" || echo "0")

# Table sizes
LARGEST_TABLES=$(run_query "
SELECT json_agg(row_to_json(t)) FROM (
    SELECT
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
    FROM pg_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 5
) t;" || echo "[]")

# Lock stats
LOCK_COUNT=$(run_query "SELECT count(*) FROM pg_locks;")
BLOCKED_QUERIES=$(run_query "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type IS NOT NULL;")

# Transaction stats
COMMITS=$(run_query "SELECT sum(xact_commit) FROM pg_stat_database WHERE datname = '$DB_NAME';" || echo "0")
ROLLBACKS=$(run_query "SELECT sum(xact_rollback) FROM pg_stat_database WHERE datname = '$DB_NAME';" || echo "0")

# Generate metrics JSON
cat > "$OUTPUT_FILE" <<EOF
{
  "collection_time": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
  "duration": "$DURATION",
  "database_url": "$DATABASE_URL",
  "database_info": {
    "host": "$DB_HOST",
    "database": "$DB_NAME",
    "version": "$(run_query "SELECT version();" | head -1)"
  },
  "connections": {
    "total": $TOTAL_CONNECTIONS,
    "active": $ACTIVE_CONNECTIONS,
    "idle": $IDLE_CONNECTIONS,
    "max_connections": $MAX_CONNECTIONS,
    "utilization_pct": $(echo "scale=2; $TOTAL_CONNECTIONS * 100 / $MAX_CONNECTIONS" | bc)
  },
  "size": {
    "database_bytes": $DB_SIZE,
    "database_human": "$DB_SIZE_HUMAN",
    "table_count": $TABLE_COUNT,
    "index_count": $INDEX_COUNT
  },
  "performance": {
    "queries_per_second": ${QUERIES_PER_SEC:-0},
    "cache_hit_ratio": ${CACHE_HIT_RATIO:-0},
    "commits": $COMMITS,
    "rollbacks": $ROLLBACKS,
    "rollback_ratio": $(echo "scale=4; $ROLLBACKS / ($COMMITS + $ROLLBACKS)" | bc || echo "0")
  },
  "locks": {
    "total_locks": $LOCK_COUNT,
    "blocked_queries": $BLOCKED_QUERIES
  },
  "largest_tables": $LARGEST_TABLES,
  "slow_queries": $(run_query "
    SELECT json_agg(row_to_json(t)) FROM (
        SELECT
            query,
            calls,
            round(total_time::numeric, 2) as total_ms,
            round(mean_time::numeric, 2) as mean_ms,
            round(max_time::numeric, 2) as max_ms
        FROM pg_stat_statements
        WHERE query NOT LIKE '%pg_stat%'
        ORDER BY mean_time DESC
        LIMIT 5
    ) t;" || echo "[]"),
  "recommendations": [
    $([ $CACHE_HIT_RATIO -lt 90 ] && echo '"Consider increasing shared_buffers for better cache performance",' || echo "")
    $([ $(echo "$TOTAL_CONNECTIONS > $MAX_CONNECTIONS * 0.8" | bc) -eq 1 ] && echo '"Connection pool near capacity, consider increasing max_connections",' || echo "")
    $([ $BLOCKED_QUERIES -gt 5 ] && echo '"High number of blocked queries detected, investigate locking issues",' || echo "")
    "Regular VACUUM and ANALYZE recommended for optimal performance"
  ]
}
EOF

echo "Metrics collected and saved to $OUTPUT_FILE"

# Pretty print summary
echo ""
echo "=== DATABASE METRICS SUMMARY ==="
echo "Connections: $ACTIVE_CONNECTIONS active, $IDLE_CONNECTIONS idle (${TOTAL_CONNECTIONS}/${MAX_CONNECTIONS} total)"
echo "Database Size: $DB_SIZE_HUMAN"
echo "Cache Hit Ratio: ${CACHE_HIT_RATIO}%"
echo "Performance: ${QUERIES_PER_SEC} queries/sec"
echo "Locks: $LOCK_COUNT total, $BLOCKED_QUERIES blocked"
