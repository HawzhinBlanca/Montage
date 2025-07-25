#!/usr/bin/env python3
"""
Evaluate database async pool canary metrics for Phase 5

DB-specific SLOs:
- P95 query latency ≤ 12.92ms (baseline 10.77ms * 1.20)
- Pool checked_out ≤ 20 (pool_size)
- No pool overflow
- Connection errors < 1%
- Deadlocks = 0
"""

import json
import sys
from datetime import datetime


def evaluate_db_canary(metrics_file: str = "db_canary_metrics.json"):
    """Evaluate DB canary metrics against Phase 5 SLOs"""

    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Phase 5 DB-specific thresholds (from db_base.json)
    P95_LATENCY_BASELINE_MS = 10.77
    P95_LATENCY_THRESHOLD_MS = 12.92  # baseline * 1.20
    POOL_SIZE = 20

    # Standard thresholds
    MAX_ERROR_RATE = 1.0  # 1% max connection errors
    MAX_DEADLOCKS = 0     # Zero deadlocks allowed

    # Extract metrics
    p95_latency = metrics.get("query_latency_p95_ms", 0)
    pool_stats = metrics.get("pool_stats", {})
    checked_out = pool_stats.get("checked_out", 0)
    overflow = pool_stats.get("overflow", 0)
    connection_errors = metrics.get("connection_error_rate", 0)
    deadlock_count = metrics.get("deadlock_count", 0)

    # Evaluate SLOs
    results = {
        "p95_latency": {
            "value": p95_latency,
            "threshold": P95_LATENCY_THRESHOLD_MS,
            "pass": p95_latency <= P95_LATENCY_THRESHOLD_MS
        },
        "pool_usage": {
            "checked_out": checked_out,
            "pool_size": POOL_SIZE,
            "pass": checked_out <= POOL_SIZE
        },
        "pool_overflow": {
            "value": overflow,
            "threshold": 0,
            "pass": overflow == 0
        },
        "connection_errors": {
            "rate": connection_errors,
            "threshold": MAX_ERROR_RATE,
            "pass": connection_errors < MAX_ERROR_RATE
        },
        "deadlocks": {
            "count": deadlock_count,
            "threshold": MAX_DEADLOCKS,
            "pass": deadlock_count == MAX_DEADLOCKS
        }
    }

    # Overall pass/fail
    all_passed = all(check["pass"] for check in results.values())

    # Generate report
    print("# Phase 5 DB Async Pool Canary Evaluation")
    print(f"Generated: {datetime.utcnow().isoformat()}")
    print(f"Metrics file: {metrics_file}")
    print()
    print(f"Overall Status: {'PASS' if all_passed else 'FAIL'}")
    print(f"Recommendation: {'PROCEED with async pool rollout' if all_passed else 'ROLLBACK to sync pool'}")
    print()
    print("## SLO Evaluation Results")

    # P95 Latency
    latency_result = results["p95_latency"]
    print(f"{'✅' if latency_result['pass'] else '❌'} P95 query latency: "
          f"{latency_result['value']}ms vs threshold {latency_result['threshold']}ms "
          f"(+{((latency_result['value']/P95_LATENCY_BASELINE_MS - 1) * 100):.1f}%) - "
          f"{'PASS' if latency_result['pass'] else 'FAIL'}")

    # Pool usage
    pool_result = results["pool_usage"]
    print(f"{'✅' if pool_result['pass'] else '❌'} Pool usage: "
          f"{pool_result['checked_out']}/{pool_result['pool_size']} connections - "
          f"{'PASS' if pool_result['pass'] else 'FAIL'}")

    # Overflow
    overflow_result = results["pool_overflow"]
    print(f"{'✅' if overflow_result['pass'] else '❌'} Pool overflow: "
          f"{overflow_result['value']} - "
          f"{'PASS' if overflow_result['pass'] else 'FAIL'}")

    # Connection errors
    error_result = results["connection_errors"]
    print(f"{'✅' if error_result['pass'] else '❌'} Connection error rate: "
          f"{error_result['rate']:.2f}% - "
          f"{'PASS' if error_result['pass'] else 'FAIL'}")

    # Deadlocks
    deadlock_result = results["deadlocks"]
    print(f"{'✅' if deadlock_result['pass'] else '❌'} Deadlocks: "
          f"{deadlock_result['count']} - "
          f"{'PASS' if deadlock_result['pass'] else 'FAIL'}")

    print()
    print("## Baseline Comparison")
    print(f"P95 latency baseline: {P95_LATENCY_BASELINE_MS}ms")
    print(f"P95 latency current: {p95_latency}ms")
    print(f"Latency increase: {((p95_latency/P95_LATENCY_BASELINE_MS - 1) * 100):.1f}%")
    print(f"Pool utilization: {(checked_out/POOL_SIZE * 100):.1f}%")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "db_canary_metrics.json"
    evaluate_db_canary(metrics_file)
