#!/usr/bin/env python3
"""
Unit tests for Phase 7 performance guard
"""

import json
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_perf_guard import PerformanceGuard


@pytest.fixture
def baseline_metrics():
    """Create test baseline metrics"""
    return {
        "latency_baseline": {
            "p95_ms": 100.0
        },
        "baseline_thresholds": {
            "latency_p95_threshold_ms": 115.0,  # +15%
            "cpu_threshold_percent": 80,
            "memory_growth_threshold_percent": 10,
            "error_rate_threshold_percent": 1.0
        },
        "resource_baseline": {
            "memory_rss_mb": {
                "avg": 1000.0
            }
        }
    }


@pytest.fixture
def temp_baseline_file(baseline_metrics):
    """Create temporary baseline file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(baseline_metrics, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def test_latency_pass(temp_baseline_file):
    """Test latency check passes when under threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {"latency_p95_ms": 110.0}  # Under 115ms threshold
    passed, status = guard.evaluate_latency(metrics)

    assert passed is True
    assert "PASS" in status
    assert guard.results['checks']['latency']['passed'] is True


def test_latency_fail(temp_baseline_file):
    """Test latency check fails when over threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {"latency_p95_ms": 120.0}  # Over 115ms threshold
    passed, status = guard.evaluate_latency(metrics)

    assert passed is False
    assert "FAIL" in status
    assert len(guard.results['violations']) == 1


def test_cpu_pass(temp_baseline_file):
    """Test CPU check passes when under threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {
        "cpu_avg_percent": 50.0,
        "cpu_max_percent": 75.0  # Under 80% threshold
    }
    passed, status = guard.evaluate_cpu(metrics)

    assert passed is True
    assert "PASS" in status


def test_cpu_fail(temp_baseline_file):
    """Test CPU check fails when over threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {
        "cpu_avg_percent": 70.0,
        "cpu_max_percent": 85.0  # Over 80% threshold
    }
    passed, status = guard.evaluate_cpu(metrics)

    assert passed is False
    assert "FAIL" in status


def test_memory_growth_pass(temp_baseline_file):
    """Test memory growth check passes when under threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {
        "memory_initial_mb": 1000.0,
        "memory_final_mb": 1090.0  # 9% growth, under 10% threshold
    }
    passed, status = guard.evaluate_memory_growth(metrics)

    assert passed is True
    assert "PASS" in status
    assert guard.results['checks']['memory_growth']['growth_percent'] == 9.0


def test_memory_growth_fail(temp_baseline_file):
    """Test memory growth check fails when over threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {
        "memory_initial_mb": 1000.0,
        "memory_final_mb": 1150.0  # 15% growth, over 10% threshold
    }
    passed, status = guard.evaluate_memory_growth(metrics)

    assert passed is False
    assert "FAIL" in status


def test_error_rate_pass(temp_baseline_file):
    """Test error rate check passes when under threshold"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    metrics = {"error_rate_percent": 0.5}  # Under 1% threshold
    passed, status = guard.evaluate_error_rate(metrics)

    assert passed is True
    assert "PASS" in status


def test_evaluate_all_pass(temp_baseline_file):
    """Test full evaluation with all checks passing"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    # Create metrics file that passes all checks
    metrics = {
        "latency_p95_ms": 110.0,
        "cpu_avg_percent": 50.0,
        "cpu_max_percent": 75.0,
        "memory_initial_mb": 1000.0,
        "memory_final_mb": 1090.0,
        "error_rate_percent": 0.5
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metrics, f)
        metrics_file = f.name

    try:
        results = guard.evaluate_all(metrics_file)
        assert results['status'] == 'PASS'
        assert len(results['violations']) == 0
        assert all(check['passed'] for check in results['checks'].values())
    finally:
        Path(metrics_file).unlink(missing_ok=True)


def test_evaluate_all_fail(temp_baseline_file):
    """Test full evaluation with some checks failing"""
    guard = PerformanceGuard(baseline_file=temp_baseline_file)

    # Create metrics file with failures
    metrics = {
        "latency_p95_ms": 120.0,  # FAIL
        "cpu_avg_percent": 70.0,
        "cpu_max_percent": 85.0,  # FAIL
        "memory_initial_mb": 1000.0,
        "memory_final_mb": 1090.0,  # PASS
        "error_rate_percent": 0.5  # PASS
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(metrics, f)
        metrics_file = f.name

    try:
        results = guard.evaluate_all(metrics_file)
        assert results['status'] == 'FAIL'
        assert len(results['violations']) == 2
        assert results['checks']['latency']['passed'] is False
        assert results['checks']['cpu']['passed'] is False
    finally:
        Path(metrics_file).unlink(missing_ok=True)
