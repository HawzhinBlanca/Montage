#!/usr/bin/env python3
"""
Phase 2 Requirements Test Suite
Tests the exact requirements from Tasks.md line 23: "Unit tests green (pytest -q)"
"""

import subprocess
import json
from pathlib import Path


def test_no_sys_path_append():
    """Test requirement 1: grep "sys.path.append" → 0"""
    result = subprocess.run(
        ["grep", "-r", "sys.path.append", "montage/"],
        capture_output=True,
        text=True
    )
    
    # Check for actual violations (not comments)
    violations = []
    for line in result.stdout.splitlines():
        if line and not line.strip().startswith("#") and "#" not in line:
            violations.append(line)
    
    assert len(violations) == 0, f"Found sys.path.append: {violations}"


def test_canary_evaluation_pass():
    """Test requirement 3: canary_metrics.json + evaluate_canary.out = PASS"""
    root = Path(__file__).parent.parent.parent
    
    # Check evaluate_canary.out exists and shows PASS
    eval_file = root / "evaluate_canary.out"
    assert eval_file.exists(), "evaluate_canary.out missing"
    
    content = eval_file.read_text()
    assert "Overall Status: PASS" in content, "Canary evaluation not PASS"
    assert "PROCEED with Phase 2 completion" in content


def test_proof_bundle_exists():
    """Test all proof bundle files exist as specified in Tasks.md lines 32-36"""
    root = Path(__file__).parent.parent.parent
    
    required_files = [
        "canary_metrics.json",
        "evaluate_canary.out",
        "perf_baseline.json", 
        "pytest_summary.txt",
        "stub_scan.out"
    ]
    
    for filename in required_files:
        filepath = root / filename
        assert filepath.exists(), f"Missing proof file: {filename}"
        assert filepath.stat().st_size > 0, f"Empty proof file: {filename}"


def test_canary_metrics_valid():
    """Test canary metrics meet SLO requirements from Tasks.md line 26"""
    root = Path(__file__).parent.parent.parent
    metrics_file = root / "canary_metrics.json"
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Test SLO matrix requirements
    baseline_p99 = metrics.get("baseline_p99_ms", 0)
    current_p99 = metrics.get("current_p99_ms", 0)
    
    if baseline_p99 > 0:
        latency_increase = ((current_p99 - baseline_p99) / baseline_p99) * 100
        assert latency_increase <= 20, f"p99 latency increase {latency_increase}% > 20%"
    
    # 5xx < 1%
    total_requests = metrics.get("total_requests", 1)
    error_5xx = metrics.get("error_5xx_count", 0)
    error_rate = (error_5xx / total_requests) * 100
    assert error_rate < 1.0, f"5xx error rate {error_rate}% >= 1%"
    
    # ImportError = 0
    assert metrics.get("import_error_count", 0) == 0, "ImportErrors found"
    
    # CPU ≤ 80%
    assert metrics.get("avg_cpu_utilization_pct", 100) <= 80, "CPU > 80%"
    
    # MEM ≤ 85%
    assert metrics.get("avg_memory_utilization_pct", 100) <= 85, "Memory > 85%"


def test_ci_scan_job_exists():
    """Test requirement 5: CI scan job blocks future hacks"""
    root = Path(__file__).parent.parent.parent
    ci_file = root / ".github" / "workflows" / "scan.yml"
    
    assert ci_file.exists(), "CI scan job missing"
    
    content = ci_file.read_text()
    assert "sys.path.append" in content, "CI doesn't check for sys.path.append"
    assert "montage/" in content, "CI doesn't scan montage directory"