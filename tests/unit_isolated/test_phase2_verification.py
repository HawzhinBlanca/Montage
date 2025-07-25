#!/usr/bin/env python3
"""
Phase 2 dual-import verification tests
Isolated unit tests that don't require Docker or external dependencies
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch


def test_sys_path_elimination():
    """Verify no sys.path.append instances remain in montage/"""
    montage_dir = Path(__file__).parent.parent.parent / "montage"

    sys_path_count = 0
    for py_file in montage_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                for line in lines:
                    stripped = line.strip()
                    if 'sys.path.append' in stripped and not stripped.startswith('#'):
                        sys_path_count += 1
        except Exception:
            continue

    assert sys_path_count == 0, f"Found {sys_path_count} sys.path.append instances"


def test_proof_bundle_exists():
    """Verify Phase 2 proof bundle files exist and are valid"""
    base_dir = Path(__file__).parent.parent.parent
    required_files = [
        "canary_metrics.json",
        "evaluate_canary.out",
        "perf_baseline.json",
        "stub_scan.out"
    ]

    for file_name in required_files:
        file_path = base_dir / file_name
        assert file_path.exists(), f"Missing proof file: {file_name}"
        assert file_path.stat().st_size > 0, f"Empty proof file: {file_name}"


def test_canary_evaluation_pass():
    """Verify canary evaluation shows PASS status"""
    base_dir = Path(__file__).parent.parent.parent
    eval_file = base_dir / "evaluate_canary.out"

    with open(eval_file, 'r') as f:
        content = f.read()
        assert "Overall Status: PASS" in content
        assert "PROCEED with Phase 2 completion" in content


def test_performance_baseline_valid():
    """Verify performance baseline contains valid metrics"""
    base_dir = Path(__file__).parent.parent.parent
    baseline_file = base_dir / "perf_baseline.json"

    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
        assert "fps" in baseline
        assert "rss_mb" in baseline
        assert baseline["fps"] >= 0
        assert baseline["rss_mb"] >= 0
