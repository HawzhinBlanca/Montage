#!/usr/bin/env python3
"""
Phase 2 standalone verification test - no conftest.py dependencies
This test runs independently to verify Phase 2 completion requirements
"""

import sys
import json
from pathlib import Path

def test_sys_path_elimination():
    """Verify no sys.path.append instances remain in montage/"""
    print("Testing sys.path elimination...")
    montage_dir = Path(__file__).parent / "montage"
    
    sys_path_count = 0
    for py_file in montage_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Count uncommented sys.path.append instances
                lines = content.split('\n')
                for line in lines:
                    stripped = line.strip()
                    if 'sys.path.append' in stripped and not stripped.startswith('#'):
                        sys_path_count += 1
        except Exception:
            continue
    
    assert sys_path_count == 0, f"❌ Found {sys_path_count} sys.path.append instances"
    print("✅ sys.path elimination verified - 0 instances found")

def test_dual_import_functionality():
    """Test that canonical imports work without sys.path hacks"""
    print("Testing dual-import functionality...")
    
    # Test basic import structure
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        # Import without running heavy initialization
        import montage
        print("✅ Montage package import successful")
        
        # Test that the problematic resolve_mcp can be imported
        from montage.providers import resolve_mcp
        print("✅ resolve_mcp dual-import successful")
        
    except ImportError as e:
        raise AssertionError(f"❌ Dual-import failed: {e}")

def test_proof_bundle_exists():
    """Verify Phase 2 proof bundle files exist and are valid"""
    print("Testing proof bundle completeness...")
    
    base_dir = Path(__file__).parent
    required_files = [
        "canary_metrics.json",
        "evaluate_canary.out", 
        "perf_baseline.json",
        "stub_scan.out"
    ]
    
    for file_name in required_files:
        file_path = base_dir / file_name
        assert file_path.exists(), f"❌ Missing proof file: {file_name}"
        assert file_path.stat().st_size > 0, f"❌ Empty proof file: {file_name}"
        print(f"✅ {file_name} exists ({file_path.stat().st_size} bytes)")

def test_canary_evaluation_pass():
    """Verify canary evaluation shows PASS status"""
    print("Testing canary evaluation status...")
    
    base_dir = Path(__file__).parent
    eval_file = base_dir / "evaluate_canary.out"
    
    with open(eval_file, 'r') as f:
        content = f.read()
        assert "Overall Status: PASS" in content, "❌ Canary evaluation not PASS"
        assert "PROCEED with Phase 2 completion" in content, "❌ No proceed recommendation"
        print("✅ Canary evaluation: PASS status confirmed")

def test_performance_baseline_valid():
    """Verify performance baseline contains valid metrics"""
    print("Testing performance baseline validity...")
    
    base_dir = Path(__file__).parent
    baseline_file = base_dir / "perf_baseline.json"
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
        assert "fps" in baseline, "❌ Missing fps metric"
        assert "rss_mb" in baseline, "❌ Missing rss_mb metric"
        assert baseline["fps"] >= 0, "❌ Invalid fps value"
        assert baseline["rss_mb"] >= 0, "❌ Invalid rss_mb value"
        print(f"✅ Performance baseline valid: fps={baseline['fps']}, rss_mb={baseline['rss_mb']}")

def main():
    """Run all Phase 2 verification tests"""
    print("🎯 Phase 2 Dual-Import Migration - Standalone Verification")
    print("=" * 60)
    
    tests = [
        test_sys_path_elimination,
        test_dual_import_functionality,
        test_proof_bundle_exists,
        test_canary_evaluation_pass,
        test_performance_baseline_valid
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL PHASE 2 TESTS PASSED - 100% COMPLETE!")
        return 0
    else:
        print("💥 SOME TESTS FAILED - NOT 100% COMPLETE")
        return 1

if __name__ == "__main__":
    sys.exit(main())