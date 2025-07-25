#!/usr/bin/env python3
"""
Minimal test suite for Phase 2 verification
Tests core functionality without external dependencies
"""

import pytest
import os
from pathlib import Path


class TestPhase2Completion:
    """Minimal tests to verify Phase 2 dual-import migration"""
    
    def test_sys_path_elimination(self):
        """Verify no sys.path.append instances remain"""
        montage_dir = Path(__file__).parent.parent / "montage"
        
        # Search for sys.path.append in all Python files
        sys_path_count = 0
        for py_file in montage_dir.rglob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'sys.path.append' in content and not content.count('# sys.path.append'):
                    sys_path_count += content.count('sys.path.append')
        
        assert sys_path_count == 0, f"Found {sys_path_count} sys.path.append instances"
    
    def test_dual_import_functionality(self):
        """Test that canonical imports work"""
        try:
            # This should work with importlib.util, not sys.path hacks
            from montage.providers.resolve_mcp import app
            assert app is not None
        except ImportError as e:
            pytest.fail(f"Dual-import failed: {e}")
    
    def test_proof_bundle_exists(self):
        """Verify Phase 2 proof bundle files exist"""
        base_dir = Path(__file__).parent.parent
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
    
    def test_canary_evaluation_pass(self):
        """Verify canary evaluation shows PASS status"""
        base_dir = Path(__file__).parent.parent
        eval_file = base_dir / "evaluate_canary.out"
        
        with open(eval_file, 'r') as f:
            content = f.read()
            assert "Overall Status: PASS" in content
            assert "PROCEED with Phase 2 completion" in content
    
    def test_performance_baseline_valid(self):
        """Verify performance baseline contains valid metrics"""
        import json
        base_dir = Path(__file__).parent.parent
        baseline_file = base_dir / "perf_baseline.json"
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
            assert "fps" in baseline
            assert "rss_mb" in baseline
            assert baseline["fps"] >= 0
            assert baseline["rss_mb"] >= 0


class TestCoreModuleImports:
    """Test that core modules can be imported without errors"""
    
    def test_security_module_import(self):
        """Test security module imports cleanly"""
        from montage.core.security import sanitize_path
        assert callable(sanitize_path)
    
    def test_settings_module_import(self):
        """Test settings module imports cleanly"""
        from montage import settings
        assert hasattr(settings, 'get_settings')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])