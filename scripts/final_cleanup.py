#!/usr/bin/env python3
"""
Final cleanup sweep - remove test artifacts, logs, and temporary files
Part of repository hygiene Pass E
"""

import os
import glob
import shutil
from pathlib import Path

def clean_build_artifacts():
    """Remove Python build artifacts"""
    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo", 
        "**/*.pyd",
        "**/.pytest_cache",
        "**/*.egg-info",
        "**/build",
        "**/dist"
    ]
    
    removed = []
    for pattern in patterns:
        for item in glob.glob(pattern, recursive=True):
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.unlink(item)
                removed.append(item)
    
    return removed

def clean_log_files():
    """Remove log and output files"""
    patterns = [
        "*.log",
        "*.out", 
        "*.tmp",
        "*_test.json",
        "*_metrics.json",
        "*_plan_*.json",
        "eval_*.out",
        "settings_stage*.json",
        "perf_*.json",
        "db_*.json",
        "mem_*.json",
        "wrk_*.log",
        "pytest_*.log",
        "full_pipeline_test.log",
        "pipeline_run.log"
    ]
    
    removed = []
    for pattern in patterns:
        for item in glob.glob(pattern):
            if os.path.exists(item) and os.path.isfile(item):
                os.unlink(item)
                removed.append(item)
    
    return removed

def clean_test_artifacts():
    """Remove test artifacts and temporary files"""
    # Test data files that can be regenerated
    test_files = [
        "test_analysis.json",
        "full_video_analysis.json", 
        "approved_story_structure.json",
        "clean_plan.json",
        "cov.json",
        "energy_test.log",
        "final_energy_test.log",
        "logging.json",
        "test_plan.json",
        "stub_scan.out"
    ]
    
    removed = []
    for file in test_files:
        if os.path.exists(file):
            os.unlink(file)
            removed.append(file)
    
    return removed

def clean_empty_directories():
    """Remove empty directories"""
    removed = []
    
    # Look for empty directories
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Directory is empty
                    os.rmdir(dir_path)
                    removed.append(dir_path)
            except (OSError, PermissionError):
                pass  # Skip if can't access or remove
    
    return removed

def preserve_important_files():
    """List of files to preserve"""
    preserve = [
        # Core project files
        "requirements.txt",
        "pyproject.toml", 
        "setup.py",
        "CLAUDE.md",
        "README.md",
        ".env.example",
        ".gitignore",
        
        # Configuration
        "pytest.ini",
        "docker-compose.yml",
        "Dockerfile",
        
        # Scripts
        "flatten_dirs.sh",
        
        # Keep one baseline for reference
        "perf_baseline.json",
        "canary_metrics.json"
    ]
    
    return preserve

def main():
    """Run final cleanup sweep"""
    print("🧹 Starting final cleanup sweep...")
    
    preserve_files = preserve_important_files()
    total_removed = []
    
    # 1. Clean build artifacts
    print("   Removing build artifacts...")
    removed = clean_build_artifacts()
    total_removed.extend(removed)
    print(f"   📦 Removed {len(removed)} build artifacts")
    
    # 2. Clean log files
    print("   Removing log files...")
    removed = clean_log_files()
    # Filter out preserved files
    removed = [f for f in removed if f not in preserve_files]
    for f in removed:
        if os.path.exists(f):
            os.unlink(f)
    total_removed.extend(removed)
    print(f"   📝 Removed {len(removed)} log files")
    
    # 3. Clean test artifacts
    print("   Removing test artifacts...")
    removed = clean_test_artifacts() 
    total_removed.extend(removed)
    print(f"   🧪 Removed {len(removed)} test artifacts")
    
    # 4. Clean empty directories
    print("   Removing empty directories...")
    removed = clean_empty_directories()
    total_removed.extend(removed)
    print(f"   📁 Removed {len(removed)} empty directories")
    
    print(f"\n✅ Final cleanup complete!")
    print(f"   Total items removed: {len(total_removed)}")
    
    if len(total_removed) < 50:  # Show details if not too many
        print(f"\n📋 Items removed:")
        for item in sorted(total_removed)[:20]:  # Show first 20
            print(f"   - {item}")
        if len(total_removed) > 20:
            print(f"   ... and {len(total_removed) - 20} more")

if __name__ == "__main__":
    main()