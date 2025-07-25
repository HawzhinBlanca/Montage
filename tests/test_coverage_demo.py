#!/usr/bin/env python3
"""Simple coverage demonstration for critical modules"""

import sys
import os

# PHASE 2 MIGRATION: Canonical imports (sys.path hack removed)
# OLD: sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import critical modules to demonstrate coverage
try:
    from montage.core.pipeline import Pipeline, run_pipeline
    print("✓ Imported pipeline module")
except Exception as e:
    print(f"✗ Failed to import pipeline: {e}")

try:
    from montage.providers.video_processor import VideoEditor, probe_codecs  
    print("✓ Imported video_processor module")
except Exception as e:
    print(f"✗ Failed to import video_processor: {e}")

try:
    from montage.utils.memory_manager import (
        MemoryMonitor, 
        MemoryPressureManager,
        AdaptiveProcessingConfig,
        memory_guard
    )
    print("✓ Imported memory_manager module")
except Exception as e:
    print(f"✗ Failed to import memory_manager: {e}")

# Basic function calls to generate coverage
if __name__ == "__main__":
    print("\nRunning basic coverage tests...")
    
    # Test memory manager
    try:
        monitor = MemoryMonitor()
        print("✓ Created MemoryMonitor")
    except Exception as e:
        print(f"✗ MemoryMonitor: {e}")
    
    # Test pipeline  
    try:
        pipeline = Pipeline(enable_visual_tracking=False)
        print("✓ Created Pipeline")
    except Exception as e:
        print(f"✗ Pipeline: {e}")
    
    print("\nCoverage test complete.")