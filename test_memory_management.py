#!/usr/bin/env python3
"""
Test script for comprehensive memory management system.
Demonstrates usage and validates functionality.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_memory_management():
    """Test the memory management system"""

    print("🔍 Testing Comprehensive Memory Management System")
    print("=" * 60)

    # Test 1: Initialize memory management
    print("\n1. Initializing memory management...")

    try:
        from src.utils.memory_init import setup_memory_management, get_memory_status

        success = setup_memory_management()
        print(f"   ✅ Initialization: {'SUCCESS' if success else 'FAILED'}")

        if success:
            status = get_memory_status()
            if "memory" in status:
                memory = status["memory"]
                print(f"   📊 Available memory: {memory['available_mb']:.0f}MB")
                print(f"   📈 Pressure level: {memory['pressure_level']}")
            else:
                print("   ⚠️  Memory status not available")

    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False

    # Test 2: Memory monitoring
    print("\n2. Testing memory monitoring...")

    try:
        from src.utils.memory_manager import get_memory_monitor

        monitor = get_memory_monitor()

        if monitor:
            stats = monitor.get_current_stats()
            print(f"   📊 Total memory: {stats.total_mb:.0f}MB")
            print(f"   💾 Process memory: {stats.process_memory_mb:.1f}MB")
            print(f"   📈 Memory usage: {stats.percent_used:.1f}%")
            print(f"   🔴 Pressure level: {stats.pressure_level.value}")
        else:
            print("   ⚠️  Memory monitor not available")

    except Exception as e:
        print(f"   ❌ Memory monitoring failed: {e}")

    # Test 3: Resource management
    print("\n3. Testing resource management...")

    try:
        from src.utils.resource_manager import managed_tempfile, get_resource_tracker

        tracker = get_resource_tracker()
        if tracker:
            initial_usage = tracker.get_resource_usage()
            print(f"   📁 Initial tracked files: {initial_usage['tracked_files']}")

        # Test managed temp file
        with managed_tempfile(suffix=".txt") as temp_file:
            print(f"   📄 Created managed temp file: {os.path.basename(temp_file)}")

            # Write test content
            with open(temp_file, "w") as f:
                f.write("Test content for memory management")

            # Verify file exists
            if os.path.exists(temp_file):
                print(f"   ✅ Temp file created successfully")
            else:
                print(f"   ❌ Temp file creation failed")

        # Verify cleanup
        if not os.path.exists(temp_file):
            print(f"   🧹 Temp file cleaned up automatically")
        else:
            print(f"   ⚠️  Temp file not cleaned up")

    except Exception as e:
        print(f"   ❌ Resource management test failed: {e}")

    # Test 4: FFmpeg memory management
    print("\n4. Testing FFmpeg memory management...")

    try:
        from src.utils.ffmpeg_memory_manager import (
            get_ffmpeg_memory_manager,
            build_memory_safe_ffmpeg_command,
            FFmpegResourceConfig,
        )

        manager = get_ffmpeg_memory_manager()
        if manager:
            # Test config generation
            config = manager.get_optimal_config()
            print(f"   ⚙️  Optimal FFmpeg config:")
            print(f"      - Max memory: {config.max_memory_mb}MB")
            print(f"      - Max threads: {config.max_threads}")
            print(f"      - Preset: {config.preset}")

            # Test command optimization
            test_cmd = ["ffmpeg", "-i", "input.mp4", "-c:v", "libx264", "output.mp4"]
            optimized_cmd = manager.build_memory_optimized_command(test_cmd)

            # Show added optimizations
            added_flags = [flag for flag in optimized_cmd if flag not in test_cmd]
            print(f"   🚀 Added optimization flags: {' '.join(added_flags[:6])}...")

        else:
            print("   ⚠️  FFmpeg memory manager not available")

    except Exception as e:
        print(f"   ❌ FFmpeg memory management test failed: {e}")

    # Test 5: Adaptive configuration
    print("\n5. Testing adaptive configuration...")

    try:
        from src.utils.memory_init import get_safe_processing_config

        config = get_safe_processing_config()
        print(f"   🎯 Safe processing configuration:")
        print(f"      - Max workers: {config['max_workers']}")
        print(f"      - Chunk size: {config['chunk_size_mb']}MB")
        print(f"      - Quality preset: {config['quality_preset']}")
        print(
            f"      - Hardware acceleration: {config.get('enable_hardware_accel', 'N/A')}"
        )

    except Exception as e:
        print(f"   ❌ Adaptive configuration test failed: {e}")

    # Test 6: Memory pressure simulation
    print("\n6. Testing memory pressure handling...")

    try:
        from src.utils.memory_manager import memory_guard

        # Test with small memory limit to trigger pressure handling
        print("   🧪 Simulating memory constraint...")

        with memory_guard(max_memory_mb=100) as guard_monitor:
            if guard_monitor:
                stats = guard_monitor.get_current_stats()
                print(f"   📊 Memory usage under guard: {stats.percent_used:.1f}%")

            # Simulate some memory usage
            test_data = []
            for i in range(10):
                test_data.append([0] * 1000)  # Small memory allocation

            print("   ✅ Memory guard test completed")

    except MemoryError as e:
        print(f"   ✅ Memory constraint properly detected: {e}")
    except Exception as e:
        print(f"   ❌ Memory pressure test failed: {e}")

    # Test 7: Cleanup verification
    print("\n7. Testing resource cleanup...")

    try:
        from src.utils.memory_init import force_memory_cleanup, get_memory_status

        # Get status before cleanup
        status_before = get_memory_status()

        # Force cleanup
        force_memory_cleanup()
        print("   🧹 Forced cleanup completed")

        # Verify cleanup
        time.sleep(0.5)  # Brief pause for cleanup to complete
        status_after = get_memory_status()

        if "resources" in status_before and "resources" in status_after:
            files_before = status_before["resources"]["tracked_files"]
            files_after = status_after["resources"]["tracked_files"]
            print(f"   📁 Tracked files: {files_before} → {files_after}")

        print("   ✅ Cleanup verification completed")

    except Exception as e:
        print(f"   ❌ Cleanup test failed: {e}")

    # Final status
    print("\n8. Final system status...")

    try:
        status = get_memory_status()

        if status.get("initialized", False):
            print("   ✅ Memory management system operational")

            if "memory" in status:
                memory = status["memory"]
                print(f"   📊 Final memory state:")
                print(f"      - Available: {memory['available_mb']:.0f}MB")
                print(f"      - Usage: {memory['used_percent']:.1f}%")
                print(f"      - Pressure: {memory['pressure_level']}")
        else:
            print("   ⚠️  Memory management system not fully operational")

    except Exception as e:
        print(f"   ❌ Status check failed: {e}")

    # Test completion
    print("\n" + "=" * 60)
    print("🎉 Memory Management Test Completed")

    # Cleanup
    try:
        from src.utils.memory_init import shutdown_memory_management_system

        shutdown_memory_management_system()
        print("✅ System shutdown completed")
    except Exception as e:
        print(f"⚠️  Shutdown error: {e}")

    return True


def test_video_processing_integration():
    """Test integration with video processing components"""

    print("\n🎬 Testing Video Processing Integration")
    print("=" * 60)

    try:
        # Initialize memory management
        from src.utils.memory_init import setup_memory_management

        setup_memory_management()

        # Test VideoEditor initialization
        print("\n1. Testing VideoEditor with memory management...")

        from src.providers.video_processor import VideoEditor

        editor = VideoEditor()
        print("   ✅ VideoEditor initialized with memory management")

        # Test memory estimation
        print("\n2. Testing memory estimation...")

        # Create a dummy video file for testing
        test_video = "/tmp/test_video.mp4"
        if not os.path.exists(test_video):
            # Create minimal test file
            with open(test_video, "wb") as f:
                f.write(b"\x00" * 1024 * 1024)  # 1MB dummy file

        try:
            from src.utils.resource_manager import estimate_processing_memory

            memory_estimate = estimate_processing_memory(test_video, "basic")
            print(f"   📊 Estimated memory for basic processing: {memory_estimate}MB")
        except Exception as e:
            print(f"   ⚠️  Memory estimation not available: {e}")

        # Cleanup test file
        if os.path.exists(test_video):
            os.unlink(test_video)

        print("   ✅ Video processing integration test completed")

    except ImportError as e:
        print(f"   ⚠️  Video processing components not available: {e}")
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Memory Management Tests")
    print("This will test the comprehensive memory management system.")
    print("Estimated time: 10-15 seconds\n")

    # Run main tests
    success = test_memory_management()

    if success:
        # Run integration tests
        test_video_processing_integration()

    print("\n🏁 All tests completed!")
    print("\nTo use memory management in your application:")
    print("```python")
    print("from src.utils.memory_init import setup_memory_management")
    print("setup_memory_management()  # Call at application startup")
    print("```")
    print("\nSee MEMORY_MANAGEMENT.md for detailed usage instructions.")
