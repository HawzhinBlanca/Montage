#!/usr/bin/env python3
"""
Resource management utilities for video processing with memory-safe operations.
Provides context managers and utilities for proper resource cleanup.
"""

import os
import gc
import cv2
import time
import psutil
import signal
import tempfile
import subprocess
import threading
import weakref
from typing import Dict, Any, Optional, List, Union, Set, ContextManager
from contextlib import contextmanager, ExitStack
from pathlib import Path
import atexit

# Import centralized logging and memory manager
try:
    from ..utils.logging_config import get_logger
    from .memory_manager import get_memory_monitor, MemoryPressureLevel
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logging_config import get_logger
    from utils.memory_manager import get_memory_monitor, MemoryPressureLevel

logger = get_logger(__name__)


class ResourceTracker:
    """Tracks and manages system resources during video processing"""

    def __init__(self):
        self._tracked_files: Set[str] = set()
        self._tracked_processes: Dict[int, subprocess.Popen] = {}
        self._tracked_opencv_objects: List[weakref.ref] = []
        self._temp_directories: Set[str] = set()
        self._file_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

        # Register cleanup on exit
        atexit.register(self.cleanup_all)

        logger.debug("Resource tracker initialized")

    def track_file(self, filepath: str, cleanup_on_exit: bool = True):
        """Track a file for automatic cleanup"""
        with self._lock:
            self._tracked_files.add(filepath)
        logger.debug(f"Tracking file: {os.path.basename(filepath)}")

    def untrack_file(self, filepath: str):
        """Stop tracking a file"""
        with self._lock:
            self._tracked_files.discard(filepath)
        logger.debug(f"Stopped tracking file: {os.path.basename(filepath)}")

    def track_process(self, process: subprocess.Popen, name: str = ""):
        """Track a process for cleanup"""
        with self._lock:
            self._tracked_processes[process.pid] = process
        logger.debug(f"Tracking process {process.pid}: {name}")

    def untrack_process(self, pid: int):
        """Stop tracking a process"""
        with self._lock:
            self._tracked_processes.pop(pid, None)
        logger.debug(f"Stopped tracking process {pid}")

    def track_opencv_object(self, obj):
        """Track OpenCV objects for cleanup"""
        if hasattr(obj, "release"):
            weak_ref = weakref.ref(obj, self._opencv_cleanup_callback)
            self._tracked_opencv_objects.append(weak_ref)

    def _opencv_cleanup_callback(self, weak_ref):
        """Callback for OpenCV object cleanup"""
        try:
            self._tracked_opencv_objects.remove(weak_ref)
        except ValueError:
            pass

    def create_temp_directory(self) -> str:
        """Create and track a temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix="montage_video_")
        with self._lock:
            self._temp_directories.add(temp_dir)
        logger.debug(f"Created temp directory: {temp_dir}")
        return temp_dir

    def cleanup_file(self, filepath: str, force: bool = False):
        """Clean up a tracked file"""
        if not force and filepath not in self._tracked_files:
            return

        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
                logger.debug(f"Cleaned up file: {os.path.basename(filepath)}")
        except OSError as e:
            logger.warning(f"Failed to cleanup file {filepath}: {e}")

        self.untrack_file(filepath)

    def cleanup_process(self, pid: int, timeout: float = 5.0):
        """Clean up a tracked process"""
        with self._lock:
            process = self._tracked_processes.get(pid)

        if process is None:
            return

        try:
            if process.poll() is None:  # Process still running
                logger.debug(f"Terminating process {pid}")
                process.terminate()

                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {pid}")
                    process.kill()
                    process.wait(timeout=2)

        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"Failed to cleanup process {pid}: {e}")

        self.untrack_process(pid)

    def cleanup_temp_directories(self):
        """Clean up all tracked temporary directories"""
        with self._lock:
            temp_dirs = list(self._temp_directories)
            self._temp_directories.clear()

        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    import shutil

                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

    def cleanup_all(self):
        """Clean up all tracked resources"""
        logger.info("Cleaning up all tracked resources")

        # Clean up processes first
        with self._lock:
            processes = list(self._tracked_processes.keys())

        for pid in processes:
            self.cleanup_process(pid)

        # Clean up files
        with self._lock:
            files = list(self._tracked_files)

        for filepath in files:
            self.cleanup_file(filepath, force=True)

        # Clean up temp directories
        self.cleanup_temp_directories()

        # Force garbage collection
        gc.collect()

        logger.info("Resource cleanup completed")

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        with self._lock:
            return {
                "tracked_files": len(self._tracked_files),
                "tracked_processes": len(self._tracked_processes),
                "temp_directories": len(self._temp_directories),
                "opencv_objects": len(
                    [ref for ref in self._tracked_opencv_objects if ref() is not None]
                ),
            }


# Global resource tracker
_global_tracker = ResourceTracker()


def get_resource_tracker() -> ResourceTracker:
    """Get global resource tracker"""
    return _global_tracker


@contextmanager
def managed_tempfile(
    suffix: str = "",
    prefix: str = "montage_",
    directory: str = None,
    text: bool = False,
):
    """Context manager for temporary files with automatic cleanup"""

    tracker = get_resource_tracker()

    # Create temporary file
    fd, filepath = tempfile.mkstemp(
        suffix=suffix, prefix=prefix, dir=directory, text=text
    )

    try:
        # Close file descriptor since we just need the path
        os.close(fd)

        # Track file for cleanup
        tracker.track_file(filepath)

        logger.debug(f"Created managed temp file: {os.path.basename(filepath)}")
        yield filepath

    finally:
        # Cleanup file
        tracker.cleanup_file(filepath)


@contextmanager
def managed_temp_directory():
    """Context manager for temporary directories with automatic cleanup"""

    tracker = get_resource_tracker()
    temp_dir = tracker.create_temp_directory()

    try:
        yield temp_dir
    finally:
        try:
            if os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except OSError as e:
            logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

        with tracker._lock:
            tracker._temp_directories.discard(temp_dir)


@contextmanager
def managed_opencv_capture(video_path: str, memory_limit_mb: Optional[int] = None):
    """Context manager for OpenCV VideoCapture with memory monitoring"""

    monitor = get_memory_monitor()
    tracker = get_resource_tracker()

    # Check memory before opening
    if memory_limit_mb:
        stats = monitor.get_current_stats()
        if stats.available_mb < memory_limit_mb:
            raise MemoryError(
                f"Insufficient memory: {stats.available_mb:.0f}MB < {memory_limit_mb}MB"
            )

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Track for cleanup
        tracker.track_opencv_object(cap)

        logger.debug(f"Opened OpenCV capture: {os.path.basename(video_path)}")
        yield cap

    finally:
        if cap is not None:
            cap.release()
            logger.debug(f"Released OpenCV capture: {os.path.basename(video_path)}")


@contextmanager
def managed_process(
    cmd: List[str],
    name: str = "",
    memory_limit_mb: Optional[int] = None,
    timeout: Optional[float] = None,
    **popen_kwargs,
):
    """Context manager for subprocess with memory monitoring and cleanup"""

    monitor = get_memory_monitor()
    tracker = get_resource_tracker()

    # Check memory pressure before starting process
    stats = monitor.get_current_stats()
    if stats.pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
        logger.warning(
            f"Starting process under {stats.pressure_level.value} memory pressure"
        )

    process = None
    try:
        # Start process
        process = subprocess.Popen(cmd, **popen_kwargs)

        # Track process
        tracker.track_process(process, name or cmd[0])

        # Monitor memory if limit specified
        if memory_limit_mb:
            monitor.track_process(process.pid, name, is_ffmpeg=("ffmpeg" in cmd[0]))

        logger.debug(f"Started managed process {process.pid}: {name or cmd[0]}")
        yield process

        # Wait for completion if still running
        if process.poll() is None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {process.pid} timed out")
                raise

    finally:
        if process is not None:
            # Cleanup process
            tracker.cleanup_process(process.pid)

            # Stop memory tracking
            if memory_limit_mb:
                monitor.untrack_process(process.pid)


@contextmanager
def memory_constrained_operation(max_memory_mb: int, operation_name: str = ""):
    """Context manager for operations with strict memory constraints"""

    monitor = get_memory_monitor()
    start_stats = monitor.get_current_stats()

    if start_stats.available_mb < max_memory_mb:
        raise MemoryError(
            f"Insufficient memory for {operation_name}: "
            f"need {max_memory_mb}MB, available {start_stats.available_mb:.0f}MB"
        )

    logger.info(
        f"Starting memory-constrained operation: {operation_name} (max {max_memory_mb}MB)"
    )

    try:
        yield

    finally:
        # Check final memory usage
        end_stats = monitor.get_current_stats()
        memory_used = start_stats.available_mb - end_stats.available_mb

        logger.info(
            f"Memory-constrained operation completed: {operation_name} (used {memory_used:.0f}MB)"
        )

        if memory_used > max_memory_mb:
            logger.warning(
                f"Memory limit exceeded: {memory_used:.0f}MB > {max_memory_mb}MB"
            )

        # Force cleanup
        gc.collect()


@contextmanager
def ffmpeg_process_manager(
    cmd: List[str],
    memory_limit_mb: int = 1024,
    timeout: float = 3600,
    capture_output: bool = True,
):
    """Specialized context manager for FFmpeg processes with memory limits"""

    monitor = get_memory_monitor()

    # Add memory-related FFmpeg options
    optimized_cmd = cmd.copy()

    # Add memory-efficient options if not already present
    if "-threads" not in optimized_cmd:
        optimized_cmd.extend(["-threads", "2"])  # Limit threads to save memory

    if "-bufsize" not in optimized_cmd:
        optimized_cmd.extend(["-bufsize", "1M"])  # Smaller buffer size

    # Use managed process with FFmpeg-specific settings
    popen_kwargs = {
        "stdout": subprocess.PIPE if capture_output else None,
        "stderr": subprocess.PIPE if capture_output else None,
        "bufsize": 0,  # Unbuffered for real-time processing
    }

    with managed_process(
        optimized_cmd,
        name="ffmpeg",
        memory_limit_mb=memory_limit_mb,
        timeout=timeout,
        **popen_kwargs,
    ) as process:

        # Monitor FFmpeg process memory usage
        def memory_monitor_thread():
            while process.poll() is None:
                try:
                    proc_info = psutil.Process(process.pid)
                    memory_mb = proc_info.memory_info().rss / 1024 / 1024

                    if memory_mb > memory_limit_mb:
                        logger.error(
                            f"FFmpeg process {process.pid} exceeded memory limit: "
                            f"{memory_mb:.0f}MB > {memory_limit_mb}MB"
                        )
                        process.terminate()
                        break

                    time.sleep(2)  # Check every 2 seconds

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                except Exception as e:
                    logger.warning(f"Memory monitoring error: {e}")
                    break

        # Start memory monitoring thread
        monitor_thread = threading.Thread(target=memory_monitor_thread, daemon=True)
        monitor_thread.start()

        try:
            yield process
        finally:
            # Ensure monitoring thread stops
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=1.0)


class VideoResourceManager:
    """High-level resource manager for video processing operations"""

    def __init__(self, max_memory_mb: int = 4096, max_processes: int = 3):
        self.max_memory_mb = max_memory_mb
        self.max_processes = max_processes
        self.monitor = get_memory_monitor()
        self.tracker = get_resource_tracker()
        self._active_operations = 0
        self._operations_lock = threading.Lock()

    @contextmanager
    def managed_video_operation(
        self, operation_name: str, estimated_memory_mb: int = 512
    ):
        """Context manager for complete video processing operations"""

        with self._operations_lock:
            if self._active_operations >= self.max_processes:
                raise RuntimeError(
                    f"Too many active operations: {self._active_operations} >= {self.max_processes}"
                )

            self._active_operations += 1

        try:
            with memory_constrained_operation(estimated_memory_mb, operation_name):
                logger.info(f"Started video operation: {operation_name}")
                yield self
                logger.info(f"Completed video operation: {operation_name}")

        finally:
            with self._operations_lock:
                self._active_operations -= 1

    def get_optimal_chunk_size(self, video_path: str) -> int:
        """Calculate optimal chunk size for processing based on available memory"""
        try:
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            stats = self.monitor.get_current_stats()

            # Use 30% of available memory, with min/max bounds
            available_memory = stats.available_mb * 0.3
            optimal_chunks = max(2, min(10, int(file_size_mb / available_memory)))

            chunk_size_mb = int(file_size_mb / optimal_chunks)
            chunk_size_mb = max(64, min(512, chunk_size_mb))  # Between 64MB and 512MB

            logger.debug(
                f"Optimal chunk size for {os.path.basename(video_path)}: {chunk_size_mb}MB"
            )
            return chunk_size_mb

        except Exception as e:
            logger.warning(f"Failed to calculate optimal chunk size: {e}")
            return 256  # Default 256MB chunks

    def should_use_streaming(self, video_path: str) -> bool:
        """Determine if video should be processed using streaming mode"""
        try:
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            stats = self.monitor.get_current_stats()

            # Use streaming if:
            # 1. File is larger than 50% of available memory
            # 2. Memory pressure is moderate or higher
            # 3. File is larger than 500MB

            return (
                file_size_mb > stats.available_mb * 0.5
                or stats.pressure_level.value != "low"
                or file_size_mb > 500
            )

        except Exception:
            return True  # Default to streaming mode on error


# Utility functions for common resource management patterns


def cleanup_video_resources():
    """Clean up all video processing resources"""
    tracker = get_resource_tracker()
    tracker.cleanup_all()

    # Additional OpenCV cleanup
    cv2.destroyAllWindows()

    # Force garbage collection
    gc.collect()

    logger.info("Video resources cleanup completed")


def get_memory_safe_worker_count() -> int:
    """Get memory-safe worker count for parallel processing"""
    monitor = get_memory_monitor()
    stats = monitor.get_current_stats()

    # Conservative worker count based on available memory
    # Assume each worker needs ~512MB
    memory_based_workers = int(stats.available_mb / 512)
    cpu_based_workers = min(os.cpu_count() or 4, 6)  # Cap at 6 workers

    optimal_workers = min(memory_based_workers, cpu_based_workers, 4)
    optimal_workers = max(1, optimal_workers)  # At least 1 worker

    logger.debug(f"Memory-safe worker count: {optimal_workers}")
    return optimal_workers


def estimate_processing_memory(video_path: str, operation_type: str = "basic") -> int:
    """Estimate memory requirements for video processing operation"""
    try:
        file_size_mb = os.path.getsize(video_path) / 1024 / 1024

        # Memory multipliers for different operation types
        multipliers = {
            "basic": 0.2,  # Basic operations: 20% of file size
            "transcoding": 0.5,  # Transcoding: 50% of file size
            "analysis": 0.3,  # Analysis: 30% of file size
            "effects": 0.8,  # Effects/filters: 80% of file size
            "cropping": 0.4,  # Intelligent cropping: 40% of file size
        }

        multiplier = multipliers.get(operation_type, 0.3)
        base_memory = file_size_mb * multiplier

        # Add base overhead (minimum 256MB)
        estimated_mb = int(max(256, base_memory + 128))

        logger.debug(f"Memory estimate for {operation_type}: {estimated_mb}MB")
        return estimated_mb

    except Exception as e:
        logger.warning(f"Failed to estimate memory: {e}")
        return 512  # Default 512MB estimate


# Create global video resource manager instance
_global_video_manager = None


def get_video_resource_manager() -> VideoResourceManager:
    """Get global video resource manager"""
    global _global_video_manager
    if _global_video_manager is None:
        _global_video_manager = VideoResourceManager()
    return _global_video_manager


if __name__ == "__main__":
    # Test resource management
    print("Testing resource management...")

    # Test temp file management
    with managed_tempfile(suffix=".txt") as temp_file:
        print(f"Created temp file: {temp_file}")
        with open(temp_file, "w") as f:
            f.write("test content")

    # Test temp directory management
    with managed_temp_directory() as temp_dir:
        print(f"Created temp directory: {temp_dir}")

    # Test memory-constrained operation
    try:
        with memory_constrained_operation(100, "test operation"):
            print("Running memory-constrained operation...")
    except MemoryError as e:
        print(f"Memory constraint error: {e}")

    # Get resource usage
    tracker = get_resource_tracker()
    usage = tracker.get_resource_usage()
    print(f"Resource usage: {usage}")

    # Get optimal worker count
    workers = get_memory_safe_worker_count()
    print(f"Optimal workers: {workers}")

    print("Resource management test completed")
