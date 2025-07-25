#!/usr/bin/env python3
"""
Performance optimization utilities for video processing pipeline
Centralized performance monitoring and optimization controls
"""

import gc
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Callable, Optional

import psutil

from .metrics import metrics

# Import centralized logging
try:
    from ..utils.logging_config import get_logger
except ImportError:
    import sys
    from pathlib import Path

    from montage.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceUsage:
    """System resource usage snapshot"""

    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_mb: float
    gpu_memory_mb: Optional[float] = None


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""

    # Processing limits
    max_workers: int = 4
    max_memory_mb: int = 8192  # 8GB
    max_cpu_percent: float = 80.0

    # Caching settings
    cache_size: int = 128
    enable_gc_optimization: bool = True

    # Video processing optimizations
    ffmpeg_threads: int = 0  # Auto-detect
    enable_hardware_acceleration: bool = True

    # Quality vs speed tradeoffs
    fast_mode: bool = False
    quality_preset: str = "medium"  # ultrafast, fast, medium, slow

    @classmethod
    def fast_preset(cls) -> "PerformanceConfig":
        """Optimized for speed over quality"""
        return cls(
            max_workers=6,
            fast_mode=True,
            quality_preset="fast",
            enable_gc_optimization=True,
        )

    @classmethod
    def quality_preset(cls) -> "PerformanceConfig":
        """Optimized for quality over speed"""
        return cls(
            max_workers=2,
            fast_mode=False,
            quality_preset="slow",
            enable_gc_optimization=False,
        )


class PerformanceMonitor:
    """Real-time performance monitoring and optimization"""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self._monitoring = False
        self._monitor_thread = None
        self._stats_lock = threading.Lock()
        self._last_stats = None

        # Initialize process monitor
        self.process = psutil.Process()

        # Try to detect GPU
        self.gpu_available = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        """Detect available GPU acceleration"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                logger.info(f"GPU detected: {gpus[0].name}")
                return True
        except ImportError:
            logger.debug("GPUtil not available, trying alternative GPU detection")

        # Check for NVIDIA
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                logger.info(f"NVIDIA GPU detected: {device_count} devices")
                return True
        except (ImportError, Exception) as e:
            logger.debug(f"NVIDIA GPU detection failed: {type(e).__name__}")

        logger.info("No GPU acceleration detected, using CPU only")
        return False

    def get_resource_usage(self) -> ResourceUsage:
        """Get current system resource usage"""
        try:
            # CPU and memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            # System memory
            sys_memory = psutil.virtual_memory()
            memory_percent = sys_memory.percent

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_mb = (
                (disk_io.read_bytes + disk_io.write_bytes) / 1024 / 1024
                if disk_io
                else 0
            )

            # GPU memory (if available)
            gpu_memory_mb = None
            if self.gpu_available:
                gpu_memory_mb = self._get_gpu_memory()

            usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_io_mb=disk_io_mb,
                gpu_memory_mb=gpu_memory_mb,
            )

            with self._stats_lock:
                self._last_stats = usage

            return usage

        except (psutil.Error, OSError, ValueError, AttributeError) as e:
            logger.debug(f"Error getting resource usage: {e}")
            return ResourceUsage(0, 0, 0, 0)

    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except (ImportError, AttributeError, OSError) as e:
            logger.debug(f"GPUtil memory query failed: {type(e).__name__}")

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used / 1024 / 1024  # Convert to MB
        except (ImportError, AttributeError, OSError) as e:
            logger.debug(f"NVIDIA memory query failed: {type(e).__name__}")

        return None

    def start_monitoring(self, interval: float = 5.0):
        """Start background resource monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                usage = self.get_resource_usage()

                # Update metrics
                metrics.set_ffmpeg_processes(
                    len([p for p in psutil.process_iter() if "ffmpeg" in p.name()])
                )

                # Check for resource pressure
                if usage.memory_percent > self.config.max_memory_mb / 1024:
                    logger.warning(f"High memory usage: {usage.memory_percent:.1f}%")

                if usage.cpu_percent > self.config.max_cpu_percent:
                    logger.warning(f"High CPU usage: {usage.cpu_percent:.1f}%")

                time.sleep(interval)

            except (psutil.Error, OSError, ValueError, AttributeError) as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track operation performance"""
        start_time = time.time()
        start_usage = self.get_resource_usage()

        try:
            yield
        finally:
            elapsed = time.time() - start_time
            end_usage = self.get_resource_usage()

            # Calculate resource deltas
            memory_delta = end_usage.memory_mb - start_usage.memory_mb
            cpu_avg = (start_usage.cpu_percent + end_usage.cpu_percent) / 2

            logger.info(
                f"Operation '{operation_name}': {elapsed:.2f}s, "
                f"CPU: {cpu_avg:.1f}%, Memory: {memory_delta:+.1f}MB"
            )

            # Track in metrics
            metrics.processing_duration.labels(stage=operation_name).observe(elapsed)


class MemoryOptimizer:
    """Memory optimization utilities"""

    @staticmethod
    def force_gc():
        """Force garbage collection and return freed memory"""
        before = psutil.Process().memory_info().rss
        gc.collect()
        after = psutil.Process().memory_info().rss
        freed_mb = (before - after) / 1024 / 1024

        if freed_mb > 1:  # Only log if significant
            logger.debug(f"Garbage collection freed {freed_mb:.1f}MB")

        return freed_mb

    @staticmethod
    @contextmanager
    def memory_guard(max_memory_mb: int = 8192):
        """Context manager to monitor memory usage"""
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            yield
        finally:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024

            if current_memory > max_memory_mb:
                logger.warning(f"Memory usage high: {current_memory:.1f}MB")
                MemoryOptimizer.force_gc()

            memory_delta = current_memory - start_memory
            if abs(memory_delta) > 10:  # Log significant changes
                logger.debug(f"Memory delta: {memory_delta:+.1f}MB")


def performance_optimized(config: PerformanceConfig = None):
    """Decorator to optimize function performance"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            perf_config = config or PerformanceConfig()

            # Enable GC optimization if configured
            if perf_config.enable_gc_optimization:
                MemoryOptimizer.force_gc()

            # Track performance
            with PerformanceMonitor().track_operation(func.__name__):
                with MemoryOptimizer.memory_guard(perf_config.max_memory_mb):
                    return func(*args, **kwargs)

        return wrapper

    return decorator


@lru_cache(maxsize=32)
def get_optimal_worker_count() -> int:
    """Calculate optimal worker count based on system resources"""
    cpu_count = os.cpu_count() or 4

    # Get available memory
    memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024

    # Conservative worker count based on resources
    memory_based = int(memory_gb / 2)  # 2GB per worker
    cpu_based = min(cpu_count, 8)  # Cap at 8 workers

    optimal = min(memory_based, cpu_based, 6)  # Never exceed 6
    optimal = max(optimal, 2)  # Always at least 2

    logger.info(
        f"Optimal worker count: {optimal} (CPU: {cpu_count}, Memory: {memory_gb:.1f}GB)"
    )
    return optimal


@lru_cache(maxsize=16)
def get_ffmpeg_optimization_flags(quality_preset: str = "medium") -> list:
    """Get optimized FFmpeg flags for current system"""
    flags = []

    # Threading
    cpu_count = os.cpu_count() or 4
    flags.extend(["-threads", str(min(cpu_count, 8))])

    # Quality presets
    preset_map = {
        "ultrafast": ["ultrafast", "0"],
        "fast": ["fast", "18"],
        "medium": ["medium", "20"],
        "slow": ["slow", "22"],
    }

    preset, crf = preset_map.get(quality_preset, ["medium", "20"])
    flags.extend(["-preset", preset, "-crf", crf])

    # Hardware acceleration (if available)
    if _check_hardware_acceleration():
        flags.extend(["-hwaccel", "auto"])

    return flags


@lru_cache(maxsize=8)
def _check_hardware_acceleration() -> bool:
    """Check if hardware acceleration is available"""
    try:
        # Check for common hardware acceleration
        result = os.popen("ffmpeg -hwaccels 2>/dev/null").read()
        return any(accel in result for accel in ["nvenc", "vaapi", "videotoolbox"])
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return False


# Global performance monitor instance
_global_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def start_performance_monitoring():
    """Start global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


# Performance utilities
def optimize_for_video_processing():
    """Apply system-wide optimizations for video processing"""
    # Force garbage collection
    MemoryOptimizer.force_gc()

    # Set process priority (if possible)
    try:
        process = psutil.Process()
        if hasattr(process, "nice"):
            process.nice(-5)  # Higher priority
    except (psutil.Error, OSError, PermissionError, AttributeError) as e:
        logger.debug(f"Failed to set process priority: {type(e).__name__} - {str(e)}")

    # Start performance monitoring
    start_performance_monitoring()

    logger.info("Video processing optimizations applied")


if __name__ == "__main__":
    # Test performance monitoring
    monitor = PerformanceMonitor()

    print(f"Optimal workers: {get_optimal_worker_count()}")
    print(f"FFmpeg flags: {get_ffmpeg_optimization_flags()}")

    usage = monitor.get_resource_usage()
    print(
        f"Current usage: CPU: {usage.cpu_percent:.1f}%, Memory: {usage.memory_mb:.1f}MB"
    )
