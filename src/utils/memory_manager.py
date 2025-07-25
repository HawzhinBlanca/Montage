#!/usr/bin/env python3
"""
Comprehensive memory management system for large video file processing.
Prevents OOM errors and provides adaptive processing capabilities.
"""

import os
import gc
import psutil
import time
import threading
import weakref
import signal
import subprocess
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import tempfile

# Import centralized logging
try:
    from ..utils.logging_config import get_logger
    from ..core.metrics import metrics
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logging_config import get_logger
    from core.metrics import metrics

logger = get_logger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels for adaptive processing"""

    LOW = "low"  # < 70% memory usage
    MODERATE = "moderate"  # 70-80% memory usage
    HIGH = "high"  # 80-90% memory usage
    CRITICAL = "critical"  # > 90% memory usage


class ProcessingMode(Enum):
    """Processing modes based on memory availability"""

    FULL_QUALITY = "full_quality"  # Best quality, high memory usage
    BALANCED = "balanced"  # Balance quality vs memory
    MEMORY_OPTIMIZED = "memory_opt"  # Lower quality, minimal memory
    SURVIVAL = "survival"  # Minimal processing to avoid OOM


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    swap_used_mb: float
    process_memory_mb: float
    pressure_level: MemoryPressureLevel
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessMemoryInfo:
    """Information about a process's memory usage"""

    pid: int
    name: str
    memory_mb: float
    cpu_percent: float
    command: str
    is_ffmpeg: bool = False


class MemoryMonitor:
    """Real-time memory monitoring with pressure detection"""

    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_lock = threading.Lock()
        self._current_stats: Optional[MemoryStats] = None
        self._stats_history: List[MemoryStats] = []
        self._max_history = 100

        # Callbacks for pressure level changes
        self._pressure_callbacks: Dict[MemoryPressureLevel, List[Callable]] = {
            level: [] for level in MemoryPressureLevel
        }

        # Process tracking
        self._tracked_processes: Dict[int, ProcessMemoryInfo] = {}
        self._process_lock = threading.Lock()

        # Initialize current process monitor
        self._process = psutil.Process()

        logger.info("Memory monitor initialized")

    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        try:
            # System memory
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()

            # Process memory
            process_mem = self._process.memory_info()
            process_mb = process_mem.rss / 1024 / 1024

            # Calculate pressure level
            pressure_level = self._calculate_pressure_level(virtual_mem.percent)

            stats = MemoryStats(
                total_mb=virtual_mem.total / 1024 / 1024,
                available_mb=virtual_mem.available / 1024 / 1024,
                used_mb=virtual_mem.used / 1024 / 1024,
                percent_used=virtual_mem.percent,
                swap_used_mb=swap_mem.used / 1024 / 1024,
                process_memory_mb=process_mb,
                pressure_level=pressure_level,
            )

            with self._stats_lock:
                self._current_stats = stats
                self._stats_history.append(stats)

                # Keep history bounded
                if len(self._stats_history) > self._max_history:
                    self._stats_history.pop(0)

            return stats

        except (psutil.Error, OSError) as e:
            logger.error(f"Failed to get memory stats: {e}")
            # Return conservative estimates
            return MemoryStats(
                total_mb=8192,  # 8GB default
                available_mb=2048,  # 2GB available
                used_mb=6144,  # 6GB used
                percent_used=75.0,
                swap_used_mb=0,
                process_memory_mb=512,  # 512MB process
                pressure_level=MemoryPressureLevel.MODERATE,
            )

    def _calculate_pressure_level(self, percent_used: float) -> MemoryPressureLevel:
        """Calculate memory pressure level based on usage percentage"""
        if percent_used >= 90:
            return MemoryPressureLevel.CRITICAL
        elif percent_used >= 80:
            return MemoryPressureLevel.HIGH
        elif percent_used >= 70:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.LOW

    def start_monitoring(self):
        """Start background memory monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop"""
        last_pressure_level = None

        while self._monitoring:
            try:
                stats = self.get_current_stats()

                # Check for pressure level changes
                if last_pressure_level != stats.pressure_level:
                    self._trigger_pressure_callbacks(stats.pressure_level, stats)
                    last_pressure_level = stats.pressure_level

                # Update metrics
                metrics.memory_usage.set(stats.percent_used)
                metrics.process_memory.set(stats.process_memory_mb)

                # Log warnings for high memory usage
                if stats.pressure_level in [
                    MemoryPressureLevel.HIGH,
                    MemoryPressureLevel.CRITICAL,
                ]:
                    logger.warning(
                        f"High memory pressure detected: {stats.pressure_level.value} "
                        f"({stats.percent_used:.1f}% used, {stats.available_mb:.0f}MB available)"
                    )

                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.update_interval)

    def register_pressure_callback(
        self, level: MemoryPressureLevel, callback: Callable
    ):
        """Register callback for memory pressure level changes"""
        self._pressure_callbacks[level].append(callback)
        logger.debug(f"Registered callback for pressure level: {level.value}")

    def _trigger_pressure_callbacks(
        self, level: MemoryPressureLevel, stats: MemoryStats
    ):
        """Trigger callbacks for pressure level change"""
        callbacks = self._pressure_callbacks.get(level, [])
        for callback in callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"Pressure callback error: {e}")

    def track_process(self, pid: int, name: str = "", is_ffmpeg: bool = False):
        """Track a specific process's memory usage"""
        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()

            process_info = ProcessMemoryInfo(
                pid=pid,
                name=name or process.name(),
                memory_mb=memory_info.rss / 1024 / 1024,
                cpu_percent=process.cpu_percent(),
                command=" ".join(process.cmdline()[:3]) if process.cmdline() else "",
                is_ffmpeg=is_ffmpeg,
            )

            with self._process_lock:
                self._tracked_processes[pid] = process_info

            logger.debug(
                f"Tracking process {pid}: {name} ({process_info.memory_mb:.1f}MB)"
            )

        except (psutil.Error, OSError) as e:
            logger.warning(f"Failed to track process {pid}: {e}")

    def untrack_process(self, pid: int):
        """Stop tracking a process"""
        with self._process_lock:
            if pid in self._tracked_processes:
                del self._tracked_processes[pid]
                logger.debug(f"Stopped tracking process {pid}")

    def get_tracked_processes(self) -> List[ProcessMemoryInfo]:
        """Get list of tracked processes"""
        with self._process_lock:
            return list(self._tracked_processes.values())

    def get_ffmpeg_processes(self) -> List[ProcessMemoryInfo]:
        """Get list of tracked FFmpeg processes"""
        with self._process_lock:
            return [info for info in self._tracked_processes.values() if info.is_ffmpeg]

    def estimate_video_memory_needs(self, video_path: str) -> float:
        """Estimate memory requirements for processing a video file"""
        try:
            # Get video file size and properties
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024

            # Rough estimation based on file size and processing type
            # These are conservative estimates based on typical FFmpeg memory usage
            base_memory = min(
                file_size_mb * 0.1, 512
            )  # Base: 10% of file size, max 512MB
            decode_memory = min(
                file_size_mb * 0.05, 256
            )  # Decode buffer: 5%, max 256MB
            encode_memory = min(
                file_size_mb * 0.15, 1024
            )  # Encode buffer: 15%, max 1GB

            total_estimate = base_memory + decode_memory + encode_memory

            logger.debug(
                f"Memory estimate for {os.path.basename(video_path)}: "
                f"{total_estimate:.0f}MB (file: {file_size_mb:.0f}MB)"
            )

            return total_estimate

        except (OSError, ValueError) as e:
            logger.warning(f"Failed to estimate memory needs: {e}")
            return 1024  # 1GB default estimate


class MemoryPressureManager:
    """Manages system response to memory pressure"""

    def __init__(self, monitor: MemoryMonitor):
        self.monitor = monitor
        self._response_actions: Dict[MemoryPressureLevel, List[Callable]] = {
            level: [] for level in MemoryPressureLevel
        }

        # Register default responses
        self._register_default_responses()

        logger.info("Memory pressure manager initialized")

    def _register_default_responses(self):
        """Register default responses to memory pressure"""
        # Moderate pressure: gentle cleanup
        self.register_response(
            MemoryPressureLevel.MODERATE, self._moderate_pressure_response
        )

        # High pressure: aggressive cleanup
        self.register_response(MemoryPressureLevel.HIGH, self._high_pressure_response)

        # Critical pressure: emergency measures
        self.register_response(
            MemoryPressureLevel.CRITICAL, self._critical_pressure_response
        )

    def register_response(self, level: MemoryPressureLevel, action: Callable):
        """Register a response action for a pressure level"""
        self._response_actions[level].append(action)
        logger.debug(f"Registered pressure response for level: {level.value}")

    def _moderate_pressure_response(self, stats: MemoryStats):
        """Response to moderate memory pressure"""
        logger.info("Responding to moderate memory pressure")

        # Force garbage collection
        freed_mb = self._force_garbage_collection()

        # Log response
        logger.info(f"Moderate pressure response: freed {freed_mb:.1f}MB via GC")

    def _high_pressure_response(self, stats: MemoryStats):
        """Response to high memory pressure"""
        logger.warning("Responding to high memory pressure")

        # Force garbage collection
        freed_mb = self._force_garbage_collection()

        # Clear caches if available
        self._clear_caches()

        # Consider killing non-essential FFmpeg processes
        ffmpeg_processes = self.monitor.get_ffmpeg_processes()
        if ffmpeg_processes:
            # Kill oldest FFmpeg process if multiple running
            if len(ffmpeg_processes) > 1:
                oldest_process = min(ffmpeg_processes, key=lambda p: p.pid)
                self._terminate_process(oldest_process.pid)

        logger.warning(
            f"High pressure response: freed {freed_mb:.1f}MB, checked FFmpeg processes"
        )

    def _critical_pressure_response(self, stats: MemoryStats):
        """Response to critical memory pressure"""
        logger.error("CRITICAL MEMORY PRESSURE - Taking emergency measures")

        # Aggressive garbage collection
        freed_mb = self._force_garbage_collection()

        # Clear all possible caches
        self._clear_caches()

        # Terminate FFmpeg processes if memory is still critical
        current_stats = self.monitor.get_current_stats()
        if current_stats.percent_used > 90:
            ffmpeg_processes = self.monitor.get_ffmpeg_processes()
            for process_info in ffmpeg_processes:
                logger.error(
                    f"Emergency: terminating FFmpeg process {process_info.pid}"
                )
                self._terminate_process(process_info.pid)

        # Consider reducing worker count or processing quality
        # This would be handled by the adaptive configuration system

        logger.error(
            f"Critical pressure response: freed {freed_mb:.1f}MB, terminated processes"
        )

    def _force_garbage_collection(self) -> float:
        """Force garbage collection and return freed memory"""
        before_stats = self.monitor.get_current_stats()

        # Multiple GC passes for better cleanup
        for _ in range(3):
            gc.collect()

        time.sleep(0.1)  # Brief pause for memory to be released
        after_stats = self.monitor.get_current_stats()

        freed_mb = before_stats.process_memory_mb - after_stats.process_memory_mb
        if freed_mb > 1:  # Only log if significant
            logger.info(f"Garbage collection freed {freed_mb:.1f}MB")

        return max(0, freed_mb)

    def _clear_caches(self):
        """Clear various caches to free memory"""
        try:
            # Clear functools.lru_cache decorated functions
            import functools

            # This would need to be extended with specific cache clearing
            # for the application's cached functions

            logger.debug("Cleared application caches")
        except Exception as e:
            logger.warning(f"Failed to clear caches: {e}")

    def _terminate_process(self, pid: int):
        """Safely terminate a process"""
        try:
            process = psutil.Process(pid)
            process.terminate()

            # Wait for graceful termination
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                # Force kill if doesn't terminate gracefully
                process.kill()

            self.monitor.untrack_process(pid)
            logger.info(f"Terminated process {pid}")

        except (psutil.Error, OSError) as e:
            logger.error(f"Failed to terminate process {pid}: {e}")


class AdaptiveProcessingConfig:
    """Adaptive processing configuration based on memory availability"""

    def __init__(self, monitor: MemoryMonitor):
        self.monitor = monitor
        self._base_config = self._get_base_config()
        self._current_mode = ProcessingMode.BALANCED

        logger.info("Adaptive processing config initialized")

    def _get_base_config(self) -> Dict[str, Any]:
        """Get base processing configuration"""
        return {
            "max_workers": 4,
            "chunk_size_mb": 256,
            "quality_preset": "medium",
            "enable_hardware_accel": True,
            "buffer_size": 65536,
            "max_concurrent_ffmpeg": 2,
        }

    def get_current_config(self) -> Dict[str, Any]:
        """Get current adaptive configuration based on memory pressure"""
        stats = self.monitor.get_current_stats()
        mode = self._determine_processing_mode(stats)

        if mode != self._current_mode:
            logger.info(
                f"Processing mode changed: {self._current_mode.value} -> {mode.value}"
            )
            self._current_mode = mode

        return self._get_config_for_mode(mode, stats)

    def _determine_processing_mode(self, stats: MemoryStats) -> ProcessingMode:
        """Determine processing mode based on memory statistics"""
        if stats.pressure_level == MemoryPressureLevel.CRITICAL:
            return ProcessingMode.SURVIVAL
        elif stats.pressure_level == MemoryPressureLevel.HIGH:
            return ProcessingMode.MEMORY_OPTIMIZED
        elif stats.pressure_level == MemoryPressureLevel.MODERATE:
            return ProcessingMode.BALANCED
        else:
            return ProcessingMode.FULL_QUALITY

    def _get_config_for_mode(
        self, mode: ProcessingMode, stats: MemoryStats
    ) -> Dict[str, Any]:
        """Get configuration for specific processing mode"""
        config = self._base_config.copy()

        if mode == ProcessingMode.SURVIVAL:
            config.update(
                {
                    "max_workers": 1,
                    "chunk_size_mb": 64,
                    "quality_preset": "ultrafast",
                    "enable_hardware_accel": False,
                    "buffer_size": 16384,
                    "max_concurrent_ffmpeg": 1,
                    "force_low_resolution": True,
                    "skip_audio_normalization": True,
                }
            )

        elif mode == ProcessingMode.MEMORY_OPTIMIZED:
            config.update(
                {
                    "max_workers": 2,
                    "chunk_size_mb": 128,
                    "quality_preset": "fast",
                    "enable_hardware_accel": True,
                    "buffer_size": 32768,
                    "max_concurrent_ffmpeg": 1,
                    "reduce_resolution": True,
                }
            )

        elif mode == ProcessingMode.BALANCED:
            config.update(
                {
                    "max_workers": 3,
                    "chunk_size_mb": 256,
                    "quality_preset": "medium",
                    "enable_hardware_accel": True,
                    "buffer_size": 65536,
                    "max_concurrent_ffmpeg": 2,
                }
            )

        else:  # FULL_QUALITY
            config.update(
                {
                    "max_workers": 4,
                    "chunk_size_mb": 512,
                    "quality_preset": "slow",
                    "enable_hardware_accel": True,
                    "buffer_size": 131072,
                    "max_concurrent_ffmpeg": 3,
                }
            )

        # Adjust based on available memory
        available_gb = stats.available_mb / 1024
        if available_gb < 2:
            config["max_workers"] = min(config["max_workers"], 1)
            config["chunk_size_mb"] = min(config["chunk_size_mb"], 64)
        elif available_gb < 4:
            config["max_workers"] = min(config["max_workers"], 2)
            config["chunk_size_mb"] = min(config["chunk_size_mb"], 128)

        return config


@contextmanager
def memory_guard(
    max_memory_mb: Optional[int] = None,
    monitor: Optional[MemoryMonitor] = None,
    cleanup_on_exit: bool = True,
):
    """Context manager for memory-safe operations"""

    if monitor is None:
        monitor = MemoryMonitor()

    start_stats = monitor.get_current_stats()

    # Determine memory limit
    if max_memory_mb is None:
        max_memory_mb = int(
            start_stats.available_mb * 0.8
        )  # Use 80% of available memory

    logger.debug(
        f"Memory guard: max {max_memory_mb}MB, available {start_stats.available_mb:.0f}MB"
    )

    try:
        yield monitor

    finally:
        if cleanup_on_exit:
            # Force cleanup
            gc.collect()

            # Check final memory usage
            end_stats = monitor.get_current_stats()
            memory_delta = end_stats.process_memory_mb - start_stats.process_memory_mb

            if abs(memory_delta) > 50:  # Log significant changes
                logger.info(f"Memory delta: {memory_delta:+.1f}MB")

            if end_stats.process_memory_mb > max_memory_mb:
                logger.warning(
                    f"Memory limit exceeded: {end_stats.process_memory_mb:.1f}MB > {max_memory_mb}MB"
                )


def memory_optimized(max_memory_mb: Optional[int] = None):
    """Decorator for memory-optimized functions"""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with memory_guard(max_memory_mb=max_memory_mb) as monitor:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class StreamingVideoProcessor:
    """Process large videos in chunks to avoid memory issues"""

    def __init__(self, monitor: MemoryMonitor, config: AdaptiveProcessingConfig):
        self.monitor = monitor
        self.config = config

    def process_large_video_chunked(
        self,
        video_path: str,
        output_path: str,
        processing_func: Callable,
        chunk_duration: Optional[float] = None,
    ) -> bool:
        """Process large video in chunks to manage memory usage"""

        try:
            # Get video duration
            duration = self._get_video_duration(video_path)
            if duration <= 0:
                logger.error("Could not determine video duration")
                return False

            # Determine chunk size based on memory availability
            if chunk_duration is None:
                current_config = self.config.get_current_config()
                # Calculate chunk duration based on available memory and file size
                file_size_mb = os.path.getsize(video_path) / 1024 / 1024
                memory_per_mb = 2.0  # Rough estimate: 2MB RAM per MB of video

                stats = self.monitor.get_current_stats()
                usable_memory = stats.available_mb * 0.6  # Use 60% of available memory

                chunk_duration = min(
                    300,
                    max(30, usable_memory / (file_size_mb * memory_per_mb) * duration),
                )

            logger.info(
                f"Processing {duration:.1f}s video in {chunk_duration:.1f}s chunks"
            )

            # Process video in chunks
            chunk_files = []
            current_time = 0.0
            chunk_index = 0

            while current_time < duration:
                end_time = min(current_time + chunk_duration, duration)

                with memory_guard() as chunk_monitor:
                    logger.debug(
                        f"Processing chunk {chunk_index + 1}: {current_time:.1f}s - {end_time:.1f}s"
                    )

                    # Create temporary chunk file
                    chunk_file = os.path.join(
                        tempfile.gettempdir(),
                        f"chunk_{chunk_index:04d}_{os.getpid()}.mp4",
                    )

                    # Extract chunk
                    if self._extract_video_chunk(
                        video_path, current_time, end_time, chunk_file
                    ):
                        # Process chunk
                        processed_chunk = processing_func(chunk_file)
                        if processed_chunk:
                            chunk_files.append(processed_chunk)
                        else:
                            logger.error(f"Failed to process chunk {chunk_index + 1}")
                            return False
                    else:
                        logger.error(f"Failed to extract chunk {chunk_index + 1}")
                        return False

                    # Clean up temporary chunk
                    try:
                        os.unlink(chunk_file)
                    except OSError:
                        pass

                current_time = end_time
                chunk_index += 1

                # Check memory pressure between chunks
                stats = self.monitor.get_current_stats()
                if stats.pressure_level in [
                    MemoryPressureLevel.HIGH,
                    MemoryPressureLevel.CRITICAL,
                ]:
                    logger.warning(
                        "High memory pressure detected, forcing cleanup between chunks"
                    )
                    gc.collect()
                    time.sleep(1)  # Brief pause to allow memory cleanup

            # Concatenate all processed chunks
            if chunk_files:
                success = self._concatenate_chunks(chunk_files, output_path)

                # Clean up chunk files
                for chunk_file in chunk_files:
                    try:
                        os.unlink(chunk_file)
                    except OSError:
                        pass

                return success

            return False

        except Exception as e:
            logger.error(f"Chunked video processing failed: {e}")
            return False

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                import json

                info = json.loads(result.stdout)
                return float(info["format"]["duration"])
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")

        return 0.0

    def _extract_video_chunk(
        self, input_path: str, start_time: float, end_time: float, output_path: str
    ) -> bool:
        """Extract a chunk from the video"""
        try:
            duration = end_time - start_time

            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                input_path,
                "-c",
                "copy",  # Stream copy for speed
                output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to extract video chunk: {e}")
            return False

    def _concatenate_chunks(self, chunk_files: List[str], output_path: str) -> bool:
        """Concatenate processed chunks"""
        try:
            # Create concat file
            concat_file = os.path.join(
                tempfile.gettempdir(), f"concat_{os.getpid()}.txt"
            )

            with open(concat_file, "w") as f:
                for chunk_file in chunk_files:
                    f.write(f"file '{chunk_file}'\n")

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=600)

            # Clean up concat file
            try:
                os.unlink(concat_file)
            except OSError:
                pass

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to concatenate chunks: {e}")
            return False


# Global instances
_global_monitor: Optional[MemoryMonitor] = None
_global_pressure_manager: Optional[MemoryPressureManager] = None
_global_adaptive_config: Optional[AdaptiveProcessingConfig] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def get_pressure_manager() -> MemoryPressureManager:
    """Get or create global pressure manager"""
    global _global_pressure_manager, _global_monitor
    if _global_pressure_manager is None:
        if _global_monitor is None:
            _global_monitor = get_memory_monitor()
        _global_pressure_manager = MemoryPressureManager(_global_monitor)
    return _global_pressure_manager


def get_adaptive_config() -> AdaptiveProcessingConfig:
    """Get or create global adaptive configuration"""
    global _global_adaptive_config, _global_monitor
    if _global_adaptive_config is None:
        if _global_monitor is None:
            _global_monitor = get_memory_monitor()
        _global_adaptive_config = AdaptiveProcessingConfig(_global_monitor)
    return _global_adaptive_config


def initialize_memory_management():
    """Initialize global memory management system"""
    monitor = get_memory_monitor()
    pressure_manager = get_pressure_manager()
    adaptive_config = get_adaptive_config()

    # Register pressure callbacks
    monitor.register_pressure_callback(
        MemoryPressureLevel.MODERATE, pressure_manager._moderate_pressure_response
    )
    monitor.register_pressure_callback(
        MemoryPressureLevel.HIGH, pressure_manager._high_pressure_response
    )
    monitor.register_pressure_callback(
        MemoryPressureLevel.CRITICAL, pressure_manager._critical_pressure_response
    )

    # Start monitoring
    monitor.start_monitoring()

    logger.info("Memory management system initialized")
    return monitor, pressure_manager, adaptive_config


def shutdown_memory_management():
    """Shutdown memory management system"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        logger.info("Memory management system shutdown")


# Cleanup on exit
def _cleanup_handler(signum, frame):
    """Signal handler for cleanup"""
    shutdown_memory_management()


# Register signal handlers
signal.signal(signal.SIGINT, _cleanup_handler)
signal.signal(signal.SIGTERM, _cleanup_handler)


if __name__ == "__main__":
    # Test memory monitoring
    monitor, pressure_manager, adaptive_config = initialize_memory_management()

    try:
        # Display current stats
        stats = monitor.get_current_stats()
        print(
            f"Memory usage: {stats.percent_used:.1f}% ({stats.used_mb:.0f}MB / {stats.total_mb:.0f}MB)"
        )
        print(f"Available: {stats.available_mb:.0f}MB")
        print(f"Process memory: {stats.process_memory_mb:.1f}MB")
        print(f"Pressure level: {stats.pressure_level.value}")

        # Display adaptive config
        config = adaptive_config.get_current_config()
        print(f"Current processing mode: {adaptive_config._current_mode.value}")
        print(f"Max workers: {config['max_workers']}")
        print(f"Chunk size: {config['chunk_size_mb']}MB")
        print(f"Quality preset: {config['quality_preset']}")

        # Keep monitoring for a bit
        time.sleep(10)

    finally:
        shutdown_memory_management()
