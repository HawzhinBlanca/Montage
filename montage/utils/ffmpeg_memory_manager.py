#!/usr/bin/env python3
"""
FFmpeg-specific memory management for large video processing.
Provides memory-aware FFmpeg command generation and process monitoring.
"""

import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import psutil

# Import memory management components
try:
    from ..utils.logging_config import get_logger
    from .memory_manager import MemoryPressureLevel, ProcessingMode, get_memory_monitor
    from .resource_manager import get_resource_tracker, managed_process
except ImportError:
    import sys
    from pathlib import Path

    from montage.utils.logging_config import get_logger
    from montage.utils.memory_manager import (
        MemoryPressureLevel,
        get_memory_monitor,
    )
    from montage.utils.resource_manager import managed_process

logger = get_logger(__name__)


class FFmpegMemoryProfile(Enum):
    """FFmpeg memory usage profiles"""

    MINIMAL = "minimal"  # Absolute minimal memory usage
    LOW = "low"  # Low memory usage
    BALANCED = "balanced"  # Balanced memory/performance
    HIGH = "high"  # High memory for best performance


@dataclass
class FFmpegResourceConfig:
    """FFmpeg resource configuration"""

    max_memory_mb: int = 1024
    max_threads: int = 4
    buffer_size_kb: int = 1024
    preset: str = "medium"
    crf: int = 23
    pixel_format: str = "yuv420p"
    enable_hwaccel: bool = False
    chunk_processing: bool = False

    @classmethod
    def for_profile(
        cls, profile: FFmpegMemoryProfile, available_memory_mb: int
    ) -> "FFmpegResourceConfig":
        """Create config for specific memory profile"""

        if profile == FFmpegMemoryProfile.MINIMAL:
            return cls(
                max_memory_mb=min(256, available_memory_mb // 4),
                max_threads=1,
                buffer_size_kb=256,
                preset="ultrafast",
                crf=28,
                enable_hwaccel=False,
                chunk_processing=True,
            )

        elif profile == FFmpegMemoryProfile.LOW:
            return cls(
                max_memory_mb=min(512, available_memory_mb // 3),
                max_threads=2,
                buffer_size_kb=512,
                preset="fast",
                crf=25,
                enable_hwaccel=True,
                chunk_processing=True,
            )

        elif profile == FFmpegMemoryProfile.BALANCED:
            return cls(
                max_memory_mb=min(1024, available_memory_mb // 2),
                max_threads=4,
                buffer_size_kb=1024,
                preset="medium",
                crf=23,
                enable_hwaccel=True,
                chunk_processing=False,
            )

        else:  # HIGH
            return cls(
                max_memory_mb=min(2048, available_memory_mb * 2 // 3),
                max_threads=8,
                buffer_size_kb=2048,
                preset="slow",
                crf=20,
                enable_hwaccel=True,
                chunk_processing=False,
            )


class FFmpegMemoryManager:
    """Manages FFmpeg processes with memory constraints and monitoring"""

    def __init__(self):
        self.monitor = get_memory_monitor()
        self._active_processes: Dict[int, subprocess.Popen] = {}
        self._process_configs: Dict[int, FFmpegResourceConfig] = {}
        self._memory_monitors: Dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

        logger.info("FFmpeg memory manager initialized")

    def get_optimal_config(
        self, video_path: str = None, processing_type: str = "basic"
    ) -> FFmpegResourceConfig:
        """Get optimal FFmpeg configuration based on current memory situation"""

        stats = self.monitor.get_current_stats()

        # Determine profile based on memory pressure
        if stats.pressure_level == MemoryPressureLevel.CRITICAL:
            profile = FFmpegMemoryProfile.MINIMAL
        elif stats.pressure_level == MemoryPressureLevel.HIGH:
            profile = FFmpegMemoryProfile.LOW
        elif stats.pressure_level == MemoryPressureLevel.MODERATE:
            profile = FFmpegMemoryProfile.BALANCED
        else:
            profile = FFmpegMemoryProfile.HIGH

        # Adjust based on file size if provided
        if video_path and os.path.exists(video_path):
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024

            # For large files, prefer chunked processing
            if file_size_mb > 500:
                if profile == FFmpegMemoryProfile.HIGH:
                    profile = FFmpegMemoryProfile.BALANCED

            # For very large files, force low memory mode
            if file_size_mb > 2000:  # 2GB+
                profile = FFmpegMemoryProfile.LOW

        config = FFmpegResourceConfig.for_profile(profile, stats.available_mb)

        logger.debug(
            f"Optimal FFmpeg config: profile={profile.value}, "
            f"memory={config.max_memory_mb}MB, threads={config.max_threads}"
        )

        return config

    def build_memory_optimized_command(
        self,
        base_cmd: List[str],
        config: Optional[FFmpegResourceConfig] = None,
        video_path: str = None,
    ) -> List[str]:
        """Build FFmpeg command with memory optimization flags"""

        if config is None:
            config = self.get_optimal_config(video_path)

        cmd = base_cmd.copy()

        # Remove existing resource-related flags to avoid conflicts
        self._remove_existing_flags(
            cmd,
            [
                "-threads",
                "-bufsize",
                "-preset",
                "-crf",
                "-pix_fmt",
                "-hwaccel",
                "-c:v",
                "-c:a",
            ],
        )

        # Add memory-optimized flags
        optimizations = [
            "-threads",
            str(config.max_threads),
            "-bufsize",
            f"{config.buffer_size_kb}k",
        ]

        # Add encoding optimizations if this is an encoding operation
        if any(flag in cmd for flag in ["-c:v", "libx264", "libx265"]):
            optimizations.extend(
                [
                    "-preset",
                    config.preset,
                    "-crf",
                    str(config.crf),
                    "-pix_fmt",
                    config.pixel_format,
                ]
            )

        # Add hardware acceleration if enabled and available
        if config.enable_hwaccel and self._check_hardware_acceleration():
            # Apple Silicon specific optimizations
            import platform
            if platform.machine() == 'arm64' and platform.system() == 'Darwin':
                # Use VideoToolbox for Apple Silicon
                optimizations.extend([
                    "-hwaccel", "videotoolbox",
                    "-hwaccel_output_format", "videotoolbox_vld"
                ])
                logger.info("Using Apple Silicon VideoToolbox acceleration")
            else:
                optimizations.extend(["-hwaccel", "auto"])

        # Insert optimizations after input files but before output
        output_index = self._find_output_index(cmd)
        if output_index > 0:
            cmd[output_index:output_index] = optimizations
        else:
            cmd.extend(optimizations)

        logger.debug(
            f"Built memory-optimized FFmpeg command with {len(optimizations)} flags"
        )
        return cmd

    def _remove_existing_flags(self, cmd: List[str], flags: List[str]):
        """Remove existing flags from command to avoid conflicts"""
        i = 0
        while i < len(cmd):
            if cmd[i] in flags and i + 1 < len(cmd):
                # Remove flag and its value
                cmd.pop(i)
                cmd.pop(i)
            else:
                i += 1

    def _find_output_index(self, cmd: List[str]) -> int:
        """Find index where output file is specified"""
        # Look for the last non-flag argument (likely the output file)
        for i in range(len(cmd) - 1, -1, -1):
            if not cmd[i].startswith("-") and i > 0 and not cmd[i - 1].startswith("-"):
                return i
        return len(cmd)

    def _check_hardware_acceleration(self) -> bool:
        """Check if hardware acceleration is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return any(
                    accel in result.stdout
                    for accel in ["nvenc", "vaapi", "videotoolbox", "qsv"]
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug(f"Hardware encoder detection failed: {type(e).__name__}")

        return False

    @contextmanager
    def managed_ffmpeg_process(
        self,
        cmd: List[str],
        config: Optional[FFmpegResourceConfig] = None,
        video_path: str = None,
        timeout: float = 3600,
    ):
        """Context manager for memory-managed FFmpeg process"""

        if config is None:
            config = self.get_optimal_config(video_path)

        # Build optimized command
        optimized_cmd = self.build_memory_optimized_command(cmd, config, video_path)

        # Use managed process with FFmpeg-specific monitoring
        with managed_process(
            optimized_cmd,
            name="ffmpeg",
            memory_limit_mb=config.max_memory_mb,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        ) as process:

            # Register process for monitoring
            with self._lock:
                self._active_processes[process.pid] = process
                self._process_configs[process.pid] = config

            # Start memory monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_process_memory,
                args=(process.pid, config.max_memory_mb),
                daemon=True,
            )
            monitor_thread.start()

            with self._lock:
                self._memory_monitors[process.pid] = monitor_thread

            try:
                yield process

            finally:
                # Cleanup monitoring
                with self._lock:
                    self._active_processes.pop(process.pid, None)
                    self._process_configs.pop(process.pid, None)
                    monitor_thread = self._memory_monitors.pop(process.pid, None)

                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=1.0)

    def _monitor_process_memory(self, pid: int, memory_limit_mb: int):
        """Monitor FFmpeg process memory usage"""
        try:
            process_obj = psutil.Process(pid)

            while process_obj.is_running():
                try:
                    memory_info = process_obj.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024

                    # Check memory limit
                    if memory_mb > memory_limit_mb:
                        logger.warning(
                            f"FFmpeg process {pid} exceeded memory limit: "
                            f"{memory_mb:.0f}MB > {memory_limit_mb}MB"
                        )

                        # Get the process from our tracking
                        with self._lock:
                            tracked_process = self._active_processes.get(pid)

                        if tracked_process:
                            logger.error(
                                f"Terminating FFmpeg process {pid} due to memory limit"
                            )
                            tracked_process.terminate()
                        break

                    # Log high memory usage
                    if memory_mb > memory_limit_mb * 0.8:
                        logger.warning(
                            f"FFmpeg process {pid} high memory usage: {memory_mb:.0f}MB"
                        )

                    time.sleep(2)  # Check every 2 seconds

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        except Exception as e:
            logger.error(f"Memory monitoring error for process {pid}: {e}")

    def process_video_chunks(
        self,
        input_path: str,
        output_path: str,
        processing_func: callable,
        chunk_duration: float = 300,
    ) -> bool:
        """Process large video in memory-safe chunks"""

        try:
            # Get video duration
            duration = self._get_video_duration(input_path)
            if duration <= 0:
                logger.error("Could not determine video duration")
                return False

            # Calculate optimal chunk size based on memory
            stats = self.monitor.get_current_stats()
            if stats.available_mb < 1024:  # Less than 1GB available
                chunk_duration = min(chunk_duration, 120)  # Max 2 minutes
            elif stats.available_mb < 2048:  # Less than 2GB available
                chunk_duration = min(chunk_duration, 300)  # Max 5 minutes

            logger.info(
                f"Processing {duration:.1f}s video in {chunk_duration:.1f}s chunks "
                f"(available memory: {stats.available_mb:.0f}MB)"
            )

            # Process chunks
            chunk_files = []
            current_time = 0.0
            chunk_index = 0

            while current_time < duration:
                end_time = min(current_time + chunk_duration, duration)

                # Check memory before processing chunk
                current_stats = self.monitor.get_current_stats()
                if current_stats.pressure_level in [
                    MemoryPressureLevel.HIGH,
                    MemoryPressureLevel.CRITICAL,
                ]:
                    logger.warning(
                        "High memory pressure - forcing cleanup before next chunk"
                    )
                    import gc

                    gc.collect()
                    time.sleep(2)

                chunk_file = self._process_chunk(
                    input_path, current_time, end_time, chunk_index, processing_func
                )

                if chunk_file:
                    chunk_files.append(chunk_file)
                    logger.debug(
                        f"Processed chunk {chunk_index + 1}: {current_time:.1f}s - {end_time:.1f}s"
                    )
                else:
                    logger.error(f"Failed to process chunk {chunk_index + 1}")
                    return False

                current_time = end_time
                chunk_index += 1

            # Concatenate chunks
            if chunk_files:
                success = self._concatenate_chunks(chunk_files, output_path)

                # Cleanup chunk files
                for chunk_file in chunk_files:
                    try:
                        os.unlink(chunk_file)
                    except OSError as e:
                        logger.debug(f"Failed to delete chunk file {chunk_file}: {e}")

                return success

            return False

        except Exception as e:
            logger.error(f"Chunked video processing failed: {e}")
            return False

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using memory-efficient probe"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                import json

                info = json.loads(result.stdout)
                return float(info["format"]["duration"])

        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")

        return 0.0

    def _process_chunk(
        self,
        input_path: str,
        start_time: float,
        end_time: float,
        chunk_index: int,
        processing_func: callable,
    ) -> Optional[str]:
        """Process a single video chunk"""

        import tempfile

        try:
            # Create chunk file
            chunk_file = os.path.join(
                tempfile.gettempdir(),
                f"ffmpeg_chunk_{chunk_index:04d}_{os.getpid()}.mp4",
            )

            duration = end_time - start_time

            # Extract chunk with minimal memory usage
            extract_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                input_path,
                "-c",
                "copy",  # Stream copy for minimal memory
                chunk_file,
            ]

            config = FFmpegResourceConfig.for_profile(
                FFmpegMemoryProfile.LOW, self.monitor.get_current_stats().available_mb
            )

            with self.managed_ffmpeg_process(extract_cmd, config) as process:
                result = process.wait()

                if result == 0 and os.path.exists(chunk_file):
                    # Process the chunk
                    processed_chunk = processing_func(chunk_file)
                    return processed_chunk
                else:
                    logger.error(f"Failed to extract chunk {chunk_index}")
                    return None

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {e}")
            return None

    def _concatenate_chunks(self, chunk_files: List[str], output_path: str) -> bool:
        """Concatenate processed chunks with minimal memory usage"""

        import tempfile

        try:
            # Create concat file
            concat_file = os.path.join(
                tempfile.gettempdir(), f"ffmpeg_concat_{os.getpid()}.txt"
            )

            with open(concat_file, "w") as f:
                for chunk_file in chunk_files:
                    f.write(f"file '{chunk_file}'\n")

            # Concatenate with minimal memory usage
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",  # Stream copy for minimal memory
                output_path,
            ]

            config = FFmpegResourceConfig.for_profile(
                FFmpegMemoryProfile.LOW, self.monitor.get_current_stats().available_mb
            )

            with self.managed_ffmpeg_process(concat_cmd, config) as process:
                result = process.wait()

                # Cleanup concat file
                try:
                    os.unlink(concat_file)
                except OSError as e:
                    logger.debug(f"Failed to delete concat file {concat_file}: {e}")

                return result == 0

        except Exception as e:
            logger.error(f"Failed to concatenate chunks: {e}")
            return False

    def estimate_memory_usage(self, cmd: List[str], video_path: str = None) -> int:
        """Estimate memory usage for FFmpeg command"""

        base_memory = 256  # Base FFmpeg memory usage

        # Add memory based on command complexity
        if "-filter_complex" in cmd or "-vf" in cmd:
            base_memory += 512  # Filters need more memory

        if any(codec in " ".join(cmd) for codec in ["libx264", "libx265", "libvpx"]):
            base_memory += 256  # Encoding needs more memory

        # Add memory based on file size
        if video_path and os.path.exists(video_path):
            file_size_mb = os.path.getsize(video_path) / 1024 / 1024
            base_memory += min(
                512, int(file_size_mb * 0.1)
            )  # 10% of file size, max 512MB

        return base_memory

    def get_active_process_count(self) -> int:
        """Get number of active FFmpeg processes"""
        with self._lock:
            return len(self._active_processes)

    def get_total_memory_usage(self) -> float:
        """Get total memory usage of all active FFmpeg processes"""
        total_memory = 0.0

        with self._lock:
            for pid in list(self._active_processes.keys()):
                try:
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    total_memory += memory_info.rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process no longer exists, clean up
                    self._active_processes.pop(pid, None)
                    self._process_configs.pop(pid, None)

        return total_memory

    def force_cleanup_processes(self):
        """Force cleanup of all FFmpeg processes"""
        with self._lock:
            processes = list(self._active_processes.items())

        for pid, process in processes:
            try:
                if process.poll() is None:  # Still running
                    logger.warning(f"Force terminating FFmpeg process {pid}")
                    process.terminate()

                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            except Exception as e:
                logger.error(f"Error terminating process {pid}: {e}")

        with self._lock:
            self._active_processes.clear()
            self._process_configs.clear()
            self._memory_monitors.clear()


# Global FFmpeg memory manager
_global_ffmpeg_manager = None


def get_ffmpeg_memory_manager() -> FFmpegMemoryManager:
    """Get global FFmpeg memory manager"""
    global _global_ffmpeg_manager
    if _global_ffmpeg_manager is None:
        _global_ffmpeg_manager = FFmpegMemoryManager()
    return _global_ffmpeg_manager


# Utility functions


def build_memory_safe_ffmpeg_command(
    base_cmd: List[str],
    video_path: str = None,
    memory_profile: FFmpegMemoryProfile = None,
) -> List[str]:
    """Build memory-safe FFmpeg command"""
    manager = get_ffmpeg_memory_manager()

    if memory_profile:
        stats = manager.monitor.get_current_stats()
        config = FFmpegResourceConfig.for_profile(memory_profile, stats.available_mb)
    else:
        config = None

    return manager.build_memory_optimized_command(base_cmd, config, video_path)


@contextmanager
def memory_safe_ffmpeg(
    cmd: List[str],
    video_path: str = None,
    max_memory_mb: int = None,
    timeout: float = 3600,
):
    """Context manager for memory-safe FFmpeg processing"""
    manager = get_ffmpeg_memory_manager()

    # Get config based on memory constraints
    if max_memory_mb:
        config = FFmpegResourceConfig(max_memory_mb=max_memory_mb)
    else:
        config = manager.get_optimal_config(video_path)

    with manager.managed_ffmpeg_process(cmd, config, video_path, timeout) as process:
        yield process


def process_large_video_safely(
    input_path: str,
    output_path: str,
    processing_func: callable,
    max_memory_mb: int = 1024,
) -> bool:
    """Process large video with memory safety"""
    manager = get_ffmpeg_memory_manager()

    # Check if chunking is needed
    stats = manager.monitor.get_current_stats()
    file_size_mb = os.path.getsize(input_path) / 1024 / 1024

    should_chunk = (
        file_size_mb > 500  # Large file
        or stats.available_mb < max_memory_mb * 2  # Limited memory
        or stats.pressure_level != MemoryPressureLevel.LOW  # Memory pressure
    )

    if should_chunk:
        logger.info(f"Using chunked processing for large video ({file_size_mb:.0f}MB)")
        return manager.process_video_chunks(input_path, output_path, processing_func)
    else:
        logger.info("Using direct processing")
        return processing_func(input_path, output_path)


if __name__ == "__main__":
    # Test FFmpeg memory management
    print("Testing FFmpeg memory management...")

    manager = get_ffmpeg_memory_manager()

    # Test configuration generation
    config = manager.get_optimal_config()
    print(f"Optimal config: {config}")

    # Test command optimization
    test_cmd = ["ffmpeg", "-i", "input.mp4", "-c:v", "libx264", "output.mp4"]

    optimized_cmd = manager.build_memory_optimized_command(test_cmd)
    print(f"Optimized command: {' '.join(optimized_cmd)}")

    # Test memory estimation
    memory_estimate = manager.estimate_memory_usage(test_cmd)
    print(f"Memory estimate: {memory_estimate}MB")

    print("FFmpeg memory management test completed")
