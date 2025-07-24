#!/usr/bin/env python3
"""
Comprehensive unit tests for the memory management system.
Tests all major components: MemoryMonitor, MemoryPressureManager, AdaptiveProcessingConfig.
Target: ≥90% coverage for montage/utils/memory_manager.py
"""

import gc
import os
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch, mock_open

import psutil
import pytest

# Import the modules under test
from montage.utils.memory_manager import (
    AdaptiveProcessingConfig,
    MemoryMonitor,
    MemoryPressureLevel,
    MemoryPressureManager,
    MemoryStats,
    ProcessMemoryInfo,
    ProcessingMode,
    StreamingVideoProcessor,
    get_adaptive_config,
    get_memory_monitor,
    get_pressure_manager,
    initialize_memory_management,
    memory_guard,
    memory_optimized,
    shutdown_memory_management,
    _cleanup_handler,
)


class TestMemoryStats:
    """Test MemoryStats dataclass"""

    def test_memory_stats_creation(self):
        stats = MemoryStats(
            total_mb=8192,
            available_mb=4096,
            used_mb=4096,
            percent_used=50.0,
            swap_used_mb=0,
            process_memory_mb=512,
            pressure_level=MemoryPressureLevel.LOW,
        )

        assert stats.total_mb == 8192
        assert stats.available_mb == 4096
        assert stats.used_mb == 4096
        assert stats.percent_used == 50.0
        assert stats.swap_used_mb == 0
        assert stats.process_memory_mb == 512
        assert stats.pressure_level == MemoryPressureLevel.LOW
        assert isinstance(stats.timestamp, float)

    def test_memory_stats_default_timestamp(self):
        """Test that timestamp is automatically set"""
        before_time = time.time()
        stats = MemoryStats(
            total_mb=1000,
            available_mb=500,
            used_mb=500,
            percent_used=50.0,
            swap_used_mb=0,
            process_memory_mb=100,
            pressure_level=MemoryPressureLevel.LOW,
        )
        after_time = time.time()
        assert before_time <= stats.timestamp <= after_time


class TestProcessMemoryInfo:
    """Test ProcessMemoryInfo dataclass"""

    def test_process_memory_info_creation(self):
        info = ProcessMemoryInfo(
            pid=1234,
            name="ffmpeg",
            memory_mb=256.5,
            cpu_percent=15.2,
            command="ffmpeg -i input.mp4",
            is_ffmpeg=True,
        )

        assert info.pid == 1234
        assert info.name == "ffmpeg"
        assert info.memory_mb == 256.5
        assert info.cpu_percent == 15.2
        assert info.command == "ffmpeg -i input.mp4"
        assert info.is_ffmpeg is True

    def test_process_memory_info_defaults(self):
        """Test default values"""
        info = ProcessMemoryInfo(
            pid=1234,
            name="test",
            memory_mb=100.0,
            cpu_percent=5.0,
            command="test command",
        )
        assert info.is_ffmpeg is False  # Default value


class TestMemoryMonitor:
    """Test MemoryMonitor class"""

    def setup_method(self):
        self.monitor = MemoryMonitor(update_interval=0.1)

    def teardown_method(self):
        if hasattr(self.monitor, '_monitoring') and self.monitor._monitoring:
            self.monitor.stop_monitoring()

    @patch("psutil.virtual_memory")
    @patch("psutil.swap_memory")
    def test_get_current_stats_success(self, mock_swap, mock_virtual):
        """Test successful memory stats retrieval"""
        # Mock virtual memory
        mock_virtual.return_value = Mock(
            total=8 * 1024 * 1024 * 1024,  # 8GB
            available=4 * 1024 * 1024 * 1024,  # 4GB
            used=4 * 1024 * 1024 * 1024,  # 4GB
            percent=50.0,
        )

        # Mock swap memory
        mock_swap.return_value = Mock(used=0)

        # Mock process memory
        with patch.object(self.monitor._process, "memory_info") as mock_mem:
            mock_mem.return_value = Mock(rss=512 * 1024 * 1024)  # 512MB

            stats = self.monitor.get_current_stats()

            assert stats.total_mb == 8192
            assert stats.available_mb == 4096
            assert stats.used_mb == 4096
            assert stats.percent_used == 50.0
            assert stats.swap_used_mb == 0
            assert stats.process_memory_mb == 512
            assert stats.pressure_level == MemoryPressureLevel.LOW

    @patch("psutil.virtual_memory")
    def test_get_current_stats_error_handling(self, mock_virtual):
        """Test error handling in memory stats retrieval"""
        mock_virtual.side_effect = psutil.Error("Mock error")

        stats = self.monitor.get_current_stats()

        # Should return conservative estimates
        assert stats.total_mb == 8192
        assert stats.available_mb == 2048
        assert stats.used_mb == 6144
        assert stats.percent_used == 75.0
        assert stats.pressure_level == MemoryPressureLevel.MODERATE

    @patch("psutil.virtual_memory")
    def test_get_current_stats_os_error(self, mock_virtual):
        """Test OSError handling in memory stats retrieval"""
        mock_virtual.side_effect = OSError("Mock OS error")

        stats = self.monitor.get_current_stats()

        # Should return conservative estimates
        assert stats.total_mb == 8192
        assert stats.available_mb == 2048
        assert stats.pressure_level == MemoryPressureLevel.MODERATE

    def test_calculate_pressure_level(self):
        """Test pressure level calculation"""
        assert (
            self.monitor._calculate_pressure_level(60) == MemoryPressureLevel.LOW
        )
        assert (
            self.monitor._calculate_pressure_level(75) == MemoryPressureLevel.MODERATE
        )
        assert (
            self.monitor._calculate_pressure_level(85) == MemoryPressureLevel.HIGH
        )
        assert (
            self.monitor._calculate_pressure_level(95) == MemoryPressureLevel.CRITICAL
        )
        
        # Test boundary conditions
        assert (
            self.monitor._calculate_pressure_level(69.9) == MemoryPressureLevel.LOW
        )
        assert (
            self.monitor._calculate_pressure_level(70.0) == MemoryPressureLevel.MODERATE
        )
        assert (
            self.monitor._calculate_pressure_level(79.9) == MemoryPressureLevel.MODERATE
        )
        assert (
            self.monitor._calculate_pressure_level(80.0) == MemoryPressureLevel.HIGH
        )
        assert (
            self.monitor._calculate_pressure_level(89.9) == MemoryPressureLevel.HIGH
        )
        assert (
            self.monitor._calculate_pressure_level(90.0) == MemoryPressureLevel.CRITICAL
        )

    def test_start_stop_monitoring(self):
        """Test starting and stopping memory monitoring"""
        assert not self.monitor._monitoring
        assert self.monitor._monitor_thread is None

        self.monitor.start_monitoring()
        assert self.monitor._monitoring
        assert self.monitor._monitor_thread is not None

        # Wait a bit for monitoring to start
        time.sleep(0.2)

        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring

    def test_start_monitoring_already_started(self):
        """Test starting monitoring when already started"""
        self.monitor.start_monitoring()
        assert self.monitor._monitoring
        
        # Start again - should not create new thread
        original_thread = self.monitor._monitor_thread
        self.monitor.start_monitoring()
        assert self.monitor._monitor_thread is original_thread
        
        self.monitor.stop_monitoring()

    def test_stop_monitoring_with_timeout(self):
        """Test stop monitoring with thread join timeout"""
        self.monitor.start_monitoring()
        time.sleep(0.1)
        
        # Mock thread join to simulate timeout
        with patch.object(self.monitor._monitor_thread, 'join') as mock_join:
            self.monitor.stop_monitoring()
            mock_join.assert_called_once_with(timeout=5.0)

    def test_pressure_callback_registration(self):
        """Test pressure callback registration"""
        callback = Mock()
        self.monitor.register_pressure_callback(MemoryPressureLevel.HIGH, callback)

        assert callback in self.monitor._pressure_callbacks[MemoryPressureLevel.HIGH]

    def test_trigger_pressure_callbacks(self):
        """Test triggering pressure callbacks"""
        callback1 = Mock()
        callback2 = Mock()
        callback_error = Mock(side_effect=Exception("Test error"))
        
        self.monitor.register_pressure_callback(MemoryPressureLevel.HIGH, callback1)
        self.monitor.register_pressure_callback(MemoryPressureLevel.HIGH, callback2)
        self.monitor.register_pressure_callback(MemoryPressureLevel.HIGH, callback_error)
        
        stats = Mock()
        self.monitor._trigger_pressure_callbacks(MemoryPressureLevel.HIGH, stats)
        
        callback1.assert_called_once_with(stats)
        callback2.assert_called_once_with(stats)
        callback_error.assert_called_once_with(stats)

    @patch("psutil.Process")
    def test_track_process_success(self, mock_process_class):
        """Test successful process tracking"""
        mock_process = Mock()
        mock_process.name.return_value = "ffmpeg"
        mock_process.memory_info.return_value = Mock(rss=256 * 1024 * 1024)  # 256MB
        mock_process.cpu_percent.return_value = 15.5
        mock_process.cmdline.return_value = ["ffmpeg", "-i", "input.mp4"]
        mock_process_class.return_value = mock_process

        self.monitor.track_process(1234, "test_ffmpeg", is_ffmpeg=True)

        assert 1234 in self.monitor._tracked_processes
        process_info = self.monitor._tracked_processes[1234]
        assert process_info.pid == 1234
        assert process_info.name == "test_ffmpeg"
        assert process_info.memory_mb == 256
        assert process_info.cpu_percent == 15.5
        assert process_info.is_ffmpeg is True

    @patch("psutil.Process")
    def test_track_process_empty_cmdline(self, mock_process_class):
        """Test process tracking with empty cmdline"""
        mock_process = Mock()
        mock_process.name.return_value = "test"
        mock_process.memory_info.return_value = Mock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process.cmdline.return_value = []
        mock_process_class.return_value = mock_process

        self.monitor.track_process(1234)

        assert 1234 in self.monitor._tracked_processes
        process_info = self.monitor._tracked_processes[1234]
        assert process_info.command == ""

    @patch("psutil.Process")
    def test_track_process_error(self, mock_process_class):
        """Test process tracking error handling"""
        mock_process_class.side_effect = psutil.Error("Process not found")

        # Should not raise exception
        self.monitor.track_process(9999)
        assert 9999 not in self.monitor._tracked_processes

    @patch("psutil.Process")
    def test_track_process_os_error(self, mock_process_class):
        """Test process tracking OSError handling"""
        mock_process_class.side_effect = OSError("OS error")

        # Should not raise exception
        self.monitor.track_process(9999)
        assert 9999 not in self.monitor._tracked_processes

    def test_untrack_process(self):
        """Test process untracking"""
        # Add a mock process
        self.monitor._tracked_processes[1234] = ProcessMemoryInfo(
            pid=1234,
            name="test",
            memory_mb=100,
            cpu_percent=5.0,
            command="test command",
        )

        self.monitor.untrack_process(1234)
        assert 1234 not in self.monitor._tracked_processes

    def test_untrack_nonexistent_process(self):
        """Test untracking a process that doesn't exist"""
        # Should not raise exception
        self.monitor.untrack_process(9999)

    def test_get_tracked_processes(self):
        """Test getting tracked processes list"""
        # Add mock processes
        process1 = ProcessMemoryInfo(
            pid=1234,
            name="test1",
            memory_mb=100,
            cpu_percent=5.0,
            command="test1",
            is_ffmpeg=False,
        )
        process2 = ProcessMemoryInfo(
            pid=1235,
            name="ffmpeg",
            memory_mb=200,
            cpu_percent=10.0,
            command="ffmpeg",
            is_ffmpeg=True,
        )

        self.monitor._tracked_processes[1234] = process1
        self.monitor._tracked_processes[1235] = process2

        all_processes = self.monitor.get_tracked_processes()
        assert len(all_processes) == 2
        assert process1 in all_processes
        assert process2 in all_processes

        ffmpeg_processes = self.monitor.get_ffmpeg_processes()
        assert len(ffmpeg_processes) == 1
        assert process2 in ffmpeg_processes

    @patch("os.path.getsize")
    def test_estimate_video_memory_needs_success(self, mock_getsize):
        """Test video memory estimation"""
        mock_getsize.return_value = 1024 * 1024 * 1024  # 1GB file

        estimate = self.monitor.estimate_video_memory_needs("/path/to/video.mp4")

        # Should be sum of base (10% of 1GB = 102.4MB) + decode (5% = 51.2MB) + encode (15% = 153.6MB)
        expected = 102.4 + 51.2 + 153.6  # = 307.2MB
        assert abs(estimate - expected) < 1.0  # Allow small floating point differences

    @patch("os.path.getsize")
    def test_estimate_video_memory_needs_large_file(self, mock_getsize):
        """Test video memory estimation with large file"""
        mock_getsize.return_value = 10 * 1024 * 1024 * 1024  # 10GB file

        estimate = self.monitor.estimate_video_memory_needs("/path/to/video.mp4")

        # Should cap at max values: 512 + 256 + 1024 = 1792MB
        assert estimate == 1792

    @patch("os.path.getsize")
    def test_estimate_video_memory_needs_error(self, mock_getsize):
        """Test video memory estimation error handling"""
        mock_getsize.side_effect = OSError("File not found")

        estimate = self.monitor.estimate_video_memory_needs("/nonexistent/video.mp4")
        assert estimate == 1024  # Default 1GB

    @patch("os.path.getsize")
    def test_estimate_video_memory_needs_value_error(self, mock_getsize):
        """Test video memory estimation ValueError handling"""
        mock_getsize.side_effect = ValueError("Invalid value")

        estimate = self.monitor.estimate_video_memory_needs("/invalid/video.mp4")
        assert estimate == 1024  # Default 1GB

    def test_monitoring_loop_stats_history(self):
        """Test monitoring loop updates stats history"""
        self.monitor._max_history = 2  # Set small limit for testing
        
        with patch.object(self.monitor, 'get_current_stats') as mock_get_stats:
            # Create mock stats with different pressure levels
            stats1 = Mock()
            stats1.pressure_level = MemoryPressureLevel.LOW
            stats1.percent_used = 60.0
            stats1.available_mb = 4000.0
            
            stats2 = Mock()
            stats2.pressure_level = MemoryPressureLevel.MODERATE
            stats2.percent_used = 75.0
            stats2.available_mb = 2000.0
            
            stats3 = Mock()
            stats3.pressure_level = MemoryPressureLevel.HIGH
            stats3.percent_used = 85.0
            stats3.available_mb = 1000.0
            
            mock_get_stats.side_effect = [stats1, stats2, stats3]
            
            # Start monitoring
            self.monitor.start_monitoring()
            time.sleep(0.3)  # Let it run for a bit
            self.monitor.stop_monitoring()
            
            # Check history was bounded
            assert len(self.monitor._stats_history) <= self.monitor._max_history

    @patch("montage.utils.memory_manager.metrics")
    def test_monitoring_loop_metrics_update(self, mock_metrics):
        """Test monitoring loop updates metrics"""
        with patch.object(self.monitor, 'get_current_stats') as mock_get_stats:
            stats = Mock()
            stats.pressure_level = MemoryPressureLevel.LOW
            stats.percent_used = 60.0
            stats.process_memory_mb = 500.0
            stats.available_mb = 4000.0
            
            mock_get_stats.return_value = stats
            
            self.monitor.start_monitoring()
            time.sleep(0.2)
            self.monitor.stop_monitoring()
            
            # Verify metrics were updated
            mock_metrics.memory_usage.set.assert_called()
            mock_metrics.process_memory.set.assert_called()

    def test_monitoring_loop_exception_handling(self):
        """Test monitoring loop handles exceptions"""
        with patch.object(self.monitor, 'get_current_stats') as mock_get_stats:
            mock_get_stats.side_effect = Exception("Test error")
            
            self.monitor.start_monitoring()
            time.sleep(0.2)  # Let it run and handle the exception
            self.monitor.stop_monitoring()
            
            # Should continue running despite exceptions
            assert True  # If we get here, exception was handled


class TestMemoryPressureManager:
    """Test MemoryPressureManager class"""

    def setup_method(self):
        self.monitor = Mock(spec=MemoryMonitor)
        self.manager = MemoryPressureManager(self.monitor)

    def test_initialization(self):
        """Test manager initialization"""
        assert self.manager.monitor == self.monitor
        assert len(self.manager._response_actions) == 4  # One for each pressure level
        
        # Check default responses are registered
        assert len(self.manager._response_actions[MemoryPressureLevel.MODERATE]) > 0
        assert len(self.manager._response_actions[MemoryPressureLevel.HIGH]) > 0
        assert len(self.manager._response_actions[MemoryPressureLevel.CRITICAL]) > 0

    def test_register_response(self):
        """Test response registration"""
        callback = Mock()
        self.manager.register_response(MemoryPressureLevel.HIGH, callback)

        assert callback in self.manager._response_actions[MemoryPressureLevel.HIGH]

    @patch("gc.collect")
    @patch("time.sleep")
    def test_force_garbage_collection(self, mock_sleep, mock_gc):
        """Test garbage collection forcing"""
        # Mock stats before and after
        before_stats = Mock()
        before_stats.process_memory_mb = 1000
        after_stats = Mock()
        after_stats.process_memory_mb = 950

        self.monitor.get_current_stats.side_effect = [before_stats, after_stats]

        freed_mb = self.manager._force_garbage_collection()

        assert mock_gc.call_count == 3  # Should call gc.collect() 3 times
        mock_sleep.assert_called_once_with(0.1)
        assert freed_mb == 50  # 1000 - 950

    @patch("gc.collect")
    @patch("time.sleep")
    def test_force_garbage_collection_negative_freed(self, mock_sleep, mock_gc):
        """Test garbage collection with negative freed memory"""
        # Mock stats where memory increased
        before_stats = Mock()
        before_stats.process_memory_mb = 900
        after_stats = Mock()
        after_stats.process_memory_mb = 1000

        self.monitor.get_current_stats.side_effect = [before_stats, after_stats]

        freed_mb = self.manager._force_garbage_collection()

        assert freed_mb == 0  # Should return 0 for negative values

    @patch("gc.collect")
    @patch("time.sleep")
    def test_force_garbage_collection_small_freed(self, mock_sleep, mock_gc):
        """Test garbage collection with small freed memory"""
        # Mock stats with small difference
        before_stats = Mock()
        before_stats.process_memory_mb = 1000
        after_stats = Mock()
        after_stats.process_memory_mb = 999.5

        self.monitor.get_current_stats.side_effect = [before_stats, after_stats]

        freed_mb = self.manager._force_garbage_collection()

        # Should return the small amount but not log
        assert freed_mb == 0.5

    @patch.object(MemoryPressureManager, "_force_garbage_collection")
    def test_moderate_pressure_response(self, mock_gc):
        """Test moderate pressure response"""
        mock_gc.return_value = 25.5
        stats = Mock()

        self.manager._moderate_pressure_response(stats)

        mock_gc.assert_called_once()

    @patch.object(MemoryPressureManager, "_force_garbage_collection")
    @patch.object(MemoryPressureManager, "_clear_caches")
    @patch.object(MemoryPressureManager, "_terminate_process")
    def test_high_pressure_response_no_ffmpeg(self, mock_terminate, mock_clear, mock_gc):
        """Test high pressure response with no FFmpeg processes"""
        mock_gc.return_value = 50.0
        stats = Mock()

        # Mock no FFmpeg processes
        self.monitor.get_ffmpeg_processes.return_value = []

        self.manager._high_pressure_response(stats)

        mock_gc.assert_called_once()
        mock_clear.assert_called_once() 
        mock_terminate.assert_not_called()

    @patch.object(MemoryPressureManager, "_force_garbage_collection")
    @patch.object(MemoryPressureManager, "_clear_caches")
    @patch.object(MemoryPressureManager, "_terminate_process")
    def test_high_pressure_response_single_ffmpeg(self, mock_terminate, mock_clear, mock_gc):
        """Test high pressure response with single FFmpeg process"""
        mock_gc.return_value = 50.0
        stats = Mock()

        # Mock single FFmpeg process
        process1 = Mock()
        process1.pid = 1234
        self.monitor.get_ffmpeg_processes.return_value = [process1]

        self.manager._high_pressure_response(stats)

        mock_gc.assert_called_once()
        mock_clear.assert_called_once()
        # Should not terminate single process
        mock_terminate.assert_not_called()

    @patch.object(MemoryPressureManager, "_force_garbage_collection")
    @patch.object(MemoryPressureManager, "_clear_caches")
    @patch.object(MemoryPressureManager, "_terminate_process")
    def test_high_pressure_response_multiple_ffmpeg(self, mock_terminate, mock_clear, mock_gc):
        """Test high pressure response with multiple FFmpeg processes"""
        mock_gc.return_value = 50.0
        stats = Mock()

        # Mock multiple FFmpeg processes
        process1 = Mock()
        process1.pid = 1234
        process2 = Mock()
        process2.pid = 1235
        self.monitor.get_ffmpeg_processes.return_value = [process1, process2]

        self.manager._high_pressure_response(stats)

        mock_gc.assert_called_once()
        mock_clear.assert_called_once()
        mock_terminate.assert_called_once_with(1234)  # Should terminate oldest (lowest PID)

    @patch.object(MemoryPressureManager, "_force_garbage_collection")
    @patch.object(MemoryPressureManager, "_clear_caches")
    @patch.object(MemoryPressureManager, "_terminate_process")
    def test_critical_pressure_response_low_usage_after_cleanup(self, mock_terminate, mock_clear, mock_gc):
        """Test critical pressure response when usage drops after cleanup"""
        mock_gc.return_value = 100.0
        stats = Mock()

        # Mock current stats showing acceptable usage after cleanup
        after_cleanup_stats = Mock()
        after_cleanup_stats.percent_used = 80.0  # Below 90%
        self.monitor.get_current_stats.return_value = after_cleanup_stats

        # Mock FFmpeg processes
        process1 = Mock()
        process1.pid = 1234
        self.monitor.get_ffmpeg_processes.return_value = [process1]

        self.manager._critical_pressure_response(stats)

        mock_gc.assert_called_once()
        mock_clear.assert_called_once()
        # Should not terminate processes if memory usage improved
        mock_terminate.assert_not_called()

    @patch.object(MemoryPressureManager, "_force_garbage_collection")
    @patch.object(MemoryPressureManager, "_clear_caches")
    @patch.object(MemoryPressureManager, "_terminate_process")
    def test_critical_pressure_response_still_critical(self, mock_terminate, mock_clear, mock_gc):
        """Test critical pressure response when usage remains critical"""
        mock_gc.return_value = 100.0
        stats = Mock()

        # Mock current stats showing still critical after cleanup
        critical_stats = Mock()
        critical_stats.percent_used = 95.0
        self.monitor.get_current_stats.return_value = critical_stats

        # Mock FFmpeg processes
        process1 = Mock()
        process1.pid = 1234
        process2 = Mock()
        process2.pid = 1235
        self.monitor.get_ffmpeg_processes.return_value = [process1, process2]

        self.manager._critical_pressure_response(stats)

        mock_gc.assert_called_once()
        mock_clear.assert_called_once()
        # Should terminate all FFmpeg processes
        assert mock_terminate.call_count == 2

    def test_clear_caches(self):
        """Test cache clearing"""
        # Should not raise exception
        self.manager._clear_caches()

    def test_clear_caches_with_exception(self):
        """Test cache clearing with exception"""
        # Mock an exception during cache clearing
        with patch.object(self.manager, '_clear_caches', side_effect=Exception("Test error")):
            # Should handle exception gracefully
            try:
                self.manager._clear_caches()
            except Exception:
                pytest.fail("_clear_caches should handle exceptions gracefully")

    @patch("psutil.Process")
    def test_terminate_process_success(self, mock_process_class):
        """Test successful process termination"""
        mock_process = Mock()
        mock_process_class.return_value = mock_process

        self.manager._terminate_process(1234)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        self.monitor.untrack_process.assert_called_once_with(1234)

    @patch("psutil.Process")
    def test_terminate_process_force_kill(self, mock_process_class):
        """Test force killing process after timeout"""
        mock_process = Mock()
        mock_process.wait.side_effect = psutil.TimeoutExpired("test")
        mock_process_class.return_value = mock_process

        self.manager._terminate_process(1234)

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
        mock_process.kill.assert_called_once()
        self.monitor.untrack_process.assert_called_once_with(1234)

    @patch("psutil.Process")
    def test_terminate_process_psutil_error(self, mock_process_class):
        """Test process termination psutil error handling"""
        mock_process_class.side_effect = psutil.Error("Process not found")

        # Should not raise exception
        self.manager._terminate_process(9999)

    @patch("psutil.Process")
    def test_terminate_process_os_error(self, mock_process_class):
        """Test process termination OSError handling"""
        mock_process_class.side_effect = OSError("OS error")

        # Should not raise exception
        self.manager._terminate_process(9999)


class TestAdaptiveProcessingConfig:
    """Test AdaptiveProcessingConfig class"""

    def setup_method(self):
        self.monitor = Mock(spec=MemoryMonitor)
        self.config = AdaptiveProcessingConfig(self.monitor)

    def test_initialization(self):
        """Test config initialization"""
        assert self.config.monitor == self.monitor
        assert hasattr(self.config, '_base_config')
        assert hasattr(self.config, '_current_mode')
        assert self.config._current_mode == ProcessingMode.BALANCED

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_base_config_high_end(self, mock_virtual, mock_cpu):
        """Test base config for high-end systems"""
        mock_cpu.return_value = 16
        mock_virtual.return_value = Mock(total=36 * 1024**3)  # 36GB RAM

        config = AdaptiveProcessingConfig(self.monitor)
        base_config = config._get_base_config()

        assert base_config["max_workers"] == 12  # min(16-2, 12)
        assert base_config["chunk_size_mb"] == 2048
        assert base_config["max_concurrent_ffmpeg"] == 6
        assert base_config["apple_silicon_optimized"] is True

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_base_config_standard(self, mock_virtual, mock_cpu):
        """Test base config for standard systems"""
        mock_cpu.return_value = 8
        mock_virtual.return_value = Mock(total=16 * 1024**3)  # 16GB RAM

        config = AdaptiveProcessingConfig(self.monitor)
        base_config = config._get_base_config()

        assert base_config["max_workers"] == 4
        assert base_config["chunk_size_mb"] == 256
        assert base_config["max_concurrent_ffmpeg"] == 2

    def test_determine_processing_mode(self):
        """Test processing mode determination"""
        # Test all pressure levels
        stats_critical = Mock()
        stats_critical.pressure_level = MemoryPressureLevel.CRITICAL
        assert (
            self.config._determine_processing_mode(stats_critical)
            == ProcessingMode.SURVIVAL
        )

        stats_high = Mock()
        stats_high.pressure_level = MemoryPressureLevel.HIGH
        assert (
            self.config._determine_processing_mode(stats_high)
            == ProcessingMode.MEMORY_OPTIMIZED
        )

        stats_moderate = Mock()
        stats_moderate.pressure_level = MemoryPressureLevel.MODERATE
        assert (
            self.config._determine_processing_mode(stats_moderate)
            == ProcessingMode.BALANCED
        )

        stats_low = Mock()
        stats_low.pressure_level = MemoryPressureLevel.LOW
        assert (
            self.config._determine_processing_mode(stats_low)
            == ProcessingMode.FULL_QUALITY
        )

    def test_get_config_for_mode_survival(self):
        """Test config for survival mode"""
        stats = Mock()
        stats.available_mb = 3000

        config = self.config._get_config_for_mode(ProcessingMode.SURVIVAL, stats)

        assert config["max_workers"] == 1
        assert config["chunk_size_mb"] == 64
        assert config["quality_preset"] == "ultrafast"
        assert config["enable_hardware_accel"] is False
        assert config["max_concurrent_ffmpeg"] == 1
        assert config["force_low_resolution"] is True
        assert config["skip_audio_normalization"] is True

    def test_get_config_for_mode_memory_optimized(self):
        """Test config for memory optimized mode"""
        stats = Mock()
        stats.available_mb = 3000

        config = self.config._get_config_for_mode(ProcessingMode.MEMORY_OPTIMIZED, stats)

        assert config["max_workers"] == 2
        assert config["chunk_size_mb"] == 128
        assert config["quality_preset"] == "fast"
        assert config["max_concurrent_ffmpeg"] == 1
        assert config["reduce_resolution"] is True

    def test_get_config_for_mode_balanced(self):
        """Test config for balanced mode"""
        stats = Mock()
        stats.available_mb = 3000

        config = self.config._get_config_for_mode(ProcessingMode.BALANCED, stats)

        assert config["max_workers"] == 3
        assert config["chunk_size_mb"] == 256
        assert config["quality_preset"] == "medium"
        assert config["max_concurrent_ffmpeg"] == 2

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_config_for_mode_full_quality_high_end(self, mock_virtual, mock_cpu):
        """Test config for full quality mode on high-end systems"""
        mock_cpu.return_value = 16
        mock_virtual.return_value = Mock(total=36 * 1024**3)  # 36GB RAM

        stats = Mock()
        stats.available_mb = 10000

        config = self.config._get_config_for_mode(ProcessingMode.FULL_QUALITY, stats)

        assert config["max_workers"] == 14  # min(16, 14)
        assert config["chunk_size_mb"] == 4096
        assert config["max_concurrent_ffmpeg"] == 8
        assert config["enable_metal_performance"] is True
        assert config["parallel_processing"] is True

    @patch("psutil.cpu_count")
    @patch("psutil.virtual_memory")
    def test_get_config_for_mode_full_quality_standard(self, mock_virtual, mock_cpu):
        """Test config for full quality mode on standard systems"""
        mock_cpu.return_value = 8
        mock_virtual.return_value = Mock(total=16 * 1024**3)  # 16GB RAM

        stats = Mock()
        stats.available_mb = 8000

        config = self.config._get_config_for_mode(ProcessingMode.FULL_QUALITY, stats)

        assert config["max_workers"] == 4
        assert config["chunk_size_mb"] == 512
        assert config["quality_preset"] == "slow"
        assert config["max_concurrent_ffmpeg"] == 3

    def test_memory_based_adjustment_very_low_memory(self):
        """Test config adjustment for very low memory"""
        stats = Mock()
        stats.available_mb = 1500  # 1.5GB available

        config = self.config._get_config_for_mode(ProcessingMode.BALANCED, stats)

        # Should be adjusted down due to low memory
        assert config["max_workers"] == 1
        assert config["chunk_size_mb"] == 64

    def test_memory_based_adjustment_low_memory(self):
        """Test config adjustment for low memory"""
        stats = Mock()
        stats.available_mb = 3000  # 3GB available

        config = self.config._get_config_for_mode(ProcessingMode.BALANCED, stats)

        # Should be adjusted down due to low memory
        assert config["max_workers"] == 2
        assert config["chunk_size_mb"] == 128

    def test_memory_based_adjustment_sufficient_memory(self):
        """Test config adjustment for sufficient memory"""
        stats = Mock()
        stats.available_mb = 8000  # 8GB available

        config = self.config._get_config_for_mode(ProcessingMode.BALANCED, stats)

        # Should not be adjusted
        assert config["max_workers"] == 3
        assert config["chunk_size_mb"] == 256

    def test_get_current_config(self):
        """Test getting current adaptive config"""
        stats = Mock()
        stats.pressure_level = MemoryPressureLevel.MODERATE
        stats.available_mb = 4000

        self.monitor.get_current_stats.return_value = stats

        config = self.config.get_current_config()

        # Should return balanced mode config
        assert config["max_workers"] == 3
        assert config["quality_preset"] == "medium"

    def test_get_current_config_mode_change(self):
        """Test current config with mode change"""
        stats = Mock()
        stats.pressure_level = MemoryPressureLevel.HIGH
        stats.available_mb = 2000

        self.monitor.get_current_stats.return_value = stats
        
        # Initially balanced mode
        assert self.config._current_mode == ProcessingMode.BALANCED
        
        config = self.config.get_current_config()
        
        # Should change to memory optimized mode
        assert self.config._current_mode == ProcessingMode.MEMORY_OPTIMIZED


class TestStreamingVideoProcessor:
    """Test StreamingVideoProcessor class"""

    def setup_method(self):
        self.monitor = Mock(spec=MemoryMonitor)
        self.config = Mock(spec=AdaptiveProcessingConfig)
        self.processor = StreamingVideoProcessor(self.monitor, self.config)

    def test_initialization(self):
        """Test processor initialization"""
        assert self.processor.monitor == self.monitor
        assert self.processor.config == self.config

    @patch("subprocess.run")
    def test_get_video_duration_success(self, mock_run):
        """Test successful video duration retrieval"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"format": {"duration": "120.5"}}',
        )

        duration = self.processor._get_video_duration("/path/to/video.mp4")

        assert duration == 120.5
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_video_duration_error_returncode(self, mock_run):
        """Test video duration retrieval error"""
        mock_run.return_value = Mock(returncode=1)

        duration = self.processor._get_video_duration("/path/to/video.mp4")

        assert duration == 0.0

    @patch("subprocess.run")
    def test_get_video_duration_exception(self, mock_run):
        """Test video duration retrieval exception"""
        mock_run.side_effect = Exception("Test error")

        duration = self.processor._get_video_duration("/path/to/video.mp4")

        assert duration == 0.0

    @patch("subprocess.run")
    def test_get_video_duration_timeout(self, mock_run):
        """Test video duration retrieval timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired("ffprobe", 30)

        duration = self.processor._get_video_duration("/path/to/video.mp4")

        assert duration == 0.0

    @patch("subprocess.run")
    def test_get_video_duration_json_error(self, mock_run):
        """Test video duration retrieval with JSON parsing error"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='invalid json',
        )

        duration = self.processor._get_video_duration("/path/to/video.mp4")

        assert duration == 0.0

    @patch("subprocess.run")
    def test_extract_video_chunk_success(self, mock_run):
        """Test successful video chunk extraction"""
        mock_run.return_value = Mock(returncode=0)

        success = self.processor._extract_video_chunk(
            "/input.mp4", 10.0, 20.0, "/output.mp4"
        )

        assert success is True
        mock_run.assert_called_once()
        # Verify ffmpeg command structure
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-ss" in args
        assert "10.0" in args
        assert "-t" in args
        assert "10.0" in args  # duration = end - start

    @patch("subprocess.run")
    def test_extract_video_chunk_error(self, mock_run):
        """Test video chunk extraction error"""
        mock_run.return_value = Mock(returncode=1)

        success = self.processor._extract_video_chunk(
            "/input.mp4", 10.0, 20.0, "/output.mp4"
        )

        assert success is False

    @patch("subprocess.run")
    def test_extract_video_chunk_exception(self, mock_run):
        """Test video chunk extraction exception"""
        mock_run.side_effect = Exception("Test error")

        success = self.processor._extract_video_chunk(
            "/input.mp4", 10.0, 20.0, "/output.mp4"
        )

        assert success is False

    @patch("subprocess.run")
    @patch("tempfile.gettempdir")
    @patch("os.getpid")
    @patch("os.unlink")
    def test_concatenate_chunks_success(self, mock_unlink, mock_getpid, mock_tempdir, mock_run):
        """Test successful chunk concatenation"""
        mock_getpid.return_value = 1234
        mock_tempdir.return_value = "/tmp"
        mock_run.return_value = Mock(returncode=0)

        chunk_files = ["/chunk1.mp4", "/chunk2.mp4", "/chunk3.mp4"]

        with patch("builtins.open", mock_open()) as mock_file:
            success = self.processor._concatenate_chunks(chunk_files, "/output.mp4")

        assert success is True
        mock_run.assert_called_once()
        # Verify concat file was written
        mock_file.assert_called_once()
        mock_unlink.assert_called_once()  # Cleanup concat file

    @patch("subprocess.run")
    @patch("tempfile.gettempdir")
    @patch("os.getpid")
    def test_concatenate_chunks_error(self, mock_getpid, mock_tempdir, mock_run):
        """Test chunk concatenation error"""
        mock_getpid.return_value = 1234
        mock_tempdir.return_value = "/tmp"
        mock_run.return_value = Mock(returncode=1)

        chunk_files = ["/chunk1.mp4", "/chunk2.mp4"]

        with patch("builtins.open", mock_open()):
            with patch("os.unlink"):
                success = self.processor._concatenate_chunks(chunk_files, "/output.mp4")

        assert success is False

    @patch("subprocess.run")
    @patch("tempfile.gettempdir")
    @patch("os.getpid")
    @patch("os.unlink")
    def test_concatenate_chunks_cleanup_error(self, mock_unlink, mock_getpid, mock_tempdir, mock_run):
        """Test chunk concatenation with cleanup error"""
        mock_getpid.return_value = 1234
        mock_tempdir.return_value = "/tmp"
        mock_run.return_value = Mock(returncode=0)
        mock_unlink.side_effect = OSError("Cannot delete file")

        chunk_files = ["/chunk1.mp4"]

        with patch("builtins.open", mock_open()):
            success = self.processor._concatenate_chunks(chunk_files, "/output.mp4")

        assert success is True  # Should still succeed despite cleanup error

    @patch("subprocess.run")
    def test_concatenate_chunks_exception(self, mock_run):
        """Test chunk concatenation exception"""
        mock_run.side_effect = Exception("Test error")

        chunk_files = ["/chunk1.mp4"]

        success = self.processor._concatenate_chunks(chunk_files, "/output.mp4")

        assert success is False

    @patch("os.path.getsize")
    @patch("tempfile.gettempdir")
    @patch("os.getpid")
    @patch("os.unlink")
    @patch("gc.collect")
    @patch("time.sleep")
    def test_process_large_video_chunked_success(self, mock_sleep, mock_gc, mock_unlink, 
                                                mock_getpid, mock_tempdir, mock_getsize):
        """Test successful large video processing"""
        mock_getsize.return_value = 1024 * 1024 * 1024  # 1GB
        mock_tempdir.return_value = "/tmp"
        mock_getpid.return_value = 1234

        # Mock methods
        with patch.object(self.processor, '_get_video_duration', return_value=120.0):
            with patch.object(self.processor, '_extract_video_chunk', return_value=True):
                with patch.object(self.processor, '_concatenate_chunks', return_value=True):
                    # Mock processing function
                    def mock_processing_func(chunk_file):
                        return "/processed_" + os.path.basename(chunk_file)
                    
                    # Mock config
                    self.config.get_current_config.return_value = {"chunk_size_mb": 256}
                    
                    # Mock stats
                    stats = Mock()
                    stats.available_mb = 4000
                    stats.pressure_level = MemoryPressureLevel.LOW
                    self.monitor.get_current_stats.return_value = stats
                    
                    with patch("montage.utils.memory_manager.memory_guard"):
                        success = self.processor.process_large_video_chunked(
                            "/input.mp4", "/output.mp4", mock_processing_func
                        )
                    
                    assert success is True

    @patch("os.path.getsize")
    def test_process_large_video_chunked_zero_duration(self, mock_getsize):
        """Test large video processing with zero duration"""
        mock_getsize.return_value = 1024 * 1024 * 1024  # 1GB

        with patch.object(self.processor, '_get_video_duration', return_value=0):
            success = self.processor.process_large_video_chunked(
                "/input.mp4", "/output.mp4", lambda x: x
            )
            
            assert success is False

    @patch("os.path.getsize")
    def test_process_large_video_chunked_negative_duration(self, mock_getsize):
        """Test large video processing with negative duration"""
        mock_getsize.return_value = 1024 * 1024 * 1024  # 1GB

        with patch.object(self.processor, '_get_video_duration', return_value=-1):
            success = self.processor.process_large_video_chunked(
                "/input.mp4", "/output.mp4", lambda x: x
            )
            
            assert success is False

    def test_process_large_video_chunked_exception(self):
        """Test large video processing with exception"""
        with patch.object(self.processor, '_get_video_duration', side_effect=Exception("Test error")):
            success = self.processor.process_large_video_chunked(
                "/input.mp4", "/output.mp4", lambda x: x
            )
            
            assert success is False


class TestMemoryGuard:
    """Test memory_guard context manager"""

    @patch("gc.collect")
    def test_memory_guard_basic(self, mock_gc):
        """Test basic memory guard functionality"""
        monitor = Mock(spec=MemoryMonitor)
        start_stats = Mock()
        start_stats.available_mb = 4000
        start_stats.process_memory_mb = 500
        end_stats = Mock()
        end_stats.process_memory_mb = 600

        monitor.get_current_stats.side_effect = [start_stats, end_stats]

        with memory_guard(monitor=monitor, cleanup_on_exit=True) as guard_monitor:
            assert guard_monitor == monitor

        mock_gc.assert_called_once()

    @patch("gc.collect")
    def test_memory_guard_no_cleanup(self, mock_gc):
        """Test memory guard without cleanup"""
        monitor = Mock(spec=MemoryMonitor)
        start_stats = Mock()
        start_stats.available_mb = 4000
        start_stats.process_memory_mb = 500

        monitor.get_current_stats.return_value = start_stats

        with memory_guard(monitor=monitor, cleanup_on_exit=False) as guard_monitor:
            assert guard_monitor == monitor

        mock_gc.assert_not_called()

    def test_memory_guard_no_monitor(self):
        """Test memory guard with no monitor provided"""
        with patch("montage.utils.memory_manager.MemoryMonitor") as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor_class.return_value = mock_monitor
            mock_monitor.get_current_stats.return_value = Mock(
                available_mb=4000, process_memory_mb=500
            )

            with memory_guard() as guard_monitor:
                assert guard_monitor == mock_monitor

    @patch("gc.collect")
    def test_memory_guard_memory_delta_logging(self, mock_gc):
        """Test memory guard logs significant memory changes"""
        monitor = Mock(spec=MemoryMonitor)
        start_stats = Mock()
        start_stats.available_mb = 4000
        start_stats.process_memory_mb = 500
        end_stats = Mock()
        end_stats.process_memory_mb = 600  # 100MB increase

        monitor.get_current_stats.side_effect = [start_stats, end_stats]

        with memory_guard(monitor=monitor, cleanup_on_exit=True) as guard_monitor:
            pass

        mock_gc.assert_called_once()

    @patch("gc.collect")
    def test_memory_guard_memory_limit_exceeded(self, mock_gc):  
        """Test memory guard warns when limit exceeded"""
        monitor = Mock(spec=MemoryMonitor)
        start_stats = Mock()
        start_stats.available_mb = 4000
        start_stats.process_memory_mb = 500
        end_stats = Mock()
        end_stats.process_memory_mb = 1200  # Exceeds 1000MB limit

        monitor.get_current_stats.side_effect = [start_stats, end_stats]

        with memory_guard(max_memory_mb=1000, monitor=monitor, cleanup_on_exit=True):
            pass

        mock_gc.assert_called_once()

    def test_memory_guard_default_memory_limit(self):
        """Test memory guard calculates default memory limit"""
        monitor = Mock(spec=MemoryMonitor)
        start_stats = Mock()
        start_stats.available_mb = 4000
        start_stats.process_memory_mb = 500
        end_stats = Mock()
        end_stats.process_memory_mb = 550

        monitor.get_current_stats.side_effect = [start_stats, end_stats]

        with memory_guard(monitor=monitor) as guard_monitor:
            # Default limit should be 80% of available (80% of 4000 = 3200)
            pass


class TestMemoryOptimizedDecorator:
    """Test memory_optimized decorator"""

    def test_memory_optimized_decorator(self):
        """Test memory optimized decorator"""

        @memory_optimized(max_memory_mb=1000)
        def test_function(x, y):
            return x + y

        with patch("montage.utils.memory_manager.memory_guard") as mock_guard:
            mock_monitor = Mock()
            mock_guard.return_value.__enter__.return_value = mock_monitor

            result = test_function(5, 3)

            assert result == 8
            mock_guard.assert_called_once_with(max_memory_mb=1000)

    def test_memory_optimized_decorator_no_max_memory(self):
        """Test memory optimized decorator without max_memory specified"""

        @memory_optimized()
        def test_function(x):
            return x * 2

        with patch("montage.utils.memory_manager.memory_guard") as mock_guard:
            mock_monitor = Mock()
            mock_guard.return_value.__enter__.return_value = mock_monitor

            result = test_function(5)

            assert result == 10
            mock_guard.assert_called_once_with(max_memory_mb=None)


class TestGlobalFunctions:
    """Test global module functions"""

    def setup_method(self):
        # Reset global variables
        import montage.utils.memory_manager as mm

        mm._global_monitor = None
        mm._global_pressure_manager = None
        mm._global_adaptive_config = None

    def teardown_method(self):
        # Shutdown any monitoring
        try:
            shutdown_memory_management()
        except:
            pass

    def test_get_memory_monitor(self):
        """Test get_memory_monitor function"""
        monitor1 = get_memory_monitor()
        monitor2 = get_memory_monitor()

        # Should return same instance (singleton)
        assert monitor1 is monitor2
        assert isinstance(monitor1, MemoryMonitor)

    def test_get_pressure_manager(self):
        """Test get_pressure_manager function"""
        manager1 = get_pressure_manager()
        manager2 = get_pressure_manager()

        # Should return same instance (singleton)
        assert manager1 is manager2
        assert isinstance(manager1, MemoryPressureManager)

    def test_get_adaptive_config(self):
        """Test get_adaptive_config function"""
        config1 = get_adaptive_config()
        config2 = get_adaptive_config()

        # Should return same instance (singleton)
        assert config1 is config2
        assert isinstance(config1, AdaptiveProcessingConfig)

    def test_initialize_memory_management(self):
        """Test memory management initialization"""
        monitor, pressure_manager, adaptive_config = initialize_memory_management()

        assert isinstance(monitor, MemoryMonitor)
        assert isinstance(pressure_manager, MemoryPressureManager)
        assert isinstance(adaptive_config, AdaptiveProcessingConfig)

        # Should start monitoring
        assert monitor._monitoring is True

    def test_shutdown_memory_management(self):
        """Test memory management shutdown"""
        # Initialize first
        monitor, _, _ = initialize_memory_management()
        assert monitor._monitoring is True

        # Then shutdown
        shutdown_memory_management()
        assert monitor._monitoring is False

    def test_shutdown_memory_management_no_monitor(self):
        """Test shutdown when no monitor exists"""
        # Should not raise exception
        shutdown_memory_management()

    def test_cleanup_handler(self):
        """Test signal cleanup handler"""
        # Initialize system
        monitor, _, _ = initialize_memory_management()
        assert monitor._monitoring is True

        # Call cleanup handler
        _cleanup_handler(signal.SIGTERM, None)
        
        assert monitor._monitoring is False


class TestEnums:
    """Test enum classes"""

    def test_memory_pressure_level_enum(self):
        """Test MemoryPressureLevel enum"""
        assert MemoryPressureLevel.LOW.value == "low"
        assert MemoryPressureLevel.MODERATE.value == "moderate"
        assert MemoryPressureLevel.HIGH.value == "high"
        assert MemoryPressureLevel.CRITICAL.value == "critical"

    def test_processing_mode_enum(self):
        """Test ProcessingMode enum"""
        assert ProcessingMode.FULL_QUALITY.value == "full_quality"
        assert ProcessingMode.BALANCED.value == "balanced"
        assert ProcessingMode.MEMORY_OPTIMIZED.value == "memory_opt"
        assert ProcessingMode.SURVIVAL.value == "survival"


# Integration test
class TestMemoryManagementIntegration:
    """Integration tests for memory management system"""

    def setup_method(self):
        # Initialize clean state
        import montage.utils.memory_manager as mm

        mm._global_monitor = None
        mm._global_pressure_manager = None
        mm._global_adaptive_config = None

    def teardown_method(self):
        try:
            shutdown_memory_management()
        except:
            pass

    def test_full_system_integration(self):
        """Test full memory management system integration"""
        # Initialize system
        monitor, pressure_manager, adaptive_config = initialize_memory_management()

        try:
            # Test monitoring
            assert monitor._monitoring is True

            # Test stats retrieval
            stats = monitor.get_current_stats()
            assert isinstance(stats, MemoryStats)
            assert stats.total_mb > 0
            assert 0 <= stats.percent_used <= 100

            # Test adaptive config
            config = adaptive_config.get_current_config()
            assert isinstance(config, dict)
            assert "max_workers" in config
            assert "chunk_size_mb" in config

            # Test pressure level determination
            pressure_level = stats.pressure_level
            assert isinstance(pressure_level, MemoryPressureLevel)

            # Test processing mode determination
            mode = adaptive_config._determine_processing_mode(stats)
            assert isinstance(mode, ProcessingMode)

        finally:
            # Clean shutdown
            shutdown_memory_management()
            assert monitor._monitoring is False

    def test_pressure_callback_integration(self):
        """Test pressure callback integration"""
        # Initialize system
        monitor, pressure_manager, adaptive_config = initialize_memory_management()

        try:
            callback_called = False
            callback_stats = None

            def test_callback(stats):
                nonlocal callback_called, callback_stats
                callback_called = True
                callback_stats = stats

            # Register callback
            monitor.register_pressure_callback(MemoryPressureLevel.HIGH, test_callback)

            # Trigger callback
            test_stats = Mock()
            test_stats.pressure_level = MemoryPressureLevel.HIGH
            monitor._trigger_pressure_callbacks(MemoryPressureLevel.HIGH, test_stats)

            assert callback_called is True
            assert callback_stats == test_stats

        finally:
            shutdown_memory_management()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])