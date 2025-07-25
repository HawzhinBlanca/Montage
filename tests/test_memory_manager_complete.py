#!/usr/bin/env python3
"""Complete memory_manager.py tests to achieve ≥75% coverage"""

import gc
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest


class TestMemoryManagerComplete:
    """Complete test coverage for memory_manager.py"""
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_monitor_all_features(self, mock_proc, mock_swap, mock_vm):
        """Test all MemoryMonitor features"""
        # Setup comprehensive mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(
            total=2*1024*1024*1024,
            used=1*1024*1024*1024,
            percent=50.0
        )
        
        # Mock process with all required attributes
        mock_proc_inst = mock.Mock()
        mock_proc_inst.memory_info.return_value = mock.Mock(rss=100*1024*1024)
        mock_proc_inst.cpu_percent.return_value = 25.0
        mock_proc_inst.cmdline.return_value = ['python', 'test.py']
        mock_proc_inst.name.return_value = 'python'
        mock_proc_inst.pid = 1234
        mock_proc.return_value = mock_proc_inst
        
        from montage.utils.memory_manager import (
            MemoryMonitor, MemoryPressureLevel, MemoryStats,
            ProcessMemoryInfo
        )
        
        # Test initialization
        monitor = MemoryMonitor(update_interval=0.1)
        assert monitor.update_interval == 0.1
        assert not monitor._monitoring
        
        # Test get_current_stats
        stats = monitor.get_current_stats()
        assert isinstance(stats, MemoryStats)
        assert stats.total_mb == 8192.0
        assert stats.available_mb == 4096.0
        assert stats.used_mb == 4096.0
        assert stats.percent_used == 50.0
        assert stats.swap_used_mb == 1024.0
        assert stats.process_memory_mb == 100.0
        assert stats.pressure_level == MemoryPressureLevel.LOW
        
        # Test pressure level calculation
        assert monitor._calculate_pressure_level(30.0) == MemoryPressureLevel.LOW
        assert monitor._calculate_pressure_level(75.0) == MemoryPressureLevel.MODERATE
        assert monitor._calculate_pressure_level(85.0) == MemoryPressureLevel.HIGH
        assert monitor._calculate_pressure_level(95.0) == MemoryPressureLevel.CRITICAL
        
        # Test monitoring lifecycle
        monitor.start_monitoring()
        assert monitor._monitoring
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()
        
        # Let monitoring run
        time.sleep(0.2)
        
        # Test pressure callbacks
        callback_triggered = {'count': 0}
        def pressure_callback(stats):
            callback_triggered['count'] += 1
        
        # Register callbacks for all levels
        for level in MemoryPressureLevel:
            monitor.register_pressure_callback(level, pressure_callback)
        
        # Trigger callbacks manually
        monitor._trigger_pressure_callbacks(MemoryPressureLevel.HIGH, stats)
        assert callback_triggered['count'] > 0
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor._monitoring
        
        # Test process tracking
        monitor.track_process(1234, "test_process", is_ffmpeg=True)
        
        # Get tracked processes
        tracked = monitor.get_tracked_processes()
        assert len(tracked) == 1
        assert tracked[0].pid == 1234
        assert tracked[0].name == "test_process"
        assert tracked[0].is_ffmpeg
        
        # Get FFmpeg processes
        ffmpeg_procs = monitor.get_ffmpeg_processes()
        assert len(ffmpeg_procs) == 1
        
        # Untrack process
        monitor.untrack_process(1234)
        assert len(monitor.get_tracked_processes()) == 0
        
        # Test video memory estimation
        with mock.patch('subprocess.run') as mock_run:
            # Success case
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=json.dumps({
                    "streams": [{
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/1",
                        "duration": "60.0",
                        "bit_rate": "4000000"
                    }],
                    "format": {
                        "duration": "60.0",
                        "bit_rate": "5000000"
                    }
                }),
                stderr=""
            )
            
            mem = monitor.estimate_video_memory_needs("/test/video.mp4")
            assert mem > 0
            
            # Error case
            mock_run.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
            mem2 = monitor.estimate_video_memory_needs("/test/error.mp4")
            assert mem2 == 1024  # Default fallback
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_pressure_manager_all_features(self, mock_proc, mock_swap, mock_vm):
        """Test all MemoryPressureManager features"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, MemoryPressureManager, MemoryPressureLevel,
            MemoryStats, ProcessingMode
        )
        
        monitor = MemoryMonitor()
        manager = MemoryPressureManager(monitor)
        
        # Test initialization
        assert manager.monitor == monitor
        assert manager._current_mode == ProcessingMode.BALANCED
        assert not manager._emergency_mode
        
        # Test response registration
        responses_called = {'moderate': 0, 'high': 0, 'critical': 0}
        
        def moderate_response(stats):
            responses_called['moderate'] += 1
        
        def high_response(stats):
            responses_called['high'] += 1
        
        def critical_response(stats):
            responses_called['critical'] += 1
        
        manager.register_response(MemoryPressureLevel.MODERATE, moderate_response)
        manager.register_response(MemoryPressureLevel.HIGH, high_response)
        manager.register_response(MemoryPressureLevel.CRITICAL, critical_response)
        
        # Test pressure responses
        low_stats = MemoryStats(8192, 6000, 2192, 26.8, 0, 100, MemoryPressureLevel.LOW, time.time())
        moderate_stats = MemoryStats(8192, 3000, 5192, 75.0, 500, 100, MemoryPressureLevel.MODERATE, time.time())
        high_stats = MemoryStats(8192, 1500, 6692, 85.0, 1000, 100, MemoryPressureLevel.HIGH, time.time())
        critical_stats = MemoryStats(8192, 500, 7692, 95.0, 1500, 100, MemoryPressureLevel.CRITICAL, time.time())
        
        # Trigger responses
        manager._moderate_pressure_response(moderate_stats)
        manager._high_pressure_response(high_stats)
        manager._critical_pressure_response(critical_stats)
        
        # Test garbage collection
        with mock.patch('gc.collect', return_value=100):
            with mock.patch('gc.get_count', return_value=(700, 10, 10)):
                freed = manager._force_garbage_collection()
                assert freed > 0
        
        # Test cache clearing
        manager._clear_caches()
        
        # Test process termination with mock
        with mock.patch('os.kill') as mock_kill:
            # Success case
            manager._terminate_process(1234)
            mock_kill.assert_called_with(1234, signal.SIGTERM)
            
            # Error case
            mock_kill.side_effect = ProcessLookupError("No such process")
            manager._terminate_process(9999)  # Should handle error gracefully
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_adaptive_processing_config_all_features(self, mock_proc, mock_swap, mock_vm):
        """Test all AdaptiveProcessingConfig features"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, AdaptiveProcessingConfig, ProcessingMode,
            MemoryPressureLevel, MemoryStats
        )
        
        monitor = MemoryMonitor()
        config = AdaptiveProcessingConfig(monitor)
        
        # Test initialization
        assert config.monitor == monitor
        assert config._base_config is not None
        
        # Test current config
        current = config.get_current_config()
        assert isinstance(current, dict)
        assert 'chunk_size_mb' in current
        assert 'max_concurrent_ffmpeg' in current
        
        # Test mode determination
        low_stats = MemoryStats(8192, 6000, 2192, 26.8, 0, 100, MemoryPressureLevel.LOW, time.time())
        moderate_stats = MemoryStats(8192, 3000, 5192, 75.0, 500, 100, MemoryPressureLevel.MODERATE, time.time())
        high_stats = MemoryStats(8192, 1500, 6692, 85.0, 1000, 100, MemoryPressureLevel.HIGH, time.time())
        critical_stats = MemoryStats(8192, 500, 7692, 95.0, 1500, 100, MemoryPressureLevel.CRITICAL, time.time())
        
        assert config._determine_processing_mode(low_stats) == ProcessingMode.FULL_QUALITY
        assert config._determine_processing_mode(moderate_stats) == ProcessingMode.BALANCED
        assert config._determine_processing_mode(high_stats) == ProcessingMode.MEMORY_OPTIMIZED
        assert config._determine_processing_mode(critical_stats) == ProcessingMode.SURVIVAL
        
        # Test config for each mode with stats
        for mode in ProcessingMode:
            mode_config = config._get_config_for_mode(mode, low_stats)
            assert isinstance(mode_config, dict)
            
            # Check expected keys based on mode
            if mode in [ProcessingMode.FULL_QUALITY, ProcessingMode.BALANCED]:
                assert mode_config['enable_hardware_accel']
            else:
                assert not mode_config['enable_hardware_accel']
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_guard_context_manager(self, mock_proc, mock_swap, mock_vm):
        """Test memory_guard context manager comprehensively"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import memory_guard, MemoryMonitor
        
        # Test with custom monitor
        custom_monitor = MemoryMonitor()
        with memory_guard(max_memory_mb=1000, monitor=custom_monitor) as guard:
            assert guard is not None
            # Context should work
        
        # Test with default monitor
        with memory_guard(max_memory_mb=2000) as guard:
            assert guard is not None
        
        # Test with low memory
        mock_vm.return_value.available = 100*1024*1024  # Only 100MB
        mock_vm.return_value.percent = 95.0
        
        with memory_guard(max_memory_mb=500) as guard:
            assert guard is not None
        
        # Test with cleanup
        with memory_guard(max_memory_mb=1000, cleanup_on_exit=True) as guard:
            assert guard is not None
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_optimized_decorator(self, mock_proc, mock_swap, mock_vm):
        """Test memory_optimized decorator"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import memory_optimized
        
        # Test with max memory
        @memory_optimized(max_memory_mb=500)
        def test_func(x, y):
            return x + y
        
        result = test_func(10, 20)
        assert result == 30
        
        # Test without max memory
        @memory_optimized()
        def test_func2():
            return "success"
        
        result2 = test_func2()
        assert result2 == "success"
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_streaming_video_processor(self, mock_proc, mock_swap, mock_vm):
        """Test StreamingVideoProcessor"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, AdaptiveProcessingConfig, StreamingVideoProcessor
        )
        
        monitor = MemoryMonitor()
        config = AdaptiveProcessingConfig(monitor)
        processor = StreamingVideoProcessor(monitor, config)
        
        assert processor.monitor == monitor
        assert processor.config == config
        assert processor.chunk_size_mb > 0
        assert processor.temp_dir is not None
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')  
    @mock.patch('psutil.Process')
    def test_global_functions_and_cleanup(self, mock_proc, mock_swap, mock_vm):
        """Test global functions and cleanup"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            initialize_memory_management,
            get_memory_monitor,
            get_pressure_manager,
            get_adaptive_config,
            shutdown_memory_management,
            _cleanup_handler
        )
        
        # Initialize
        initialize_memory_management()
        
        # Get singletons
        monitor = get_memory_monitor()
        manager = get_pressure_manager()
        config = get_adaptive_config()
        
        assert monitor is not None
        assert manager is not None
        assert config is not None
        
        # Verify they're singletons
        assert get_memory_monitor() is monitor
        assert get_pressure_manager() is manager
        assert get_adaptive_config() is config
        
        # Test cleanup handler
        with mock.patch('montage.utils.memory_manager.shutdown_memory_management') as mock_shutdown:
            _cleanup_handler(signal.SIGTERM, None)
            mock_shutdown.assert_called_once()
        
        # Shutdown
        shutdown_memory_management()
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_edge_cases_and_errors(self, mock_proc, mock_swap, mock_vm):
        """Test edge cases and error handling"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=4*1024*1024*1024,
            percent=50.0,
            used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, MemoryStats, ProcessMemoryInfo
        )
        
        # Test MemoryStats dataclass
        stats = MemoryStats(
            total_mb=8192,
            available_mb=4096,
            used_mb=4096,
            percent_used=50.0,
            swap_used_mb=1024,
            process_memory_mb=100,
            pressure_level=MemoryPressureLevel.LOW
        )
        assert stats.total_mb == 8192
        assert stats.timestamp > 0
        
        # Test ProcessMemoryInfo dataclass
        proc_info = ProcessMemoryInfo(
            pid=1234,
            name="ffmpeg",
            memory_mb=500.0,
            cpu_percent=50.0,
            command="ffmpeg -i input.mp4",
            is_ffmpeg=True
        )
        assert proc_info.pid == 1234
        assert proc_info.is_ffmpeg
        
        # Test monitoring error handling
        monitor = MemoryMonitor()
        
        # Mock error in get_current_stats
        with mock.patch('psutil.virtual_memory', side_effect=Exception("Memory error")):
            # Should handle error in monitoring thread
            monitor.start_monitoring()
            time.sleep(0.1)
            monitor.stop_monitoring()