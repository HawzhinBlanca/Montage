#!/usr/bin/env python3
"""Final memory_manager tests to achieve 75% coverage"""

import gc  
import json
import os
import signal
import subprocess
import tempfile
import time
from unittest import mock

import pytest


class TestMemoryManagerFinal:
    """Final tests for memory_manager.py coverage"""
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_monitor_complete(self, mock_proc, mock_swap, mock_vm):
        """Comprehensive MemoryMonitor test"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(
            total=2*1024*1024*1024, used=1*1024*1024*1024, percent=50.0
        )
        mock_proc_inst = mock.Mock()
        mock_proc_inst.memory_info.return_value = mock.Mock(rss=100*1024*1024)
        mock_proc_inst.cpu_percent.return_value = 25.0
        mock_proc_inst.cmdline.return_value = ['python', 'test.py']
        mock_proc_inst.name.return_value = 'python'
        mock_proc_inst.pid = 1234
        mock_proc.return_value = mock_proc_inst
        
        from montage.utils.memory_manager import (
            MemoryMonitor, MemoryPressureLevel, MemoryStats
        )
        
        monitor = MemoryMonitor(update_interval=0.1)
        
        # Test basic functionality
        stats = monitor.get_current_stats()
        assert isinstance(stats, MemoryStats)
        assert stats.total_mb == 8192.0
        
        # Test pressure levels
        assert monitor._calculate_pressure_level(30.0) == MemoryPressureLevel.LOW
        assert monitor._calculate_pressure_level(75.0) == MemoryPressureLevel.MODERATE
        assert monitor._calculate_pressure_level(85.0) == MemoryPressureLevel.HIGH
        assert monitor._calculate_pressure_level(95.0) == MemoryPressureLevel.CRITICAL
        
        # Test monitoring
        monitor.start_monitoring()
        assert monitor._monitoring is True
        time.sleep(0.1)
        monitor.stop_monitoring()
        assert monitor._monitoring is False
        
        # Test process tracking
        monitor.track_process(1234, "test_process", is_ffmpeg=True)
        tracked = monitor.get_tracked_processes()
        assert len(tracked) > 0
        
        ffmpeg_procs = monitor.get_ffmpeg_processes()
        assert len(ffmpeg_procs) > 0
        
        monitor.untrack_process(1234)
        assert len(monitor.get_tracked_processes()) == 0
        
        # Test callbacks
        callback_called = {'count': 0}
        def test_callback(stats):
            callback_called['count'] += 1
        
        monitor.register_pressure_callback(MemoryPressureLevel.LOW, test_callback)
        monitor._trigger_pressure_callbacks(MemoryPressureLevel.LOW, stats)
        assert callback_called['count'] > 0
        
        # Test video memory estimation
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=json.dumps({
                    "streams": [{
                        "width": 1920, "height": 1080,
                        "r_frame_rate": "30/1", "duration": "60.0"
                    }]
                })
            )
            mem = monitor.estimate_video_memory_needs("/test/video.mp4")
            assert mem > 0
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_pressure_manager(self, mock_proc, mock_swap, mock_vm):
        """Test MemoryPressureManager"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, MemoryPressureManager, MemoryPressureLevel,
            MemoryStats
        )
        
        monitor = MemoryMonitor()
        manager = MemoryPressureManager(monitor)
        
        # Test response registration
        def test_response(stats):
            pass
        
        manager.register_response(MemoryPressureLevel.LOW, test_response)
        
        # Test pressure responses
        low_stats = MemoryStats(8192, 6000, 2192, 26.8, 0, 100, MemoryPressureLevel.LOW, time.time())
        moderate_stats = MemoryStats(8192, 3000, 5192, 75.0, 500, 100, MemoryPressureLevel.MODERATE, time.time())
        high_stats = MemoryStats(8192, 1500, 6692, 85.0, 1000, 100, MemoryPressureLevel.HIGH, time.time())
        critical_stats = MemoryStats(8192, 500, 7692, 95.0, 1500, 100, MemoryPressureLevel.CRITICAL, time.time())
        
        # Test each response
        manager._moderate_pressure_response(moderate_stats)
        manager._high_pressure_response(high_stats) 
        manager._critical_pressure_response(critical_stats)
        
        # Test garbage collection
        freed = manager._force_garbage_collection()
        assert freed >= 0
        
        # Test cache clearing
        manager._clear_caches()
        
        # Test process termination with error handling
        with mock.patch('os.kill', side_effect=ProcessLookupError("No such process")):
            manager._terminate_process(9999)  # Should handle error gracefully
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_adaptive_processing_config(self, mock_proc, mock_swap, mock_vm):
        """Test AdaptiveProcessingConfig"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
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
        
        # Test current config
        current = config.get_current_config()
        assert isinstance(current, dict)
        assert 'chunk_size_mb' in current
        
        # Test mode determination
        low_stats = MemoryStats(8192, 6000, 2192, 26.8, 0, 100, MemoryPressureLevel.LOW, time.time())
        moderate_stats = MemoryStats(8192, 3000, 5192, 75.0, 500, 100, MemoryPressureLevel.MODERATE, time.time())
        high_stats = MemoryStats(8192, 1500, 6692, 85.0, 1000, 100, MemoryPressureLevel.HIGH, time.time())
        critical_stats = MemoryStats(8192, 500, 7692, 95.0, 1500, 100, MemoryPressureLevel.CRITICAL, time.time())
        
        assert config._determine_processing_mode(low_stats) == ProcessingMode.FULL_QUALITY
        assert config._determine_processing_mode(moderate_stats) == ProcessingMode.BALANCED
        assert config._determine_processing_mode(high_stats) == ProcessingMode.MEMORY_OPTIMIZED
        assert config._determine_processing_mode(critical_stats) == ProcessingMode.SURVIVAL
        
        # Test config for each mode
        for mode in ProcessingMode:
            mode_config = config._get_config_for_mode(mode, low_stats)
            assert isinstance(mode_config, dict)
            assert 'chunk_size_mb' in mode_config
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_context_managers_and_decorators(self, mock_proc, mock_swap, mock_vm):
        """Test memory_guard and memory_optimized"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import memory_guard, memory_optimized
        
        # Test memory_guard
        with memory_guard(max_memory_mb=1000) as guard:
            assert guard is not None
        
        # Test memory_optimized decorator
        @memory_optimized(max_memory_mb=500)
        def test_func(x, y):
            return x + y
        
        result = test_func(5, 10)
        assert result == 15
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_streaming_video_processor(self, mock_proc, mock_swap, mock_vm):
        """Test StreamingVideoProcessor"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
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
        
        # Test _get_video_duration method
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=json.dumps({
                    "format": {"duration": "120.5"}
                })
            )
            duration = processor._get_video_duration("/test/video.mp4")
            assert duration == 120.5
            
            # Test error case
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
            duration = processor._get_video_duration("/test/error.mp4")
            assert duration == 0.0
        
        # Test _extract_video_chunk method
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            success = processor._extract_video_chunk("/input.mp4", 0.0, 30.0, "/output.mp4")
            assert success is True
            
            # Test error case
            mock_run.return_value = mock.Mock(returncode=1)
            success = processor._extract_video_chunk("/input.mp4", 0.0, 30.0, "/output.mp4")
            assert success is False
        
        # Test _concatenate_chunks method
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            with mock.patch('builtins.open', mock.mock_open()) as mock_file:
                success = processor._concatenate_chunks(["/chunk1.mp4", "/chunk2.mp4"], "/output.mp4")
                assert success is True
                
                # Test error case
                mock_run.return_value = mock.Mock(returncode=1)
                success = processor._concatenate_chunks(["/chunk1.mp4"], "/output.mp4")
                assert success is False
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_global_functions(self, mock_proc, mock_swap, mock_vm):
        """Test global singleton functions"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
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
        
        # Test cleanup handler
        with mock.patch('montage.utils.memory_manager.shutdown_memory_management') as mock_shutdown:
            _cleanup_handler(signal.SIGTERM, None)
            mock_shutdown.assert_called_once()
        
        # Shutdown
        shutdown_memory_management()
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_stats_and_process_info(self, mock_proc, mock_swap, mock_vm):
        """Test MemoryStats and ProcessMemoryInfo dataclasses"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024  
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryStats, ProcessMemoryInfo, MemoryPressureLevel
        )
        
        # Test MemoryStats
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
        
        # Test ProcessMemoryInfo
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
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_streaming_video_chunked_processing(self, mock_proc, mock_swap, mock_vm):
        """Test chunked video processing to boost coverage"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024, available=4*1024*1024*1024,
            percent=50.0, used=4*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, AdaptiveProcessingConfig, StreamingVideoProcessor,
            MemoryPressureLevel, MemoryStats
        )
        
        monitor = MemoryMonitor()
        config = AdaptiveProcessingConfig(monitor)
        processor = StreamingVideoProcessor(monitor, config)
        
        # Mock the duration to be short to avoid long loops
        with mock.patch.object(processor, '_get_video_duration', return_value=10.0):
            with mock.patch.object(processor, '_extract_video_chunk', return_value=True):
                with mock.patch.object(processor, '_concatenate_chunks', return_value=True):
                    with mock.patch('os.path.getsize', return_value=100*1024*1024):  # 100MB file
                        with mock.patch('os.unlink'):
                            def mock_processing_func(chunk_file):
                                return f"processed_{chunk_file}"
                            
                            # Test successful chunked processing
                            result = processor.process_large_video_chunked(
                                "/test/input.mp4",
                                "/test/output.mp4", 
                                mock_processing_func,
                                chunk_duration=5.0
                            )
                            assert result is True
        
        # Test error cases
        with mock.patch.object(processor, '_get_video_duration', return_value=0.0):
            result = processor.process_large_video_chunked(
                "/test/input.mp4", "/test/output.mp4", lambda x: x
            )
            assert result is False
            
        # Test chunk extraction failure
        with mock.patch.object(processor, '_get_video_duration', return_value=10.0):
            with mock.patch.object(processor, '_extract_video_chunk', return_value=False):
                with mock.patch('os.path.getsize', return_value=100*1024*1024):
                    result = processor.process_large_video_chunked(
                        "/test/input.mp4", "/test/output.mp4", lambda x: x
                    )
                    assert result is False
    
