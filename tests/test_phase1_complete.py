#!/usr/bin/env python3
"""Phase 1 Complete Coverage Tests - Achieve ≥90% on all critical modules"""

import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest


class TestVideoProcessorComplete:
    """Complete coverage for video_processor.py"""
    
    def test_video_editor_all_paths(self):
        """Test all code paths in VideoEditor"""
        # Create a real temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test file not found
            with pytest.raises(FileNotFoundError):
                from montage.providers.video_processor import VideoEditor
                VideoEditor("/nonexistent/file.mp4")
            
            # Test successful creation and processing
            with mock.patch('montage.providers.video_processor.probe_codecs') as mock_probe:
                mock_probe.return_value = {"video": "h264", "audio": "aac"}
                
                editor = VideoEditor(tmp_path)
                assert editor.source == tmp_path
                assert editor.codecs["video"] == "h264"
                
                # Mock subprocess for clip processing
                with mock.patch('subprocess.run') as mock_run:
                    with mock.patch('tempfile.mkdtemp') as mock_mkdtemp:
                        # Create a real temp dir for testing
                        real_tmp_dir = tempfile.mkdtemp()
                        mock_mkdtemp.return_value = real_tmp_dir
                        
                        try:
                            mock_run.return_value = mock.Mock(returncode=0)
                            
                            # Test with multiple clips including filters
                            clips = [
                                {"start": 0.0, "end": 5.0, "filters": "scale=1280:720"},
                                {"start": 10.0, "end": 15.0},
                                {"start": 20.0, "end": 25.0, "filters": "crop=640:480:0:0"}
                            ]
                            
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as out_tmp:
                                output_path = out_tmp.name
                            
                            try:
                                editor.process_clips(clips, output_path)
                                
                                # Verify ffmpeg commands were constructed correctly
                                calls = mock_run.call_args_list
                                
                                # Should have 3 clip extractions + 1 concat
                                assert len(calls) == 4
                                
                                # Check first clip with filter
                                first_call = calls[0][0][0]
                                assert 'ffmpeg' in first_call
                                assert '-vf' in first_call
                                assert 'scale=1280:720' in first_call
                                
                                # Check codec preservation
                                assert '-c:v' in first_call
                                assert 'h264' in first_call
                                assert '-c:a' in first_call
                                assert 'aac' in first_call
                                
                                # Check concat command
                                concat_call = calls[-1][0][0]
                                assert '-f' in concat_call
                                assert 'concat' in concat_call
                                assert output_path in concat_call
                                
                            finally:
                                if os.path.exists(output_path):
                                    os.unlink(output_path)
                                    
                        finally:
                            # Cleanup temp dir
                            import shutil
                            if os.path.exists(real_tmp_dir):
                                shutil.rmtree(real_tmp_dir)
                        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_probe_codecs_error(self):
        """Test probe_codecs error handling"""
        with mock.patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'ffprobe')
            
            from montage.providers.video_processor import probe_codecs
            with pytest.raises(subprocess.CalledProcessError):
                probe_codecs("/test.mp4")


class TestMemoryManagerComplete:
    """Complete coverage for memory_manager.py"""
    
    def test_all_memory_manager_features(self):
        """Test all features of memory manager for coverage"""
        # Mock all psutil calls
        with mock.patch('psutil.virtual_memory') as mock_vm:
            with mock.patch('psutil.swap_memory') as mock_swap:
                with mock.patch('psutil.Process') as mock_proc_class:
                    with mock.patch('psutil.process_iter') as mock_proc_iter:
                        # Setup mocks
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
                        
                        # Mock process
                        mock_proc = mock.Mock()
                        mock_proc.memory_info.return_value = mock.Mock(rss=100*1024*1024)
                        mock_proc.cpu_percent.return_value = 25.0
                        mock_proc.cmdline.return_value = ['ffmpeg', '-i', 'test.mp4']
                        mock_proc.name.return_value = 'ffmpeg'
                        mock_proc.pid = 1234
                        mock_proc.info = {
                            'pid': 1234,
                            'name': 'ffmpeg',
                            'memory_info': mock_proc.memory_info.return_value
                        }
                        mock_proc_class.return_value = mock_proc
                        mock_proc_iter.return_value = [mock_proc]
                        
                        from montage.utils.memory_manager import (
                            MemoryMonitor, MemoryPressureLevel, MemoryStats,
                            MemoryPressureManager, AdaptiveProcessingConfig,
                            ProcessingMode, memory_guard, memory_optimized,
                            StreamingVideoProcessor, get_memory_monitor,
                            get_pressure_manager, get_adaptive_config,
                            initialize_memory_management, shutdown_memory_management
                        )
                        
                        # Test singleton initialization
                        initialize_memory_management()
                        
                        # Test MemoryMonitor
                        monitor = get_memory_monitor()
                        assert monitor is not None
                        
                        # Get stats multiple times to test history
                        stats1 = monitor.get_current_stats()
                        time.sleep(0.1)
                        stats2 = monitor.get_current_stats()
                        assert stats1.percent_used == 50.0
                        
                        # Test pressure level calculation
                        assert monitor._calculate_pressure_level(25.0) == MemoryPressureLevel.LOW
                        assert monitor._calculate_pressure_level(75.0) == MemoryPressureLevel.MODERATE
                        assert monitor._calculate_pressure_level(85.0) == MemoryPressureLevel.HIGH
                        assert monitor._calculate_pressure_level(95.0) == MemoryPressureLevel.CRITICAL
                        
                        # Test monitoring thread
                        monitor.start_monitoring()
                        time.sleep(0.2)  # Let it run a bit
                        monitor.stop_monitoring()
                        
                        # Test callbacks
                        callback_called = False
                        def test_callback(stats):
                            nonlocal callback_called
                            callback_called = True
                        
                        monitor.register_pressure_callback(MemoryPressureLevel.HIGH, test_callback)
                        monitor._trigger_pressure_callbacks(MemoryPressureLevel.HIGH, stats1)
                        assert callback_called
                        
                        # Test process tracking
                        monitor.track_process(1234, "ffmpeg", is_ffmpeg=True)
                        tracked = monitor.get_tracked_processes()
                        assert len(tracked) > 0
                        
                        ffmpeg_procs = monitor.get_ffmpeg_processes()
                        assert len(ffmpeg_procs) > 0
                        
                        monitor.untrack_process(1234)
                        
                        # Test memory estimation
                        with mock.patch('subprocess.run') as mock_run:
                            mock_run.return_value = mock.Mock(
                                returncode=0,
                                stdout=json.dumps({
                                    "streams": [{
                                        "width": 1920,
                                        "height": 1080,
                                        "r_frame_rate": "30/1",
                                        "duration": "60.0"
                                    }]
                                })
                            )
                            mem = monitor.estimate_video_memory_needs("/test.mp4")
                            assert mem > 0
                        
                        # Test MemoryPressureManager
                        manager = get_pressure_manager()
                        
                        # Test all pressure responses
                        low_stats = MemoryStats(8192, 6000, 2192, 26.8, 0, 100, MemoryPressureLevel.LOW, time.time())
                        moderate_stats = MemoryStats(8192, 3000, 5192, 75.0, 500, 100, MemoryPressureLevel.MODERATE, time.time())
                        high_stats = MemoryStats(8192, 1500, 6692, 85.0, 1000, 100, MemoryPressureLevel.HIGH, time.time())
                        critical_stats = MemoryStats(8192, 500, 7692, 95.0, 1500, 100, MemoryPressureLevel.CRITICAL, time.time())
                        
                        manager._moderate_pressure_response(moderate_stats)
                        manager._high_pressure_response(high_stats)
                        manager._critical_pressure_response(critical_stats)
                        
                        # Test garbage collection
                        freed = manager._force_garbage_collection()
                        assert freed >= 0
                        
                        # Test cache clearing
                        manager._clear_caches()
                        
                        # Test process termination
                        with mock.patch('os.kill'):
                            manager._terminate_process(9999)  # Non-existent process
                        
                        # Test AdaptiveProcessingConfig
                        config = get_adaptive_config()
                        
                        # Test mode determination
                        assert config._determine_processing_mode(low_stats) == ProcessingMode.FULL_QUALITY
                        assert config._determine_processing_mode(moderate_stats) == ProcessingMode.BALANCED
                        assert config._determine_processing_mode(high_stats) == ProcessingMode.MEMORY_OPTIMIZED
                        assert config._determine_processing_mode(critical_stats) == ProcessingMode.SURVIVAL
                        
                        # Test config for each mode
                        for mode in ProcessingMode:
                            cfg = config._get_config_for_mode(mode)
                            assert 'resolution' in cfg
                            assert 'fps' in cfg
                            assert 'bitrate' in cfg
                        
                        current_cfg = config.get_current_config()
                        assert 'max_memory_mb' in current_cfg
                        
                        # Test StreamingVideoProcessor
                        processor = StreamingVideoProcessor(monitor, config)
                        assert processor.monitor == monitor
                        assert processor.config == config
                        
                        # Test memory guard context manager
                        with memory_guard(max_memory_mb=1000):
                            pass
                        
                        # Test memory optimized decorator
                        @memory_optimized(max_memory_mb=500)
                        def test_func():
                            return "success"
                        
                        result = test_func()
                        assert result == "success"
                        
                        # Cleanup
                        shutdown_memory_management()