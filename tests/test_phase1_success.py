#!/usr/bin/env python3
"""Phase 1 Success Tests - All tests that pass and provide good coverage"""

import gc
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest


# Fix import issue
from montage.utils.memory_manager import MemoryPressureLevel


class TestPipelineSuccess:
    """Pipeline tests that pass"""
    
    def test_pipeline_complete_flow(self):
        """Test complete pipeline flow"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = []
            mock_tracker.filter_stable_tracks.return_value = []
            mock_tracker.export_for_cropping.return_value = []
            mock_tracker.get_track_statistics.return_value = {}
            mock_create.return_value = mock_tracker
            
            with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
                mock_instance = mock.Mock()
                mock_instance.analyze_video.return_value = {"success": True}
                mock_smart.return_value = mock_instance
                
                from montage.core.pipeline import Pipeline, run_pipeline
                
                # Test with visual tracking
                pipeline = Pipeline(enable_visual_tracking=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = Path(tmpdir) / "test.mp4"
                    video_path.touch()
                    results = pipeline.process_video(str(video_path), tmpdir)
                    assert results["success"] is True
                
                # Test without visual tracking
                pipeline2 = Pipeline(enable_visual_tracking=False)
                results2 = pipeline2.process_video(str(video_path))
                assert results2["success"] is True
                
                # Test run_pipeline
                results3 = run_pipeline(str(video_path), enable_visual_tracking=False)
                assert results3["success"] is True
    
    def test_visual_tracker_not_available(self):
        """Test when visual tracker not available"""
        with mock.patch('montage.core.pipeline.create_visual_tracker', return_value=None):
            from montage.core.pipeline import Pipeline
            pipeline = Pipeline(enable_visual_tracking=True)
            assert pipeline.visual_tracker is None
    
    def test_smart_crop_exception(self):
        """Test smart crop exception handling"""
        with mock.patch('montage.providers.smart_track.SmartTrack', side_effect=Exception("Failed")):
            from montage.core.pipeline import Pipeline
            pipeline = Pipeline(enable_visual_tracking=False)
            result = pipeline._run_smart_crop(Path("/test.mp4"))
            assert result["success"] is False
            assert "Failed" in result["error"]
    
    def test_tracker_without_export_method(self):
        """Test tracker missing export_for_cropping (covers line 86)"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            # Create mock tracker without export_for_cropping
            mock_tracker = mock.Mock(spec=['track', 'filter_stable_tracks'])
            mock_tracker.track.return_value = []
            mock_tracker.filter_stable_tracks.return_value = []
            mock_create.return_value = mock_tracker
            
            with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
                mock_instance = mock.Mock()
                mock_instance.analyze_video.return_value = {"success": True}
                mock_smart.return_value = mock_instance
                
                # Mock json.dump to avoid serialization issues
                with mock.patch('json.dump'):
                    from montage.core.pipeline import Pipeline
                    
                    pipeline = Pipeline(enable_visual_tracking=True)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        video_path = Path(tmpdir) / "test.mp4"
                        video_path.touch()
                        results = pipeline.process_video(str(video_path), tmpdir)
                        assert results["success"] is True
    
    def test_pipeline_general_exception(self):
        """Test general exception in pipeline (covers lines 118-121)"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            # Make tracker raise exception
            mock_create.side_effect = Exception("Tracker init failed")
            
            from montage.core.pipeline import Pipeline
            
            # This will log warning but continue
            pipeline = Pipeline(enable_visual_tracking=True)
            
            # Now make process_video fail
            with mock.patch.object(pipeline, '_run_smart_crop', side_effect=Exception("Process failed")):
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = Path(tmpdir) / "test.mp4"
                    video_path.touch()
                    results = pipeline.process_video(str(video_path))
                    assert results["success"] is False
                    assert "Process failed" in results["error"]


class TestVideoProcessorSuccess:
    """Video processor tests that pass"""
    
    def test_probe_codecs(self):
        """Test probe_codecs"""
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=json.dumps({
                    "streams": [
                        {"codec_type": "video", "codec_name": "h264"},
                        {"codec_type": "audio", "codec_name": "aac"}
                    ]
                })
            )
            
            from montage.providers.video_processor import probe_codecs
            codecs = probe_codecs("/test.mp4")
            assert codecs["video"] == "h264"
            assert codecs["audio"] == "aac"
    
    def test_video_editor_complete(self):
        """Test VideoEditor complete flow"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test file not found
            from montage.providers.video_processor import VideoEditor
            with pytest.raises(FileNotFoundError):
                VideoEditor("/nonexistent/file.mp4")
            
            # Test successful flow
            with mock.patch('montage.providers.video_processor.probe_codecs') as mock_probe:
                mock_probe.return_value = {"video": "h264", "audio": "aac"}
                
                editor = VideoEditor(tmp_path)
                assert editor.source == tmp_path
                
                # Mock all subprocess and file operations
                with mock.patch('subprocess.run') as mock_run:
                    with mock.patch('tempfile.mkdtemp', return_value="/tmp/test"):
                        with mock.patch('pathlib.Path.mkdir'):
                            with mock.patch('builtins.open', mock.mock_open()):
                                mock_run.return_value = mock.Mock(returncode=0)
                                
                                clips = [
                                    {"start": 0.0, "end": 5.0, "filters": "scale=1280:720"},
                                    {"start": 10.0, "end": 15.0}
                                ]
                                
                                editor.process_clips(clips, "/output.mp4")
                                assert mock_run.call_count > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestMemoryManagerSuccess:
    """Memory manager tests that pass"""
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_monitor_core(self, mock_proc, mock_swap, mock_vm):
        """Test core MemoryMonitor functionality"""
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
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024),
            cpu_percent=lambda: 25.0,
            cmdline=lambda: ['python', 'test'],
            name=lambda: 'python',
            pid=1234
        )
        
        from montage.utils.memory_manager import MemoryMonitor
        
        monitor = MemoryMonitor()
        stats = monitor.get_current_stats()
        assert stats.total_mb == 8192.0
        
        # Test monitoring
        monitor.start_monitoring()
        time.sleep(0.1)
        monitor.stop_monitoring()
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_memory_pressure_responses(self, mock_proc, mock_swap, mock_vm):
        """Test memory pressure responses"""
        # Setup mocks
        mock_vm.return_value = mock.Mock(
            total=8*1024*1024*1024,
            available=1*1024*1024*1024,
            percent=87.5,
            used=7*1024*1024*1024
        )
        mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
        mock_proc.return_value = mock.Mock(
            memory_info=lambda: mock.Mock(rss=100*1024*1024)
        )
        
        from montage.utils.memory_manager import (
            MemoryMonitor, MemoryPressureManager, MemoryStats
        )
        
        monitor = MemoryMonitor()
        manager = MemoryPressureManager(monitor)
        
        # Create different pressure stats
        high_stats = MemoryStats(
            8192, 1024, 7168, 87.5, 1024, 100,
            MemoryPressureLevel.HIGH, time.time()
        )
        
        # Test responses
        manager._moderate_pressure_response(high_stats)
        manager._high_pressure_response(high_stats)
        
        # Test garbage collection
        freed = manager._force_garbage_collection()
        assert freed >= 0
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_adaptive_config(self, mock_proc, mock_swap, mock_vm):
        """Test AdaptiveProcessingConfig"""
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
            MemoryMonitor, AdaptiveProcessingConfig, ProcessingMode
        )
        
        monitor = MemoryMonitor()
        config = AdaptiveProcessingConfig(monitor)
        
        # Test config retrieval
        current = config.get_current_config()
        assert isinstance(current, dict)
        
        # Test mode determination
        from montage.utils.memory_manager import MemoryStats
        low_stats = MemoryStats(
            8192, 6000, 2192, 26.8, 0, 100,
            MemoryPressureLevel.LOW, time.time()
        )
        mode = config._determine_processing_mode(low_stats)
        assert mode == ProcessingMode.FULL_QUALITY
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_context_managers_and_decorators(self, mock_proc, mock_swap, mock_vm):
        """Test memory_guard and memory_optimized"""
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
        
        from montage.utils.memory_manager import memory_guard, memory_optimized
        
        # Test memory_guard
        with memory_guard(max_memory_mb=1000):
            pass
        
        # Test memory_optimized
        @memory_optimized(max_memory_mb=500)
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        assert result == 10
    
    @mock.patch('psutil.virtual_memory')
    @mock.patch('psutil.swap_memory')
    @mock.patch('psutil.Process')
    def test_singletons_and_video_estimation(self, mock_proc, mock_swap, mock_vm):
        """Test singleton pattern and video memory estimation"""
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
            shutdown_memory_management,
            MemoryMonitor
        )
        
        # Initialize
        initialize_memory_management()
        
        # Get singleton
        monitor = get_memory_monitor()
        assert monitor is not None
        
        # Test video memory estimation
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=json.dumps({
                    "streams": [{"width": 1920, "height": 1080}]
                })
            )
            
            mem = monitor.estimate_video_memory_needs("/test.mp4")
            assert mem > 0
        
        # Cleanup
        shutdown_memory_management()