#!/usr/bin/env python3
"""Phase 1 Final Tests - Achieve required coverage targets"""

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest


# PIPELINE TESTS - Target ≥90% coverage
class TestPipelineFinal:
    """Tests to achieve ≥90% coverage on core/pipeline.py"""
    
    def test_pipeline_complete_flow_with_tracking(self):
        """Test complete pipeline with visual tracking"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = []
            mock_tracker.filter_stable_tracks.return_value = []
            mock_tracker.export_for_cropping.return_value = []
            mock_tracker.get_track_statistics.return_value = {}
            mock_create.return_value = mock_tracker
            
            with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
                mock_smart_instance = mock.Mock()
                mock_smart_instance.analyze_video.return_value = {"success": True}
                mock_smart.return_value = mock_smart_instance
                
                from montage.core.pipeline import Pipeline, run_pipeline
                
                pipeline = Pipeline(enable_visual_tracking=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = Path(tmpdir) / "test.mp4"
                    video_path.touch()
                    results = pipeline.process_video(str(video_path), tmpdir)
                    assert results["success"] is True
    
    def test_pipeline_no_tracking(self):
        """Test pipeline without visual tracking"""
        with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
            mock_smart_instance = mock.Mock()
            mock_smart_instance.analyze_video.return_value = {"success": True}
            mock_smart.return_value = mock_smart_instance
            
            from montage.core.pipeline import Pipeline
            
            pipeline = Pipeline(enable_visual_tracking=False)
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / "test.mp4"
                video_path.touch()
                results = pipeline.process_video(str(video_path))
                assert results["success"] is True
    
    def test_visual_tracker_not_available(self):
        """Test when visual tracker creation fails (line 32)"""
        with mock.patch('montage.core.pipeline.create_visual_tracker', return_value=None):
            from montage.core.pipeline import Pipeline
            pipeline = Pipeline(enable_visual_tracking=True)
            assert pipeline.visual_tracker is None
    
    def test_smart_crop_exception(self):
        """Test smart crop exception handling (lines 146-148)"""
        with mock.patch('montage.providers.smart_track.SmartTrack', side_effect=Exception("Failed")):
            from montage.core.pipeline import Pipeline
            pipeline = Pipeline(enable_visual_tracking=False)
            result = pipeline._run_smart_crop(Path("/test.mp4"))
            assert result["success"] is False
            assert "Failed" in result["error"]
    
    def test_pipeline_exception_handling(self):
        """Test pipeline exception handling (lines 118-121)"""
        with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
            mock_smart.side_effect = Exception("Pipeline error")
            
            from montage.core.pipeline import Pipeline
            pipeline = Pipeline(enable_visual_tracking=False)
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / "test.mp4"
                video_path.touch()
                results = pipeline.process_video(str(video_path))
                assert results["success"] is False
                assert "Pipeline error" in results["error"]
    
    def test_run_pipeline_function(self):
        """Test run_pipeline convenience function"""
        with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
            mock_smart_instance = mock.Mock()
            mock_smart_instance.analyze_video.return_value = {"success": True}
            mock_smart.return_value = mock_smart_instance
            
            from montage.core.pipeline import run_pipeline
            
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / "test.mp4"
                video_path.touch()
                results = run_pipeline(str(video_path), enable_visual_tracking=False)
                assert results["success"] is True


# VIDEO PROCESSOR TESTS - Target ≥90% coverage
class TestVideoProcessorFinal:
    """Tests to achieve ≥90% coverage on providers/video_processor.py"""
    
    def test_probe_codecs(self):
        """Test probe_codecs function"""
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
    
    def test_video_editor_file_not_found(self):
        """Test VideoEditor with non-existent file"""
        from montage.providers.video_processor import VideoEditor
        with pytest.raises(FileNotFoundError):
            VideoEditor("/nonexistent/file.mp4")
    
    def test_video_editor_process_clips(self):
        """Test VideoEditor clip processing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with mock.patch('montage.providers.video_processor.probe_codecs') as mock_probe:
                mock_probe.return_value = {"video": "h264", "audio": "aac"}
                
                from montage.providers.video_processor import VideoEditor
                editor = VideoEditor(tmp_path)
                
                with mock.patch('subprocess.run') as mock_run:
                    with mock.patch('tempfile.mkdtemp', return_value="/tmp/test"):
                        with mock.patch('pathlib.Path.exists', return_value=True):
                            with mock.patch('builtins.open', mock.mock_open()):
                                mock_run.return_value = mock.Mock(returncode=0)
                                
                                clips = [
                                    {"start": 0.0, "end": 5.0, "filters": "scale=1280:720"},
                                    {"start": 10.0, "end": 15.0}
                                ]
                                
                                editor.process_clips(clips, "/output.mp4")
                                
                                # Verify subprocess was called
                                assert mock_run.call_count > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# MEMORY MANAGER TESTS - Target ≥75% coverage
class TestMemoryManagerFinal:
    """Tests to achieve ≥75% coverage on utils/memory_manager.py"""
    
    def test_memory_monitor_basic(self):
        """Test MemoryMonitor basic functionality"""
        with mock.patch('psutil.virtual_memory') as mock_vm:
            with mock.patch('psutil.swap_memory') as mock_swap:
                with mock.patch('psutil.Process') as mock_proc:
                    mock_vm.return_value = mock.Mock(
                        total=8*1024*1024*1024, available=4*1024*1024*1024,
                        percent=50.0, used=4*1024*1024*1024
                    )
                    mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
                    mock_proc.return_value = mock.Mock(
                        memory_info=lambda: mock.Mock(rss=100*1024*1024)
                    )
                    
                    from montage.utils.memory_manager import MemoryMonitor, MemoryPressureLevel
                    
                    monitor = MemoryMonitor()
                    stats = monitor.get_current_stats()
                    assert stats.total_mb == 8192.0
                    assert stats.available_mb == 4096.0
                    
                    # Test pressure levels
                    assert monitor._calculate_pressure_level(30.0) == MemoryPressureLevel.LOW
                    assert monitor._calculate_pressure_level(75.0) == MemoryPressureLevel.MODERATE
                    assert monitor._calculate_pressure_level(85.0) == MemoryPressureLevel.HIGH
                    assert monitor._calculate_pressure_level(95.0) == MemoryPressureLevel.CRITICAL
                    
                    # Test monitoring
                    monitor.start_monitoring()
                    time.sleep(0.1)
                    monitor.stop_monitoring()
                    
                    # Test process tracking
                    monitor.track_process(1234, "test", is_ffmpeg=True)
                    tracked = monitor.get_tracked_processes()
                    assert len(tracked) > 0
                    monitor.untrack_process(1234)
    
    def test_memory_pressure_manager(self):
        """Test MemoryPressureManager"""
        with mock.patch('psutil.virtual_memory') as mock_vm:
            with mock.patch('psutil.swap_memory') as mock_swap:
                with mock.patch('psutil.Process') as mock_proc:
                    mock_vm.return_value = mock.Mock(
                        total=8*1024*1024*1024, available=4*1024*1024*1024,
                        percent=50.0, used=4*1024*1024*1024
                    )
                    mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
                    mock_proc.return_value = mock.Mock(
                        memory_info=lambda: mock.Mock(rss=100*1024*1024)
                    )
                    
                    from montage.utils.memory_manager import (
                        MemoryMonitor, MemoryPressureManager, MemoryPressureLevel
                    )
                    
                    monitor = MemoryMonitor()
                    manager = MemoryPressureManager(monitor)
                    
                    # Test response registration
                    def test_response(stats):
                        pass
                    
                    manager.register_response(MemoryPressureLevel.HIGH, test_response)
                    
                    # Test garbage collection
                    freed = manager._force_garbage_collection()
                    assert freed >= 0
    
    def test_memory_guard_context(self):
        """Test memory_guard context manager"""
        with mock.patch('psutil.virtual_memory') as mock_vm:
            with mock.patch('psutil.swap_memory') as mock_swap:
                with mock.patch('psutil.Process') as mock_proc:
                    mock_vm.return_value = mock.Mock(
                        total=8*1024*1024*1024, available=4*1024*1024*1024,
                        percent=50.0, used=4*1024*1024*1024
                    )
                    mock_swap.return_value = mock.Mock(used=1*1024*1024*1024)
                    mock_proc.return_value = mock.Mock(
                        memory_info=lambda: mock.Mock(rss=100*1024*1024)
                    )
                    
                    from montage.utils.memory_manager import memory_guard
                    
                    with memory_guard(max_memory_mb=1000) as guard:
                        assert guard is not None
    
    def test_singletons(self):
        """Test singleton functions"""
        with mock.patch('psutil.virtual_memory') as mock_vm:
            with mock.patch('psutil.swap_memory') as mock_swap:
                with mock.patch('psutil.Process') as mock_proc:
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
                        shutdown_memory_management
                    )
                    
                    initialize_memory_management()
                    
                    monitor = get_memory_monitor()
                    manager = get_pressure_manager()
                    config = get_adaptive_config()
                    
                    assert monitor is not None
                    assert manager is not None
                    assert config is not None
                    
                    shutdown_memory_management()