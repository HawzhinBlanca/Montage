#!/usr/bin/env python3
"""Phase 1 Coverage Tests - Simplified approach to achieve ≥90% coverage"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestPipelineCoverage:
    """Tests to achieve ≥90% coverage on core/pipeline.py"""
    
    def test_pipeline_complete_flow(self):
        """Test complete pipeline flow for coverage"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            # Mock visual tracker
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = [{"id": 1, "frames": [1, 2, 3]}]
            mock_tracker.filter_stable_tracks.return_value = [{"id": 1, "frames": [1, 2, 3]}]
            mock_tracker.export_for_cropping.return_value = [{"frame": 1, "crop": "data"}]
            mock_tracker.get_track_statistics.return_value = {"total": 1}
            mock_create.return_value = mock_tracker
            
            with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
                mock_smart_instance = mock.Mock()
                mock_smart_instance.analyze_video.return_value = {"success": True}
                mock_smart.return_value = mock_smart_instance
                
                # Import here to avoid issues
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
                
                # Test run_pipeline function
                results3 = run_pipeline(str(video_path), enable_visual_tracking=False)
                assert results3["success"] is True


class TestVideoProcessorCoverage:
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
    
    def test_video_editor_complete(self):
        """Test VideoEditor class"""
        with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_source:
            with mock.patch('subprocess.run') as mock_run:
                # First call is probe, rest are processing
                mock_run.side_effect = [
                    # probe_codecs call
                    mock.Mock(returncode=0, stdout=json.dumps({
                        "streams": [{"codec_type": "video", "codec_name": "h264"}]
                    })),
                    # clip extraction calls
                    mock.Mock(returncode=0),
                    mock.Mock(returncode=0),
                    # concat call
                    mock.Mock(returncode=0)
                ]
                
                from montage.providers.video_processor import VideoEditor
                editor = VideoEditor(tmp_source.name)
                
                # Process clips with filters
                clips = [
                    {"start": 0.0, "end": 5.0, "filters": "scale=1280:720"},
                    {"start": 10.0, "end": 15.0}
                ]
                
                with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp_out:
                    editor.process_clips(clips, tmp_out.name)
                    
                    # Verify all subprocess calls were made
                    assert mock_run.call_count == 4


class TestMemoryManagerCoverage:
    """Tests to achieve ≥90% coverage on utils/memory_manager.py"""
    
    def test_memory_monitor_complete(self):
        """Test MemoryMonitor functionality"""
        with mock.patch('psutil.virtual_memory') as mock_vm:
            with mock.patch('psutil.swap_memory') as mock_swap:
                with mock.patch('psutil.Process') as mock_proc:
                    # Mock system memory
                    mock_vm.return_value = mock.Mock(
                        total=8*1024*1024*1024,
                        available=4*1024*1024*1024,
                        percent=50.0,
                        used=4*1024*1024*1024
                    )
                    mock_swap.return_value = mock.Mock(used=1024*1024*1024)
                    
                    # Mock process
                    mock_proc_inst = mock.Mock()
                    mock_proc_inst.memory_info.return_value = mock.Mock(rss=100*1024*1024)
                    mock_proc_inst.cpu_percent.return_value = 25.0
                    mock_proc_inst.cmdline.return_value = ['python']
                    mock_proc_inst.name.return_value = 'python'
                    mock_proc_inst.pid = 1234
                    mock_proc.return_value = mock_proc_inst
                    
                    from montage.utils.memory_manager import (
                        MemoryMonitor, MemoryPressureManager, 
                        AdaptiveProcessingConfig, memory_guard
                    )
                    
                    # Test MemoryMonitor
                    monitor = MemoryMonitor()
                    stats = monitor.get_current_stats()
                    assert stats is not None
                    
                    # Test monitoring
                    monitor.start_monitoring()
                    monitor.stop_monitoring()
                    
                    # Test process tracking
                    monitor.track_process(1234, "test", is_ffmpeg=True)
                    processes = monitor.get_tracked_processes()
                    assert len(processes) > 0
                    
                    ffmpeg_procs = monitor.get_ffmpeg_processes()
                    monitor.untrack_process(1234)
                    
                    # Test video memory estimation
                    with mock.patch('subprocess.run') as mock_run:
                        mock_run.return_value = mock.Mock(
                            returncode=0,
                            stdout=json.dumps({
                                "streams": [{"width": 1920, "height": 1080}]
                            })
                        )
                        mem_needed = monitor.estimate_video_memory_needs("/test.mp4")
                        assert mem_needed > 0
                    
                    # Test pressure manager
                    manager = MemoryPressureManager(monitor)
                    manager._force_garbage_collection()
                    
                    # Test adaptive config
                    config = AdaptiveProcessingConfig(monitor)
                    current_cfg = config.get_current_config()
                    assert isinstance(current_cfg, dict)
                    
                    # Test memory guard
                    with memory_guard(max_memory_mb=1000):
                        pass