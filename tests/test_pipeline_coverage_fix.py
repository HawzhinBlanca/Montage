#!/usr/bin/env python3
"""Fix pipeline.py coverage to reach ≥90%"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestPipelineCoverageFix:
    """Tests to cover missing lines in pipeline.py"""
    
    def test_visual_tracker_not_available(self):
        """Test when create_visual_tracker returns None (line 32)"""
        with mock.patch('montage.core.pipeline.create_visual_tracker', return_value=None):
            from montage.core.pipeline import Pipeline
            
            # This should trigger the warning on line 32
            pipeline = Pipeline(enable_visual_tracking=True)
            assert pipeline.visual_tracker is None
    
    def test_visual_tracker_missing_export_method(self):
        """Test when visual tracker doesn't have export_for_cropping (line 86)"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            # Create tracker without export_for_cropping method
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = [{"id": 1, "frames": [1, 2, 3]}]
            mock_tracker.filter_stable_tracks.return_value = [{"id": 1, "frames": [1, 2, 3]}]
            # Explicitly remove the method
            if hasattr(mock_tracker, 'export_for_cropping'):
                delattr(mock_tracker, 'export_for_cropping')
            mock_create.return_value = mock_tracker
            
            with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
                mock_smart_instance = mock.Mock()
                mock_smart_instance.analyze_video.return_value = {"success": True}
                mock_smart.return_value = mock_smart_instance
                
                from montage.core.pipeline import Pipeline
                
                pipeline = Pipeline(enable_visual_tracking=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = Path(tmpdir) / "test.mp4"
                    video_path.touch()
                    
                    results = pipeline.process_video(str(video_path), tmpdir)
                    
                    # Verify we hit the code path
                    assert results["success"] is True
                    # The test is to ensure line 86 (crop_data = []) is executed
                    # We can't check the file since it may have Mock objects
    
    def test_pipeline_general_exception(self):
        """Test exception handling in process_video (lines 118-121)"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            # Create tracker that will raise an exception
            mock_tracker = mock.Mock()
            mock_tracker.track.side_effect = Exception("Tracking failed!")
            mock_create.return_value = mock_tracker
            
            with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
                mock_smart_instance = mock.Mock()
                mock_smart_instance.analyze_video.return_value = {"success": True}
                mock_smart.return_value = mock_smart_instance
                
                from montage.core.pipeline import Pipeline
                
                pipeline = Pipeline(enable_visual_tracking=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = Path(tmpdir) / "test.mp4"
                    video_path.touch()
                    
                    results = pipeline.process_video(str(video_path), tmpdir)
                    
                    # Should catch exception and set success=False
                    assert results["success"] is False
                    assert "error" in results
                    assert "Tracking failed!" in results["error"]
    
    def test_smart_crop_exception(self):
        """Test smart crop exception handling (lines 146-148)"""
        with mock.patch('montage.providers.smart_track.SmartTrack') as mock_smart:
            # Make SmartTrack constructor fail
            mock_smart.side_effect = Exception("SmartTrack initialization failed!")
            
            from montage.core.pipeline import Pipeline
            
            pipeline = Pipeline(enable_visual_tracking=False)
            video_path = Path("/tmp/test.mp4")
            
            # Call _run_smart_crop directly to test exception handling
            result = pipeline._run_smart_crop(video_path)
            
            assert result["success"] is False
            assert "error" in result
            assert "SmartTrack initialization failed!" in result["error"]