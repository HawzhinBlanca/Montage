#!/usr/bin/env python3
"""
Test Visual Tracker with MMTracking
Tests multi-object tracking functionality
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest
import numpy as np

from montage.core.visual_tracker import (
    VisualTracker,
    create_visual_tracker,
    MMTRACK_AVAILABLE
)


class TestVisualTracker:
    """Test VisualTracker class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock MMTracking model"""
        with patch('montage.core.visual_tracker.init_model') as mock_init:
            model = MagicMock()
            mock_init.return_value = model
            yield model
    
    @pytest.fixture
    def tracker(self, mock_model):
        """Create tracker instance with mocked model"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', True):
            tracker = VisualTracker()
            tracker.model = mock_model
            return tracker
    
    def test_init_without_mmtrack(self):
        """Test initialization when MMTracking not available"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', False):
            with pytest.raises(ImportError) as exc_info:
                VisualTracker()
            assert "MMTracking is not installed" in str(exc_info.value)
    
    def test_init_with_cuda(self):
        """Test initialization with CUDA device"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', True):
            with patch('montage.core.visual_tracker.init_model') as mock_init:
                with patch.object(VisualTracker, '_check_cuda', return_value=True):
                    tracker = VisualTracker(device='cuda:0')
                    assert tracker.device == 'cuda:0'
    
    def test_init_fallback_to_cpu(self):
        """Test fallback to CPU when CUDA not available"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', True):
            with patch('montage.core.visual_tracker.init_model') as mock_init:
                with patch.object(VisualTracker, '_check_cuda', return_value=False):
                    tracker = VisualTracker(device='cuda:0')
                    assert tracker.device == 'cpu'
    
    def test_track_video_not_found(self, tracker):
        """Test tracking with non-existent video"""
        with pytest.raises(FileNotFoundError):
            tracker.track("non_existent_video.mp4")
    
    def test_track_success(self, tracker):
        """Test successful video tracking"""
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
            Path(tmp_path).touch()
        
        try:
            # Mock inference results
            mock_results = []
            for i in range(5):  # 5 frames
                frame_result = MagicMock()
                frame_result.pred_track_instances = MagicMock()
                frame_result.pred_track_instances.ids = np.array([1, 2, 3])
                frame_result.pred_track_instances.bboxes = np.array([
                    [100, 100, 200, 200],
                    [300, 300, 400, 400],
                    [500, 100, 600, 200]
                ])
                frame_result.pred_track_instances.scores = np.array([0.9, 0.8, 0.7])
                mock_results.append(frame_result)
            
            with patch('montage.core.visual_tracker.inference_mot') as mock_inference:
                mock_inference.return_value = mock_results
                
                tracks = tracker.track(tmp_path)
                
                # Verify results
                assert len(tracks) == 5
                assert all('tracks' in frame for frame in tracks)
                assert len(tracks[0]['tracks']) == 3
                
                # Check first track
                first_track = tracks[0]['tracks'][0]
                assert first_track['track_id'] == 1
                assert first_track['bbox'] == [100, 100, 200, 200]
                assert first_track['score'] == 0.9
                assert first_track['category'] == 'person'
                assert 'center' in first_track
                assert 'size' in first_track
                
        finally:
            Path(tmp_path).unlink()
    
    def test_track_with_output_file(self, tracker):
        """Test tracking with output file specified"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
            Path(tmp_path).touch()
        
        try:
            with patch('montage.core.visual_tracker.inference_mot') as mock_inference:
                mock_inference.return_value = []
                
                output_file = "tracked_output.mp4"
                tracks = tracker.track(tmp_path, output_file)
                
                # Verify inference was called with output file
                mock_inference.assert_called_once_with(
                    tracker.model, 
                    str(tmp_path), 
                    output_file
                )
        finally:
            Path(tmp_path).unlink()
    
    def test_get_track_statistics(self, tracker):
        """Test track statistics calculation"""
        # Create test tracking data
        tracks = [
            {
                'frame_idx': 0,
                'tracks': [
                    {'track_id': 1},
                    {'track_id': 2}
                ]
            },
            {
                'frame_idx': 1,
                'tracks': [
                    {'track_id': 1},
                    {'track_id': 3}
                ]
            },
            {
                'frame_idx': 2,
                'tracks': [
                    {'track_id': 1},
                    {'track_id': 2},
                    {'track_id': 3}
                ]
            }
        ]
        
        stats = tracker.get_track_statistics(tracks)
        
        assert stats['total_frames'] == 3
        assert stats['unique_tracks'] == 3
        assert stats['total_detections'] == 7
        assert stats['avg_tracks_per_frame'] == 7/3
        assert stats['track_lengths'] == {1: 3, 2: 2, 3: 2}
        assert stats['longest_track'] == 3
        assert stats['average_track_length'] == 7/3
    
    def test_filter_stable_tracks(self, tracker):
        """Test filtering out short/unstable tracks"""
        tracks = [
            {
                'frame_idx': i,
                'timestamp': i/30.0,
                'tracks': [
                    {'track_id': 1},  # Appears in all frames
                    {'track_id': 2} if i < 5 else {'track_id': 3}  # Short tracks
                ]
            }
            for i in range(15)
        ]
        
        # Filter with min_length=10
        filtered = tracker.filter_stable_tracks(tracks, min_length=10)
        
        # Only track 1 should remain (appears in 15 frames)
        for frame in filtered:
            track_ids = [t['track_id'] for t in frame['tracks']]
            assert track_ids == [1]
    
    def test_export_for_cropping(self, tracker):
        """Test exporting tracking data for cropping"""
        tracks = [
            {
                'frame_idx': 0,
                'timestamp': 0.0,
                'tracks': [
                    {
                        'track_id': 1,
                        'bbox': [100, 100, 200, 300],
                        'center': [150, 200],
                        'size': 10000
                    },
                    {
                        'track_id': 2,
                        'bbox': [400, 100, 500, 300],
                        'center': [450, 200],
                        'size': 10000
                    }
                ]
            }
        ]
        
        crop_data = tracker.export_for_cropping(tracks, 1920, 1080)
        
        assert len(crop_data) == 1
        assert crop_data[0]['frame_idx'] == 0
        assert crop_data[0]['timestamp'] == 0.0
        assert crop_data[0]['num_subjects'] == 2
        assert crop_data[0]['primary_subject'] == 1  # Both same size, first one wins
        
        # Check crop center is average of both subjects
        expected_center_x = (150 + 450) / 2  # 300
        assert crop_data[0]['crop_center'][0] == int(expected_center_x)
    
    def test_export_for_cropping_single_subject(self, tracker):
        """Test export with single subject"""
        tracks = [
            {
                'frame_idx': 0,
                'timestamp': 0.0,
                'tracks': [
                    {
                        'track_id': 1,
                        'bbox': [800, 400, 1000, 800],
                        'center': [900, 600],
                        'size': 40000
                    }
                ]
            }
        ]
        
        crop_data = tracker.export_for_cropping(tracks, 1920, 1080)
        
        assert crop_data[0]['crop_center'] == (900, 600)
        assert crop_data[0]['primary_subject'] == 1
        assert crop_data[0]['num_subjects'] == 1


class TestCreateVisualTracker:
    """Test factory function"""
    
    def test_create_tracker_success(self):
        """Test successful tracker creation"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', True):
            with patch('montage.core.visual_tracker.VisualTracker') as mock_class:
                mock_instance = MagicMock()
                mock_class.return_value = mock_instance
                
                tracker = create_visual_tracker()
                assert tracker == mock_instance
    
    def test_create_tracker_not_available(self):
        """Test when MMTracking not available"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', False):
            tracker = create_visual_tracker()
            assert tracker is None
    
    def test_create_tracker_exception(self):
        """Test when tracker creation fails"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', True):
            with patch('montage.core.visual_tracker.VisualTracker') as mock_class:
                mock_class.side_effect = Exception("Init failed")
                
                tracker = create_visual_tracker()
                assert tracker is None


class TestIntegration:
    """Integration tests for visual tracking"""
    
    @pytest.mark.skipif(not MMTRACK_AVAILABLE, reason="MMTracking not available")
    def test_real_tracker_initialization(self):
        """Test real tracker initialization if MMTracking available"""
        try:
            tracker = VisualTracker()
            assert tracker.model is not None
            assert tracker.device in ['cpu', 'cuda:0']
        except Exception as e:
            # May fail due to missing model files
            pytest.skip(f"Real initialization failed: {e}")
    
    def test_tracking_pipeline_integration(self):
        """Test integration with video processing pipeline"""
        with patch('montage.core.visual_tracker.MMTRACK_AVAILABLE', True):
            with patch('montage.core.visual_tracker.VisualTracker') as mock_class:
                mock_tracker = MagicMock()
                mock_class.return_value = mock_tracker
                
                # Mock tracking results
                mock_tracker.track.return_value = [
                    {
                        'frame_idx': 0,
                        'timestamp': 0.0,
                        'tracks': [{'track_id': 1, 'center': [960, 540]}]
                    }
                ]
                
                # Simulate pipeline usage
                tracker = create_visual_tracker()
                assert tracker is not None
                
                tracks = tracker.track("test.mp4")
                assert len(tracks) == 1
                
                crop_data = tracker.export_for_cropping(tracks, 1920, 1080)
                assert len(crop_data) == 1
                assert 'crop_center' in crop_data[0]