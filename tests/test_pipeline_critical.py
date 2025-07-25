#!/usr/bin/env python3
"""
Comprehensive test suite for critical pipeline components
Tests core Pipeline and VideoProcessor with ≥90% coverage target
Target: ≥90% coverage for montage/core/pipeline.py and montage/providers/video_processor.py
"""

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, Mock, patch, mock_open, call

import pytest

from montage.core.pipeline import Pipeline, run_pipeline
from montage.providers.video_processor import VideoEditor, probe_codecs


class TestPipeline:
    """Comprehensive test suite for core Pipeline class"""

    @pytest.fixture
    def mock_visual_tracker(self):
        """Mock visual tracker with comprehensive functionality"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = [
                {"id": 1, "frames": list(range(60))},
                {"id": 2, "frames": list(range(30, 90))}
            ]
            mock_tracker.filter_stable_tracks.return_value = [
                {"id": 1, "frames": list(range(60))}
            ]
            mock_tracker.export_for_cropping.return_value = [
                {"frame": 0, "x": 100, "y": 100, "w": 300, "h": 300}
            ]
            mock_tracker.get_track_statistics.return_value = {
                "total_tracks": 2,
                "stable_tracks": 1,
                "avg_track_length": 45
            }
            mock_create.return_value = mock_tracker
            yield mock_tracker

    @pytest.fixture
    def mock_smart_track(self):
        """Mock SmartTrack with comprehensive functionality"""
        with mock.patch('montage.providers.smart_track.SmartTrack') as mock_class:
            mock_instance = mock.Mock()
            mock_instance.analyze_video.return_value = {
                "faces_detected": 5,
                "motion_score": 0.7,
                "crop_params": {"x": 0, "y": 0, "w": 1920, "h": 1080}
            }
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def temp_video_path(self):
        """Create temporary video path"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = Path(f.name)
        yield video_path
        if video_path.exists():
            video_path.unlink()

    def test_pipeline_init_with_visual_tracking_enabled(self, mock_visual_tracker):
        """Test Pipeline initialization with visual tracking enabled"""
        pipeline = Pipeline(enable_visual_tracking=True)
        
        assert pipeline.enable_visual_tracking is True
        assert pipeline.visual_tracker is not None
        assert pipeline.visual_tracker == mock_visual_tracker

    def test_pipeline_init_with_visual_tracking_disabled(self):
        """Test Pipeline initialization with visual tracking disabled"""
        pipeline = Pipeline(enable_visual_tracking=False)
        
        assert pipeline.enable_visual_tracking is False
        assert pipeline.visual_tracker is None

    def test_pipeline_init_visual_tracking_unavailable(self):
        """Test Pipeline when visual tracking creation returns None"""
        with mock.patch('montage.core.pipeline.create_visual_tracker', return_value=None):
            pipeline = Pipeline(enable_visual_tracking=True)
            
            assert pipeline.enable_visual_tracking is True
            assert pipeline.visual_tracker is None

    def test_process_video_complete_success_with_output_dir(
        self, mock_visual_tracker, mock_smart_track, temp_video_path
    ):
        """Test complete successful video processing with specified output directory"""
        output_dir = temp_video_path.parent / "test_output"
        
        with mock.patch('builtins.open', mock_open()) as mock_file:
            with mock.patch('json.dump') as mock_json_dump:
                pipeline = Pipeline(enable_visual_tracking=True)
                results = pipeline.process_video(str(temp_video_path), str(output_dir))
        
        # Verify results structure
        assert results["success"] is True
        assert results["video_path"] == str(temp_video_path)
        assert results["output_dir"] == str(output_dir)
        assert len(results["pipeline_steps"]) == 2
        
        # Verify smart crop step
        smart_crop_step = results["pipeline_steps"][0]
        assert smart_crop_step["step"] == "smart_crop"
        assert smart_crop_step["success"] is True
        assert "result" in smart_crop_step
        
        # Verify visual tracking step
        tracking_step = results["pipeline_steps"][1]
        assert tracking_step["step"] == "visual_tracking"
        assert tracking_step["success"] is True
        assert tracking_step["tracks_count"] == 1
        assert "output_file" in tracking_step
        
        # Verify tracker method calls
        mock_visual_tracker.track.assert_called_once_with(str(temp_video_path))
        mock_visual_tracker.filter_stable_tracks.assert_called_once()
        mock_visual_tracker.export_for_cropping.assert_called_once_with(
            mock_visual_tracker.filter_stable_tracks.return_value, 1920, 1080
        )
        mock_visual_tracker.get_track_statistics.assert_called_once()
        
        # Verify file operations
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()

    def test_process_video_success_without_output_dir(
        self, mock_visual_tracker, mock_smart_track, temp_video_path
    ):
        """Test video processing without specifying output directory"""
        with mock.patch('builtins.open', mock_open()):
            with mock.patch('json.dump'):
                pipeline = Pipeline(enable_visual_tracking=True)
                results = pipeline.process_video(str(temp_video_path))
        
        assert results["success"] is True
        assert results["output_dir"] == str(temp_video_path.parent)

    def test_process_video_visual_tracking_disabled(
        self, mock_smart_track, temp_video_path
    ):
        """Test video processing with visual tracking disabled"""
        pipeline = Pipeline(enable_visual_tracking=False)
        results = pipeline.process_video(str(temp_video_path))
        
        assert results["success"] is True
        assert len(results["pipeline_steps"]) == 2
        
        # Smart crop should succeed
        smart_crop_step = results["pipeline_steps"][0]
        assert smart_crop_step["step"] == "smart_crop"
        assert smart_crop_step["success"] is True
        
        # Visual tracking step should show disabled
        tracking_step = results["pipeline_steps"][1]
        assert tracking_step["step"] == "visual_tracking"
        assert tracking_step["success"] is False
        assert tracking_step["reason"] == "disabled_or_unavailable"

    def test_process_video_visual_tracking_unavailable(
        self, mock_smart_track, temp_video_path
    ):
        """Test video processing when visual tracker is None"""
        pipeline = Pipeline(enable_visual_tracking=True)
        pipeline.visual_tracker = None  # Simulate unavailable tracker
        
        results = pipeline.process_video(str(temp_video_path))
        
        assert results["success"] is True
        
        # Visual tracking step should show unavailable
        tracking_step = results["pipeline_steps"][1]
        assert tracking_step["step"] == "visual_tracking"
        assert tracking_step["success"] is False
        assert tracking_step["reason"] == "disabled_or_unavailable"

    def test_process_video_smart_crop_import_error(self, mock_visual_tracker, temp_video_path):
        """Test pipeline when SmartTrack import fails"""
        with mock.patch('montage.core.pipeline.SmartTrack', side_effect=ImportError("Module not found")):
            pipeline = Pipeline(enable_visual_tracking=True)
            
            with mock.patch('builtins.open', mock_open()):
                with mock.patch('json.dump'):
                    results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is True  # Pipeline continues despite smart crop failure
            smart_crop_step = results["pipeline_steps"][0]
            assert smart_crop_step["success"] is False
            assert "Module not found" in smart_crop_step["result"]["error"]

    def test_process_video_smart_crop_runtime_error(self, mock_visual_tracker, temp_video_path):
        """Test pipeline when SmartTrack.analyze_video raises runtime error"""
        with mock.patch('montage.providers.smart_track.SmartTrack') as mock_class:
            mock_instance = mock.Mock()
            mock_instance.analyze_video.side_effect = RuntimeError("Analysis failed")
            mock_class.return_value = mock_instance
            
            pipeline = Pipeline(enable_visual_tracking=True)
            
            with mock.patch('builtins.open', mock_open()):
                with mock.patch('json.dump'):
                    results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is True  # Pipeline continues
            smart_crop_step = results["pipeline_steps"][0]
            assert smart_crop_step["success"] is False
            assert "Analysis failed" in smart_crop_step["result"]["error"]

    def test_process_video_visual_tracking_track_error(
        self, mock_smart_track, temp_video_path
    ):
        """Test pipeline when visual tracker.track() raises error"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.side_effect = Exception("Tracking failed")
            mock_create.return_value = mock_tracker
            
            pipeline = Pipeline(enable_visual_tracking=True)
            results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is False
            assert "error" in results
            assert "Tracking failed" in results["error"]

    def test_process_video_visual_tracking_filter_error(
        self, mock_smart_track, temp_video_path
    ):
        """Test pipeline when visual tracker.filter_stable_tracks() raises error"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = [{"id": 1}]
            mock_tracker.filter_stable_tracks.side_effect = Exception("Filter failed")
            mock_create.return_value = mock_tracker
            
            pipeline = Pipeline(enable_visual_tracking=True)
            results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is False
            assert "Filter failed" in results["error"]

    def test_process_video_visual_tracking_file_write_error(
        self, mock_visual_tracker, mock_smart_track, temp_video_path
    ):
        """Test pipeline when track file writing fails"""
        with mock.patch('builtins.open', side_effect=IOError("Write failed")):
            pipeline = Pipeline(enable_visual_tracking=True)
            results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is False
            assert "Write failed" in results["error"]

    def test_process_video_visual_tracker_missing_export_method(
        self, mock_smart_track, temp_video_path
    ):
        """Test pipeline when visual tracker missing export_for_cropping method"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = [{"id": 1, "frames": [1, 2, 3]}]
            mock_tracker.filter_stable_tracks.return_value = [{"id": 1, "stable": True}]
            mock_tracker.get_track_statistics.return_value = {"total": 1}
            # Remove export_for_cropping method
            del mock_tracker.export_for_cropping
            mock_create.return_value = mock_tracker
            
            pipeline = Pipeline(enable_visual_tracking=True)
            
            with mock.patch('builtins.open', mock_open()):
                with mock.patch('json.dump') as mock_json_dump:
                    results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is True
            # Verify crop_data is empty list when method is missing
            call_args = mock_json_dump.call_args[0][0]
            assert call_args['crop_data'] == []

    def test_process_video_visual_tracker_missing_statistics_method(
        self, mock_smart_track, temp_video_path
    ):
        """Test pipeline when visual tracker missing get_track_statistics method"""
        with mock.patch('montage.core.pipeline.create_visual_tracker') as mock_create:
            mock_tracker = mock.Mock()
            mock_tracker.track.return_value = [{"id": 1, "frames": [1, 2, 3]}]
            mock_tracker.filter_stable_tracks.return_value = [{"id": 1, "stable": True}]
            mock_tracker.export_for_cropping.return_value = []
            # Remove get_track_statistics method
            del mock_tracker.get_track_statistics
            mock_create.return_value = mock_tracker
            
            pipeline = Pipeline(enable_visual_tracking=True)
            
            with mock.patch('builtins.open', mock_open()):
                with mock.patch('json.dump') as mock_json_dump:
                    results = pipeline.process_video(str(temp_video_path))
            
            assert results["success"] is True
            # Verify statistics is empty dict when method is missing
            call_args = mock_json_dump.call_args[0][0]
            assert call_args['statistics'] == {}

    def test_process_video_output_directory_creation(
        self, mock_visual_tracker, mock_smart_track, temp_video_path
    ):
        """Test that output directory is created when it doesn't exist"""
        output_dir = temp_video_path.parent / "new_test_output"
        
        with mock.patch('builtins.open', mock_open()):
            with mock.patch('json.dump'):
                pipeline = Pipeline(enable_visual_tracking=True)
                results = pipeline.process_video(str(temp_video_path), str(output_dir))
        
        assert output_dir.exists()
        assert results["success"] is True
        
        # Cleanup
        if output_dir.exists():
            output_dir.rmdir()

    def test_process_video_path_conversion(self, mock_smart_track):
        """Test that string paths are converted to Path objects correctly"""
        pipeline = Pipeline(enable_visual_tracking=False)
        
        # Test with string path
        with mock.patch('montage.core.pipeline.Path') as mock_path_class:
            mock_video_path = Mock()
            mock_video_path.parent = Path("/parent")
            mock_output_path = Mock()
            mock_output_path.mkdir = Mock()
            mock_path_class.side_effect = [mock_video_path, mock_output_path]
            
            results = pipeline.process_video("/test/video.mp4", "/test/output")
            
            # Verify Path was called with string arguments
            mock_path_class.assert_any_call("/test/video.mp4")
            mock_path_class.assert_any_call("/test/output")
            mock_output_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_run_smart_crop_success(self):
        """Test successful _run_smart_crop execution"""
        pipeline = Pipeline(enable_visual_tracking=False)
        
        with mock.patch('montage.core.pipeline.SmartTrack') as mock_smart_track_class:
            mock_smart_track = Mock()
            mock_smart_track.analyze_video.return_value = {"crop_data": "test_data"}
            mock_smart_track_class.return_value = mock_smart_track
            
            result = pipeline._run_smart_crop(Path("/test/video.mp4"))
            
            assert result["success"] is True
            assert result["result"] == {"crop_data": "test_data"}
            mock_smart_track.analyze_video.assert_called_once_with("/test/video.mp4")

    def test_run_smart_crop_import_failure(self):
        """Test _run_smart_crop when SmartTrack import fails"""
        pipeline = Pipeline(enable_visual_tracking=False)
        
        with mock.patch('montage.core.pipeline.SmartTrack', side_effect=ImportError("Import failed")):
            result = pipeline._run_smart_crop(Path("/test/video.mp4"))
            
            assert result["success"] is False
            assert "Import failed" in result["error"]

    def test_run_smart_crop_analyze_failure(self):
        """Test _run_smart_crop when analyze_video fails"""
        pipeline = Pipeline(enable_visual_tracking=False)
        
        with mock.patch('montage.core.pipeline.SmartTrack') as mock_smart_track_class:
            mock_smart_track = Mock()
            mock_smart_track.analyze_video.side_effect = ValueError("Analysis error")
            mock_smart_track_class.return_value = mock_smart_track
            
            result = pipeline._run_smart_crop(Path("/test/video.mp4"))
            
            assert result["success"] is False
            assert "Analysis error" in result["error"]


class TestRunPipeline:
    """Test run_pipeline convenience function"""

    def test_run_pipeline_with_visual_tracking_enabled(self):
        """Test run_pipeline with visual tracking enabled"""
        with mock.patch('montage.core.pipeline.Pipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.process_video.return_value = {"success": True, "data": "test"}
            mock_pipeline_class.return_value = mock_pipeline
            
            result = run_pipeline("/test/video.mp4", "/test/output", enable_visual_tracking=True)
            
            mock_pipeline_class.assert_called_once_with(enable_visual_tracking=True)
            mock_pipeline.process_video.assert_called_once_with("/test/video.mp4", "/test/output")
            assert result == {"success": True, "data": "test"}

    def test_run_pipeline_with_visual_tracking_disabled(self):
        """Test run_pipeline with visual tracking disabled"""
        with mock.patch('montage.core.pipeline.Pipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.process_video.return_value = {"success": True}
            mock_pipeline_class.return_value = mock_pipeline
            
            result = run_pipeline("/test/video.mp4", enable_visual_tracking=False)
            
            mock_pipeline_class.assert_called_once_with(enable_visual_tracking=False)
            mock_pipeline.process_video.assert_called_once_with("/test/video.mp4", None)

    def test_run_pipeline_default_parameters(self):
        """Test run_pipeline with default parameters"""
        with mock.patch('montage.core.pipeline.Pipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.process_video.return_value = {"success": True}
            mock_pipeline_class.return_value = mock_pipeline
            
            result = run_pipeline("/test/video.mp4")
            
            mock_pipeline_class.assert_called_once_with(enable_visual_tracking=True)
            mock_pipeline.process_video.assert_called_once_with("/test/video.mp4", None)

    def test_run_pipeline_with_output_dir_none(self):
        """Test run_pipeline with explicit None output_dir"""
        with mock.patch('montage.core.pipeline.Pipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline.process_video.return_value = {"success": True}
            mock_pipeline_class.return_value = mock_pipeline
            
            result = run_pipeline("/test/video.mp4", output_dir=None, enable_visual_tracking=False)
            
            mock_pipeline.process_video.assert_called_once_with("/test/video.mp4", None)


class TestProbeCodecs:
    """Test probe_codecs function comprehensively"""

    @patch('subprocess.run')
    def test_probe_codecs_success_full_data(self, mock_run):
        """Test successful codec probing with complete data"""
        mock_output = {
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "subtitle", "codec_name": "srt"}
            ],
            "format": {"format_name": "mp4"}
        }
        
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output)
        mock_run.return_value = mock_result
        
        result = probe_codecs("/test/video.mp4")
        
        expected = {"video": "h264", "audio": "aac", "subtitle": "srt"}
        assert result == expected
        
        expected_cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            "/test/video.mp4"
        ]
        mock_run.assert_called_once_with(expected_cmd, capture_output=True, text=True, check=True)

    @patch('subprocess.run')
    def test_probe_codecs_no_streams(self, mock_run):
        """Test codec probing with no streams in response"""
        mock_output = {"format": {"format_name": "mp4"}}
        
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output)
        mock_run.return_value = mock_result
        
        result = probe_codecs("/test/video.mp4")
        
        assert result == {}

    @patch('subprocess.run')
    def test_probe_codecs_empty_streams(self, mock_run):
        """Test codec probing with empty streams array"""
        mock_output = {"streams": [], "format": {}}
        
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output)
        mock_run.return_value = mock_result
        
        result = probe_codecs("/test/video.mp4")
        
        assert result == {}

    @patch('subprocess.run')
    def test_probe_codecs_missing_codec_name(self, mock_run):
        """Test codec probing with missing codec_name field"""
        mock_output = {
            "streams": [
                {"codec_type": "video"},  # Missing codec_name
                {"codec_type": "audio", "codec_name": "aac"},
                {"codec_type": "data", "codec_name": "unknown"}  # Different type
            ]
        }
        
        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_output)
        mock_run.return_value = mock_result
        
        result = probe_codecs("/test/video.mp4")
        
        assert result == {"video": None, "audio": "aac", "data": "unknown"}

    @patch('subprocess.run')
    def test_probe_codecs_subprocess_error(self, mock_run):
        """Test codec probing when subprocess raises CalledProcessError"""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["ffprobe"], "Error output")
        
        with pytest.raises(subprocess.CalledProcessError):
            probe_codecs("/test/video.mp4")

    @patch('subprocess.run')
    def test_probe_codecs_json_decode_error(self, mock_run):
        """Test codec probing when JSON decoding fails"""
        mock_result = Mock()
        mock_result.stdout = "invalid json output"
        mock_run.return_value = mock_result
        
        with pytest.raises(json.JSONDecodeError):
            probe_codecs("/test/video.mp4")

    @patch('subprocess.run')
    def test_probe_codecs_empty_response(self, mock_run):
        """Test codec probing with empty JSON response"""
        mock_result = Mock()
        mock_result.stdout = "{}"
        mock_run.return_value = mock_result
        
        result = probe_codecs("/test/video.mp4")
        
        assert result == {}


class TestVideoEditor:
    """Comprehensive test suite for VideoEditor class"""

    def setup_method(self):
        self.test_video = "/path/to/test_video.mp4"
        self.test_clips = [
            {"start": 0.0, "end": 5.0},
            {"start": 10.0, "end": 15.0, "filters": "scale=1280:720"},
        ]

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    def test_video_editor_init_success(self, mock_probe, mock_path):
        """Test successful VideoEditor initialization"""
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "h264", "audio": "aac"}
        
        editor = VideoEditor(self.test_video)
        
        assert editor.source == self.test_video
        assert editor.codecs == {"video": "h264", "audio": "aac"}
        mock_path.assert_called_once_with(self.test_video)
        mock_path_obj.is_file.assert_called_once()
        mock_probe.assert_called_once_with(self.test_video)

    @patch('montage.providers.video_processor.Path')
    def test_video_editor_init_file_not_found(self, mock_path):
        """Test VideoEditor initialization with missing file"""
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = False
        mock_path.return_value = mock_path_obj
        
        with pytest.raises(FileNotFoundError, match="Source not found"):
            VideoEditor(self.test_video)

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    @patch('montage.providers.video_processor.subprocess.run')
    @patch('montage.providers.video_processor.tempfile.mkdtemp')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_clips_success_with_filters(self, mock_file, mock_mkdtemp, mock_subprocess, mock_probe, mock_path):
        """Test successful clip processing with filters"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "h264", "audio": "aac"}
        mock_mkdtemp.return_value = "/tmp/montage_test"
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Mock Path operations for temp directory
        with patch('montage.providers.video_processor.Path') as mock_path_class:
            def path_constructor(path_str):
                if path_str == "/tmp/montage_test":
                    mock_tmp_dir = Mock()
                    mock_seg_path = Mock()
                    mock_seg_path.__str__ = Mock(return_value=f"/tmp/montage_test/seg_000.mp4")
                    mock_list_txt = Mock()
                    mock_list_txt.__str__ = Mock(return_value="/tmp/montage_test/segments.txt")
                    
                    call_count = getattr(path_constructor, 'call_count', 0)
                    path_constructor.call_count = call_count + 1
                    
                    if call_count < 2:  # First two calls for segments
                        mock_tmp_dir.__truediv__ = Mock(return_value=mock_seg_path)
                    else:  # Third call for list file
                        mock_tmp_dir.__truediv__ = Mock(return_value=mock_list_txt)
                    
                    return mock_tmp_dir
                else:
                    return mock_path_obj
            
            mock_path_class.side_effect = path_constructor
            
            editor = VideoEditor(self.test_video)
            editor.process_clips(self.test_clips, "/output/video.mp4")
            
            # Verify subprocess calls: 2 segments + 1 concat = 3 calls
            assert mock_subprocess.call_count == 3
            
            # Verify file writing for concat list
            mock_file.assert_called()

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    @patch('montage.providers.video_processor.subprocess.run')
    @patch('montage.providers.video_processor.tempfile.mkdtemp')
    @patch('montage.providers.video_processor.shlex.quote')
    @patch('montage.providers.video_processor.shlex.split')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_clips_command_structure(self, mock_file, mock_split, mock_quote, mock_mkdtemp, 
                                           mock_subprocess, mock_probe, mock_path):
        """Test the structure of generated ffmpeg commands"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "libx264", "audio": "aac"}
        mock_mkdtemp.return_value = "/tmp/montage_test"
        mock_subprocess.return_value = Mock(returncode=0)
        mock_quote.side_effect = lambda x: f"'{x}'"
        mock_split.side_effect = lambda x: x.split()
        
        with patch('montage.providers.video_processor.Path') as mock_path_class:
            mock_tmp_dir = Mock()
            mock_seg_path = Mock()
            mock_seg_path.__str__ = Mock(return_value="/tmp/montage_test/seg_000.mp4")
            mock_list_txt = Mock()
            mock_list_txt.__str__ = Mock(return_value="/tmp/montage_test/segments.txt")
            
            def truediv_side_effect(arg):
                if "seg_" in str(arg):
                    return mock_seg_path
                else:
                    return mock_list_txt
            
            mock_tmp_dir.__truediv__.side_effect = truediv_side_effect
            mock_path_class.side_effect = lambda arg: mock_tmp_dir if arg == "/tmp/montage_test" else mock_path_obj
            
            editor = VideoEditor(self.test_video)
            editor.process_clips(self.test_clips, "/output/video.mp4")
            
            # Verify command structure for segments
            segment_calls = mock_split.call_args_list[:-1]  # Exclude concat call
            
            for i, call_args in enumerate(segment_calls):
                cmd = call_args[0][0]
                
                # Verify basic command structure
                assert "ffmpeg -y" in cmd
                assert f"-ss {self.test_clips[i]['start']}" in cmd
                assert f"-t {self.test_clips[i]['end'] - self.test_clips[i]['start']}" in cmd
                assert f"-i '{self.test_video}'" in cmd
                assert "-c:v libx264" in cmd
                assert "-c:a aac" in cmd
                
                # Verify filters for second clip
                if i == 1 and "filters" in self.test_clips[i]:
                    assert f"-vf '{self.test_clips[i]['filters']}'" in cmd
            
            # Verify concat command
            concat_call = mock_split.call_args_list[-1]
            concat_cmd = concat_call[0][0]
            assert "ffmpeg -y" in concat_cmd
            assert "-f concat" in concat_cmd
            assert "-safe 0" in concat_cmd
            assert "-c copy" in concat_cmd

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    @patch('montage.providers.video_processor.subprocess.run')
    @patch('montage.providers.video_processor.tempfile.mkdtemp')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_clips_without_filters(self, mock_file, mock_mkdtemp, mock_subprocess, mock_probe, mock_path):
        """Test clip processing for clips without filters"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "h264", "audio": "aac"}
        mock_mkdtemp.return_value = "/tmp/montage_test"
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('montage.providers.video_processor.Path') as mock_path_class:
            mock_tmp_dir = Mock()
            mock_seg_path = Mock()
            mock_seg_path.__str__ = Mock(return_value="/tmp/montage_test/seg_000.mp4")
            mock_list_txt = Mock()
            mock_list_txt.__str__ = Mock(return_value="/tmp/montage_test/segments.txt")
            
            def truediv_side_effect(arg):
                if "seg_" in str(arg):
                    return mock_seg_path
                else:
                    return mock_list_txt
            
            mock_tmp_dir.__truediv__.side_effect = truediv_side_effect
            mock_path_class.side_effect = lambda arg: mock_tmp_dir if arg == "/tmp/montage_test" else mock_path_obj
            
            editor = VideoEditor(self.test_video)
            clips_without_filters = [{"start": 0.0, "end": 5.0}]
            editor.process_clips(clips_without_filters, "/output/video.mp4")
            
            # Should complete successfully
            assert mock_subprocess.call_count == 2  # 1 segment + 1 concat

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    @patch('montage.providers.video_processor.subprocess.run')
    @patch('montage.providers.video_processor.tempfile.mkdtemp')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_clips_empty_list(self, mock_file, mock_mkdtemp, mock_subprocess, mock_probe, mock_path):
        """Test clip processing with empty clips list"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "h264", "audio": "aac"}
        mock_mkdtemp.return_value = "/tmp/montage_test"
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('montage.providers.video_processor.Path') as mock_path_class:
            mock_tmp_dir = Mock()
            mock_list_txt = Mock()
            mock_list_txt.__str__ = Mock(return_value="/tmp/montage_test/segments.txt")
            mock_tmp_dir.__truediv__ = Mock(return_value=mock_list_txt)
            mock_path_class.side_effect = lambda arg: mock_tmp_dir if arg == "/tmp/montage_test" else mock_path_obj
            
            editor = VideoEditor(self.test_video)
            editor.process_clips([], "/output/video.mp4")
            
            # Should still create concat file and run concat command
            assert mock_subprocess.call_count == 1  # Just concat
            mock_file.assert_called()

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    @patch('montage.providers.video_processor.subprocess.run')
    @patch('montage.providers.video_processor.tempfile.mkdtemp')
    def test_process_clips_subprocess_error_first_clip(self, mock_mkdtemp, mock_subprocess, mock_probe, mock_path):
        """Test clip processing when first clip subprocess fails"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "h264", "audio": "aac"}
        mock_mkdtemp.return_value = "/tmp/montage_test"
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffmpeg", "Error")
        
        with patch('montage.providers.video_processor.Path') as mock_path_class:
            mock_tmp_dir = Mock()
            mock_seg_path = Mock()
            mock_seg_path.__str__ = Mock(return_value="/tmp/montage_test/seg_000.mp4")
            mock_tmp_dir.__truediv__ = Mock(return_value=mock_seg_path)
            mock_path_class.side_effect = lambda arg: mock_tmp_dir if arg == "/tmp/montage_test" else mock_path_obj
            
            editor = VideoEditor(self.test_video)
            
            with pytest.raises(subprocess.CalledProcessError):
                editor.process_clips([{"start": 0.0, "end": 5.0}], "/output/video.mp4")

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    @patch('montage.providers.video_processor.subprocess.run')
    @patch('montage.providers.video_processor.tempfile.mkdtemp')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_clips_concat_subprocess_error(self, mock_file, mock_mkdtemp, mock_subprocess, mock_probe, mock_path):
        """Test clip processing when concat subprocess fails"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        mock_probe.return_value = {"video": "h264", "audio": "aac"}
        mock_mkdtemp.return_value = "/tmp/montage_test"
        
        # First call (segment) succeeds, second call (concat) fails
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # Segment extraction succeeds
            subprocess.CalledProcessError(1, "ffmpeg", "Concat error")  # Concat fails
        ]
        
        with patch('montage.providers.video_processor.Path') as mock_path_class:
            mock_tmp_dir = Mock()
            mock_seg_path = Mock()
            mock_seg_path.__str__ = Mock(return_value="/tmp/montage_test/seg_000.mp4")
            mock_list_txt = Mock()
            mock_list_txt.__str__ = Mock(return_value="/tmp/montage_test/segments.txt")
            
            def truediv_side_effect(arg):
                if "seg_" in str(arg):
                    return mock_seg_path
                else:
                    return mock_list_txt
            
            mock_tmp_dir.__truediv__.side_effect = truediv_side_effect
            mock_path_class.side_effect = lambda arg: mock_tmp_dir if arg == "/tmp/montage_test" else mock_path_obj
            
            editor = VideoEditor(self.test_video)
            
            with pytest.raises(subprocess.CalledProcessError):
                editor.process_clips([{"start": 0.0, "end": 5.0}], "/output/video.mp4")

    @patch('montage.providers.video_processor.Path')
    @patch('montage.providers.video_processor.probe_codecs')
    def test_process_clips_with_default_fallback_codecs(self, mock_probe, mock_path):
        """Test clip processing when probe returns no codecs (using get() default)"""
        # Setup mocks
        mock_path_obj = Mock()
        mock_path_obj.is_file.return_value = True
        mock_path.return_value = mock_path_obj
        
        # Return empty codecs dict
        mock_probe.return_value = {}
        
        with patch('montage.providers.video_processor.subprocess.run') as mock_subprocess:
            with patch('montage.providers.video_processor.tempfile.mkdtemp') as mock_mkdtemp:
                with patch('builtins.open', mock_open()):
                    mock_mkdtemp.return_value = "/tmp/montage_test"
                    mock_subprocess.return_value = Mock(returncode=0)
                    
                    with patch('montage.providers.video_processor.Path') as mock_path_class:
                        mock_tmp_dir = Mock()
                        mock_seg_path = Mock()
                        mock_seg_path.__str__ = Mock(return_value="/tmp/montage_test/seg_000.mp4")
                        mock_list_txt = Mock()
                        mock_list_txt.__str__ = Mock(return_value="/tmp/montage_test/segments.txt")
                        
                        def truediv_side_effect(arg):
                            if "seg_" in str(arg):
                                return mock_seg_path
                            else:
                                return mock_list_txt
                        
                        mock_tmp_dir.__truediv__.side_effect = truediv_side_effect
                        mock_path_class.side_effect = lambda arg: mock_tmp_dir if arg == "/tmp/montage_test" else mock_path_obj
                        
                        editor = VideoEditor(self.test_video)
                        editor.process_clips([{"start": 0.0, "end": 5.0}], "/output/video.mp4")
                        
                        # Should use default codecs: libx264 for video, aac for audio
                        assert mock_subprocess.call_count == 2  # 1 segment + 1 concat


class TestIntegration:
    """Integration tests for pipeline components"""

    @patch('montage.core.pipeline.create_visual_tracker') 
    @patch('montage.core.pipeline.SmartTrack')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_full_pipeline_integration_success(self, mock_json_dump, mock_file, mock_smart_track_class, mock_create_tracker):
        """Test complete pipeline integration with all components"""
        # Setup visual tracker
        mock_tracker = Mock()
        mock_tracker.track.return_value = [{"id": 1, "frames": list(range(60))}]
        mock_tracker.filter_stable_tracks.return_value = [{"id": 1, "stable": True}]
        mock_tracker.export_for_cropping.return_value = [{"x": 100, "y": 100}]
        mock_tracker.get_track_statistics.return_value = {"total": 1}
        mock_create_tracker.return_value = mock_tracker
        
        # Setup smart track
        mock_smart_track = Mock()
        mock_smart_track.analyze_video.return_value = {"analysis": "complete"}
        mock_smart_track_class.return_value = mock_smart_track
        
        # Setup path operations
        with patch('montage.core.pipeline.Path') as mock_path:
            mock_video_path = Mock()
            mock_video_path.parent = Path("/parent")
            mock_output_dir = Mock()
            mock_output_dir.mkdir = Mock()
            mock_track_file = Mock()
            mock_output_dir.__truediv__ = Mock(return_value=mock_track_file)
            mock_path.side_effect = [mock_video_path, mock_output_dir]
            
            result = run_pipeline("/test/video.mp4", "/test/output", enable_visual_tracking=True)
            
            assert result["success"] is True
            assert len(result["pipeline_steps"]) == 2
            
            # Verify both pipeline steps
            steps = {step["step"]: step for step in result["pipeline_steps"]}
            assert "smart_crop" in steps
            assert "visual_tracking" in steps
            assert steps["smart_crop"]["success"] is True
            assert steps["visual_tracking"]["success"] is True
            
            # Verify method calls
            mock_smart_track.analyze_video.assert_called_once()
            mock_tracker.track.assert_called_once()
            mock_file.assert_called_once()
            mock_json_dump.assert_called_once()

    @patch('montage.core.pipeline.create_visual_tracker')
    @patch('montage.core.pipeline.SmartTrack')
    def test_pipeline_error_propagation(self, mock_smart_track_class, mock_create_tracker):
        """Test error propagation through pipeline"""
        # Setup tracker to fail
        mock_create_tracker.return_value = None
        
        # Setup smart track to fail
        mock_smart_track_class.side_effect = Exception("Pipeline failure")
        
        with patch('montage.core.pipeline.Path') as mock_path:
            mock_video_path = Mock()
            mock_video_path.parent = Path("/parent")
            mock_path.return_value = mock_video_path
            
            result = run_pipeline("/test/video.mp4", enable_visual_tracking=True)
            
            assert result["success"] is True  # Pipeline continues despite smart crop failure
            
            # Smart crop should have failed
            smart_crop_step = result["pipeline_steps"][0]
            assert smart_crop_step["success"] is False
            assert "Pipeline failure" in smart_crop_step["result"]["error"]
            
            # Visual tracking should show unavailable
            tracking_step = result["pipeline_steps"][1]
            assert tracking_step["success"] is False
            assert tracking_step["reason"] == "disabled_or_unavailable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])