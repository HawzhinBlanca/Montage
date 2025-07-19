"""Test video validation and pre-flight checks"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from src.utils.video_validator import (
    VideoValidator,
    VideoMetadata,
    CorruptedVideoError,
    perform_preflight_check,
)
from src.core.db import Database


class TestVideoValidator:
    """Test video validator functionality"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return VideoValidator()

    @pytest.fixture
    def mock_ffprobe_output(self):
        """Mock ffprobe output for a valid video"""
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_name": "h264",
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "color_space": "bt709",
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                },
                {
                    "index": 1,
                    "codec_name": "aac",
                    "codec_type": "audio",
                    "channels": 2,
                    "sample_rate": "48000",
                },
            ],
            "format": {
                "duration": "300.5",
                "bit_rate": "8000000",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            },
        }

    def test_validate_file_not_exists(self, validator):
        """Test validation of non-existent file"""
        is_valid, metadata, error = validator.validate_file("/nonexistent/file.mp4")

        assert not is_valid
        assert metadata is None
        assert "does not exist" in error

    def test_validate_empty_file(self, validator):
        """Test validation of empty file"""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            is_valid, metadata, error = validator.validate_file(tmp_path)

            assert not is_valid
            assert "empty" in error.lower()
        finally:
            os.unlink(tmp_path)

    @patch("subprocess.run")
    def test_validate_valid_video(self, mock_run, validator, mock_ffprobe_output):
        """Test validation of valid video file"""
        # Mock successful ffprobe run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_output)
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake video data")

        try:
            is_valid, metadata, error = validator.validate_file(tmp_path)

            assert is_valid
            assert metadata is not None
            assert error is None

            # Check metadata
            assert metadata.duration == 300.5
            assert metadata.codec == "h264"
            assert metadata.resolution == "1920x1080"
            assert metadata.fps == 30.0
            assert metadata.color_space == "bt709"
            assert metadata.has_audio
            assert metadata.audio_codec == "aac"

        finally:
            os.unlink(tmp_path)

    @patch("subprocess.run")
    def test_detect_moov_atom_corruption(self, mock_run, validator):
        """Test detection of moov atom corruption"""
        # Mock ffprobe with moov atom error
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "moov atom not found"
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            with pytest.raises(CorruptedVideoError) as exc_info:
                validator._run_ffprobe(tmp.name)

            assert "moov atom not found" in str(exc_info.value)

    @patch("subprocess.run")
    def test_detect_hdr_video(self, mock_run, validator):
        """Test detection of HDR video (should fail validation)"""
        # Mock HDR video output
        hdr_output = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 3840,
                    "height": 2160,
                    "r_frame_rate": "24/1",
                    "color_space": "bt2020nc",  # HDR color space
                    "color_transfer": "smpte2084",  # HDR transfer
                }
            ],
            "format": {"duration": "600.0", "bit_rate": "15000000"},
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(hdr_output)
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            is_valid, metadata, error = validator.validate_file(tmp.name)

            assert not is_valid
            assert "HDR input not supported" in error

    def test_validate_metadata_duration_limits(self, validator):
        """Test duration validation limits"""
        # Test too long duration
        metadata = VideoMetadata(
            duration=7201,  # Over 2 hours
            codec="h264",
            resolution="1920x1080",
            fps=30,
            bitrate=8000000,
            color_space="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
            has_audio=True,
        )

        errors = validator._validate_metadata(metadata)
        assert any("exceeds 2 hours limit" in e for e in errors)

        # Test zero duration
        metadata.duration = 0
        errors = validator._validate_metadata(metadata)
        assert any("Invalid duration" in e for e in errors)

    def test_validate_metadata_resolution_limits(self, validator):
        """Test resolution validation limits"""
        # Test 4K limit
        metadata = VideoMetadata(
            duration=300,
            codec="h264",
            resolution="4096x2160",  # Over 4K
            fps=30,
            bitrate=8000000,
            color_space="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
            has_audio=True,
        )

        errors = validator._validate_metadata(metadata)
        assert any("exceeds 4K limit" in e for e in errors)

    def test_validate_metadata_fps_limits(self, validator):
        """Test frame rate validation limits"""
        # Test high FPS
        metadata = VideoMetadata(
            duration=300,
            codec="h264",
            resolution="1920x1080",
            fps=144,  # Over 120fps
            bitrate=8000000,
            color_space="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
            has_audio=True,
        )

        errors = validator._validate_metadata(metadata)
        assert any("exceeds 120fps limit" in e for e in errors)

    def test_validate_metadata_unsupported_codec(self, validator):
        """Test unsupported codec detection"""
        metadata = VideoMetadata(
            duration=300,
            codec="wmv3",  # Unsupported codec
            resolution="1920x1080",
            fps=30,
            bitrate=8000000,
            color_space="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
            has_audio=True,
            audio_codec="wmav2",  # Unsupported audio codec
        )

        errors = validator._validate_metadata(metadata)
        assert any("Unsupported video codec" in e for e in errors)
        assert any("Unsupported audio codec" in e for e in errors)

    def test_calculate_file_hash(self, validator):
        """Test file hash calculation"""
        # Create temp file with known content
        test_content = b"test video content"

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name

        try:
            # Calculate hash
            file_hash = validator.calculate_file_hash(tmp_path)

            # Verify it's a valid SHA256 hash (64 hex chars)
            assert len(file_hash) == 64
            assert all(c in "0123456789abcdef" for c in file_hash)

            # Hash should be consistent
            hash2 = validator.calculate_file_hash(tmp_path)
            assert file_hash == hash2

        finally:
            os.unlink(tmp_path)

    @patch("subprocess.run")
    def test_validate_and_store(self, mock_run, validator, mock_ffprobe_output):
        """Test complete validation and database storage"""
        # Mock ffprobe
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(mock_ffprobe_output)
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create test job
        db = Database()
        job_id = db.insert(
            "video_job",
            {
                "src_hash": "pending",
                "status": "queued",
                "input_path": "/test/video.mp4",
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            # Write some data
            tmp.write(b"fake video content")
            tmp.flush()

            # Validate and store
            result = validator.validate_and_store(job_id, tmp.name)

            assert result is True

            # Check database was updated
            job = db.find_one("video_job", {"id": job_id})
            assert job["status"] == "validated"
            assert job["duration"] == 300.5
            assert job["codec"] == "h264"
            assert job["color_space"] == "bt709"
            assert len(job["src_hash"]) == 64  # SHA256 hash

    @patch("subprocess.run")
    def test_validate_and_store_failure(self, mock_run, validator):
        """Test validation failure updates database correctly"""
        # Mock ffprobe with error
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Invalid data found when processing input"
        mock_run.return_value = mock_result

        # Create test job
        db = Database()
        job_id = db.insert(
            "video_job",
            {
                "src_hash": "pending",
                "status": "queued",
                "input_path": "/test/corrupted.mp4",
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            result = validator.validate_and_store(job_id, tmp.name)

            assert result is False

            # Check database was updated with failure
            job = db.find_one("video_job", {"id": job_id})
            assert job["status"] == "failed"
            assert "Pre-flight check" in job["error_message"]


class TestPreflightIntegration:
    """Test preflight check integration function"""

    @patch("subprocess.run")
    def test_perform_preflight_check_success(self, mock_run):
        """Test successful preflight check"""
        # Mock ffprobe
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/1",
                        "color_space": "bt709",
                        "color_primaries": "bt709",
                        "color_transfer": "bt709",
                    }
                ],
                "format": {"duration": "120.5", "bit_rate": "5000000"},
            }
        )
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create test job
        db = Database()
        job_id = db.insert(
            "video_job",
            {
                "src_hash": "pending",
                "status": "queued",
                "input_path": "/test/video.mp4",
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(b"test data")
            tmp.flush()

            result = perform_preflight_check(job_id, tmp.name)

            assert result["valid"] is True
            assert result["duration"] == 120.5
            assert result["codec"] == "h264"
            assert result["color_space"] == "bt709"

    @patch("subprocess.run")
    def test_perform_preflight_check_failure(self, mock_run):
        """Test failed preflight check"""
        # Mock ffprobe with HDR video
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "hevc",
                        "width": 3840,
                        "height": 2160,
                        "r_frame_rate": "24/1",
                        "color_space": "bt2020nc",
                        "color_transfer": "smpte2084",
                    }
                ],
                "format": {"duration": "600.0", "bit_rate": "15000000"},
            }
        )
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Create test job
        db = Database()
        job_id = db.insert(
            "video_job",
            {
                "src_hash": "pending",
                "status": "queued",
                "input_path": "/test/hdr_video.mp4",
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            result = perform_preflight_check(job_id, tmp.name)

            assert result["valid"] is False
            assert "HDR input not supported" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
