"""
P1-01: Upload security validation tests

Tests the upload validator security controls:
- File size limits and streaming validation
- MIME type detection and validation
- Extension validation and sanitization
- Rate limiting per API key
- Malicious file detection
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import UploadFile

from montage.core.upload_validator import (
    UploadLimits,
    UploadValidationError,
    UploadValidator,
)


class TestUploadValidator:
    """Test upload validation security controls"""

    def setup_method(self):
        """Setup test environment"""
        self.validator = UploadValidator()
        self.test_api_key = "test-key-12345"

    def test_file_size_validation_within_limit(self):
        """Test file size validation passes for files within limit"""
        # Create mock UploadFile with valid size
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.mp4"
        mock_file.headers = {"content-length": "100000000"}  # 100MB

        # Should not raise exception
        self.validator._validate_file_size(mock_file)

    def test_file_size_validation_exceeds_limit(self):
        """Test file size validation fails for files exceeding limit"""
        # Create mock UploadFile with excessive size
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "huge_video.mp4"
        mock_file.headers = {"content-length": str(self.validator.limits.MAX_FILE_SIZE_BYTES + 1)}

        with pytest.raises(UploadValidationError) as exc_info:
            self.validator._validate_file_size(mock_file)

        assert exc_info.value.error_code == "FILE_TOO_LARGE"
        assert "exceeds limit" in exc_info.value.message

    def test_filename_validation_clean(self):
        """Test filename validation passes for clean filenames"""
        test_cases = [
            "video.mp4",
            "my-video_2025.mov",
            "presentation.avi",
            "test123.mkv"
        ]

        for filename in test_cases:
            result = self.validator._validate_filename(filename)
            assert result == filename

    def test_filename_validation_dangerous(self):
        """Test filename validation blocks dangerous filenames"""
        dangerous_filenames = [
            "../../../etc/passwd",
            "video.mp4/../../../etc/passwd",
            "video.php.mp4",
            "video.mp4.exe",
            "test<script>.mp4",
            "video|cmd.mp4",
            "video?.mp4"
        ]

        for filename in dangerous_filenames:
            with pytest.raises(UploadValidationError) as exc_info:
                self.validator._validate_filename(filename)
            assert exc_info.value.error_code in ["INVALID_FILENAME", "INVALID_EXTENSION"]

    def test_filename_validation_invalid_extension(self):
        """Test filename validation blocks invalid extensions"""
        invalid_files = [
            "video.exe",
            "script.php",
            "document.txt",
            "image.png",
            "malware.bat"
        ]

        for filename in invalid_files:
            with pytest.raises(UploadValidationError) as exc_info:
                self.validator._validate_filename(filename)
            assert exc_info.value.error_code == "INVALID_EXTENSION"

    def test_rate_limiting_within_limits(self):
        """Test rate limiting allows requests within limits"""
        # Should pass for first few requests
        for i in range(5):
            self.validator._check_rate_limits(self.test_api_key)

    def test_rate_limiting_hourly_exceeded(self):
        """Test rate limiting blocks after hourly limit exceeded"""
        # Simulate max uploads in last hour
        now = datetime.utcnow()
        self.validator._upload_tracking[self.test_api_key] = [
            now - timedelta(minutes=30)
        ] * self.validator.limits.MAX_UPLOADS_PER_HOUR

        with pytest.raises(UploadValidationError) as exc_info:
            self.validator._check_rate_limits(self.test_api_key)

        assert exc_info.value.error_code == "RATE_LIMIT_HOURLY"

    def test_rate_limiting_daily_exceeded(self):
        """Test rate limiting blocks after daily limit exceeded"""
        # Simulate max uploads in last day
        now = datetime.utcnow()
        self.validator._upload_tracking[self.test_api_key] = [
            now - timedelta(hours=12)
        ] * self.validator.limits.MAX_UPLOADS_PER_DAY

        with pytest.raises(UploadValidationError) as exc_info:
            self.validator._check_rate_limits(self.test_api_key)

        assert exc_info.value.error_code == "RATE_LIMIT_DAILY"

    def test_rate_limiting_cleanup_old_entries(self):
        """Test rate limiting cleans up old tracking entries"""
        now = datetime.utcnow()

        # Add old entries that should be cleaned up
        self.validator._upload_tracking[self.test_api_key] = [
            now - timedelta(days=2),
            now - timedelta(hours=25),
            now - timedelta(minutes=30)  # This should remain
        ]

        # Check rate limits (should trigger cleanup)
        self.validator._check_rate_limits(self.test_api_key)

        # Should only have recent entry plus the new one
        remaining = self.validator._upload_tracking[self.test_api_key]
        assert len(remaining) == 2

    @patch('src.core.upload_validator.magic.Magic')
    def test_mime_validation_valid_video(self, mock_magic):
        """Test MIME validation passes for valid video files"""
        # Mock libmagic to return valid video MIME type
        mock_mime = Mock()
        mock_mime.from_file.return_value = "video/mp4"
        mock_magic.return_value = mock_mime

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            file_path = Path(tmp_file.name)

            result = self.validator._validate_mime_type(file_path, "video/mp4")
            assert result == "video/mp4"

    @patch('src.core.upload_validator.magic.Magic')
    def test_mime_validation_spoofed_file(self, mock_magic):
        """Test MIME validation detects spoofed files"""
        # Mock libmagic to return different MIME type than extension suggests
        mock_mime = Mock()
        mock_mime.from_file.return_value = "application/x-executable"
        mock_magic.return_value = mock_mime

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
            file_path = Path(tmp_file.name)

            with pytest.raises(UploadValidationError) as exc_info:
                self.validator._validate_mime_type(file_path)

            assert exc_info.value.error_code in ["INVALID_MIME_TYPE", "MIME_EXTENSION_MISMATCH"]

    def test_upload_stats_tracking(self):
        """Test upload statistics tracking"""
        # Initial stats should be zero
        stats = self.validator.get_upload_stats(self.test_api_key)
        assert stats["uploads_last_hour"] == 0
        assert stats["uploads_last_day"] == 0

        # Add some upload records
        now = datetime.utcnow()
        self.validator._upload_tracking[self.test_api_key] = [
            now - timedelta(minutes=30),  # Within hour
            now - timedelta(hours=2),     # Within day
            now - timedelta(hours=25)     # Outside day (should be ignored)
        ]

        stats = self.validator.get_upload_stats(self.test_api_key)
        assert stats["uploads_last_hour"] == 1
        assert stats["uploads_last_day"] == 2

    def test_malware_scan_suspicious_small_file(self):
        """Test malware scanning detects suspiciously small files"""
        # Enable scanning for this test
        original_setting = self.validator.limits.SCAN_FOR_MALWARE
        self.validator.limits.SCAN_FOR_MALWARE = True

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_file:
                # Write very small amount of data (suspicious for video)
                tmp_file.write(b"tiny")
                tmp_file.flush()

                file_path = Path(tmp_file.name)

                with pytest.raises(UploadValidationError) as exc_info:
                    self.validator._scan_for_malware(file_path)

                assert exc_info.value.error_code == "SUSPICIOUS_FILE_SIZE"

        finally:
            self.validator.limits.SCAN_FOR_MALWARE = original_setting

    @pytest.mark.asyncio
    async def test_full_validation_pipeline_success(self):
        """Test complete validation pipeline with valid file"""
        # Create a proper temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            # Write some video-like data
            tmp_file.write(b"fake video content" * 1000)  # Make it reasonable size
            tmp_file.flush()

            try:
                file_path = Path(tmp_file.name)

                # Mock UploadFile
                mock_file = Mock(spec=UploadFile)
                mock_file.filename = "test_video.mp4"
                mock_file.content_type = "video/mp4"

                # Mock MIME detection to return valid type
                with patch('src.core.upload_validator.magic.Magic') as mock_magic:
                    mock_mime = Mock()
                    mock_mime.from_file.return_value = "video/mp4"
                    mock_magic.return_value = mock_mime

                    # Should complete successfully
                    filename, mime_type = await self.validator.validate_upload(
                        mock_file, self.test_api_key, file_path
                    )

                    assert filename == "test_video.mp4"
                    assert mime_type == "video/mp4"

            finally:
                # Clean up
                if file_path.exists():
                    file_path.unlink()


class TestUploadLimits:
    """Test upload limits configuration"""

    def test_default_limits(self):
        """Test default upload limits are reasonable"""
        limits = UploadLimits()

        assert limits.MAX_FILE_SIZE_MB == 600
        assert limits.MAX_DURATION_SECONDS == 3600  # 1 hour
        assert limits.MAX_UPLOADS_PER_HOUR == 10
        assert limits.MAX_UPLOADS_PER_DAY == 50
        assert limits.REQUIRE_MIME_VALIDATION is True

    def test_allowed_file_types(self):
        """Test allowed file types include common video formats"""
        limits = UploadLimits()

        expected_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        for ext in expected_extensions:
            assert ext in limits.ALLOWED_EXTENSIONS

        expected_mimes = [
            "video/mp4", "video/quicktime", "video/x-msvideo",
            "video/x-matroska", "video/webm"
        ]
        for mime in expected_mimes:
            assert mime in limits.ALLOWED_MIME_TYPES
