#!/usr/bin/env python3
"""
Test suite for the comprehensive error propagation system.
Demonstrates error handling, context preservation, and recovery strategies.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import error handling system
from src.core.exceptions import (
    MontageError,
    VideoProcessingError,
    VideoDecodeError,
    VideoEncodeError,
    FFmpegError,
    TranscriptionError,
    WhisperError,
    DeepgramError,
    ResourceError,
    MemoryError,
    DiskSpaceError,
    DatabaseError,
    ConnectionError,
    QueryError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
    ErrorCategory,
    ErrorSeverity,
    RetryStrategy,
    ErrorContext,
    handle_error,
    create_error_context,
)
from src.utils.error_handler import (
    with_error_context,
    with_retry,
    safe_execute,
    error_recovery_context,
    aggregate_errors,
    ErrorCollector,
    handle_ffmpeg_error,
    handle_api_error,
)


class TestErrorHierarchy:
    """Test the custom exception hierarchy"""

    def test_base_error_creation(self):
        """Test creating base MontageError with context"""
        context = create_error_context(
            job_id="test_job_123",
            video_path="/path/to/video.mp4",
            component="TestComponent",
            operation="test_operation",
        )

        error = MontageError("Test error message", context=context)

        assert error.message == "Test error message"
        assert error.context.job_id == "test_job_123"
        assert error.context.video_path == "/path/to/video.mp4"
        assert error.category == ErrorCategory.SYSTEM_ERROR
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.retry_strategy == RetryStrategy.NO_RETRY

    def test_error_with_suggestions(self):
        """Test adding suggestions to errors"""
        error = VideoDecodeError("Cannot decode video", video_path="/corrupted.mp4")

        assert len(error.suggestions) >= 2
        assert "corrupted" in error.suggestions[0]
        assert "codec" in error.suggestions[1]

    def test_error_chaining(self):
        """Test exception chaining for context preservation"""
        original = ValueError("Original error")

        error = VideoProcessingError("Processing failed", cause=original)

        assert error.__cause__ == original
        assert error.cause == original

    def test_error_context_metadata(self):
        """Test adding metadata to error context"""
        error = FFmpegError(
            "FFmpeg failed",
            command=["ffmpeg", "-i", "input.mp4"],
            stderr="Invalid codec",
        )

        error.add_context(duration=120.5, resolution="1920x1080", codec="h264")

        assert error.context.metadata["duration"] == 120.5
        assert error.context.metadata["resolution"] == "1920x1080"
        assert error.context.metadata["codec"] == "h264"

    def test_error_serialization(self):
        """Test error serialization for API responses"""
        error = RateLimitError(
            "API rate limit exceeded", service="deepgram", retry_after=60
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "RateLimitError"
        assert error_dict["message"] == "API rate limit exceeded"
        assert error_dict["category"] == "api_error"
        assert error_dict["severity"] == "medium"
        assert error_dict["retry_strategy"] == "exponential_backoff"
        assert error_dict["context"]["metadata"]["service"] == "deepgram"
        assert error_dict["context"]["metadata"]["retry_after"] == 60


class TestErrorHandlingDecorators:
    """Test error handling decorators"""

    def test_error_context_decorator(self):
        """Test automatic error context injection"""

        @with_error_context("TestComponent", "test_operation")
        def failing_function(video_path: str, job_id: str = None):
            raise ValueError("Test error")

        with pytest.raises(MontageError) as exc_info:
            failing_function("/test/video.mp4", job_id="job_123")

        error = exc_info.value
        assert error.context.component == "TestComponent"
        assert error.context.operation == "test_operation"
        assert error.context.job_id == "job_123"
        assert error.context.video_path == "/test/video.mp4"

    def test_retry_decorator_with_strategy(self):
        """Test retry decorator respecting retry strategies"""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                # First two calls fail with retryable error
                raise RateLimitError("Rate limited", service="test")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_no_retry(self):
        """Test retry decorator respecting NO_RETRY strategy"""
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0.1)
        def non_retryable_function():
            nonlocal call_count
            call_count += 1
            raise ConfigurationError("Bad config", config_key="test")

        with pytest.raises(ConfigurationError):
            non_retryable_function()

        # Should only be called once (no retries)
        assert call_count == 1

    def test_safe_execute(self):
        """Test safe execution with default values"""
        # Successful execution
        result = safe_execute(lambda: 42, default=0)
        assert result == 42

        # Failed execution returns default
        result = safe_execute(lambda: 1 / 0, default=-1, log_errors=False)
        assert result == -1

        # Critical errors are re-raised
        def critical_error():
            raise AuthenticationError("Auth failed", service="test")

        with pytest.raises(AuthenticationError):
            safe_execute(critical_error, raise_on_critical=True)


class TestErrorRecovery:
    """Test error recovery mechanisms"""

    def test_error_recovery_context(self):
        """Test error recovery with cleanup"""
        cleanup_called = False
        fallback_called = False

        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"

        # Test successful operation
        with error_recovery_context("test_op", cleanup=cleanup, fallback=fallback):
            result = "success"

        assert not cleanup_called
        assert not fallback_called

        # Test failed operation with recovery
        try:
            with error_recovery_context("test_op", cleanup=cleanup, fallback=fallback):
                raise ValueError("Test error")
        except ValueError:
            pass

        assert cleanup_called
        assert fallback_called

    def test_error_collector(self):
        """Test batch error collection"""
        collector = ErrorCollector()

        # Collect some errors
        with collector.collect(item_id=1):
            raise ValueError("Error 1")

        with collector.collect(item_id=2):
            pass  # No error

        with collector.collect(item_id=3):
            raise KeyError("Error 3")

        assert collector.has_errors()
        assert len(collector.get_errors()) == 2

        # Test aggregated error
        with pytest.raises(MontageError) as exc_info:
            collector.raise_if_errors("Batch processing failed")

        error = exc_info.value
        assert "Multiple errors occurred" in error.message
        assert len(error.context.metadata["batch_errors"]) == 2

    def test_aggregate_errors(self):
        """Test error aggregation"""
        errors = [
            VideoDecodeError("Decode failed", video_path="/bad.mp4"),
            DiskSpaceError("No space", required_gb=10, available_gb=5),
            MemoryError("Out of memory", required_mb=8192, available_mb=4096),
        ]

        aggregated = aggregate_errors(errors)

        # Should use most severe error's properties
        assert aggregated.severity == ErrorSeverity.CRITICAL  # From DiskSpaceError
        assert aggregated.category == ErrorCategory.RESOURCE_ERROR
        assert "Multiple errors occurred (3 total)" in aggregated.message
        assert len(aggregated.context.metadata["all_errors"]) == 3


class TestSpecializedErrorHandlers:
    """Test specialized error handlers"""

    def test_handle_ffmpeg_error(self):
        """Test FFmpeg error handling"""
        # Test disk space error detection
        error = handle_ffmpeg_error(
            command=["ffmpeg", "-i", "input.mp4", "-o", "output.mp4"],
            return_code=1,
            stderr="No space left on device",
            operation="Encoding video",
        )

        assert isinstance(error, DiskSpaceError)
        assert error.retry_strategy == RetryStrategy.NO_RETRY

        # Test invalid data error
        error = handle_ffmpeg_error(
            command=["ffmpeg", "-i", "corrupt.mp4"],
            return_code=1,
            stderr="Invalid data found when processing input",
            operation="Decoding video",
        )

        assert isinstance(error, VideoDecodeError)
        assert error.context.video_path == "corrupt.mp4"

        # Test generic FFmpeg error
        error = handle_ffmpeg_error(
            command=["ffmpeg"],
            return_code=255,
            stderr="Unknown error",
            operation="Processing",
        )

        assert isinstance(error, FFmpegError)
        assert error.context.metadata["stderr"] == "Unknown error"

    def test_handle_api_error(self):
        """Test API error handling"""
        # Test rate limit error
        error = handle_api_error(
            service="deepgram",
            error=Exception("Rate limit exceeded"),
            status_code=429,
            response_body='{"error": "rate_limit"}',
        )

        assert isinstance(error, RateLimitError)
        assert error.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert error.context.metadata["service"] == "deepgram"

        # Test authentication error
        error = handle_api_error(
            service="openai", error=Exception("Unauthorized"), status_code=401
        )

        assert isinstance(error, AuthenticationError)
        assert error.retry_strategy == RetryStrategy.NO_RETRY
        assert error.severity == ErrorSeverity.CRITICAL

        # Test server error (should retry)
        error = handle_api_error(
            service="gemini", error=Exception("Internal server error"), status_code=500
        )

        assert isinstance(error, APIError)
        assert error.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF


class TestProductionScenarios:
    """Test real-world production error scenarios"""

    @patch("subprocess.run")
    def test_video_processing_pipeline_errors(self, mock_run):
        """Test error propagation in video processing pipeline"""
        from src.providers.video_processor_v2 import VideoEditor, VideoSegment

        # Simulate FFmpeg not found
        editor = VideoEditor()
        editor.ffmpeg_path = "/nonexistent/ffmpeg"

        segments = [
            VideoSegment(0, 10, "/input.mp4"),
            VideoSegment(20, 30, "/input.mp4"),
        ]

        with pytest.raises(ConfigurationError) as exc_info:
            with patch("shutil.which", return_value=None):
                editor.extract_and_concatenate_efficient(
                    "/input.mp4", segments, "/output.mp4"
                )

        error = exc_info.value
        assert "FFmpeg not found" in error.message
        assert (
            error.suggestions[0]
            == "Install FFmpeg or set FFMPEG_PATH environment variable"
        )

    def test_transcription_fallback_chain(self):
        """Test transcription error handling with fallbacks"""
        from src.core.analyze_video_v2 import _analyze_audio_chunk

        # Mock both transcription engines to fail
        with patch("src.core.analyze_video_v2.transcribe_whisper") as mock_whisper:
            with patch(
                "src.core.analyze_video_v2.transcribe_deepgram"
            ) as mock_deepgram:
                mock_whisper.side_effect = WhisperError(
                    "Whisper failed", audio_path="/test.wav"
                )
                mock_deepgram.side_effect = DeepgramError(
                    "Deepgram failed", status_code=500
                )

                # Should aggregate errors when both fail
                with pytest.raises(MontageError) as exc_info:
                    _analyze_audio_chunk("/test.wav", job_id="test_job")

                error = exc_info.value
                assert "All transcription engines failed" in error.message

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure errors"""
        error = MemoryError(
            "Insufficient memory for video processing",
            required_mb=8192,
            available_mb=2048,
        )

        assert error.retry_strategy == RetryStrategy.LINEAR_BACKOFF
        assert error.severity == ErrorSeverity.HIGH
        assert "Close other applications" in error.suggestions[0]
        assert "smaller video segments" in error.suggestions[1]

    def test_database_connection_resilience(self):
        """Test database error handling without failing operations"""
        from src.core.analyze_video_v2 import get_cached_transcript, cache_transcript

        # Mock database connection to fail
        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError("Connection failed")

            # Should return None instead of raising
            result = get_cached_transcript("test_hash")
            assert result is None

            # Cache should silently fail
            cache_transcript("test_hash", '{"test": "data"}')  # Should not raise

    def test_partial_batch_failure_recovery(self):
        """Test recovery from partial batch failures"""
        from src.providers.video_processor_v2 import VideoEditor, VideoSegment

        editor = VideoEditor()
        segments = [
            VideoSegment(0, 10, "/input.mp4"),
            VideoSegment(20, 30, "/input.mp4"),
            VideoSegment(40, 50, "/input.mp4"),
        ]

        # Mock some segments to fail
        with patch.object(editor, "extract_segments_parallel") as mock_extract:
            # Return mix of valid and invalid FIFOs
            mock_extract.return_value = [
                "/tmp/fifo1",  # exists
                "/tmp/fifo2",  # missing
                "/tmp/fifo3",  # exists
            ]

            with patch("os.path.exists") as mock_exists:
                mock_exists.side_effect = lambda p: p != "/tmp/fifo2"

                with patch.object(editor, "_create_concat_list") as mock_concat:
                    mock_concat.return_value = "/tmp/concat.txt"

                    # Should continue with available segments
                    with patch("subprocess.Popen"):
                        editor.concatenate_segments_fifo(
                            mock_extract.return_value, "/output.mp4"
                        )

                    # Should have been called with only existing FIFOs
                    mock_concat.assert_called_once()
                    call_args = mock_concat.call_args[0][0]
                    assert len(call_args) == 2
                    assert "/tmp/fifo2" not in call_args


class TestErrorMetricsIntegration:
    """Test error integration with metrics and monitoring"""

    def test_error_logging_with_context(self, caplog):
        """Test that errors are logged with full context"""
        import logging

        caplog.set_level(logging.WARNING)

        error = VideoProcessingError("Test processing error").add_context(
            job_id="job_123", video_path="/test.mp4", duration=120.5
        )

        # Error creation triggers logging
        assert len(caplog.records) > 0
        record = caplog.records[-1]

        assert record.levelname == "WARNING"
        assert "Test processing error" in record.message
        assert record.error_type == "VideoProcessingError"
        assert record.context["job_id"] == "job_123"
        assert record.context["metadata"]["duration"] == 120.5

    def test_critical_error_alerting(self, caplog):
        """Test that critical errors trigger appropriate alerts"""
        import logging

        caplog.set_level(logging.CRITICAL)

        error = DiskSpaceError(
            "Critical: No disk space", required_gb=50, available_gb=0.1
        )

        # Should log at CRITICAL level
        critical_records = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert len(critical_records) > 0
        assert "Critical error" in critical_records[0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
