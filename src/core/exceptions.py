#!/usr/bin/env python3
"""
Comprehensive exception hierarchy for Montage video processing pipeline.

This module provides:
- Domain-specific exceptions for video processing
- Error context preservation with exception chaining
- Structured error data with relevant context
- Error classification (transient, permanent, user, system)
- Retry strategy integration
- Production debugging capabilities
"""

import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categorize errors for proper handling and recovery"""

    USER_ERROR = "user_error"  # User input/config errors
    SYSTEM_ERROR = "system_error"  # System/infrastructure errors
    RESOURCE_ERROR = "resource_error"  # Resource availability errors
    API_ERROR = "api_error"  # External API errors
    DATA_ERROR = "data_error"  # Data format/corruption errors
    PROCESSING_ERROR = "processing_error"  # Video processing errors


class ErrorSeverity(Enum):
    """Error severity levels for alerting and monitoring"""

    CRITICAL = "critical"  # Immediate attention required
    HIGH = "high"  # Significant impact on functionality
    MEDIUM = "medium"  # Degraded functionality
    LOW = "low"  # Minor issues


class RetryStrategy(Enum):
    """Retry strategies for different error types"""

    NO_RETRY = "no_retry"  # Error should not be retried
    IMMEDIATE_RETRY = "immediate_retry"  # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff
    LINEAR_BACKOFF = "linear_backoff"  # Linear backoff
    CUSTOM = "custom"  # Custom retry logic


@dataclass
class ErrorContext:
    """Structured error context for debugging and monitoring"""

    job_id: Optional[str] = None
    video_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class MontageError(Exception):
    """Base exception for all Montage errors with rich context"""

    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_strategy: RetryStrategy = RetryStrategy.NO_RETRY

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause
        self.suggestions = suggestions or []

        # Capture stack trace if not provided
        if not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()

        # Chain the cause if provided
        if cause:
            self.__cause__ = cause

        # Log the error with context
        self._log_error()

    def _log_error(self):
        """Log error with full context"""
        log_data = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "retry_strategy": self.retry_strategy.value,
            "context": self.context.to_dict() if self.context else None,
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
        }

        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {self.message}", extra=log_data)
        else:
            logger.warning(f"Error: {self.message}", extra=log_data)

    def add_context(self, **kwargs) -> "MontageError":
        """Add additional context to the error"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value
        return self

    def with_suggestion(self, suggestion: str) -> "MontageError":
        """Add a suggestion for resolving the error"""
        self.suggestions.append(suggestion)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "retry_strategy": self.retry_strategy.value,
            "context": self.context.to_dict() if self.context else None,
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
        }


# Video Processing Errors
class VideoProcessingError(MontageError):
    """Base error for video processing operations"""

    category = ErrorCategory.PROCESSING_ERROR
    severity = ErrorSeverity.HIGH


class VideoDecodeError(VideoProcessingError):
    """Error decoding video file"""

    retry_strategy = RetryStrategy.NO_RETRY

    def __init__(self, message: str, video_path: str, **kwargs):
        super().__init__(message, context=ErrorContext(video_path=video_path, **kwargs))
        self.with_suggestion("Check if the video file is corrupted")
        self.with_suggestion("Verify the video codec is supported")


class VideoEncodeError(VideoProcessingError):
    """Error encoding video file"""

    retry_strategy = RetryStrategy.IMMEDIATE_RETRY

    def __init__(self, message: str, output_path: str, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(metadata={"output_path": output_path}, **kwargs),
        )
        self.with_suggestion("Check available disk space")
        self.with_suggestion("Verify output directory permissions")


class FFmpegError(VideoProcessingError):
    """FFmpeg command execution error"""

    def __init__(self, message: str, command: List[str], stderr: str = "", **kwargs):
        super().__init__(
            message,
            context=ErrorContext(
                metadata={"command": " ".join(command), "stderr": stderr}, **kwargs
            ),
        )
        if "No space left" in stderr:
            self.retry_strategy = RetryStrategy.NO_RETRY
            self.with_suggestion("Free up disk space")
        elif "Permission denied" in stderr:
            self.retry_strategy = RetryStrategy.NO_RETRY
            self.with_suggestion("Check file permissions")
        else:
            self.retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF


# Transcription Errors
class TranscriptionError(MontageError):
    """Base error for transcription operations"""

    category = ErrorCategory.PROCESSING_ERROR
    severity = ErrorSeverity.HIGH


class WhisperError(TranscriptionError):
    """Faster-whisper transcription error"""

    retry_strategy = RetryStrategy.IMMEDIATE_RETRY

    def __init__(self, message: str, audio_path: str, **kwargs):
        super().__init__(
            message, context=ErrorContext(metadata={"audio_path": audio_path}, **kwargs)
        )


class DeepgramError(TranscriptionError):
    """Deepgram API error"""

    category = ErrorCategory.API_ERROR

    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(metadata={"status_code": status_code}, **kwargs),
        )
        if status_code == 429:
            self.retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
            self.with_suggestion("Rate limit exceeded - will retry with backoff")
        elif status_code and 500 <= status_code < 600:
            self.retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
            self.with_suggestion("Server error - will retry")
        else:
            self.retry_strategy = RetryStrategy.NO_RETRY


# Resource Errors
class ResourceError(MontageError):
    """Base error for resource-related issues"""

    category = ErrorCategory.RESOURCE_ERROR
    severity = ErrorSeverity.HIGH


class MemoryError(ResourceError):
    """Insufficient memory error"""

    retry_strategy = RetryStrategy.LINEAR_BACKOFF

    def __init__(self, message: str, required_mb: int, available_mb: int, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(
                metadata={"required_mb": required_mb, "available_mb": available_mb},
                **kwargs,
            ),
        )
        self.with_suggestion("Close other applications to free memory")
        self.with_suggestion("Consider processing smaller video segments")


class DiskSpaceError(ResourceError):
    """Insufficient disk space error"""

    retry_strategy = RetryStrategy.NO_RETRY
    severity = ErrorSeverity.CRITICAL

    def __init__(self, message: str, required_gb: float, available_gb: float, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(
                metadata={"required_gb": required_gb, "available_gb": available_gb},
                **kwargs,
            ),
        )
        self.with_suggestion(
            f"Free at least {required_gb - available_gb:.1f}GB of disk space"
        )


# Database Errors
class DatabaseError(MontageError):
    """Base error for database operations"""

    category = ErrorCategory.SYSTEM_ERROR
    severity = ErrorSeverity.HIGH


class ConnectionError(DatabaseError):
    """Database connection error"""

    retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF

    def __init__(self, message: str, database_url: str = None, **kwargs):
        # Sanitize database URL to avoid leaking credentials
        if database_url:
            import re

            sanitized_url = re.sub(r"://[^@]+@", "://***:***@", database_url)
        else:
            sanitized_url = "unknown"

        super().__init__(
            message,
            context=ErrorContext(metadata={"database_url": sanitized_url}, **kwargs),
        )
        self.with_suggestion("Check database connection settings")
        self.with_suggestion("Verify network connectivity")


class QueryError(DatabaseError):
    """Database query error"""

    def __init__(self, message: str, query: str = None, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(
                metadata={"query": query[:200] if query else None}, **kwargs
            ),
        )


# API Errors
class APIError(MontageError):
    """Base error for external API calls"""

    category = ErrorCategory.API_ERROR
    severity = ErrorSeverity.MEDIUM


class RateLimitError(APIError):
    """API rate limit exceeded"""

    retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF

    def __init__(
        self, message: str, service: str, retry_after: Optional[int] = None, **kwargs
    ):
        super().__init__(
            message,
            context=ErrorContext(
                metadata={"service": service, "retry_after": retry_after}, **kwargs
            ),
        )
        if retry_after:
            self.with_suggestion(f"Wait {retry_after} seconds before retrying")


class AuthenticationError(APIError):
    """API authentication failed"""

    retry_strategy = RetryStrategy.NO_RETRY
    severity = ErrorSeverity.CRITICAL

    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(
            message, context=ErrorContext(metadata={"service": service}, **kwargs)
        )
        self.with_suggestion(f"Check {service} API credentials")
        self.with_suggestion("Verify API key is not expired")


# Configuration Errors
class ConfigurationError(MontageError):
    """Configuration-related errors"""

    category = ErrorCategory.USER_ERROR
    severity = ErrorSeverity.HIGH
    retry_strategy = RetryStrategy.NO_RETRY

    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message, context=ErrorContext(metadata={"config_key": config_key}, **kwargs)
        )
        if config_key:
            self.with_suggestion(f"Check configuration for '{config_key}'")


# Validation Errors
class ValidationError(MontageError):
    """Input validation errors"""

    category = ErrorCategory.USER_ERROR
    severity = ErrorSeverity.LOW
    retry_strategy = RetryStrategy.NO_RETRY

    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(
                metadata={"field": field, "value": str(value)[:100] if value else None},
                **kwargs,
            ),
        )


# Checkpoint Errors
class CheckpointError(MontageError):
    """Checkpoint-related errors"""

    category = ErrorCategory.SYSTEM_ERROR
    severity = ErrorSeverity.MEDIUM

    def __init__(self, message: str, checkpoint_id: str = None, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(metadata={"checkpoint_id": checkpoint_id}, **kwargs),
        )


# Helper functions for error handling
def handle_error(
    error: Exception, context: Optional[ErrorContext] = None
) -> MontageError:
    """Convert any exception to a MontageError with context"""
    if isinstance(error, MontageError):
        if context:
            error.context = context
        return error

    # Map common exceptions to specific MontageErrors
    if isinstance(error, MemoryError):
        return MemoryError(
            f"Memory error: {str(error)}",
            required_mb=0,  # Unknown
            available_mb=0,  # Unknown
            context=context,
        )
    elif isinstance(error, OSError) and "No space left" in str(error):
        return DiskSpaceError(
            f"Disk space error: {str(error)}",
            required_gb=0,  # Unknown
            available_gb=0,  # Unknown
            context=context,
        )
    elif isinstance(error, ConnectionError):
        return ConnectionError(f"Connection error: {str(error)}", context=context)
    else:
        # Generic system error for unknown exceptions
        return MontageError(
            f"Unexpected error: {str(error)}", context=context, cause=error
        )


def create_error_context(
    job_id: Optional[str] = None,
    video_path: Optional[str] = None,
    component: Optional[str] = None,
    operation: Optional[str] = None,
    **metadata,
) -> ErrorContext:
    """Create an error context with the provided information"""
    return ErrorContext(
        job_id=job_id,
        video_path=video_path,
        component=component,
        operation=operation,
        metadata=metadata,
    )
