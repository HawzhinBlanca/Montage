#!/usr/bin/env python3
"""
Custom exception hierarchy for Montage
Provides structured error handling with proper context
"""
from typing import Optional, Dict, Any


class MontageError(Exception):
    """Base exception for all Montage errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(MontageError):
    """Raised when configuration is invalid or missing"""

    pass


class SecretError(ConfigurationError):
    """Raised when secrets/API keys are missing or invalid"""

    pass


class ValidationError(MontageError):
    """Raised when input validation fails"""

    pass


class VideoProcessingError(MontageError):
    """Base class for video processing errors"""

    pass


class TranscriptionError(VideoProcessingError):
    """Raised when transcription fails"""

    pass


class AnalysisError(VideoProcessingError):
    """Raised when video analysis fails"""

    pass


class BudgetExceededError(MontageError):
    """Raised when API cost budget is exceeded"""

    def __init__(
        self, spent: float, limit: float, details: Optional[Dict[str, Any]] = None
    ):
        message = f"Budget exceeded: ${spent:.2f} spent, limit was ${limit:.2f}"
        super().__init__(message, details)
        self.spent = spent
        self.limit = limit


class ExternalServiceError(MontageError):
    """Base class for external service errors"""

    def __init__(
        self, service: str, message: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"{service}: {message}", details)
        self.service = service


class OpenAIError(ExternalServiceError):
    """Raised when OpenAI API fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("OpenAI", message, details)


class AnthropicError(ExternalServiceError):
    """Raised when Anthropic API fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("Anthropic", message, details)


class DeepgramError(ExternalServiceError):
    """Raised when Deepgram API fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("Deepgram", message, details)


class DatabaseError(MontageError):
    """Raised when database operations fail"""

    pass


class FileSystemError(MontageError):
    """Raised when file system operations fail"""

    def __init__(
        self, path: str, operation: str, details: Optional[Dict[str, Any]] = None
    ):
        message = f"File system error during {operation} on {path}"
        super().__init__(message, details)
        self.path = path
        self.operation = operation


class MCPServerError(MontageError):
    """Raised when MCP server operations fail"""

    pass


# Error handling utilities
def handle_error(error: Exception, context: str = "") -> MontageError:
    """
    Convert generic exceptions to MontageError hierarchy

    Args:
        error: The original exception
        context: Additional context about where the error occurred

    Returns:
        MontageError subclass instance
    """
    # Already a MontageError
    if isinstance(error, MontageError):
        return error

    # Map common exceptions
    error_message = str(error)
    details = {"original_type": type(error).__name__, "context": context}

    if "api key" in error_message.lower() or "authentication" in error_message.lower():
        return SecretError(error_message, details)

    if (
        "file not found" in error_message.lower()
        or "permission denied" in error_message.lower()
    ):
        return FileSystemError("unknown", "access", details)

    if "connection" in error_message.lower() or "timeout" in error_message.lower():
        return ExternalServiceError("Unknown", error_message, details)

    # Generic error
    return MontageError(
        f"{context}: {error_message}" if context else error_message, details
    )
