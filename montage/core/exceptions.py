#!/usr/bin/env python3
"""
Centralized exception handling for Montage video processing pipeline
Provides custom exceptions with proper error codes and user-friendly messages
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import traceback


logger = logging.getLogger(__name__)


class MontageError(Exception):
    """
    Base exception for all Montage errors
    
    Attributes:
        code: Unique error code for tracking
        message: User-friendly error message
        details: Additional error context
        timestamp: When the error occurred
    """
    
    def __init__(
        self, 
        message: str, 
        code: str = "MONTAGE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        
        # Log the error
        logger.error(
            f"{self.code}: {self.message}",
            extra={
                "error_code": self.code,
                "error_details": self.details,
                "error_type": self.__class__.__name__
            }
        )
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp.isoformat(),
                "type": self.__class__.__name__
            }
        }


# Video Processing Errors
class VideoProcessingError(MontageError):
    """Base class for video processing errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VIDEO_PROCESSING_ERROR", details)


class InvalidVideoFormatError(VideoProcessingError):
    """Raised when video format is not supported"""
    
    def __init__(self, format: str, supported_formats: List[str]):
        super().__init__(
            f"Video format '{format}' is not supported",
            details={
                "format": format,
                "supported_formats": supported_formats
            }
        )
        self.code = "INVALID_VIDEO_FORMAT"


class VideoTooLargeError(VideoProcessingError):
    """Raised when video exceeds size limits"""
    
    def __init__(self, size_mb: float, max_size_mb: float):
        super().__init__(
            f"Video size {size_mb:.1f}MB exceeds maximum allowed size of {max_size_mb:.1f}MB",
            details={
                "size_mb": size_mb,
                "max_size_mb": max_size_mb
            }
        )
        self.code = "VIDEO_TOO_LARGE"


class VideoTooLongError(VideoProcessingError):
    """Raised when video duration exceeds limits"""
    
    def __init__(self, duration_seconds: float, max_duration_seconds: float):
        super().__init__(
            f"Video duration {duration_seconds:.1f}s exceeds maximum allowed duration of {max_duration_seconds:.1f}s",
            details={
                "duration_seconds": duration_seconds,
                "max_duration_seconds": max_duration_seconds
            }
        )
        self.code = "VIDEO_TOO_LONG"


class TranscriptionError(VideoProcessingError):
    """Raised when transcription fails"""
    
    def __init__(self, provider: str, reason: str):
        super().__init__(
            f"Transcription failed with {provider}: {reason}",
            details={
                "provider": provider,
                "reason": reason
            }
        )
        self.code = "TRANSCRIPTION_ERROR"


# API and Cost Errors
class APIError(MontageError):
    """Base class for API-related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "API_ERROR", details)


class APIKeyError(APIError):
    """Raised when API key is invalid or missing"""
    
    def __init__(self, provider: str):
        super().__init__(
            f"Invalid or missing API key for {provider}",
            details={"provider": provider}
        )
        self.code = "API_KEY_ERROR"


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, limit_type: str, current: int, limit: int, reset_time: Optional[datetime] = None):
        super().__init__(
            f"Rate limit exceeded: {current}/{limit} {limit_type}",
            details={
                "limit_type": limit_type,
                "current": current,
                "limit": limit,
                "reset_time": reset_time.isoformat() if reset_time else None
            }
        )
        self.code = "RATE_LIMIT_ERROR"


class CostLimitError(APIError):
    """Raised when cost limit is exceeded"""
    
    def __init__(self, current_cost: float, limit: float, provider: Optional[str] = None):
        super().__init__(
            f"Cost limit exceeded: ${current_cost:.2f} / ${limit:.2f}" + 
            (f" for {provider}" if provider else ""),
            details={
                "current_cost": current_cost,
                "limit": limit,
                "provider": provider
            }
        )
        self.code = "COST_LIMIT_ERROR"


# Database Errors
class DatabaseError(MontageError):
    """Base class for database errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details)


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Failed to connect to database: {reason}",
            details={"reason": reason}
        )
        self.code = "DATABASE_CONNECTION_ERROR"


class RecordNotFoundError(DatabaseError):
    """Raised when a database record is not found"""
    
    def __init__(self, entity: str, identifier: str):
        super().__init__(
            f"{entity} not found: {identifier}",
            details={
                "entity": entity,
                "identifier": identifier
            }
        )
        self.code = "RECORD_NOT_FOUND"


# Authentication and Authorization Errors
class AuthError(MontageError):
    """Base class for auth errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTH_ERROR", details)


class AuthenticationError(AuthError):
    """Raised when authentication fails"""
    
    def __init__(self, reason: str):
        super().__init__(
            f"Authentication failed: {reason}",
            details={"reason": reason}
        )
        self.code = "AUTHENTICATION_ERROR"


class AuthorizationError(AuthError):
    """Raised when user lacks required permissions"""
    
    def __init__(self, action: str, resource: str):
        super().__init__(
            f"Not authorized to {action} {resource}",
            details={
                "action": action,
                "resource": resource
            }
        )
        self.code = "AUTHORIZATION_ERROR"


# Configuration Errors
class ConfigurationError(MontageError):
    """Raised when configuration is invalid"""
    
    def __init__(self, setting: str, reason: str):
        super().__init__(
            f"Invalid configuration for {setting}: {reason}",
            details={
                "setting": setting,
                "reason": reason
            }
        )
        self.code = "CONFIGURATION_ERROR"


# Resource Errors
class ResourceError(MontageError):
    """Base class for resource-related errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "RESOURCE_ERROR", details)


class InsufficientResourcesError(ResourceError):
    """Raised when system resources are insufficient"""
    
    def __init__(self, resource: str, available: float, required: float):
        super().__init__(
            f"Insufficient {resource}: {available:.1f} available, {required:.1f} required",
            details={
                "resource": resource,
                "available": available,
                "required": required
            }
        )
        self.code = "INSUFFICIENT_RESOURCES"


class DiskSpaceError(InsufficientResourcesError):
    """Raised when disk space is insufficient"""
    
    def __init__(self, available_gb: float, required_gb: float):
        super().__init__("disk space", available_gb, required_gb)
        self.code = "INSUFFICIENT_DISK_SPACE"


class MemoryError(InsufficientResourcesError):
    """Raised when memory is insufficient"""
    
    def __init__(self, available_gb: float, required_gb: float):
        super().__init__("memory", available_gb, required_gb)
        self.code = "INSUFFICIENT_MEMORY"


# Job and Task Errors
class JobError(MontageError):
    """Base class for job-related errors"""
    
    def __init__(self, message: str, job_id: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["job_id"] = job_id
        super().__init__(message, "JOB_ERROR", details)


class JobNotFoundError(JobError):
    """Raised when a job is not found"""
    
    def __init__(self, job_id: str):
        super().__init__(
            f"Job not found: {job_id}",
            job_id
        )
        self.code = "JOB_NOT_FOUND"


class JobTimeoutError(JobError):
    """Raised when a job times out"""
    
    def __init__(self, job_id: str, timeout_seconds: int):
        super().__init__(
            f"Job timed out after {timeout_seconds} seconds",
            job_id,
            details={"timeout_seconds": timeout_seconds}
        )
        self.code = "JOB_TIMEOUT"


class JobCancelledError(JobError):
    """Raised when a job is cancelled"""
    
    def __init__(self, job_id: str, reason: Optional[str] = None):
        super().__init__(
            f"Job cancelled" + (f": {reason}" if reason else ""),
            job_id,
            details={"reason": reason} if reason else None
        )
        self.code = "JOB_CANCELLED"


# Validation Errors (already exists in security module, but adding here for completeness)
class ValidationError(MontageError):
    """Base class for validation errors"""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation failed for {field}: {reason}",
            code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value)[:100],  # Truncate long values
                "reason": reason
            }
        )


# Error Handler
class ErrorHandler:
    """
    Centralized error handler with recovery strategies
    """
    
    @staticmethod
    def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error and return appropriate response
        
        Args:
            error: The exception to handle
            context: Additional context about where the error occurred
            
        Returns:
            Dictionary with error information suitable for API response
        """
        # Log the full traceback for debugging
        logger.exception(
            "Error occurred",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context
            }
        )
        
        # Handle Montage errors
        if isinstance(error, MontageError):
            return error.to_dict()
        
        # Handle standard Python errors
        error_mapping = {
            FileNotFoundError: ("FILE_NOT_FOUND", "File not found"),
            PermissionError: ("PERMISSION_DENIED", "Permission denied"),
            TimeoutError: ("TIMEOUT", "Operation timed out"),
            MemoryError: ("OUT_OF_MEMORY", "Out of memory"),
            KeyboardInterrupt: ("CANCELLED", "Operation cancelled by user"),
        }
        
        for error_type, (code, default_message) in error_mapping.items():
            if isinstance(error, error_type):
                return {
                    "error": {
                        "code": code,
                        "message": str(error) or default_message,
                        "type": error_type.__name__,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
        
        # Generic error for unhandled exceptions
        return {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "type": type(error).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "message": str(error),
                    "traceback": traceback.format_exc() if context and context.get("debug") else None
                }
            }
        }
    
    @staticmethod
    def get_user_friendly_message(error: Exception) -> str:
        """
        Get a user-friendly error message
        
        Args:
            error: The exception
            
        Returns:
            User-friendly error message
        """
        if isinstance(error, MontageError):
            return error.message
        
        # Map technical errors to user-friendly messages
        user_messages = {
            FileNotFoundError: "The requested file could not be found.",
            PermissionError: "You don't have permission to perform this action.",
            TimeoutError: "The operation took too long to complete. Please try again.",
            ConnectionError: "Unable to connect to the service. Please check your connection.",
        }
        
        for error_type, message in user_messages.items():
            if isinstance(error, error_type):
                return message
        
        return "An error occurred while processing your request. Please try again."


# Export main items
__all__ = [
    # Base
    "MontageError",
    "ErrorHandler",
    
    # Video Processing
    "VideoProcessingError",
    "InvalidVideoFormatError",
    "VideoTooLargeError",
    "VideoTooLongError",
    "TranscriptionError",
    
    # API
    "APIError",
    "APIKeyError",
    "RateLimitError",
    "CostLimitError",
    
    # Database
    "DatabaseError",
    "DatabaseConnectionError",
    "RecordNotFoundError",
    
    # Auth
    "AuthError",
    "AuthenticationError",
    "AuthorizationError",
    
    # Configuration
    "ConfigurationError",
    
    # Resources
    "ResourceError",
    "InsufficientResourcesError",
    "DiskSpaceError",
    "MemoryError",
    
    # Jobs
    "JobError",
    "JobNotFoundError",
    "JobTimeoutError",
    "JobCancelledError",
    
    # Validation
    "ValidationError",
]