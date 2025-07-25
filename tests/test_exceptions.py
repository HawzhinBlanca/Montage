#!/usr/bin/env python3
"""
Test exception handling
"""
import pytest
from datetime import datetime

from montage.core.exceptions import (
    MontageError,
    ErrorHandler,
    VideoTooLargeError,
    APIKeyError,
    RateLimitError,
    JobNotFoundError,
    AuthenticationError,
    ValidationError,
)


class TestMontageError:
    """Test base Montage error"""
    
    def test_montage_error_creation(self):
        """Test creating a Montage error"""
        error = MontageError(
            message="Test error",
            code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert isinstance(error.timestamp, datetime)
    
    def test_montage_error_to_dict(self):
        """Test converting error to dictionary"""
        error = MontageError("Test error", "TEST_CODE")
        result = error.to_dict()
        
        assert result["error"]["code"] == "TEST_CODE"
        assert result["error"]["message"] == "Test error"
        assert result["error"]["type"] == "MontageError"
        assert "timestamp" in result["error"]


class TestSpecificErrors:
    """Test specific error types"""
    
    def test_video_too_large_error(self):
        """Test video size error"""
        error = VideoTooLargeError(size_mb=150.5, max_size_mb=100.0)
        
        assert "150.5MB exceeds maximum allowed size of 100.0MB" in error.message
        assert error.code == "VIDEO_TOO_LARGE"
        assert error.details["size_mb"] == 150.5
        assert error.details["max_size_mb"] == 100.0
    
    def test_api_key_error(self):
        """Test API key error"""
        error = APIKeyError("OpenAI")
        
        assert "Invalid or missing API key for OpenAI" in error.message
        assert error.code == "API_KEY_ERROR"
        assert error.details["provider"] == "OpenAI"
    
    def test_rate_limit_error(self):
        """Test rate limit error"""
        reset_time = datetime.utcnow()
        error = RateLimitError(
            limit_type="requests",
            current=100,
            limit=60,
            reset_time=reset_time
        )
        
        assert "Rate limit exceeded: 100/60 requests" in error.message
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.details["current"] == 100
        assert error.details["limit"] == 60
        assert error.details["reset_time"] == reset_time.isoformat()
    
    def test_job_not_found_error(self):
        """Test job not found error"""
        error = JobNotFoundError("job-123")
        
        assert "Job not found: job-123" in error.message
        assert error.code == "JOB_NOT_FOUND"
        assert error.details["job_id"] == "job-123"
    
    def test_authentication_error(self):
        """Test authentication error"""
        error = AuthenticationError("Invalid API key")
        
        assert "Authentication failed: Invalid API key" in error.message
        assert error.code == "AUTHENTICATION_ERROR"
        assert error.details["reason"] == "Invalid API key"
    
    def test_validation_error(self):
        """Test validation error"""
        error = ValidationError(
            field="email",
            value="invalid-email",
            reason="Invalid email format"
        )
        
        assert "Validation failed for email: Invalid email format" in error.message
        assert error.code == "VALIDATION_ERROR"
        assert error.details["field"] == "email"
        assert error.details["value"] == "invalid-email"
        assert error.details["reason"] == "Invalid email format"


class TestErrorHandler:
    """Test error handler"""
    
    def test_handle_montage_error(self):
        """Test handling Montage errors"""
        error = VideoTooLargeError(100, 50)
        result = ErrorHandler.handle_error(error)
        
        assert result["error"]["code"] == "VIDEO_TOO_LARGE"
        assert result["error"]["type"] == "VideoTooLargeError"
        assert "100.0MB exceeds" in result["error"]["message"]
    
    def test_handle_standard_error(self):
        """Test handling standard Python errors"""
        error = FileNotFoundError("test.txt not found")
        result = ErrorHandler.handle_error(error)
        
        assert result["error"]["code"] == "FILE_NOT_FOUND"
        assert result["error"]["type"] == "FileNotFoundError"
        assert "test.txt not found" in result["error"]["message"]
    
    def test_handle_unknown_error(self):
        """Test handling unknown errors"""
        error = RuntimeError("Something went wrong")
        result = ErrorHandler.handle_error(error)
        
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["error"]["type"] == "RuntimeError"
        assert result["error"]["message"] == "An unexpected error occurred"
        assert result["error"]["details"]["message"] == "Something went wrong"
    
    def test_handle_error_with_debug_context(self):
        """Test error handling with debug context"""
        error = ValueError("Test error")
        result = ErrorHandler.handle_error(
            error, 
            context={"debug": True, "path": "/test"}
        )
        
        assert "traceback" in result["error"]["details"]
        assert result["error"]["details"]["traceback"] is not None
    
    def test_get_user_friendly_message_montage_error(self):
        """Test getting user-friendly message for Montage error"""
        error = VideoTooLargeError(100, 50)
        message = ErrorHandler.get_user_friendly_message(error)
        
        assert "100.0MB exceeds maximum allowed size of 50.0MB" in message
    
    def test_get_user_friendly_message_standard_error(self):
        """Test getting user-friendly message for standard error"""
        error = PermissionError("Access denied")
        message = ErrorHandler.get_user_friendly_message(error)
        
        assert message == "You don't have permission to perform this action."
    
    def test_get_user_friendly_message_unknown_error(self):
        """Test getting user-friendly message for unknown error"""
        error = RuntimeError("Complex technical error")
        message = ErrorHandler.get_user_friendly_message(error)
        
        assert message == "An error occurred while processing your request. Please try again."