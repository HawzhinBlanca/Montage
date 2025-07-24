#!/usr/bin/env python3
"""
Test security controls
"""
import pytest
from pathlib import Path

from montage.core.security import (
    sanitize_path,
    validate_filename,
    validate_identifier,
    sanitize_value,
    validate_job_id,
    validate_email,
    validate_url,
    sanitize_user_input,
    generate_token,
    PathTraversalError,
    SQLInjectionError,
    InputValidationError,
)


class TestPathSecurity:
    """Test path sanitization and validation"""
    
    def test_valid_paths(self):
        """Test that valid paths are accepted"""
        valid_path = "/tmp/test.mp4"
        result = sanitize_path(valid_path)
        assert isinstance(result, Path)
        assert str(result).endswith("test.mp4")
    
    def test_path_traversal_detection(self):
        """Test that path traversal attempts are blocked"""
        dangerous_paths = [
            "../../../etc/passwd",
            "~/../../root/.ssh/id_rsa",
            "/tmp/../../../etc/shadow",
            "video.mp4;rm -rf /",
            "video$(whoami).mp4",
        ]
        
        for path in dangerous_paths:
            with pytest.raises(PathTraversalError):
                sanitize_path(path)
    
    def test_filename_validation(self):
        """Test filename sanitization"""
        # Valid filenames
        assert validate_filename("video.mp4") == "video.mp4"
        assert validate_filename("my-video_123.mov") == "my-video_123.mov"
        
        # Invalid filenames
        with pytest.raises(InputValidationError):
            validate_filename("")
        with pytest.raises(InputValidationError):
            validate_filename(".")
        with pytest.raises(InputValidationError):
            validate_filename("..")
        
        # Sanitized filenames
        assert validate_filename("video;rm -rf /.mp4") == ".mp4"
        assert validate_filename("../../../etc/passwd") == "passwd"


class TestSQLInjectionPrevention:
    """Test SQL injection prevention"""
    
    def test_table_validation(self):
        """Test table name validation"""
        # Valid tables
        assert validate_identifier("job", "table") == "job"
        assert validate_identifier("video_job", "table") == "video_job"
        
        # Invalid tables
        with pytest.raises(SQLInjectionError):
            validate_identifier("users; DROP TABLE users;--", "table")
        with pytest.raises(SQLInjectionError):
            validate_identifier("users' OR '1'='1", "table")
    
    def test_column_validation(self):
        """Test column name validation"""
        # Valid columns
        assert validate_identifier("id", "column") == "id"
        assert validate_identifier("user_id", "column") == "user_id"
        
        # Invalid columns
        with pytest.raises(SQLInjectionError):
            validate_identifier("id; DROP TABLE users", "column")
    
    def test_value_sanitization(self):
        """Test value sanitization"""
        # Normal values should pass
        assert sanitize_value("normal text") == "normal text"
        assert sanitize_value(123) == 123
        assert sanitize_value(None) is None
        
        # SQL injection attempts should be caught
        with pytest.raises(SQLInjectionError):
            sanitize_value("'; DROP TABLE users;--")
        with pytest.raises(SQLInjectionError):
            sanitize_value("admin'--")
        with pytest.raises(SQLInjectionError):
            sanitize_value("1; DELETE FROM users WHERE 1=1")


class TestInputValidation:
    """Test general input validation"""
    
    def test_job_id_validation(self):
        """Test job ID validation"""
        # Valid job IDs
        assert validate_job_id("abc123") == "abc123"
        assert validate_job_id("job_123") == "job_123"
        assert validate_job_id("12345-abcde-67890") == "12345-abcde-67890"
        
        # Invalid job IDs
        with pytest.raises(InputValidationError):
            validate_job_id("job;rm -rf /")
        with pytest.raises(InputValidationError):
            validate_job_id("job' OR '1'='1")
        with pytest.raises(InputValidationError):
            validate_job_id("../../../etc/passwd")
    
    def test_email_validation(self):
        """Test email validation"""
        # Valid emails
        assert validate_email("user@example.com") == "user@example.com"
        assert validate_email("Test.User@Example.Com") == "test.user@example.com"
        
        # Invalid emails
        with pytest.raises(InputValidationError):
            validate_email("not-an-email")
        with pytest.raises(InputValidationError):
            validate_email("@example.com")
        with pytest.raises(InputValidationError):
            validate_email("user@")
    
    def test_url_validation(self):
        """Test URL validation"""
        # Valid URLs
        assert validate_url("https://example.com") == "https://example.com"
        assert validate_url("http://api.service.com/endpoint") == "http://api.service.com/endpoint"
        
        # Invalid URLs
        with pytest.raises(InputValidationError):
            validate_url("javascript:alert('xss')")
        with pytest.raises(InputValidationError):
            validate_url("file:///etc/passwd")
    
    def test_user_input_sanitization(self):
        """Test user input sanitization"""
        # Normal text
        assert sanitize_user_input("Hello world!") == "Hello world!"
        
        # Remove null bytes
        assert sanitize_user_input("Text with \x00 null bytes") == "Text with null bytes"
        
        # Normalize whitespace
        assert sanitize_user_input("Too    many     spaces") == "Too many spaces"
        
        # Truncate long text
        long_text = "x" * 200
        result = sanitize_user_input(long_text, max_length=100)
        assert len(result) == 100


class TestTokenGeneration:
    """Test secure token generation"""
    
    def test_token_generation(self):
        """Test that tokens are unique and secure"""
        tokens = set()
        for _ in range(100):
            token = generate_token()
            assert len(token) >= 32
            assert token not in tokens
            tokens.add(token)
    
    def test_token_length(self):
        """Test custom token lengths"""
        short_token = generate_token(16)
        long_token = generate_token(64)
        
        # URL-safe base64 encoding makes tokens longer than requested bytes
        assert len(short_token) >= 16
        assert len(long_token) >= 64