#!/usr/bin/env python3
"""
Test secure logging functionality
"""
import pytest
import logging
import json
from io import StringIO

from montage.utils.secure_logging import (
    SensitiveDataFilter,
    SecureJSONFormatter,
    configure_secure_logging,
    get_secure_logger,
)


class TestSensitiveDataFilter:
    """Test sensitive data masking"""
    
    def test_api_key_masking(self):
        """Test API key patterns are masked"""
        filter = SensitiveDataFilter()
        
        # OpenAI style keys
        assert filter.mask_sensitive_data("sk-1234567890abcdef1234567890abcdef") == "***MASKED***"
        assert "***MASKED***" in filter.mask_sensitive_data("My API key is sk-abc123def456ghi789jkl012mno345")
        
        # Anthropic style keys
        assert "***MASKED***" in filter.mask_sensitive_data("Using key: sk-ant-1234567890abcdef1234567890")
        
        # Key-value pairs
        assert filter.mask_sensitive_data("api_key=secret123") == "api_key=***MASKED***"
        assert filter.mask_sensitive_data("api-key: 'my-secret-key'") == "api-key: '***MASKED***'"
    
    def test_password_masking(self):
        """Test password patterns are masked"""
        filter = SensitiveDataFilter()
        
        assert filter.mask_sensitive_data("password=mysecret123") == "password=***MASKED***"
        assert filter.mask_sensitive_data("pwd: supersecret") == "pwd: ***MASKED***"
        assert filter.mask_sensitive_data('{"password": "hidden"}') == '{"password": "***MASKED***"}'
    
    def test_token_masking(self):
        """Test token patterns are masked"""
        filter = SensitiveDataFilter()
        
        assert filter.mask_sensitive_data("Bearer abc123def456") == "Bearer ***MASKED***"
        assert filter.mask_sensitive_data("token=xyz789") == "token=***MASKED***"
        assert filter.mask_sensitive_data("refresh_token: 'abc-123-def'") == "refresh_token: '***MASKED***'"
    
    def test_database_url_masking(self):
        """Test database URL credentials are masked"""
        filter = SensitiveDataFilter()
        
        masked = filter.mask_sensitive_data("postgresql://user:pass@localhost:5432/db")
        assert masked == "postgresql://***MASKED***@localhost:5432/db"
        
        masked = filter.mask_sensitive_data("mongodb://admin:secret@mongo.local:27017")
        assert masked == "mongodb://***MASKED***@mongo.local:27017"
    
    def test_credit_card_masking(self):
        """Test credit card numbers are masked"""
        filter = SensitiveDataFilter()
        
        assert filter.mask_sensitive_data("Card: 4111-1111-1111-1111") == "Card: ***MASKED***"
        assert filter.mask_sensitive_data("Payment: 5555 5555 5555 4444") == "Payment: ***MASKED***"
        assert filter.mask_sensitive_data("4242424242424242") == "***MASKED***"
    
    def test_aws_key_masking(self):
        """Test AWS keys are masked"""
        filter = SensitiveDataFilter()
        
        assert filter.mask_sensitive_data("AKIAIOSFODNN7EXAMPLE") == "***MASKED***"
        assert filter.mask_sensitive_data("AWS Key: AKIAJ1234567890ABCD") == "AWS Key: ***MASKED***"
    
    def test_mixed_content_masking(self):
        """Test masking in mixed content"""
        filter = SensitiveDataFilter()
        
        text = """
        Configuration:
        - API Key: sk-1234567890abcdef
        - Password: mysecret123
        - Database: postgresql://user:pass@localhost/db
        - Token: Bearer abc123
        """
        
        masked = filter.mask_sensitive_data(text)
        assert "sk-1234567890abcdef" not in masked
        assert "mysecret123" not in masked
        assert "user:pass" not in masked
        assert "abc123" not in masked
        assert "***MASKED***" in masked
    
    def test_masking_disabled(self):
        """Test that masking can be disabled"""
        filter = SensitiveDataFilter()
        filter.enabled = False
        
        sensitive = "password=secret123"
        assert filter.mask_sensitive_data(sensitive) == sensitive


class TestSecureJSONFormatter:
    """Test JSON formatter with masking"""
    
    def test_json_format_with_masking(self):
        """Test JSON formatter masks sensitive data"""
        formatter = SecureJSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="API call with key: %s",
            args=("sk-secret123",),
            exc_info=None
        )
        
        # Format the record
        json_output = formatter.format(record)
        data = json.loads(json_output)
        
        # Check that sensitive data is masked
        assert "sk-secret123" not in data["message"]
        assert "***MASKED***" in data["message"]
    
    def test_json_format_extra_fields(self):
        """Test JSON formatter handles extra fields"""
        formatter = SecureJSONFormatter(include_extra_fields=True)
        
        # Create a log record with extra fields
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="User action",
            args=None,
            exc_info=None
        )
        
        # Add extra fields
        record.user_id = "user123"
        record.api_key = "sk-secret456"
        record.correlation_id = "corr-123"
        
        # Format the record
        json_output = formatter.format(record)
        data = json.loads(json_output)
        
        # Check fields are included and sensitive ones are masked
        assert data["user_id"] == "user123"
        assert "sk-secret456" not in json_output
        assert data["correlation_id"] == "corr-123"


class TestSecureLoggingIntegration:
    """Test secure logging configuration"""
    
    def test_configure_secure_logging(self, tmp_path):
        """Test secure logging configuration"""
        log_file = tmp_path / "test.log"
        
        # Configure logging
        configure_secure_logging(
            level="INFO",
            log_file=log_file,
            use_json=True,
            mask_secrets=True
        )
        
        # Get a logger and log sensitive data
        logger = get_secure_logger("test")
        logger.info("Password is: %s", "mysecret123")
        logger.warning("API key: sk-1234567890")
        
        # Check log file exists
        assert log_file.exists()
        
        # Read log file and verify masking
        log_content = log_file.read_text()
        assert "mysecret123" not in log_content
        assert "sk-1234567890" not in log_content
        assert "***MASKED***" in log_content
    
    def test_log_record_filtering(self):
        """Test that log records are filtered properly"""
        # Create a logger with handler
        logger = logging.getLogger("test.filtering")
        logger.setLevel(logging.DEBUG)
        
        # Add string handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        # Add sensitive data filter
        filter = SensitiveDataFilter()
        handler.addFilter(filter)
        logger.addHandler(handler)
        
        # Log sensitive data
        logger.info("Database URL: postgresql://admin:password@localhost/db")
        
        # Check output
        output = stream.getvalue()
        assert "admin:password" not in output
        assert "***MASKED***" in output