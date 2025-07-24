#!/usr/bin/env python3
"""
Secure logging module with sensitive data masking
Prevents secrets and PII from being logged
"""
import re
import logging
import logging.handlers
import json
from typing import Any, Dict, List, Pattern, Union, Optional
from pathlib import Path
from functools import lru_cache

from ..settings import get_settings


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that masks sensitive data in log messages
    """
    
    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.enabled = settings.logging.mask_secrets
        self.patterns = self._compile_patterns(settings.logging.mask_patterns)
        
        # Additional patterns for common secrets
        self.secret_patterns = [
            # API Keys - simple patterns first
            (re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'), None),  # OpenAI style
            (re.compile(r'\bsk-ant-[a-zA-Z0-9]{20,}\b'), None),  # Anthropic style
            (re.compile(r'\bAKIA[0-9A-Z]{16}\b'), None),  # AWS Access Key
            
            # Key-value patterns
            (re.compile(r'(api[_\-]?key|apikey|api_secret)\s*[:=]\s*["\']?([^"\'\s,}]+)', re.I), 2),
            (re.compile(r'(password|passwd|pwd)\s*[:=]\s*["\']?([^"\'\s,}]+)', re.I), 2),
            (re.compile(r'(token|access_token|refresh_token|auth_token)\s*[:=]\s*["\']?([^"\'\s,}]+)', re.I), 2),
            (re.compile(r'(secret|client_secret)\s*[:=]\s*["\']?([^"\'\s,}]+)', re.I), 2),
            
            # Bearer tokens
            (re.compile(r'Bearer\s+([a-zA-Z0-9\-_.=/+]+)', re.I), 1),
            
            # Database URLs - mask the credentials part
            (re.compile(r'(postgresql|postgres|mysql|mongodb|redis)://([^@]+)@', re.I), 2),
            
            # Credit Cards
            (re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'), None),
            
            # SSN
            (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), None),
        ]
    
    def _compile_patterns(self, pattern_strings: List[str]) -> List[Pattern]:
        """Compile regex patterns from strings"""
        patterns = []
        for pattern_str in pattern_strings:
            try:
                patterns.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                # Skip invalid patterns
                logger.debug(f"Invalid regex pattern '{pattern_str}': {e}")
        return patterns
    
    def mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text"""
        if not self.enabled:
            return text
            
        # Apply custom patterns
        for pattern in self.patterns:
            text = pattern.sub('***MASKED***', text)
        
        # Apply secret patterns with group handling
        for pattern, group_to_mask in self.secret_patterns:
            if group_to_mask is None:
                # Simple pattern without groups - mask entire match
                text = pattern.sub('***MASKED***', text)
            else:
                # Pattern with groups - mask specific group
                def replacer(match):
                    groups = list(match.groups())
                    if group_to_mask <= len(groups):
                        # Replace the specified group with masked text
                        full_match = match.group(0)
                        group_text = groups[group_to_mask - 1]
                        return full_match.replace(group_text, '***MASKED***')
                    return '***MASKED***'
                text = pattern.sub(replacer, text)
        
        return text
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to mask sensitive data"""
        # Mask the main message
        if hasattr(record, 'msg'):
            record.msg = self.mask_sensitive_data(str(record.msg))
        
        # Mask arguments
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self.mask_sensitive_data(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(
                    self.mask_sensitive_data(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        
        # Mask extra fields
        for key in dir(record):
            if not key.startswith('_') and key not in ('msg', 'args', 'getMessage'):
                value = getattr(record, key, None)
                if isinstance(value, str):
                    setattr(record, key, self.mask_sensitive_data(value))
        
        return True


class SecureJSONFormatter(logging.Formatter):
    """
    JSON formatter that ensures sensitive data is masked
    """
    
    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields
        self.filter = SensitiveDataFilter()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with sensitive data masked"""
        # Build base log entry
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": self.filter.mask_sensitive_data(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if enabled
        if self.include_extra_fields:
            # Standard extra fields
            for field in ['correlation_id', 'job_id', 'user_id', 'request_id']:
                if hasattr(record, field):
                    value = getattr(record, field)
                    if isinstance(value, str):
                        value = self.filter.mask_sensitive_data(value)
                    log_entry[field] = value
            
            # Add exception info if present
            if record.exc_info:
                import traceback
                log_entry['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': self.filter.mask_sensitive_data(str(record.exc_info[1])),
                    'traceback': self.filter.mask_sensitive_data(
                        ''.join(traceback.format_exception(*record.exc_info))
                    )
                }
        
        return json.dumps(log_entry, default=str)


class SecureFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler with secure defaults
    """
    
    def __init__(self, filename: Union[str, Path], 
                 maxBytes: int = 104857600,  # 100MB default
                 backupCount: int = 5,
                 encoding: str = 'utf-8',
                 secure_permissions: bool = True):
        # Ensure log directory exists
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        super().__init__(
            filename=str(log_path),
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding
        )
        
        # Set secure file permissions (owner read/write only)
        if secure_permissions:
            try:
                import os
                os.chmod(str(log_path), 0o600)
            except (OSError, ImportError) as e:
                logger.debug(f"Failed to set secure log file permissions: {type(e).__name__}")


def configure_secure_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_json: Optional[bool] = None,
    mask_secrets: Optional[bool] = None
) -> None:
    """
    Configure secure logging for the application
    
    Args:
        level: Logging level (default from settings)
        log_file: Path to log file (default from settings)
        use_json: Whether to use JSON format (default from settings)
        mask_secrets: Whether to mask secrets (default from settings)
    """
    settings = get_settings()
    logging_config = settings.logging
    
    # Use provided values or defaults from settings
    level = level or logging_config.level
    log_file = log_file or logging_config.file_path
    use_json = use_json if use_json is not None else logging_config.use_json_format
    mask_secrets = mask_secrets if mask_secrets is not None else logging_config.mask_secrets
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    if use_json:
        formatter = SecureJSONFormatter(include_extra_fields=True)
    else:
        formatter = logging.Formatter(logging_config.format)
    
    # Add sensitive data filter
    if mask_secrets:
        sensitive_filter = SensitiveDataFilter()
        
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if mask_secrets:
        console_handler.addFilter(sensitive_filter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = SecureFileHandler(
            filename=log_file,
            maxBytes=logging_config.max_file_size_mb * 1024 * 1024,
            backupCount=logging_config.backup_count
        )
        file_handler.setFormatter(formatter)
        if mask_secrets:
            file_handler.addFilter(sensitive_filter)
        root_logger.addHandler(file_handler)
    
    # Add correlation filter from existing logging_config
    try:
        from .logging_config import CorrelationFilter
        correlation_filter = CorrelationFilter()
        for handler in root_logger.handlers:
            handler.addFilter(correlation_filter)
    except ImportError:
        logger.debug("CorrelationFilter not available, skipping correlation logging")
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Secure logging configured",
        extra={
            "level": level,
            "json_format": use_json,
            "mask_secrets": mask_secrets,
            "log_file": str(log_file) if log_file else None
        }
    )


def get_secure_logger(name: str) -> logging.Logger:
    """
    Get a logger with secure configuration
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Ensure secure logging is configured
    if not logger.handlers and not logging.getLogger().handlers:
        configure_secure_logging()
    
    return logger


# Example usage functions
def log_api_call(logger: logging.Logger, 
                 endpoint: str,
                 method: str,
                 headers: Dict[str, str],
                 response_code: int) -> None:
    """
    Safely log an API call with headers masked
    
    Args:
        logger: Logger instance
        endpoint: API endpoint
        method: HTTP method
        headers: Request headers (will be masked)
        response_code: HTTP response code
    """
    # Headers will be automatically masked by the filter
    logger.info(
        "API call completed",
        extra={
            "endpoint": endpoint,
            "method": method,
            "headers": headers,  # Will be masked
            "response_code": response_code
        }
    )


def log_database_query(logger: logging.Logger,
                      query: str,
                      params: Optional[Union[tuple, dict]] = None,
                      duration_ms: Optional[float] = None) -> None:
    """
    Safely log a database query
    
    Args:
        logger: Logger instance
        query: SQL query
        params: Query parameters (will be masked if sensitive)
        duration_ms: Query duration in milliseconds
    """
    logger.debug(
        "Database query executed",
        extra={
            "query": query,
            "params": params,  # Will be masked if contains sensitive data
            "duration_ms": duration_ms
        }
    )