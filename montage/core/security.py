#!/usr/bin/env python3
"""
Centralized security module for path sanitization, SQL injection prevention, and input validation
"""
import os
import re
import hashlib
import secrets
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from urllib.parse import urlparse

from ..settings import get_settings

# Security settings
settings = get_settings()
security_config = settings.security


class SecurityError(Exception):
    """Base exception for security violations"""
    def __init__(self, message="Security violation detected"):
        super().__init__(message)
        self.message = message


class PathTraversalError(SecurityError):
    """Path traversal attempt detected"""
    def __init__(self, path="", message=None):
        if message is None:
            message = f"Path traversal attempt detected: {path}"
        super().__init__(message)
        self.path = path


class SQLInjectionError(SecurityError):
    """SQL injection attempt detected"""
    def __init__(self, query="", message=None):
        if message is None:
            message = f"SQL injection attempt detected in query: {query[:100]}..."
        super().__init__(message)
        self.query = query


class InputValidationError(SecurityError):
    """Input validation failed"""
    def __init__(self, field="", value="", message=None):
        if message is None:
            message = f"Input validation failed for field '{field}' with value '{str(value)[:50]}...'"
        super().__init__(message)
        self.field = field
        self.value = value


class PathSanitizer:
    """Secure path handling and validation"""
    
    def __init__(self):
        # Get allowed paths from config
        allowed_paths = list(security_config.allowed_upload_paths)
        
        # Always include temp directory for processing
        allowed_paths.append(tempfile.gettempdir())
        
        self.allowed_bases = [
            Path(base).resolve() for base in allowed_paths
        ]
        self.max_depth = security_config.max_path_depth
        
    def sanitize_path(self, path: Union[str, Path]) -> Path:
        """
        Sanitize and validate a file path
        
        Args:
            path: Path to sanitize
            
        Returns:
            Sanitized absolute path
            
        Raises:
            PathTraversalError: If path traversal detected
        """
        if not security_config.enable_path_sanitization:
            return Path(path).resolve()
            
        try:
            # Convert to Path and resolve
            clean_path = Path(path).resolve()
            
            # Check for path traversal patterns
            path_str = str(path)
            dangerous_patterns = [
                "..",
                "~",
                "${",
                "$(", 
                "`",
                ";",
                "|",
                "&",
                "\x00",  # Null byte
                "\n",
                "\r",
            ]
            
            for pattern in dangerous_patterns:
                if pattern in path_str:
                    raise PathTraversalError(
                        f"Dangerous pattern '{pattern}' detected in path"
                    )
            
            # Check path depth
            parts = clean_path.parts
            if len(parts) > self.max_depth:
                raise PathTraversalError(
                    f"Path too deep: {len(parts)} > {self.max_depth}"
                )
            
            # Verify path is within allowed directories
            path_allowed = False
            for base in self.allowed_bases:
                try:
                    clean_path.relative_to(base)
                    path_allowed = True
                    break
                except ValueError:
                    continue
                    
            if not path_allowed:
                raise PathTraversalError(
                    f"Path outside allowed directories: {clean_path}"
                )
            
            return clean_path
            
        except (OSError, ValueError, AttributeError) as e:
            raise PathTraversalError(f"Invalid path: {e}")
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize a filename
        
        Args:
            filename: Filename to validate
            
        Returns:
            Sanitized filename
            
        Raises:
            InputValidationError: If filename is invalid
        """
        # Remove directory separators
        filename = os.path.basename(filename)
        
        # Check for empty or special names
        if not filename or filename in (".", "..", "~"):
            raise InputValidationError("Invalid filename")
        
        # Remove dangerous characters
        safe_chars = re.compile(r'[^a-zA-Z0-9._\-]')
        clean_name = safe_chars.sub('_', filename)
        
        # Ensure extension is preserved
        name, ext = os.path.splitext(filename)
        if ext and not clean_name.endswith(ext):
            clean_name = safe_chars.sub('_', name) + ext
            
        # Limit length
        if len(clean_name) > 255:
            name, ext = os.path.splitext(clean_name)
            clean_name = name[:255-len(ext)] + ext
            
        return clean_name


class SQLSanitizer:
    """SQL injection prevention utilities"""
    
    # Whitelist of allowed table names
    ALLOWED_TABLES = {
        "video_job",
        "api_cost_log", 
        "job_checkpoint",
        "transcript_cache",
        "highlight",
        "alert_log",
        "job",
        "checkpoint",
        "cost",
        "user",
        "session",
    }
    
    # Whitelist of allowed column names
    ALLOWED_COLUMNS = {
        "id", "user_id", "job_id", "video_id", "status", "created_at", 
        "updated_at", "data", "cost", "provider", "model", "tokens",
        "duration", "error", "result", "config", "metadata", "path",
        "filename", "size", "mime_type", "hash", "transcript", "highlights",
        "start_time", "end_time", "score", "title", "description",
        # Additional columns found in codebase
        "src_hash", "codec", "stage", "error_message", "checkpoint",
        "completed", "timestamp", "message", "level", "name", "value",
        "expires_at", "budget", "used", "priority", "retry_count",
    }
    
    # SQL keywords that should never appear in user input
    DANGEROUS_KEYWORDS = {
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "GRANT", 
        "REVOKE", "EXEC", "EXECUTE", "UNION", "SELECT", "INSERT",
        "UPDATE", "MERGE", "--", "/*", "*/", "xp_", "sp_"
    }
    
    @classmethod
    def validate_identifier(
        cls, 
        identifier: str, 
        identifier_type: str = "table"
    ) -> str:
        """
        Validate a SQL identifier (table/column name)
        
        Args:
            identifier: The identifier to validate
            identifier_type: 'table' or 'column'
            
        Returns:
            Validated identifier
            
        Raises:
            SQLInjectionError: If identifier is invalid
        """
        if not security_config.enable_sql_injection_protection:
            return identifier
            
        # Check if empty
        if not identifier:
            raise SQLInjectionError(f"Empty {identifier_type} name")
            
        # Convert to lowercase for comparison
        identifier_lower = identifier.lower()
        
        # Check against whitelist
        if identifier_type == "table":
            if identifier_lower not in cls.ALLOWED_TABLES:
                raise SQLInjectionError(
                    f"Table '{identifier}' not in allowed list"
                )
        elif identifier_type == "column":
            if identifier_lower not in cls.ALLOWED_COLUMNS:
                raise SQLInjectionError(
                    f"Column '{identifier}' not in allowed list"
                )
                
        # Check for dangerous patterns
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', identifier):
            raise SQLInjectionError(
                f"Invalid {identifier_type} name format: {identifier}"
            )
            
        # Check length
        if len(identifier) > 64:
            raise SQLInjectionError(
                f"{identifier_type} name too long: {len(identifier)} > 64"
            )
            
        return identifier
    
    @classmethod
    def sanitize_value(cls, value: Any) -> Any:
        """
        Sanitize a value for SQL queries
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized value
            
        Raises:
            SQLInjectionError: If dangerous patterns detected
        """
        if not security_config.enable_sql_injection_protection:
            return value
            
        if value is None:
            return None
            
        # Convert to string for checking
        str_value = str(value)
        upper_value = str_value.upper()
        
        # Check for dangerous keywords
        for keyword in cls.DANGEROUS_KEYWORDS:
            if keyword in upper_value:
                raise SQLInjectionError(
                    f"Dangerous SQL keyword '{keyword}' detected"
                )
                
        # Check for SQL comment patterns
        if "--" in str_value or "/*" in str_value or "*/" in str_value:
            raise SQLInjectionError("SQL comment pattern detected")
            
        # Check for semicolon (statement terminator)
        if ";" in str_value and not isinstance(value, str):
            raise SQLInjectionError("Unexpected semicolon in value")
            
        return value
    
    @classmethod
    def build_safe_query(
        cls,
        query_template: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Optional[tuple]]:
        """
        Build a safe parameterized query
        
        Args:
            query_template: Query template with %s placeholders
            params: Parameters to bind
            
        Returns:
            Tuple of (query, params)
        """
        if params:
            # Sanitize all parameter values
            safe_params = {
                k: cls.sanitize_value(v) for k, v in params.items()
            }
            # Convert to tuple for psycopg2
            param_tuple = tuple(safe_params.values())
            return query_template, param_tuple
        else:
            return query_template, None


class InputValidator:
    """General input validation utilities"""
    
    @staticmethod
    def validate_url(url: str) -> str:
        """
        Validate and sanitize a URL
        
        Args:
            url: URL to validate
            
        Returns:
            Validated URL
            
        Raises:
            InputValidationError: If URL is invalid
        """
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ('http', 'https', 'ftp', 'ftps'):
                raise InputValidationError(
                    f"Invalid URL scheme: {parsed.scheme}"
                )
                
            # Check for localhost/private IPs in production
            if settings.environment == "production":
                hostname = parsed.hostname or ""
                if hostname in ('localhost', '127.0.0.1', '0.0.0.0'):
                    raise InputValidationError(
                        "Localhost URLs not allowed in production"
                    )
                    
            return url
            
        except Exception as e:
            raise InputValidationError(f"Invalid URL: {e}")
    
    @staticmethod
    def validate_email(email: str) -> str:
        """
        Validate email address
        
        Args:
            email: Email to validate
            
        Returns:
            Validated email
            
        Raises:
            InputValidationError: If email is invalid
        """
        # Basic email regex
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise InputValidationError("Invalid email format")
            
        # Check length
        if len(email) > 254:
            raise InputValidationError("Email too long")
            
        return email.lower()
    
    @staticmethod
    def validate_job_id(job_id: str) -> str:
        """
        Validate job ID format
        
        Args:
            job_id: Job ID to validate
            
        Returns:
            Validated job ID
            
        Raises:
            InputValidationError: If job ID is invalid
        """
        # Expected format: alphanumeric with hyphens/underscores
        if not re.match(r'^[a-zA-Z0-9_\-]+$', job_id):
            raise InputValidationError("Invalid job ID format")
            
        if len(job_id) > 64:
            raise InputValidationError("Job ID too long")
            
        return job_id
    
    @staticmethod
    def sanitize_user_input(
        text: str, 
        max_length: int = 1000,
        allow_newlines: bool = True
    ) -> str:
        """
        Sanitize general user text input
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            allow_newlines: Whether to allow newlines
            
        Returns:
            Sanitized text
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters except newlines/tabs
        if allow_newlines:
            text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        else:
            text = re.sub(r'[\x00-\x1F\x7F]', '', text)
            
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length]
            
        return text.strip()


class TokenGenerator:
    """Secure token generation"""
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        return f"sk-{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash a password with salt
        
        Returns:
            Tuple of (hash, salt)
        """
        if not salt:
            salt = secrets.token_hex(16)
            
        hash_obj = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return hash_obj.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hash_hex: str, salt: str) -> bool:
        """Verify a password against its hash"""
        new_hash, _ = TokenGenerator.hash_password(password, salt)
        return secrets.compare_digest(new_hash, hash_hex)


# Convenience instances
path_sanitizer = PathSanitizer()
sql_sanitizer = SQLSanitizer()
input_validator = InputValidator()
token_generator = TokenGenerator()

# Export main functions
sanitize_path = path_sanitizer.sanitize_path
validate_filename = path_sanitizer.validate_filename
validate_identifier = sql_sanitizer.validate_identifier
sanitize_value = sql_sanitizer.sanitize_value
build_safe_query = sql_sanitizer.build_safe_query
validate_url = input_validator.validate_url
validate_email = input_validator.validate_email
validate_job_id = input_validator.validate_job_id
sanitize_user_input = input_validator.sanitize_user_input
generate_token = token_generator.generate_token
generate_api_key = token_generator.generate_api_key