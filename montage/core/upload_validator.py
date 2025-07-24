"""
P1-01: Upload validation, hard limits, and MIME type security controls

This module provides comprehensive upload security to prevent:
- File size bombs and resource exhaustion attacks
- Malicious file uploads with fake extensions
- Directory traversal attacks via filenames
- MIME type spoofing and polyglot file attacks
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import magic
from fastapi import HTTPException, UploadFile

from ..utils.logging_config import get_logger
from ..utils.secret_loader import get

logger = get_logger(__name__)

class FileType(str, Enum):
    """Allowed file types for upload"""
    MP4 = "video/mp4"
    MOV = "video/quicktime"
    AVI = "video/x-msvideo"
    MKV = "video/x-matroska"
    WEBM = "video/webm"
    # Audio-only formats for testing
    WAV = "audio/wav"
    MP3 = "audio/mpeg"

@dataclass
class UploadLimits:
    """P1-01: Upload security limits configuration"""

    # File size limits
    MAX_FILE_SIZE_MB: int = int(get("MAX_UPLOAD_SIZE_MB", "600"))  # 600MB default
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

    # Duration limits (seconds)
    MAX_DURATION_SECONDS: int = int(get("MAX_VIDEO_DURATION_SEC", "3600"))  # 1 hour max

    # Rate limiting per user
    MAX_UPLOADS_PER_HOUR: int = int(get("MAX_UPLOADS_PER_HOUR", "10"))
    MAX_UPLOADS_PER_DAY: int = int(get("MAX_UPLOADS_PER_DAY", "50"))

    # Content validation
    REQUIRE_MIME_VALIDATION: bool = get("REQUIRE_MIME_VALIDATION", "true").lower() == "true"
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv", ".webm"])
    ALLOWED_MIME_TYPES: List[str] = field(default_factory=lambda: [ft.value for ft in FileType])

    # Security
    SCAN_FOR_MALWARE: bool = get("SCAN_FOR_MALWARE", "false").lower() == "true"
    QUARANTINE_SUSPICIOUS: bool = get("QUARANTINE_SUSPICIOUS", "true").lower() == "true"

    # Performance limits
    MAX_CONCURRENT_UPLOADS: int = int(get("MAX_CONCURRENT_UPLOADS", "5"))

class UploadValidationError(Exception):
    """Custom exception for upload validation failures"""

    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

class UploadValidator:
    """
    P1-01: Comprehensive upload validation and security controls
    
    Features:
    - File size and duration limits
    - MIME type validation with magic byte detection
    - Extension validation and normalization
    - Rate limiting per API key/user
    - Malware scanning hooks
    - Upload tracking and metrics
    """

    def __init__(self):
        self.limits = UploadLimits()
        self._upload_tracking: Dict[str, List[datetime]] = {}
        logger.info(f"Upload validator initialized with {self.limits.MAX_FILE_SIZE_MB}MB limit")

    def _get_real_mime_type(self, file_path: Path) -> str:
        """
        Get actual MIME type using libmagic (magic bytes detection)
        This prevents MIME type spoofing attacks
        """
        try:
            # Use python-magic to detect actual file type
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_file(str(file_path))
            logger.debug(f"Detected MIME type for {file_path.name}: {detected_mime}")
            return detected_mime
        except Exception as e:
            logger.warning(f"Failed to detect MIME type for {file_path}: {e}")
            return "application/octet-stream"

    def _validate_file_size(self, file: UploadFile) -> None:
        """
        P1-01: Validate file size doesn't exceed limits
        Prevents file size bomb attacks and resource exhaustion
        """
        # Try to get content length from headers
        content_length = None
        if hasattr(file, 'headers') and file.headers:
            content_length = file.headers.get('content-length')

        if content_length:
            try:
                size_bytes = int(content_length)
                if size_bytes > self.limits.MAX_FILE_SIZE_BYTES:
                    raise UploadValidationError(
                        f"File size {size_bytes/1024/1024:.1f}MB exceeds limit of {self.limits.MAX_FILE_SIZE_MB}MB",
                        "FILE_TOO_LARGE"
                    )
            except ValueError:
                logger.warning(f"Invalid content-length header: {content_length}")

        # Note: Final size validation happens during streaming read

    def _validate_filename(self, filename: str) -> str:
        """
        P1-01: Validate and sanitize filename
        Prevents directory traversal and invalid character attacks
        """
        if not filename:
            raise UploadValidationError("Filename cannot be empty", "INVALID_FILENAME")

        # Extract just the filename component (no directory paths)
        clean_filename = Path(filename).name

        if not clean_filename:
            raise UploadValidationError("Invalid filename provided", "INVALID_FILENAME")

        # Check for dangerous patterns
        dangerous_patterns = ["..", "\\", "/", ":", "*", "?", '"', "<", ">", "|"]
        if any(pattern in clean_filename for pattern in dangerous_patterns):
            raise UploadValidationError(
                "Filename contains invalid characters",
                "INVALID_FILENAME"
            )

        # Check extension
        extension = Path(clean_filename).suffix.lower()
        if extension not in self.limits.ALLOWED_EXTENSIONS:
            allowed = ", ".join(self.limits.ALLOWED_EXTENSIONS)
            raise UploadValidationError(
                f"Invalid file extension '{extension}'. Allowed: {allowed}",
                "INVALID_EXTENSION"
            )

        logger.debug(f"Filename validated: {clean_filename}")
        return clean_filename

    def _validate_mime_type(self, file_path: Path, declared_mime: Optional[str] = None) -> str:
        """
        P1-01: Validate MIME type using both declaration and magic bytes
        Prevents MIME type spoofing and polyglot file attacks
        """
        if not self.limits.REQUIRE_MIME_VALIDATION:
            logger.debug("MIME validation disabled by configuration")
            return declared_mime or "video/mp4"

        # Get actual MIME type from file content
        actual_mime = self._get_real_mime_type(file_path)

        # Check if actual MIME type is allowed
        if actual_mime not in self.limits.ALLOWED_MIME_TYPES:
            # Special handling for common variations
            mime_mappings = {
                "video/mp4": ["video/mp4", "application/mp4"],
                "video/quicktime": ["video/quicktime", "video/mov"],
                "video/x-msvideo": ["video/x-msvideo", "video/avi"],
                "video/x-matroska": ["video/x-matroska", "video/mkv"],
            }

            # Check if it's a known variant
            is_allowed = False
            for standard_mime, variants in mime_mappings.items():
                if actual_mime in variants and standard_mime in self.limits.ALLOWED_MIME_TYPES:
                    actual_mime = standard_mime
                    is_allowed = True
                    break

            if not is_allowed:
                allowed = ", ".join(self.limits.ALLOWED_MIME_TYPES)
                raise UploadValidationError(
                    f"Invalid file type '{actual_mime}'. Allowed: {allowed}",
                    "INVALID_MIME_TYPE"
                )

        # Check for MIME/extension mismatch (security concern)
        extension = file_path.suffix.lower()
        expected_mimes = {
            ".mp4": ["video/mp4", "application/mp4"],
            ".mov": ["video/quicktime", "video/mov"],
            ".avi": ["video/x-msvideo", "video/avi"],
            ".mkv": ["video/x-matroska", "video/mkv"],
            ".webm": ["video/webm"],
        }

        if extension in expected_mimes:
            if actual_mime not in expected_mimes[extension]:
                logger.warning(
                    f"MIME/extension mismatch: {extension} file has MIME type {actual_mime}"
                )
                if self.limits.QUARANTINE_SUSPICIOUS:
                    raise UploadValidationError(
                        f"File extension '{extension}' doesn't match detected type '{actual_mime}'",
                        "MIME_EXTENSION_MISMATCH"
                    )

        logger.info(f"MIME type validated: {actual_mime} for {file_path.name}")
        return actual_mime

    def _check_rate_limits(self, api_key: str) -> None:
        """
        P1-01: Check upload rate limits per API key
        Prevents abuse and resource exhaustion
        """
        now = datetime.utcnow()

        # Initialize tracking for this API key
        if api_key not in self._upload_tracking:
            self._upload_tracking[api_key] = []

        uploads = self._upload_tracking[api_key]

        # Clean old entries
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        uploads[:] = [upload_time for upload_time in uploads if upload_time > day_ago]

        # Check hourly limit
        recent_uploads = [upload_time for upload_time in uploads if upload_time > hour_ago]
        if len(recent_uploads) >= self.limits.MAX_UPLOADS_PER_HOUR:
            raise UploadValidationError(
                f"Upload rate limit exceeded: {len(recent_uploads)}/{self.limits.MAX_UPLOADS_PER_HOUR} uploads in last hour",
                "RATE_LIMIT_HOURLY"
            )

        # Check daily limit
        if len(uploads) >= self.limits.MAX_UPLOADS_PER_DAY:
            raise UploadValidationError(
                f"Upload rate limit exceeded: {len(uploads)}/{self.limits.MAX_UPLOADS_PER_DAY} uploads in last day",
                "RATE_LIMIT_DAILY"
            )

        # Record this upload
        uploads.append(now)
        logger.debug(f"Upload rate check passed for {api_key}: {len(recent_uploads)}/hr, {len(uploads)}/day")

    def _scan_for_malware(self, file_path: Path) -> None:
        """
        P1-01: Optional malware scanning hook
        Can be integrated with ClamAV or other scanners
        """
        if not self.limits.SCAN_FOR_MALWARE:
            return

        # Placeholder for malware scanning integration
        # In production, integrate with:
        # - ClamAV daemon
        # - VirusTotal API
        # - Custom ML-based detection
        logger.debug(f"Malware scan placeholder for {file_path.name}")

        # Example: Check file size patterns that might indicate malicious content
        file_size = file_path.stat().st_size
        if file_size < 1024:  # Files under 1KB are suspicious for videos
            logger.warning(f"Suspiciously small video file: {file_size} bytes")
            if self.limits.QUARANTINE_SUSPICIOUS:
                raise UploadValidationError(
                    "File appears to be corrupted or malicious (too small)",
                    "SUSPICIOUS_FILE_SIZE"
                )

    async def validate_upload(
        self,
        file: UploadFile,
        api_key: str,
        temp_path: Path
    ) -> Tuple[str, str]:
        """
        P1-01: Main upload validation pipeline
        
        Returns:
            Tuple of (validated_filename, validated_mime_type)
            
        Raises:
            UploadValidationError: If validation fails
            HTTPException: For critical security issues
        """
        try:
            logger.info(f"Starting upload validation for {file.filename}")

            # 1. Rate limiting check
            self._check_rate_limits(api_key)

            # 2. File size validation (initial)
            self._validate_file_size(file)

            # 3. Filename validation and sanitization
            clean_filename = self._validate_filename(file.filename or "unknown")

            # 4. Content-based validation (after file is written to temp_path)
            if temp_path.exists():
                # Validate actual file size on disk
                file_size = temp_path.stat().st_size
                if file_size > self.limits.MAX_FILE_SIZE_BYTES:
                    temp_path.unlink(missing_ok=True)  # Clean up
                    raise UploadValidationError(
                        f"File size {file_size/1024/1024:.1f}MB exceeds limit of {self.limits.MAX_FILE_SIZE_MB}MB",
                        "FILE_TOO_LARGE"
                    )

                # MIME type validation
                mime_type = self._validate_mime_type(temp_path, file.content_type)

                # Malware scanning
                self._scan_for_malware(temp_path)

            else:
                logger.error(f"Temporary file not found: {temp_path}")
                raise UploadValidationError(
                    "File upload incomplete",
                    "UPLOAD_INCOMPLETE"
                )

            logger.info(f"Upload validation successful: {clean_filename} ({mime_type})")
            return clean_filename, mime_type

        except UploadValidationError:
            # Clean up temp file on validation failure
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upload validation: {e}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Upload validation failed due to internal error"
            )

    def get_upload_stats(self, api_key: str) -> Dict[str, int]:
        """Get current upload statistics for an API key"""
        if api_key not in self._upload_tracking:
            return {"uploads_last_hour": 0, "uploads_last_day": 0}

        now = datetime.utcnow()
        uploads = self._upload_tracking[api_key]

        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        uploads_last_hour = len([u for u in uploads if u > hour_ago])
        uploads_last_day = len([u for u in uploads if u > day_ago])

        return {
            "uploads_last_hour": uploads_last_hour,
            "uploads_last_day": uploads_last_day,
            "hourly_limit": self.limits.MAX_UPLOADS_PER_HOUR,
            "daily_limit": self.limits.MAX_UPLOADS_PER_DAY
        }

# Global validator instance
upload_validator = UploadValidator()
