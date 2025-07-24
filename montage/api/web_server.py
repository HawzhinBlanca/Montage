"""
Production FastAPI server for Montage video processing
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..core.db import Database
from ..core.highlight_merger import rover_merger
from ..core.resource_watchdog import resource_watchdog
from ..core.upload_validator import UploadValidationError, upload_validator
from ..core.security import (
    sanitize_path,
    validate_filename,
    validate_job_id,
    sanitize_user_input,
    PathTraversalError,
    InputValidationError,
)
from ..core.exceptions import (
    ErrorHandler,
    JobNotFoundError,
    RateLimitError,
    AuthenticationError,
    VideoTooLargeError,
    VideoTooLongError,
    InvalidVideoFormatError,
)
from ..utils.ffmpeg_process_manager import ffmpeg_process_manager
from ..utils.secret_loader import (
    get,
    get_secret_sources_status,
    validate_required_secrets,
)
from .auth import require_api_key, validate_api_key
from .celery_app import process_video_task

# Configure logger
logger = logging.getLogger(__name__)

# P0-05: Custom rate limiting key function combining IP + API key
def rate_limit_key_func(request: Request):
    """
    P0-05 Requirement: Rate limit per key & IP (30/minute 500/day)
    Create composite key from IP address and API key for granular rate limiting
    """
    client_ip = get_remote_address(request)

    # Try to get API key for more granular rate limiting
    api_key = None
    try:
        # Extract API key if present (don't require it here as some endpoints may not need it)
        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
        if api_key and validate_api_key(api_key):
            # Use last 8 characters of API key for identification (don't log full key)
            api_key_suffix = api_key[-8:] if len(api_key) >= 8 else api_key
            return f"{client_ip}:{api_key_suffix}"
    except:
        pass  # Fallback to IP-only rate limiting

    return client_ip

# P0-05: Initialize rate limiter with custom key function
limiter = Limiter(key_func=rate_limit_key_func)

from .auth import verify_key

app = FastAPI(
    title="Montage Video Processing API",
    version="3.5.0",
    description="AI-powered video highlight extraction service",
    dependencies=[Depends(verify_key)]
)

# Import unified settings for CORS configuration
from ..settings import settings

# P0-05: Add CORS middleware with secure defaults
if settings.environment == "development":
    # Development: Allow all origins but warn
    cors_origins = ["*"]
    allow_credentials = False
    logger.warning("⚠️ CORS: Allowing all origins in development mode")
else:
    # Production: Strict origin control
    cors_origins = [
        "https://yourdomain.com",  # Replace with actual domain
        "https://app.yourdomain.com"
    ]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# P0-05: Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Custom exception handler for Montage errors
@app.exception_handler(Exception)
async def montage_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions with consistent error responses"""
    from fastapi.responses import JSONResponse
    
    # Get error response from centralized handler
    error_response = ErrorHandler.handle_error(
        exc, 
        context={
            "path": request.url.path,
            "method": request.method,
            "debug": settings.debug
        }
    )
    
    # Determine status code
    status_code = 500  # Default to internal server error
    error_code = error_response["error"]["code"]
    
    # Map error codes to HTTP status codes
    status_mapping = {
        "AUTHENTICATION_ERROR": 401,
        "AUTHORIZATION_ERROR": 403,
        "RECORD_NOT_FOUND": 404,
        "JOB_NOT_FOUND": 404,
        "FILE_NOT_FOUND": 404,
        "RATE_LIMIT_ERROR": 429,
        "VALIDATION_ERROR": 400,
        "INVALID_VIDEO_FORMAT": 400,
        "VIDEO_TOO_LARGE": 413,
        "VIDEO_TOO_LONG": 413,
        "COST_LIMIT_ERROR": 402,
        "INSUFFICIENT_RESOURCES": 503,
        "DATABASE_CONNECTION_ERROR": 503,
    }
    
    if error_code in status_mapping:
        status_code = status_mapping[error_code]
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )

# Initialize database
db = Database()

# P0-04: FastAPI startup secret validation
@app.on_event("startup")
async def validate_secrets_on_startup():
    """
    Validate all required secrets are available on FastAPI startup
    This ensures the application cannot start without proper configuration
    """
    logger.info("🔑 Validating API secrets on startup...")

    validation_results = validate_required_secrets()
    sources_status = get_secret_sources_status()

    logger.info(f"📊 Secret sources status: {sources_status}")

    if not validation_results.get("all_valid", False):
        missing_secrets = [
            key for key, valid in validation_results.items()
            if key != "all_valid" and not valid
        ]
        error_msg = f"🚨 STARTUP FAILED: Missing required secrets: {missing_secrets}"
        logger.critical(error_msg)

        # In development, warn but continue. In production, this would raise an exception
        if settings.environment == "production":
            from ..core.exceptions import ConfigurationError
            raise ConfigurationError(f"Missing required secrets in production: {missing_secrets}", "MISSING_SECRETS")
        else:
            logger.warning("⚠️ Development mode: continuing with missing secrets")
    else:
        logger.info("✅ All required secrets validated successfully")

        # Specifically verify OPENAI_API_KEY as required by task
        openai_key = settings.api_keys.openai.get_secret_value() if settings.api_keys.openai else None
        if openai_key and not openai_key.startswith('PLACEHOLDER'):
            logger.info("✅ OPENAI_API_KEY environment verification passed")
        else:
            logger.warning("⚠️ OPENAI_API_KEY not properly set in environment")

# Ensure upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_path(filename: str, base_dir: Path) -> Path:
    """Validate and return safe file path using security module"""
    try:
        # Sanitize filename first
        clean_name = validate_filename(filename)
        
        # Build full path
        target = base_dir / clean_name
        
        # Validate with security module
        safe_target = sanitize_path(target)
        
        # Additional check to ensure it's within base_dir
        safe_target.relative_to(base_dir.resolve())
        
        return safe_target
    except (PathTraversalError, ValueError) as e:
        from ..core.exceptions import ValidationError
        raise ValidationError(f"Invalid path: {e}", "INVALID_PATH")


@app.get("/health")
@limiter.limit("100/minute")  # Health checks don't need API key but should be rate limited
async def health_check(request: Request):
    """Health check endpoint"""
    try:
        # Check database connection
        db.execute("SELECT 1")

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.5.0",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/process")
@limiter.limit("30/minute")  # P0-05: Rate limit 30/minute per key & IP
@limiter.limit("500/day")    # P0-05: Rate limit 500/day per key & IP
async def process_video(
    request: Request,
    file: UploadFile,
    mode: str = "smart",
    vertical: bool = False,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """
    Upload and process a video file with P1-01 security controls

    Args:
        file: Video file to process
        mode: Processing mode (smart or premium)
        vertical: Output vertical format for social media

    Returns:
        Job ID and status
    """
    # P1-01: Basic file validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Generate job ID early for temp file naming
    job_id = str(uuid.uuid4())
    temp_upload_path = None

    try:
        # P1-01: Create temporary upload path
        temp_filename = f"{job_id}_{file.filename}"
        temp_upload_path = safe_path(temp_filename, UPLOAD_DIR)

        # Stream file to disk with size monitoring
        total_size = 0
        max_size = upload_validator.limits.MAX_FILE_SIZE_BYTES

        async with aiofiles.open(temp_upload_path, "wb") as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                total_size += len(chunk)

                # P1-01: Real-time size check during streaming
                if total_size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: {total_size/1024/1024:.1f}MB exceeds {max_size/1024/1024:.1f}MB limit"
                    )

                await f.write(chunk)

        # P1-01: Comprehensive upload validation
        try:
            validated_filename, mime_type = await upload_validator.validate_upload(
                file, api_key, temp_upload_path
            )
            logger.info(f"Upload validated: {validated_filename} ({mime_type})")

        except UploadValidationError as e:
            logger.warning(f"Upload validation failed: {e.message}")
            raise HTTPException(
                status_code=400,
                detail=f"Upload validation failed: {e.message}"
            )

        # Rename to final path with validated filename
        final_upload_path = safe_path(f"{job_id}_{validated_filename}", UPLOAD_DIR)
        if temp_upload_path != final_upload_path:
            temp_upload_path.rename(final_upload_path)
            temp_upload_path = final_upload_path  # Update for cleanup reference

    except HTTPException:
        # Clean up temp file on HTTP errors
        if temp_upload_path and temp_upload_path.exists():
            temp_upload_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        # Clean up temp file on any error
        if temp_upload_path and temp_upload_path.exists():
            temp_upload_path.unlink(missing_ok=True)
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # Create job in database with validation metadata
    try:
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "input_path": str(final_upload_path),
            "mode": mode,
            "vertical": vertical,
            "created_at": datetime.utcnow(),
            "metadata": json.dumps({
                "original_filename": file.filename,
                "validated_filename": validated_filename,
                "mime_type": mime_type,
                "file_size_bytes": total_size,
                "validation_timestamp": datetime.utcnow().isoformat()
            })
        }
        db.insert("jobs", job_data)

    except Exception as e:
        # Clean up file on database error
        if final_upload_path.exists():
            final_upload_path.unlink(missing_ok=True)
        logger.error(f"Database error during job creation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

    # Queue processing task
    process_video_task.delay(job_id, str(final_upload_path), mode, vertical)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video processing started",
        "file_info": {
            "filename": validated_filename,
            "size_mb": round(total_size / 1024 / 1024, 2),
            "mime_type": mime_type
        }
    }


@app.get("/status/{job_id}")
@limiter.limit("120/minute")  # Status checks can be more frequent but still limited
async def get_job_status(
    request: Request,
    job_id: str,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Get status of a processing job"""
    try:
        # Validate job ID format
        job_id = validate_job_id(job_id)
        
        job = db.find_one("jobs", {"job_id": job_id})
        if not job:
            raise JobNotFoundError(job_id)

        status = job["status"]
        output_path = job["output_path"]
        error = job["error_message"]
        created = job["created_at"]
        completed = job["completed_at"]
        mode = job["mode"]
        vertical = job["vertical"]
        metrics = job["metrics"]

        result = {
            "job_id": job_id,
            "status": status,
            "mode": mode,
            "vertical": vertical,
            "created_at": created.isoformat() if created else None,
            "completed_at": completed.isoformat() if completed else None,
        }

        if error:
            result["error"] = error

        if output_path and Path(output_path).exists():
            result["download_url"] = f"/download/{job_id}"

        if metrics:
            result["metrics"] = metrics

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.get("/download/{job_id}")
@limiter.limit("60/minute")   # Downloads need rate limiting
@limiter.limit("200/day")     # Daily download limit
async def download_result(
    request: Request,
    job_id: str,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Download processed video"""
    try:
        # Validate job ID format
        job_id = validate_job_id(job_id)
        
        job = db.find_one("jobs", {"job_id": job_id})
        if not job:
            raise JobNotFoundError(job_id)

        output_path = job["output_path"]
        status = job["status"]

        if status != "completed":
            raise HTTPException(
                    status_code=400, detail=f"Job not completed, status: {status}"
                )

        if not output_path or not Path(output_path).exists():
            raise HTTPException(status_code=404, detail="Output file not found")

        return FileResponse(
            output_path, media_type="video/mp4", filename=f"montage_{job_id}.mp4"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download: {str(e)}")


@app.get("/metrics")
@limiter.limit("30/minute")   # Metrics access should be limited
async def get_metrics(
    request: Request,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Get system metrics"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get job statistics
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_jobs,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing,
                    COUNT(CASE WHEN status = 'queued' THEN 1 END) as queued,
                    AVG(CASE 
                        WHEN status = 'completed' AND completed_at IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (completed_at - created_at))
                    END) as avg_processing_time_sec
                FROM jobs
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """
            )

            stats = cursor.fetchone()

            return {
                "jobs_24h": {
                    "total": stats[0],
                    "completed": stats[1],
                    "failed": stats[2],
                    "processing": stats[3],
                    "queued": stats[4],
                    "avg_processing_time_sec": round(stats[5], 2) if stats[5] else None,
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/upload-stats")
@limiter.limit("60/minute")   # P1-01: Upload statistics endpoint
async def get_upload_stats(
    request: Request,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Get current upload statistics and limits for the API key"""
    try:
        stats = upload_validator.get_upload_stats(api_key)
        return {
            "api_key_suffix": api_key[-8:] if len(api_key) >= 8 else api_key,
            "current_usage": {
                "uploads_last_hour": stats["uploads_last_hour"],
                "uploads_last_day": stats["uploads_last_day"]
            },
            "limits": {
                "max_uploads_per_hour": stats["hourly_limit"],
                "max_uploads_per_day": stats["daily_limit"],
                "max_file_size_mb": upload_validator.limits.MAX_FILE_SIZE_MB,
                "max_duration_seconds": upload_validator.limits.MAX_DURATION_SECONDS
            },
            "allowed_formats": {
                "extensions": upload_validator.limits.ALLOWED_EXTENSIONS,
                "mime_types": upload_validator.limits.ALLOWED_MIME_TYPES
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get upload stats: {str(e)}")


@app.get("/worker-stats")
@limiter.limit("30/minute")   # P1-02: Resource monitoring endpoint
async def get_worker_stats(
    request: Request,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Get current Celery worker resource usage and statistics"""
    try:
        resource_stats = resource_watchdog.get_resource_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "worker_status": {
                "resource_monitoring_active": resource_watchdog.monitoring_active,
                "uptime_hours": resource_stats["uptime_hours"]
            },
            "current_usage": resource_stats["current"],
            "limits": resource_stats["limits"],
            "statistics": resource_stats["statistics"],
            "recent_alerts": resource_stats["recent_alerts"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get worker stats: {str(e)}")


@app.get("/process-stats")
@limiter.limit("30/minute")   # P1-03: FFmpeg process monitoring endpoint
async def get_process_stats(
    request: Request,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Get current FFmpeg process statistics and resource usage"""
    try:
        process_stats = ffmpeg_process_manager.get_process_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "ffmpeg_processes": process_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get process stats: {str(e)}")


@app.get("/algorithm-stats")
@limiter.limit("30/minute")   # P1-04: ROVER algorithm performance monitoring
async def get_algorithm_stats(
    request: Request,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Get ROVER highlight merge algorithm performance statistics"""
    try:
        rover_stats = rover_merger.get_performance_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "rover_performance": rover_stats,
            "algorithm_info": {
                "name": "ROVER (Rapid Overlap Verification and Efficient Ranking)",
                "complexity": "O(n log n)",
                "max_highlights": rover_merger.max_highlights,
                "min_duration_ms": rover_merger.min_duration_ms,
                "max_duration_ms": rover_merger.max_duration_ms
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm stats: {str(e)}")


@app.delete("/cleanup")
@limiter.limit("5/hour")      # Cleanup is admin operation, very restricted
async def cleanup_old_files(
    request: Request,
    days: int = 2,
    api_key: str = Depends(require_api_key)  # P0-05: Require API key
):
    """Clean up old files (requires API key)"""
    import time

    cutoff_time = time.time() - (days * 24 * 60 * 60)

    cleaned = {"uploads": 0, "outputs": 0}

    # Clean uploads
    for file in UPLOAD_DIR.iterdir():
        if file.is_file() and file.stat().st_mtime < cutoff_time:
            file.unlink()
            cleaned["uploads"] += 1

    # Clean outputs
    for file in OUTPUT_DIR.iterdir():
        if file.is_file() and file.stat().st_mtime < cutoff_time:
            file.unlink()
            cleaned["outputs"] += 1

    return {"message": f"Cleaned up files older than {days} days", "cleaned": cleaned}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
