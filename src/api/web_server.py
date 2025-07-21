"""
Production FastAPI server for Montage video processing
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles

from ..core.db import Database
from ..utils.secret_loader import get
from .celery_app import process_video_task

app = FastAPI(
    title="Montage Video Processing API",
    version="3.5.0",
    description="AI-powered video highlight extraction service",
)

# Initialize database
db = Database()

# Ensure upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_path(filename: str, base_dir: Path) -> Path:
    """Validate and return safe file path"""
    # Remove any path components, keep only filename
    clean_name = Path(filename).name
    if not clean_name:
        raise ValueError("Invalid filename")

    target = base_dir / clean_name
    # Ensure path doesn't escape base directory
    try:
        target.resolve().relative_to(base_dir.resolve())
    except ValueError:
        raise ValueError("Path traversal attempt detected")

    return target


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        with db.get_connection() as conn:
            conn.execute("SELECT 1")

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "3.5.0",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/process")
async def process_video(file: UploadFile, mode: str = "smart", vertical: bool = False):
    """
    Upload and process a video file

    Args:
        file: Video file to process
        mode: Processing mode (smart or premium)
        vertical: Output vertical format for social media

    Returns:
        Job ID and status
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    try:
        upload_path = safe_path(f"{job_id}_{file.filename}", UPLOAD_DIR)

        async with aiofiles.open(upload_path, "wb") as f:
            content = await file.read()
            await f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create job in database
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO jobs (job_id, status, input_path, mode, vertical, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """,
                (job_id, "queued", str(upload_path), mode, vertical),
            )
            conn.commit()
    except Exception as e:
        # Clean up file on error
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

    # Queue processing task
    process_video_task.delay(job_id, str(upload_path), mode, vertical)

    return {"job_id": job_id, "status": "queued", "message": "Video processing started"}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a processing job"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, output_path, error_message, created_at, completed_at,
                       mode, vertical, metrics
                FROM jobs
                WHERE job_id = %s
            """,
                (job_id,),
            )

            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")

            status, output_path, error, created, completed, mode, vertical, metrics = (
                row
            )

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
async def download_result(job_id: str):
    """Download processed video"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT output_path, status
                FROM jobs
                WHERE job_id = %s
            """,
                (job_id,),
            )

            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Job not found")

            output_path, status = row

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
async def get_metrics():
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


@app.delete("/cleanup")
async def cleanup_old_files(days: int = 2):
    """Clean up old files (requires API key)"""
    # TODO: Add API key authentication

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
