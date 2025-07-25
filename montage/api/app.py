"""
FastAPI/Uvicorn app for Montage API
Tasks.md Step 1: Real API implementation with Gunicorn/Uvicorn
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
from typing import Optional

# Add parent directory to path

app = FastAPI(
    title="Montage API",
    description="AI-powered video processing pipeline",
    version="1.0.0"
)

class VideoProcessRequest(BaseModel):
    video_path: str
    job_id: str
    use_premium: bool = False
    variant: str = "control"

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Montage API", "status": "operational"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/process")
async def process_video(request: VideoProcessRequest):
    """Submit video for processing"""
    try:
        # Import Celery task
        from montage.api.celery_tasks import process_video

        # Submit task
        task = process_video.delay(
            request.video_path,
            request.job_id,
            {
                'use_premium': request.use_premium,
                'variant': request.variant
            }
        )

        return {
            "job_id": request.job_id,
            "task_id": task.id,
            "status": "submitted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get task status"""
    try:
        from montage.api.celery_tasks import process_video

        task = process_video.AsyncResult(task_id)

        return {
            "task_id": task_id,
            "state": task.state,
            "result": task.result if task.state == "SUCCESS" else None,
            "info": task.info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found"}
    )

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
