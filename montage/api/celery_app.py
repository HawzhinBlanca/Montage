"""
Celery app for async video processing with P1-02 resource controls
"""

import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure

from ..core.db import Database
from ..core.checkpoint import CheckpointManager
from ..core.resource_watchdog import resource_watchdog
from ..providers.video_processor import VideoEditor
from ..core.analyze_video import analyze_video
from ..core.cost import budget_manager, BudgetExceededError
from ..core.metrics import metrics
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Redis configuration
from ..settings import settings
REDIS_URL = settings.redis.url.get_secret_value()

# Create Celery app
celery = Celery(
    "montage",
    broker=REDIS_URL,
    backend=REDIS_URL,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# P1-02: Enhanced Celery configuration with resource limits
celery.conf.update(
    # Task timing and limits
    task_time_limit=resource_watchdog.limits.MAX_TASK_RUNTIME_SEC,  # Hard limit from config
    task_soft_time_limit=resource_watchdog.limits.SOFT_TASK_LIMIT_SEC,  # Soft limit from config
    task_acks_late=True,

    # Worker resource management
    worker_prefetch_multiplier=1,  # Process one task at a time to control memory
    worker_max_tasks_per_child=5,  # P1-02: Restart workers more frequently to prevent memory leaks
    worker_max_memory_per_child=resource_watchdog.limits.MAX_MEMORY_MB * 1024,  # Memory limit in KB

    # Concurrency and performance
    worker_concurrency=settings.video.worker_concurrency,
    worker_enable_remote_control=True,  # Allow remote worker management

    # Task routing and priorities
    task_default_queue="default",
    task_default_routing_key="default",
    task_annotations={
        "montage.process_video": {
            "rate_limit": "10/m",  # Limit video processing tasks
            "time_limit": resource_watchdog.limits.MAX_TASK_RUNTIME_SEC,
            "soft_time_limit": resource_watchdog.limits.SOFT_TASK_LIMIT_SEC,
        }
    },

    # Resource monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_track_started=True,

    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
)

# Initialize services
db = Database()
checkpoint_manager = CheckpointManager()


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status in database"""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Build update query dynamically
            updates = ["status = %s", "updated_at = NOW()"]
            values = [status]

            if "error_message" in kwargs:
                updates.append("error_message = %s")
                values.append(kwargs["error_message"])

            if "output_path" in kwargs:
                updates.append("output_path = %s")
                values.append(kwargs["output_path"])

            if "metrics" in kwargs:
                updates.append("metrics = %s")
                values.append(json.dumps(kwargs["metrics"]))

            if status == "completed":
                updates.append("completed_at = NOW()")

            values.append(job_id)

            query = f"""
                UPDATE jobs 
                SET {', '.join(updates)}
                WHERE job_id = %s
            """

            cursor.execute(query, values)
            conn.commit()

    except Exception as e:
        logger.error(f"Failed to update job status: {e}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, **kwargs):
    """Called before task execution"""
    if args and len(args) > 0:
        job_id = args[0]
        update_job_status(job_id, "processing")
        logger.info(f"Starting processing for job {job_id}")


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, args=None, **kwargs
):
    """Called on task failure"""
    if args and len(args) > 0:
        job_id = args[0]
        error_msg = str(exception) if exception else "Unknown error"
        update_job_status(job_id, "failed", error_message=error_msg)
        logger.error(f"Job {job_id} failed: {error_msg}")


@celery.task(bind=True, name="montage.process_video")
def process_video_task(
    self, job_id: str, input_path: str, mode: str = "smart", vertical: bool = False
):
    """
    Process video asynchronously with P1-02 resource monitoring

    Args:
        job_id: Unique job identifier
        input_path: Path to input video
        mode: Processing mode (smart or premium)
        vertical: Whether to output vertical format
    """
    start_time = time.time()
    output_path = None
    temp_files = []  # Track temporary files for cleanup

    try:
        logger.info(f"Processing job {job_id}: {input_path}")

        # P1-02: Register input file for resource tracking
        input_file_path = Path(input_path)
        resource_watchdog.register_temp_file(input_file_path)
        temp_files.append(input_file_path)

        # Update progress
        self.update_state(state="PROGRESS", meta={
            "current": 10,
            "total": 100,
            "resource_stats": resource_watchdog.get_resource_stats()
        })

        # P1-02: Check resources before starting heavy processing
        resource_stats = resource_watchdog.get_resource_stats()
        if resource_stats["current"]["memory_mb"] > resource_watchdog.limits.MEMORY_WARNING_MB:
            logger.warning(f"Starting task with high memory usage: {resource_stats['current']['memory_mb']:.1f}MB")

        # Initialize video editor
        editor = VideoEditor()

        # Analyze video
        logger.info("Analyzing video content...")
        self.update_state(state="PROGRESS", meta={
            "current": 20,
            "total": 100,
            "step": "analyzing_video"
        })

        analysis = analyze_video(input_path, use_premium=(mode == "premium"))

        # P1-02: Check resources after analysis (most memory-intensive step)
        resource_stats = resource_watchdog.get_resource_stats()
        logger.info(f"Memory usage after analysis: {resource_stats['current']['memory_mb']:.1f}MB")

        # Extract highlights
        self.update_state(state="PROGRESS", meta={
            "current": 40,
            "total": 100,
            "step": "extracting_highlights"
        })

        if not analysis or "highlights" not in analysis:
            from ..core.exceptions import VideoProcessingError
            raise VideoProcessingError("Video analysis failed to produce highlights", "ANALYSIS_FAILED")

        highlights = analysis["highlights"]
        logger.info(f"Found {len(highlights)} highlights")

        # Create output path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_filename = f"montage_{job_id}.mp4"
        output_path = output_dir / output_filename

        # Register output file for tracking
        resource_watchdog.register_temp_file(output_path)
        temp_files.append(output_path)

        # Process video
        self.update_state(state="PROGRESS", meta={
            "current": 60,
            "total": 100,
            "step": "processing_video"
        })

        if vertical:
            logger.info("Creating vertical format video...")
            # Use the same editor for now - vertical crop would be applied in production
            result = editor.create_highlight_montage(
                input_path=input_path,
                output_path=str(output_path),
                highlights=highlights,
            )
        else:
            logger.info("Creating highlight montage...")
            result = editor.create_highlight_montage(
                input_path=input_path,
                output_path=str(output_path),
                highlights=highlights,
            )

        # Verify output
        self.update_state(state="PROGRESS", meta={
            "current": 90,
            "total": 100,
            "step": "finalizing"
        })

        if not output_path.exists():
            from ..core.exceptions import VideoProcessingError
            raise VideoProcessingError("Output file was not created", "OUTPUT_MISSING")

        # Calculate metrics with resource usage
        processing_time = time.time() - start_time
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        final_resource_stats = resource_watchdog.get_resource_stats()

        job_metrics = {
            "processing_time_sec": round(processing_time, 2),
            "output_size_mb": round(file_size_mb, 2),
            "highlights_count": len(highlights),
            "mode": mode,
            "vertical": vertical,
            "cost_usd": budget_manager.get_total_cost(),
            # P1-02: Include resource usage metrics
            "max_memory_mb": final_resource_stats["statistics"]["max_memory_used_mb"],
            "max_cpu_percent": final_resource_stats["statistics"]["max_cpu_used_percent"],
            "resource_alerts": len(final_resource_stats["recent_alerts"])
        }

        # Update job as completed
        update_job_status(
            job_id, "completed", output_path=str(output_path), metrics=job_metrics
        )

        # Record metrics
        metrics.record("jobs_completed", 1)
        metrics.record("processing_time", processing_time)
        metrics.record("peak_memory_mb", final_resource_stats["statistics"]["max_memory_used_mb"])

        logger.info(f"Job {job_id} completed successfully in {processing_time:.1f}s")

        return {
            "job_id": job_id,
            "status": "completed",
            "output_path": str(output_path),
            "metrics": job_metrics,
        }

    except BudgetExceededError as e:
        logger.error(f"Budget exceeded for job {job_id}: {e}")
        update_job_status(job_id, "failed", error_message=f"Budget exceeded: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        logger.error(traceback.format_exc())

        # P1-02: Include resource stats in error reporting
        error_resource_stats = resource_watchdog.get_resource_stats()
        error_message = f"{str(e)} (Peak memory: {error_resource_stats['statistics']['max_memory_used_mb']:.1f}MB)"

        update_job_status(job_id, "failed", error_message=error_message)
        raise

    finally:
        # P1-02: Enhanced cleanup with resource tracking
        try:
            for temp_file in temp_files:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
                    resource_watchdog.unregister_temp_file(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")

            # Force cleanup after task completion
            resource_watchdog._gentle_cleanup()

            logger.info(f"Cleanup completed for job {job_id}")

        except Exception as e:
            logger.error(f"Failed to clean up files for job {job_id}: {e}")
            # Don't raise - task might have succeeded


@celery.task(name="montage.cleanup_old_files")
def cleanup_old_files_task(days: int = 2):
    """Periodic task to clean up old files"""
    import time

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    cleaned = {"uploads": 0, "outputs": 0, "database": 0}

    # Clean upload directory
    upload_dir = Path("uploads")
    if upload_dir.exists():
        for file in upload_dir.iterdir():
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    cleaned["uploads"] += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")

    # Clean output directory
    output_dir = Path("output")
    if output_dir.exists():
        for file in output_dir.iterdir():
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    cleaned["outputs"] += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {e}")

    # Clean old database entries
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM jobs
                WHERE created_at < NOW() - INTERVAL '%s days'
                AND status IN ('completed', 'failed')
            """,
                (days,),
            )
            cleaned["database"] = cursor.rowcount
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to clean database: {e}")

    logger.info(f"Cleanup completed: {cleaned}")
    return cleaned


# Configure periodic tasks
celery.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "montage.cleanup_old_files",
        "schedule": 3600.0,  # Every hour
        "args": (2,),  # Clean files older than 2 days
    },
}
