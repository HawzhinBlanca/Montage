"""Redis-based checkpointing system for crash recovery"""

import json
import pickle
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import redis
from redis.exceptions import RedisError

# Import centralized logging
try:
    from ..utils.logging_config import correlation_context, get_logger
except ImportError:
    import sys
    from pathlib import Path

    from montage.utils.logging_config import get_logger

try:
    from ..settings import settings
    from ..utils.secret_loader import get
    from .db import Database
    REDIS_URL = settings.redis.url.get_secret_value()
except ImportError:
    import sys
    from pathlib import Path

    from montage.settings import settings
    from montage.utils.secret_loader import get
    from montage.core.db import Database
    REDIS_URL = settings.redis.url.get_secret_value()

logger = get_logger(__name__)


class CheckpointError(Exception):
    """Base exception for checkpoint operations"""

    def __init__(self, message="Checkpoint operation failed"):
        super().__init__(message)
        self.message = message


class CheckpointManager:
    """Manages job state checkpoints in Redis for crash recovery"""

    def __init__(self):
        try:
            # Use REDIS_URL if available, otherwise construct from individual values
            if REDIS_URL:
                self.redis_client = redis.from_url(
                    REDIS_URL,
                    decode_responses=False,  # We'll handle encoding/decoding
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            else:
                # Parse Redis URL from settings
                import urllib.parse
                parsed = urllib.parse.urlparse(REDIS_URL)
                self.redis_client = redis.Redis(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 6379,
                    db=int(parsed.path.lstrip('/')) if parsed.path and parsed.path != '/' else 0,
                    password=parsed.password,
                    decode_responses=False,  # We'll handle encoding/decoding
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis checkpoint manager initialized")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CheckpointError(f"Redis connection failed: {e}")

        self.db = Database()

    def _get_checkpoint_key(self, job_id: str, stage: str) -> str:
        """Generate Redis key for a checkpoint"""
        return f"checkpoint:{job_id}:{stage}"

    def _get_job_stages_key(self, job_id: str) -> str:
        """Generate Redis key for job stages list"""
        return f"job_stages:{job_id}"

    def save_checkpoint(self, job_id: str, stage: str, data: Dict[str, Any]) -> None:
        """Save a checkpoint for a job stage"""
        try:
            # Prepare checkpoint data
            checkpoint = {
                "job_id": job_id,
                "stage": stage,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0",
            }

            # Serialize data
            serialized = pickle.dumps(checkpoint)

            # Save to Redis with TTL
            key = self._get_checkpoint_key(job_id, stage)
            self.redis_client.setex(key, 3600, serialized)  # 1 hour TTL

            # Add stage to job's stage list
            stages_key = self._get_job_stages_key(job_id)
            self.redis_client.sadd(stages_key, stage)
            self.redis_client.expire(stages_key, 3600)  # 1 hour TTL

            # Also save to PostgreSQL for backup
            self.db.insert(
                "job_checkpoint",
                {
                    "job_id": job_id,
                    "stage": stage,
                    "checkpoint_data": json.dumps(data, default=str),
                },
            )

            logger.info(f"Saved checkpoint for job {job_id} at stage '{stage}'")

        except (
            RedisError,
            pickle.PickleError,
            ValueError,
            TypeError,
        ) as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            # Don't raise error - continue processing without checkpoints

    def load_checkpoint(self, job_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint for a job stage"""
        try:
            key = self._get_checkpoint_key(job_id, stage)
            serialized = self.redis_client.get(key)

            if not serialized:
                # Try to restore from PostgreSQL
                return self._restore_from_postgres(job_id, stage)

            checkpoint = pickle.loads(serialized)
            logger.info(f"Loaded checkpoint for job {job_id} at stage '{stage}'")
            return checkpoint["data"]

        except (
            RedisError,
            pickle.UnpicklingError,
            pickle.PickleError,
            ValueError,
        ) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _restore_from_postgres(
        self, job_id: str, stage: str
    ) -> Optional[Dict[str, Any]]:
        """Restore checkpoint from PostgreSQL if not in Redis"""
        try:
            record = self.db.find_one(
                "job_checkpoint", {"job_id": job_id, "stage": stage}
            )

            if record:
                data = json.loads(record["checkpoint_data"])
                # Re-save to Redis
                self.save_checkpoint(job_id, stage, data)
                logger.info(
                    f"Restored checkpoint from PostgreSQL for job {job_id} stage '{stage}'"
                )
                return data

        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to restore from PostgreSQL: {e}")

        return None

    def get_last_successful_stage(self, job_id: str) -> Optional[str]:
        """Get the last successfully completed stage for a job"""
        try:
            # Define stage order
            stage_order = [
                "validation",
                "analysis",
                "transcription",
                "highlight_detection",
                "editing",
                "audio_normalization",
                "color_correction",
                "export",
            ]

            # Get all stages for this job
            stages_key = self._get_job_stages_key(job_id)
            completed_stages = self.redis_client.smembers(stages_key)

            if not completed_stages:
                # Try PostgreSQL
                records = self.db.find_many(
                    "job_checkpoint",
                    where={"job_id": job_id},
                    order_by="created_at DESC",
                )
                if records:
                    completed_stages = {r["stage"].encode() for r in records}
                else:
                    return None

            # Find the last stage in order that was completed
            completed_stages = {
                s.decode() if isinstance(s, bytes) else s for s in completed_stages
            }

            for stage in reversed(stage_order):
                if stage in completed_stages:
                    return stage

            return None

        except (
            RedisError,
            ValueError,
            TypeError,
            UnicodeDecodeError,
            AttributeError,
        ) as e:
            logger.error(f"Failed to get last successful stage: {e}")
            return None

    def delete_job_checkpoints(self, job_id: str) -> None:
        """Delete all checkpoints for a job"""
        try:
            # Get all stages
            stages_key = self._get_job_stages_key(job_id)
            stages = self.redis_client.smembers(stages_key)

            # Delete each checkpoint
            for stage in stages:
                stage_str = stage.decode() if isinstance(stage, bytes) else stage
                key = self._get_checkpoint_key(job_id, stage_str)
                self.redis_client.delete(key)

            # Delete stages list
            self.redis_client.delete(stages_key)

            logger.info(f"Deleted all checkpoints for job {job_id}")

        except (RedisError, UnicodeDecodeError, AttributeError) as e:
            logger.error(f"Failed to delete checkpoints: {e}")

    def exists(self, job_id: str, stage: str) -> bool:
        """Check if a checkpoint exists"""
        key = self._get_checkpoint_key(job_id, stage)
        return bool(self.redis_client.exists(key))

    @contextmanager
    def atomic_checkpoint(self, job_id: str, stage: str):
        """Context manager for atomic checkpoint operations"""
        temp_key = f"temp:{job_id}:{stage}:{time.time()}"
        completed = False

        try:
            yield
            completed = True
        finally:
            if completed:
                # Save successful checkpoint
                self.save_checkpoint(job_id, stage, {"completed": True})
            else:
                # Clean up any temporary data
                self.redis_client.delete(temp_key)

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a job"""
        try:
            stages_key = self._get_job_stages_key(job_id)
            completed_stages = self.redis_client.smembers(stages_key)
            completed_stages = {
                s.decode() if isinstance(s, bytes) else s for s in completed_stages
            }

            # Get checkpoint details
            checkpoints = []
            for stage in completed_stages:
                key = self._get_checkpoint_key(job_id, stage)
                ttl = self.redis_client.ttl(key)
                checkpoints.append(
                    {
                        "stage": stage,
                        "ttl_seconds": ttl,
                        "expires_at": (
                            (datetime.utcnow() + timedelta(seconds=ttl)).isoformat()
                            if ttl > 0
                            else None
                        ),
                    }
                )

            return {
                "job_id": job_id,
                "completed_stages": list(completed_stages),
                "last_stage": self.get_last_successful_stage(job_id),
                "checkpoints": checkpoints,
            }

        except (
            RedisError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            AttributeError,
        ) as e:
            logger.error(f"Failed to get job progress: {e}")
            return {"job_id": job_id, "error": str(e)}

    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.redis_client.ping()
        except (RedisError, ConnectionError, TimeoutError):
            return False


class SmartVideoEditorCheckpoint:
    """Integration class for SmartVideoEditor checkpointing"""

    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager

    def save_stage_data(self, job_id: str, stage: str, **kwargs) -> None:
        """Save stage-specific data"""
        self.checkpoint_manager.save_checkpoint(job_id, stage, kwargs)

    def load_stage_data(self, job_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Load stage-specific data"""
        return self.checkpoint_manager.load_checkpoint(job_id, stage)

    def should_skip_stage(self, job_id: str, stage: str) -> bool:
        """Check if a stage should be skipped based on checkpoints"""
        return self.checkpoint_manager.exists(job_id, stage)

    def get_resume_point(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the point from which to resume processing"""
        last_stage = self.checkpoint_manager.get_last_successful_stage(job_id)

        if not last_stage:
            return None

        # Map stages to next stage
        stage_progression = {
            "validation": "analysis",
            "analysis": "transcription",
            "transcription": "highlight_detection",
            "highlight_detection": "editing",
            "editing": "audio_normalization",
            "audio_normalization": "color_correction",
            "color_correction": "export",
        }

        next_stage = stage_progression.get(last_stage)

        if next_stage:
            # Load data from last successful stage
            checkpoint_data = self.load_stage_data(job_id, last_stage)

            return {
                "resume_from_stage": next_stage,
                "last_completed_stage": last_stage,
                "checkpoint_data": checkpoint_data,
            }

        return None


# Example usage in SmartVideoEditor
def example_usage():
    """Example of how to use checkpointing in SmartVideoEditor"""

    checkpoint_mgr = CheckpointManager()
    editor_checkpoint = SmartVideoEditorCheckpoint(checkpoint_mgr)

    job_id = "test-job-123"

    # Check if we should resume
    resume_info = editor_checkpoint.get_resume_point(job_id)

    if resume_info:
        print(f"Resuming from stage: {resume_info['resume_from_stage']}")
        # Skip to the appropriate stage

    # During processing, save checkpoints

    # After validation stage
    editor_checkpoint.save_stage_data(
        job_id, "validation", duration=120.5, codec="h264", resolution="1920x1080"
    )

    # After analysis stage
    editor_checkpoint.save_stage_data(
        job_id, "analysis", segments_found=15, total_score=0.85, processing_time=45.2
    )

    # Check if should skip a stage
    if editor_checkpoint.should_skip_stage(job_id, "analysis"):
        print("Skipping analysis stage - already completed")
        # Load previous results
        _ = editor_checkpoint.load_stage_data(job_id, "analysis")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    example_usage()
