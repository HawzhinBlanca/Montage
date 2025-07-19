"""Main orchestration script for AI Video Processing Pipeline"""

import argparse
import logging
import sys
import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.core.db import Database
from src.core.checkpoint import CheckpointManager
from src.utils.video_validator import VideoValidator
from src.providers.video_processor import SmartVideoEditor
from src.providers.transcript_analyzer import analyze_video_content
from src.providers.smart_crop import apply_smart_crop
from src.utils.budget_guard import (
    budget_manager,
    BudgetExceededError,
    with_budget_tracking,
)
from src.core.metrics import metrics
from src.utils.monitoring_integration import (
    MonitoringServer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoProcessingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self):
        self.db = Database()
        self.checkpoint_manager = CheckpointManager()
        self.validator = VideoValidator()
        self.editor = SmartVideoEditor()
        self.monitoring = MonitoringServer()

    def start_monitoring(self):
        """Start monitoring servers"""
        logger.info("Starting monitoring servers...")
        self.monitoring.start()
        time.sleep(2)  # Let servers start

    def create_job(
        self,
        input_path: str,
        output_path: str,
        edit_plan: Optional[Dict] = None,
        options: Dict[str, Any] = None,
    ) -> str:
        """Create a new processing job"""
        # Calculate source hash
        src_hash = self.validator._calculate_file_hash(input_path)

        # Check if job already exists
        existing = self.db.find_one("video_job", {"src_hash": src_hash})
        if existing and existing["status"] in ["completed", "processing"]:
            logger.info(f"Job already exists for this video: {existing['id']}")
            return existing["id"]

        # Create new job
        job_id = str(uuid.uuid4())
        self.db.insert(
            "video_job",
            {
                "id": job_id,
                "src_hash": src_hash,
                "status": "queued",
                "input_path": input_path,
                "output_path": output_path,
                "metadata": json.dumps(
                    {"edit_plan": edit_plan, "options": options or {}}
                ),
            },
        )

        logger.info(f"Created job: {job_id}")
        return job_id

    def process_job(self, job_id: str) -> Dict[str, Any]:
        """Process a single job with all stages"""
        logger.info(f"Processing job: {job_id}")

        # Load job
        job = self.db.find_one("video_job", {"id": job_id})
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(job_id)
        start_stage = checkpoint["stage"] if checkpoint else "validation"

        logger.info(f"Starting from stage: {start_stage}")

        # Update job status
        self.db.update("video_job", {"id": job_id}, {"status": "processing"})

        try:
            # Use budget tracking
            with with_budget_tracking(job_id):
                # Execute stages
                result = self._execute_stages(job_id, job, start_stage)

                # Update job completion
                self.db.update(
                    "video_job",
                    {"id": job_id},
                    {"status": "completed", "completed_at": datetime.utcnow()},
                )

                # Track completion
                metrics.track_job_complete(job_id, "completed")

                return result

        except BudgetExceededError as e:
            logger.error(f"Budget exceeded for job {job_id}: {e}")
            self._fail_job(job_id, str(e))
            raise

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self._fail_job(job_id, str(e))
            raise

    def execute_stage(
        self, stage: str, job_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage"""
        stage_result = {}

        if stage == "validation":
            # Pre-flight checks
            is_valid, reason = self.validator.validate_input(context["input_path"])
            if not is_valid:
                raise ValueError(f"Input validation failed: {reason}")

            # Store video metadata
            video_info = self.validator.get_video_info(context["input_path"])
            self.db.update(
                "video_job",
                {"id": job_id},
                {
                    "duration": video_info.duration,
                    "codec": video_info.codec,
                    "color_space": video_info.color_space,
                },
            )
            stage_result = {"status": "passed", "info": video_info}

        elif stage == "analysis":
            # Skip if we have edit plan
            if context.get("edit_plan"):
                logger.info("Skipping analysis - edit plan provided")
                stage_result = {"status": "skipped"}
            else:
                # Analyze video content
                highlights = analyze_video_content(
                    job_id,
                    context["input_path"],
                    context["input_path"],  # Use same for audio
                )
                stage_result = {
                    "status": "completed",
                    "highlights_found": len(highlights),
                }
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    job_id, "analysis", {"highlights": highlights}
                )
                context["highlights"] = highlights

        elif stage == "highlights":
            # Generate edit plan from highlights
            if not context.get("edit_plan"):
                highlights = context.get("highlights")
                if not highlights:
                    checkpoint = self.checkpoint_manager.load_checkpoint(job_id)
                    if checkpoint and checkpoint["stage"] == "analysis":
                        highlights = checkpoint["data"]["highlights"]
                    else:
                        # Re-analyze if needed
                        highlights = analyze_video_content(
                            job_id, context["input_path"], context["input_path"]
                        )
                # Convert to edit plan
                context["edit_plan"] = self._highlights_to_edit_plan(
                    highlights[:5]
                )  # Top 5

            stage_result = {
                "status": "completed",
                "segments": len(context["edit_plan"]["segments"]),
            }

        elif stage == "editing":
            # Execute edit
            if context["options"].get("smart_crop"):
                # Apply smart cropping for vertical
                temp_output = context["output_path"].replace(".mp4", "_temp.mp4")
                self.editor.execute_edit(
                    job_id, context["input_path"], context["edit_plan"], temp_output
                )

                crop_stats = apply_smart_crop(
                    temp_output,
                    context["output_path"],
                    aspect_ratio=context["options"].get("aspect_ratio", "9:16"),
                )
                os.remove(temp_output)
                stage_result["smart_crop"] = crop_stats
            else:
                # Regular edit
                self.editor.execute_edit(
                    job_id,
                    context["input_path"],
                    context["edit_plan"],
                    context["output_path"],
                )

            stage_result["status"] = "completed"

        elif stage == "encoding":
            # Already handled in editing
            stage_result = {"status": "completed"}

        elif stage == "finalization":
            # Verify output
            if not os.path.exists(context["output_path"]):
                raise ValueError("Output file not created")

            output_info = self.validator.get_video_info(context["output_path"])
            stage_result["output"] = {
                "path": context["output_path"],
                "duration": output_info.duration,
                "size_mb": os.path.getsize(context["output_path"]) / (1024 * 1024),
            }

        return stage_result

    def dispatch_stage(
        self,
        stages: List[str],
        job_id: str,
        context: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Dispatch stages and handle checkpointing"""
        for stage in stages:
            logger.info(f"Executing stage: {stage}")

            # Execute the stage
            stage_result = self.execute_stage(stage, job_id, context)
            result[stage] = stage_result

            # Save checkpoint after each stage
            self.checkpoint_manager.save_checkpoint(job_id, stage, result)

    def _execute_stages(
        self, job_id: str, job: Dict, start_stage: str
    ) -> Dict[str, Any]:
        """Execute pipeline stages"""
        stages = [
            "validation",
            "analysis",
            "highlights",
            "editing",
            "encoding",
            "finalization",
        ]

        # Find starting index
        start_index = stages.index(start_stage) if start_stage in stages else 0

        # Build context
        metadata = json.loads(job.get("metadata", "{}"))
        context = {
            "input_path": job.get("input_path"),
            "output_path": job.get("output_path"),
            "edit_plan": metadata.get("edit_plan"),
            "options": metadata.get("options", {}),
        }

        result = {}

        # Execute stages
        self.dispatch_stage(stages[start_index:], job_id, context, result)

        # Clear checkpoints on success
        self.checkpoint_manager.clear_checkpoint(job_id)

        return result

    def _highlights_to_edit_plan(self, highlights: List[Dict]) -> Dict[str, List[Dict]]:
        """Convert highlights to edit plan"""
        segments = []

        for i, highlight in enumerate(highlights):
            # Ensure 15-60 second duration
            duration = highlight["end_time"] - highlight["start_time"]
            if duration < 15:
                # Extend to 15s
                highlight["end_time"] = highlight["start_time"] + 15
            elif duration > 60:
                # Trim to 60s
                highlight["end_time"] = highlight["start_time"] + 60

            segments.append(
                {
                    "start": highlight["start_time"],
                    "end": highlight["end_time"],
                    "transition": "fade" if i < len(highlights) - 1 else None,
                    "transition_duration": 0.5,
                }
            )

        return {"segments": segments}

    def _fail_job(self, job_id: str, error_message: str):
        """Mark job as failed"""
        self.db.update(
            "video_job",
            {"id": job_id},
            {
                "status": "failed",
                "error_message": error_message,
                "completed_at": datetime.utcnow(),
            },
        )

        metrics.track_job_complete(job_id, "failed")

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current job status"""
        job = self.db.find_one("video_job", {"id": job_id})
        if not job:
            return {"error": "Job not found"}

        # Get cost breakdown
        cost_breakdown = budget_manager.get_job_cost_breakdown(job_id)

        return {
            "id": job["id"],
            "status": job["status"],
            "created_at": job["created_at"].isoformat() if job["created_at"] else None,
            "completed_at": (
                job["completed_at"].isoformat() if job.get("completed_at") else None
            ),
            "duration": job.get("duration"),
            "total_cost": job.get("total_cost", 0),
            "cost_breakdown": cost_breakdown,
            "error": job.get("error_message"),
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Video Processing Pipeline")

    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    parser.add_argument("--edit-plan", help="JSON file with edit plan")
    parser.add_argument(
        "--smart-crop", action="store_true", help="Apply smart cropping"
    )
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio")
    parser.add_argument("--job-id", help="Resume existing job")
    parser.add_argument("--status", action="store_true", help="Check job status")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VideoProcessingPipeline()

    # Start monitoring
    pipeline.start_monitoring()

    try:
        if args.status and args.job_id:
            # Check status
            status = pipeline.get_job_status(args.job_id)
            print(json.dumps(status, indent=2))

        else:
            # Load edit plan if provided
            edit_plan = None
            if args.edit_plan:
                with open(args.edit_plan, "r") as f:
                    edit_plan = json.load(f)

            # Create or resume job
            if args.job_id:
                job_id = args.job_id
            else:
                job_id = pipeline.create_job(
                    args.input,
                    args.output,
                    edit_plan=edit_plan,
                    options={
                        "smart_crop": args.smart_crop,
                        "aspect_ratio": args.aspect_ratio,
                    },
                )

            # Process job
            logger.info(f"Processing job {job_id}...")
            result = pipeline.process_job(job_id)

            # Print result
            print("\n✅ Processing complete!")
            print(json.dumps(result, indent=2))
            print(f"\nOutput saved to: {args.output}")

            # Get final status
            status = pipeline.get_job_status(job_id)
            print(f"\nTotal cost: ${status['total_cost']:.2f}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
