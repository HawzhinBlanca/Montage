"""Smart Video Editor Pipeline - AI Creative Director Integration"""
import os
import json
import logging
from typing import Dict, Optional
from ..ai.director import AICreativeDirector

logger = logging.getLogger(__name__)

def create_smart_video(
    video_path: str,
    target_duration: int = 60,
    output_path: Optional[str] = None,
    platform: str = "tiktok"
) -> Dict:
    """
    Create smart video using AI Creative Director

    Args:
        video_path: Input video file path
        target_duration: Target duration in seconds
        output_path: Output file path (optional)
        platform: Target platform (tiktok, youtube, etc.)

    Returns:
        Result dictionary with success status and paths
    """
    try:
        logger.info(f"🎬 AI Creative Director starting analysis of {video_path}")

        # Initialize AI Creative Director
        director = AICreativeDirector()

        # Create intelligent edit plan
        edit_plan = director.create_smart_edit(video_path, target_duration)

        # Generate output paths
        if not output_path:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"/tmp/smart_edit_{base_name}_{platform}.mp4"

        plan_path = output_path.replace(".mp4", "_plan.json")

        # Save edit plan
        with open(plan_path, 'w') as f:
            json.dump(edit_plan, f, indent=2)

        logger.info(f"📋 Edit plan saved: {plan_path}")
        logger.info(f"🎯 Selected {len(edit_plan['clips'])} highlights")

        # Execute the plan using existing pipeline
        success = execute_edit_plan(edit_plan, output_path, platform)

        return {
            "success": success,
            "output_video": output_path if success else None,
            "edit_plan": plan_path,
            "clips_selected": len(edit_plan['clips']),
            "total_duration": edit_plan['metadata']['total_duration'],
            "ai_director_version": edit_plan['metadata']['ai_director']
        }

    except Exception as e:
        logger.error(f"Smart video creation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "output_video": None
        }

def execute_edit_plan(edit_plan: Dict, output_path: str, platform: str) -> bool:
    """Execute the AI-generated edit plan"""
    try:
        # Use existing execute_plan_from_cli logic
        from ..cli.run_pipeline import execute_plan_from_cli

        # Convert AI edit plan to CLI-compatible format
        cli_plan = convert_to_cli_plan(edit_plan)

        # Save temporary plan file
        temp_plan_path = "/tmp/ai_edit_plan.json"
        with open(temp_plan_path, 'w') as f:
            json.dump(cli_plan, f)

        # Execute using existing pipeline
        execute_plan_from_cli(temp_plan_path, output_path)

        # Cleanup
        os.unlink(temp_plan_path)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0

    except Exception as e:
        logger.error(f"Edit plan execution failed: {e}")
        return False

def convert_to_cli_plan(ai_plan: Dict) -> Dict:
    """Convert AI Creative Director plan to CLI format"""
    cli_clips = []

    for clip in ai_plan["clips"]:
        cli_clips.append({
            "start": clip["start_time"],
            "end": clip["end_time"],
            "start_time": clip["start_time"],
            "end_time": clip["end_time"]
        })

    return {
        "version": "1.0",
        "source": ai_plan["source_video"],
        "source_video_path": ai_plan["source_video"],
        "clips": cli_clips,
        "actions": cli_clips  # Compatibility with both formats
    }
