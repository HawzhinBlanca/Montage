"""Professional video effects and transitions using real FFmpeg functionality"""

import logging
import os
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def create_professional_video(
    clips: List[Dict],
    source_video: str,
    output_path: str,
    vertical_format: bool = False,
    enhance_colors: bool = True,
    add_transitions: bool = True,
    subtitles_path: Optional[str] = None,
) -> bool:
    """Create professional video with real FFmpeg transitions and effects

    Args:
        clips: List of clip dictionaries with start_ms, end_ms
        source_video: Path to source video file
        output_path: Path for output video
        vertical_format: Whether to crop for vertical (9:16) format
        enhance_colors: Apply basic color enhancement
        add_transitions: Add fade transitions between clips
        subtitles_path: Path to SRT subtitles file (optional)

    Returns:
        bool: True if successful, False otherwise
    """

    try:
        # Create temp directory for processing
        temp_dir = "/tmp/montage_pro"
        os.makedirs(temp_dir, exist_ok=True)

        # Step 1: Extract and process individual clips
        processed_clips = []
        for i, clip in enumerate(clips):
            # Extract clip with padding for transitions
            clip_path = f"{temp_dir}/clip_{i:03d}.mp4"

            # Add 0.5s padding for transitions
            start_ms = max(0, clip["start_ms"] - 500)
            end_ms = clip["end_ms"] + 500
            duration_ms = end_ms - start_ms

            # Extract with fade effects
            extract_cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_ms / 1000),
                "-t",
                str(duration_ms / 1000),
                "-i",
                source_video,
            ]

            # Add video filters
            filters = []

            # Fade in/out for smooth transitions
            fade_duration = 0.3
            filters.append(f"fade=t=in:st=0:d={fade_duration}")
            filters.append(
                f"fade=t=out:st={duration_ms/1000 - fade_duration}:d={fade_duration}"
            )

            # Add subtle zoom effect for visual interest
            if i == 0:  # First clip gets zoom in
                filters.append(
                    "scale=iw*1.05:ih*1.05,zoompan=z='min(zoom+0.0005,1.05)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=125"
                )
            elif i == len(clips) - 1:  # Last clip gets zoom out
                filters.append(
                    "scale=iw*1.05:ih*1.05,zoompan=z='1.05-0.05*in/125':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=125"
                )

            # Apply vertical format if needed
            if vertical_format:
                # Use intelligent cropping for better vertical format
                try:
                    from .intelligent_crop import IntelligentCropper
                    cropper = IntelligentCropper()
                    analysis = cropper.analyze_video_content(
                        source_video, clip["start_ms"], clip["end_ms"]
                    )
                    crop_center = analysis["crop_center"]
                except ImportError:
                    # Fallback if IntelligentCropper not available
                    crop_center = {"x": 960, "y": 540}

                # Get actual dimensions from video metadata
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0",
                    source_video
                ]
                try:
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    dimensions = result.stdout.strip().split('x')
                    input_width = int(dimensions[0])
                    input_height = int(dimensions[1])
                except Exception:
                    # Fallback to common dimensions if probe fails
                    input_width, input_height = 1920, 1080

                if 'cropper' in locals():
                    crop_filter = cropper.generate_crop_filter(
                        input_width, input_height, crop_center
                    )
                else:
                    # Fallback crop filter for 9:16 vertical format
                    crop_width = min(input_width, int(input_height * 9 / 16))
                    crop_height = input_height
                    crop_x = (input_width - crop_width) // 2
                    crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:0"

                filters.append(crop_filter)
                # Add subtle vignette for social media appeal
                filters.append("vignette=PI/8")
            else:
                # Standard scaling for landscape
                filters.append("scale=1920:1080")

            # Basic color enhancement for professional look
            filters.append("eq=brightness=0.02:saturation=1.1:contrast=1.05")

            # Combine filters
            if filters:
                extract_cmd.extend(["-vf", ",".join(filters)])

            # High quality encoding
            extract_cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    clip_path,
                ]
            )

            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                processed_clips.append(
                    {"path": clip_path, "clip": clip, "duration": duration_ms / 1000}
                )
            else:
                logger.error(f"Failed to process clip {i}: {result.stderr}")
                return False

        # Step 2: Create transition segments
        transitions = []
        for i in range(len(processed_clips) - 1):
            trans_path = f"{temp_dir}/trans_{i:03d}.mp4"

            # Use consistent smooth transitions
            transition_type = "xfade"
            duration = 0.3  # Standard transition duration

            # Create transition using xfade filter
            trans_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                processed_clips[i]["path"],
                "-i",
                processed_clips[i + 1]["path"],
                "-filter_complex",
                f"[0:v][1:v]{transition_type}=transition=fade:duration={duration}:offset={processed_clips[i]['duration'] - duration}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                trans_path,
            ]

            subprocess.run(trans_cmd, capture_output=True)
            transitions.append(trans_path)

        # Step 3: Final assembly with audio
        concat_list = f"{temp_dir}/final_concat.txt"
        with open(concat_list, "w") as f:
            for i, clip in enumerate(processed_clips):
                f.write(f"file '{clip['path']}'\n")
                if i < len(transitions):
                    f.write(f"file '{transitions[i]}'\n")

        # Final render command
        final_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
        ]

        # Add subtitles if provided
        if subtitles_path and os.path.exists(subtitles_path):
            # Burn in stylish captions
            subtitle_style = "FontName=Arial,FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=30"
            if vertical_format:
                # Larger text for mobile viewing
                subtitle_style = "FontName=Arial Black,FontSize=32,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=1,Outline=3,Shadow=2,MarginV=100"

            final_cmd.extend(
                ["-vf", f"subtitles={subtitles_path}:force_style='{subtitle_style}'"]
            )

        # Professional encoding settings
        final_cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "17",  # High quality
                "-profile:v",
                "high",
                "-level",
                "4.2",
                "-c:a",
                "aac",
                "-b:a",
                "256k",  # High quality audio
                "-ar",
                "48000",
                "-movflags",
                "+faststart",  # Optimize for streaming
                output_path,
            ]
        )

        # Execute final render
        result = subprocess.run(final_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Clean up temp files
            for clip in processed_clips:
                try:
                    os.unlink(clip["path"])
                except Exception as e:
                    logger.debug(f"Failed to delete clip file {clip['path']}: {e}")
            for trans in transitions:
                try:
                    os.unlink(trans)
                except Exception as e:
                    logger.debug(f"Failed to delete transition file {trans}: {e}")

            logger.info(f"✅ Professional video created: {output_path}")
            return True
        else:
            logger.error(f"Final render failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Professional video creation failed: {e}")
        return False
