#!/usr/bin/env python3
"""
FFmpeg utilities for video processing
"""
import os
import tempfile
import subprocess
from typing import List, Dict, Any
import ffmpeg

# Import centralized logging
try:
    from ..utils.logging_config import get_logger
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logging_config import get_logger

logger = get_logger(__name__)


def make_concat_file(video_clips: List[Dict[str, Any]], source_video: str) -> str:
    """Create FFmpeg concat file for multiple video segments"""

    # Create temporary concat file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name

        for clip in video_clips:
            start_ms = clip["start_ms"]
            end_ms = clip["end_ms"]

            # Write segment info
            f.write(f"file '{source_video}'\n")
            f.write(f"inpoint {start_ms/1000.0}\n")
            f.write(f"outpoint {end_ms/1000.0}\n")

    return concat_file


def extract_video_segment(
    source_video: str, start_ms: int, end_ms: int, output_path: str
) -> bool:
    """Extract a specific segment from video"""
    try:
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0

        (
            ffmpeg.input(source_video, ss=start_sec, t=duration_sec)
            .output(output_path, c="copy")  # Copy without re-encoding for speed
            .overwrite_output()
            .run(quiet=True)
        )

        return True

    except (ffmpeg.Error, OSError, ValueError, TypeError) as e:
        logger.error(
            "Failed to extract video segment",
            extra={"error_type": type(e).__name__},
            exc_info=True,
        )
        return False


def concatenate_video_segments(
    video_clips: List[Dict[str, Any]],
    source_video: str,
    output_path: str,
    vertical_format: bool = False,
    professional: bool = True,
) -> bool:
    """Concatenate multiple video segments into one video"""

    # Use professional video creation if enabled
    if professional and len(video_clips) > 0:
        from .video_effects import create_professional_video

        # Check if we have story metadata
        story_theme = video_clips[0].get("story_theme", None) if video_clips else None
        music_mood = (
            video_clips[0].get("suggested_music", None) if video_clips else None
        )

        # Try professional creation first
        if create_professional_video(
            video_clips,
            source_video,
            output_path,
            vertical_format,
            story_theme,
            music_mood,
        ):
            return True
        else:
            logger.warning(
                "Professional video creation failed, falling back to standard processing"
            )

    # Standard concatenation
    try:
        # Create temporary files for each segment
        temp_files = []

        for i, clip in enumerate(video_clips):
            temp_file = f"/tmp/segment_{i}.mp4"

            if extract_video_segment(
                source_video, clip["start_ms"], clip["end_ms"], temp_file
            ):
                temp_files.append(temp_file)
            else:
                logger.error(
                    "Failed to extract video segment during batch processing",
                    extra={"segment_index": i},
                )
                return False

        if not temp_files:
            logger.error("No segments available for concatenation")
            return False

        # Create concat file
        concat_file = "/tmp/concat_list.txt"
        with open(concat_file, "w") as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")

        # Concatenate segments with optional vertical scaling
        if vertical_format:
            # Use intelligent cropping for vertical format
            from .intelligent_crop import create_intelligent_vertical_video

            logger.info(
                "Starting intelligent face detection and content-aware cropping"
            )

            if create_intelligent_vertical_video(
                video_clips, source_video, output_path
            ):
                return True
            else:
                print(
                    "⚠️  Intelligent cropping failed, falling back to basic vertical scaling"
                )

                # Fallback to improved vertical scaling (still better than black bars)
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file,
                    "-vf",
                    "scale='if(gt(a,9/16),1080,-1)':'if(gt(a,9/16),-1,1920)',crop=1080:1920,fps=30",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    output_path,
                ]
        else:
            # Standard concatenation
            cmd = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                "-y",
                output_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Concatenated {len(temp_files)} segments to {output_path}")
            success = True
        else:
            print(f"❌ Concatenation failed: {result.stderr}")
            success = False

        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError, PermissionError):
                pass

        try:
            os.unlink(concat_file)
        except (OSError, FileNotFoundError, PermissionError):
            pass

        return success

    except (
        subprocess.CalledProcessError,
        OSError,
        ValueError,
        FileNotFoundError,
        ffmpeg.Error,
    ) as e:
        print(f"❌ Concatenation error: {e}")
        return False


def create_subtitle_file(
    words: List[Dict[str, Any]], start_ms: int, end_ms: int, output_path: str
) -> bool:
    """Create SRT subtitle file for a video segment"""
    try:
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        # Filter words within the segment
        segment_words = [
            word for word in words if start_sec <= word["start"] <= end_sec
        ]

        if not segment_words:
            print("❌ No words in segment for subtitles")
            return False

        # Group words into subtitle lines (every 3-5 seconds) - FIXED TIMING
        subtitle_lines = []
        current_line = []
        line_start = segment_words[0]["start"]
        prev_word = None
        word_idx = 0

        for word_idx, word in enumerate(segment_words):
            # Skip consecutive duplicate words
            if prev_word and word["word"].lower().strip() == prev_word.lower().strip():
                continue
            current_line.append(word["word"])
            prev_word = word["word"]

            # End line after 3-5 seconds or at sentence boundary
            if (
                word["end"] - line_start >= 3
                and word["word"].endswith((".", "!", "?", ","))
            ) or (word["end"] - line_start >= 5):

                # Clean duplicate words in the final text as well
                text_words = " ".join(current_line).split()
                cleaned_text = []
                prev = None
                for w in text_words:
                    if prev is None or w.lower() != prev.lower():
                        cleaned_text.append(w)
                    prev = w

                # Calculate relative times
                rel_start = line_start - start_sec
                rel_end = word["end"] - start_sec

                # Ensure no overlap with previous subtitle
                if subtitle_lines and rel_start < subtitle_lines[-1]["end"]:
                    rel_start = subtitle_lines[-1]["end"] + 0.1  # 100ms gap

                # Ensure end is after start
                if rel_end <= rel_start:
                    rel_end = rel_start + 0.5  # Minimum 500ms duration

                subtitle_lines.append(
                    {
                        "text": " ".join(cleaned_text),
                        "start": rel_start,
                        "end": rel_end,
                    }
                )

                current_line = []
                # Set next line start - FIXED: use proper next word
                if word_idx + 1 < len(segment_words):
                    line_start = segment_words[word_idx + 1]["start"]

        # Add any remaining words - FIXED TIMING
        if current_line:
            # Clean duplicate words in the final text as well
            text_words = " ".join(current_line).split()
            cleaned_text = []
            prev = None
            for w in text_words:
                if prev is None or w.lower() != prev.lower():
                    cleaned_text.append(w)
                prev = w

            # Calculate relative times with validation
            rel_start = line_start - start_sec
            rel_end = segment_words[-1]["end"] - start_sec

            # Ensure no overlap with previous subtitle
            if subtitle_lines and rel_start < subtitle_lines[-1]["end"]:
                rel_start = subtitle_lines[-1]["end"] + 0.1  # 100ms gap

            # Ensure end is after start
            if rel_end <= rel_start:
                rel_end = rel_start + 0.5  # Minimum 500ms duration

            subtitle_lines.append(
                {
                    "text": " ".join(cleaned_text),
                    "start": rel_start,
                    "end": rel_end,
                }
            )

        # FINAL VALIDATION: Ensure no timing issues
        for i in range(len(subtitle_lines) - 1):
            current = subtitle_lines[i]
            next_line = subtitle_lines[i + 1]

            # Fix any remaining overlaps
            if current["end"] > next_line["start"]:
                current["end"] = next_line["start"] - 0.05  # 50ms gap

            # Ensure minimum duration
            if current["end"] - current["start"] < 0.2:
                current["end"] = current["start"] + 0.2

        # Write SRT file
        with open(output_path, "w", encoding="utf-8") as f:
            for i, line in enumerate(subtitle_lines, 1):
                start_time = format_srt_time(line["start"])
                end_time = format_srt_time(line["end"])

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{line['text']}\n\n")

        print(f"✅ Created subtitle file: {output_path} ({len(subtitle_lines)} lines)")
        return True

    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        print(f"❌ Subtitle creation failed: {e}")
        return False


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def burn_captions(video_path: str, srt_path: str, output_path: str) -> bool:
    """Burn subtitles into video"""
    try:
        if not os.path.exists(srt_path):
            print(f"❌ Subtitle file not found: {srt_path}")
            return False

        # Use ffmpeg to burn subtitles
        (
            ffmpeg.input(video_path)
            .output(
                output_path,
                vf=f"subtitles={srt_path}:force_style='Fontsize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'",
                c="libx264",
                crf=23,
            )
            .overwrite_output()
            .run(quiet=True)
        )

        print(f"✅ Burned subtitles into: {output_path}")
        return True

    except (ffmpeg.Error, OSError, FileNotFoundError, ValueError) as e:
        print(f"❌ Subtitle burning failed: {e}")
        return False


def apply_smart_crop(
    video_path: str, output_path: str, crop_zone: str = "center"
) -> bool:
    """Apply smart crop to video"""
    try:
        # Get video dimensions
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )

        if not video_stream:
            print("❌ No video stream found")
            return False

        width = int(video_stream["width"])
        height = int(video_stream["height"])

        # Calculate crop parameters for 9:16 aspect ratio
        target_width = min(width, int(height * 9 / 16))
        target_height = height

        if crop_zone == "center":
            x_offset = (width - target_width) // 2
        elif crop_zone == "left":
            x_offset = 0
        elif crop_zone == "right":
            x_offset = width - target_width
        else:
            x_offset = (width - target_width) // 2  # Default to center

        # Apply crop
        (
            ffmpeg.input(video_path)
            .filter("crop", target_width, target_height, x_offset, 0)
            .output(output_path, c="libx264", crf=23)
            .overwrite_output()
            .run(quiet=True)
        )

        print(
            f"✅ Smart crop applied: {crop_zone} crop to {target_width}x{target_height}"
        )
        return True

    except (ffmpeg.Error, OSError, ValueError, KeyError, TypeError) as e:
        print(f"❌ Smart crop failed: {e}")
        return False


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "audio"),
            None,
        )

        info = {
            "duration": float(probe["format"]["duration"]),
            "size": int(probe["format"]["size"]),
            "bitrate": int(probe["format"]["bit_rate"]),
        }

        if video_stream:
            info.update(
                {
                    "width": int(video_stream["width"]),
                    "height": int(video_stream["height"]),
                    "fps": eval(video_stream["r_frame_rate"]),
                    "video_codec": video_stream["codec_name"],
                }
            )

        if audio_stream:
            info.update(
                {
                    "audio_codec": audio_stream["codec_name"],
                    "sample_rate": int(audio_stream["sample_rate"]),
                    "channels": int(audio_stream["channels"]),
                }
            )

        return info

    except (ffmpeg.Error, OSError, ValueError, KeyError, TypeError) as e:
        print(f"❌ Failed to get video info: {e}")
        return {}
