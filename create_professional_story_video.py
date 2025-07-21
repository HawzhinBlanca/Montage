#!/usr/bin/env python3
"""
Create professional video using EXISTING proven pipeline + approved story structure
Use the working components, not amateur rewrites
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Disable external APIs to avoid hangs
os.environ["DEEPGRAM_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""


def load_approved_story():
    """Load the creative director approved story structure"""
    try:
        with open("approved_story_structure.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ No approved story structure found")
        return None


def convert_to_highlight_format(story_data):
    """Convert approved story segments to existing highlight format"""

    approved_segments = story_data.get("approved_segments", [])

    highlights = []
    for i, segment in enumerate(approved_segments):
        highlight = {
            "title": segment["beat_type"].replace("_", " ").title()[:40],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "start_ms": int(segment["start_time"] * 1000),
            "end_ms": int(segment["end_time"] * 1000),
            "duration": segment["duration"],
            "text": segment["text"][:100] + "...",
            "score": segment["score"],
            "purpose": segment["purpose"],
        }
        highlights.append(highlight)

    return highlights


def create_professional_video_with_existing_tools(highlights, output_path):
    """Use EXISTING professional tools to create the video"""

    print(f"🎬 Creating professional video with {len(highlights)} story segments...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Import EXISTING professional components
        from src.utils.intelligent_crop import create_intelligent_vertical_video
        from src.providers.video_processor import VideoEditor

        print("✅ Using existing professional video processing system")

        # Convert highlights to format expected by intelligent_crop
        video_clips = []
        for highlight in highlights:
            clip = {
                "start_ms": highlight["start_ms"],
                "end_ms": highlight["end_ms"],
                "title": highlight["title"],
                "text": highlight.get("text", ""),
                "score": highlight.get("score", 1.0),
            }
            video_clips.append(clip)

        print(f"   📊 Processing {len(video_clips)} clips...")
        for i, clip in enumerate(video_clips):
            duration = (clip["end_ms"] - clip["start_ms"]) / 1000
            print(f"   {i+1}. {clip['title']} ({duration:.1f}s)")

        # Use the EXISTING professional system
        success = create_intelligent_vertical_video(
            video_clips=video_clips,
            source_video="/Users/hawzhin/Montage/test_video.mp4",
            output_path=output_path,
        )

        return success

    except ImportError as e:
        print(f"❌ Could not import existing tools: {e}")
        print("   Falling back to direct FFmpeg approach...")
        return create_video_direct_ffmpeg(highlights, output_path)


def create_video_direct_ffmpeg(highlights, output_path):
    """Fallback: Direct FFmpeg using proven commands from existing system"""

    print("🔧 Using direct FFmpeg with proven settings...")

    # Create individual clips using EXISTING proven approach
    clip_files = []

    for i, highlight in enumerate(highlights):
        clip_file = f"temp_professional_clip_{i:03d}.mp4"

        start_time = highlight["start_time"]
        duration = highlight["end_time"] - highlight["start_time"]

        print(f"   Creating clip {i+1}: {highlight['title']} ({duration:.1f}s)")

        # Use PROVEN FFmpeg settings from existing system (no amateur fades)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            "/Users/hawzhin/Montage/test_video.mp4",
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
            "-avoid_negative_ts",
            "make_zero",
            clip_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(clip_file):
            clip_files.append(clip_file)
        else:
            print(f"   ❌ Failed to create clip {i+1}: {result.stderr}")
            return False

    if not clip_files:
        print("❌ No clips created successfully")
        return False

    # Concatenate using PROVEN method from existing system
    concat_file = "professional_concat.txt"
    with open(concat_file, "w") as f:
        for clip_file in clip_files:
            f.write(f"file '{clip_file}'\n")

    print(f"🔗 Concatenating {len(clip_files)} professional clips...")

    # Use PROVEN concatenation settings
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "quiet",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c",
        "copy",  # Don't re-encode (preserves quality)
        "-movflags",
        "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Cleanup temp files
    for clip_file in clip_files:
        try:
            os.unlink(clip_file)
        except:
            pass
    try:
        os.unlink(concat_file)
    except:
        pass

    if result.returncode == 0:
        print("✅ Professional video concatenation complete")
        return True
    else:
        print(f"❌ Concatenation failed: {result.stderr}")
        return False


def analyze_output_quality(video_path):
    """Analyze the actual output quality"""

    if not os.path.exists(video_path):
        return {"error": "Video file does not exist"}

    print(f"🔍 Analyzing output quality: {video_path}")

    # Get comprehensive video info
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": "Could not analyze video"}

    try:
        video_info = json.loads(result.stdout)
        format_info = video_info.get("format", {})

        # Find video and audio streams
        video_stream = None
        audio_stream = None

        for stream in video_info.get("streams", []):
            if stream.get("codec_type") == "video" and not video_stream:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and not audio_stream:
                audio_stream = stream

        if not video_stream:
            return {"error": "No video stream found"}

        # Extract quality metrics
        analysis = {
            "file_size_mb": round(int(format_info.get("size", 0)) / (1024 * 1024), 2),
            "duration_seconds": round(float(format_info.get("duration", 0)), 1),
            "video": {
                "codec": video_stream.get("codec_name", "unknown"),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "aspect_ratio": video_stream.get("sample_aspect_ratio", "1:1"),
                "frame_rate": (
                    eval(video_stream.get("avg_frame_rate", "30/1"))
                    if "/" in video_stream.get("avg_frame_rate", "30/1")
                    else 30
                ),
                "bitrate": (
                    int(video_stream.get("bit_rate", 0))
                    if video_stream.get("bit_rate")
                    else None
                ),
            },
            "audio": {
                "codec": (
                    audio_stream.get("codec_name", "none") if audio_stream else "none"
                ),
                "sample_rate": (
                    int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
                ),
                "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
                "bitrate": (
                    int(audio_stream.get("bit_rate", 0))
                    if audio_stream and audio_stream.get("bit_rate")
                    else None
                ),
            },
        }

        # Quality assessment
        issues = []
        quality_score = 10

        # Check video quality
        if analysis["video"]["width"] < 1920 or analysis["video"]["height"] < 1080:
            issues.append(
                f"Low resolution: {analysis['video']['width']}x{analysis['video']['height']}"
            )
            quality_score -= 2

        if analysis["video"]["aspect_ratio"] != "1:1":
            issues.append(
                f"Incorrect aspect ratio: {analysis['video']['aspect_ratio']} (should be 1:1)"
            )
            quality_score -= 3

        if analysis["video"]["frame_rate"] < 24:
            issues.append(f"Low frame rate: {analysis['video']['frame_rate']} fps")
            quality_score -= 1

        # Check audio quality
        if audio_stream is None:
            issues.append("No audio stream")
            quality_score -= 5
        elif analysis["audio"]["channels"] < 2:
            issues.append(f"Mono audio (should be stereo)")
            quality_score -= 2
        elif analysis["audio"]["sample_rate"] < 44100:
            issues.append(f"Low audio quality: {analysis['audio']['sample_rate']} Hz")
            quality_score -= 1

        # File size check
        duration = analysis["duration_seconds"]
        if duration > 0:
            mb_per_minute = analysis["file_size_mb"] / (duration / 60)
            if mb_per_minute < 10:
                issues.append(f"Low bitrate: {mb_per_minute:.1f} MB/min")
                quality_score -= 1
            elif mb_per_minute > 100:
                issues.append(f"Very high bitrate: {mb_per_minute:.1f} MB/min")

        analysis["quality_score"] = max(0, quality_score)
        analysis["issues"] = issues
        analysis["quality_rating"] = (
            "PROFESSIONAL"
            if quality_score >= 9
            else (
                "GOOD"
                if quality_score >= 7
                else "ACCEPTABLE" if quality_score >= 5 else "POOR"
            )
        )

        return analysis

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return {"error": f"Analysis failed: {e}"}


def main():
    """Create professional video using approved story and existing tools"""

    print("🎬 PROFESSIONAL STORY VIDEO CREATION")
    print("=" * 60)
    print("Using EXISTING proven pipeline + Creative Director approved story")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load approved story
    story_data = load_approved_story()
    if not story_data:
        print("🛑 No approved story structure found")
        return

    print(f"✅ Loaded approved story: {len(story_data['approved_segments'])} segments")
    print(f"   Quality rating: {story_data['quality_analysis']['rating']}/10")
    print(f"   Total duration: {story_data['story_structure']['total_duration']:.1f}s")

    # Convert to highlight format
    highlights = convert_to_highlight_format(story_data)

    # Create output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/professional_story_{timestamp}.mp4"

    # Create professional video
    print(f"\n🎯 Creating professional video...")
    success = create_professional_video_with_existing_tools(highlights, output_path)

    if success and os.path.exists(output_path):
        # Analyze output quality
        print(f"\n🔍 QUALITY ANALYSIS:")
        quality = analyze_output_quality(output_path)

        if "error" in quality:
            print(f"   ❌ {quality['error']}")
        else:
            print(f"   📁 File: {output_path}")
            print(f"   💾 Size: {quality['file_size_mb']} MB")
            print(f"   ⏱️  Duration: {quality['duration_seconds']}s")
            print(
                f"   📺 Resolution: {quality['video']['width']}x{quality['video']['height']}"
            )
            print(f"   🎭 Aspect: {quality['video']['aspect_ratio']}")
            print(f"   🎬 Codec: {quality['video']['codec']}")
            print(
                f"   🎵 Audio: {quality['audio']['codec']} {quality['audio']['channels']}ch"
            )
            print(f"   📊 Quality Score: {quality['quality_score']}/10")
            print(f"   🏆 Rating: {quality['quality_rating']}")

            if quality["issues"]:
                print(f"   ⚠️  Issues:")
                for issue in quality["issues"]:
                    print(f"      • {issue}")

            # Overall verdict
            print(f"\n" + "=" * 60)
            if quality["quality_score"] >= 8:
                print(f"✅ PROFESSIONAL QUALITY ACHIEVED")
                print(f"   Video meets professional standards")
            elif quality["quality_score"] >= 6:
                print(f"⚠️  GOOD QUALITY - Minor issues")
                print(f"   Video is usable but has some issues")
            else:
                print(f"❌ POOR QUALITY - Major issues")
                print(f"   Video needs significant improvement")

            # Test playback
            print(f"\n🎥 Testing playback...")
            test_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                output_path,
            ]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True)

            if test_result.returncode == 0:
                packets = test_result.stdout.strip()
                print(f"   ✅ Video is playable ({packets} video packets)")
            else:
                print(f"   ❌ Video may have playback issues")

        print(f"\n🎭 Professional video creation complete!")
        print(f"   📂 Output: {output_path}")

    else:
        print(f"❌ Professional video creation failed")


if __name__ == "__main__":
    main()
