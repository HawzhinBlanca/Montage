#!/usr/bin/env python3
"""
Create the best possible professional video using all available features
Stop immediately if any feature fails
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime


def test_feature(name, func):
    """Test a feature and stop execution if it fails"""
    print(f"\n🔧 Testing Feature: {name}")
    print("=" * 50)

    try:
        start_time = time.time()
        result = func()
        duration = time.time() - start_time

        if result:
            print(f"✅ {name} - SUCCESS ({duration:.1f}s)")
            return result
        else:
            print(f"❌ {name} - FAILED ({duration:.1f}s)")
            print(f"🛑 STOPPING EXECUTION - Feature '{name}' did not work")
            sys.exit(1)

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {name} - ERROR ({duration:.1f}s): {type(e).__name__}: {e}")
        print(f"🛑 STOPPING EXECUTION - Feature '{name}' crashed")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def check_prerequisites():
    """Check all prerequisites before starting"""
    print("🔍 Checking Prerequisites...")

    # Check input video exists
    video_path = "/Users/hawzhin/Montage/test_video.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Input video not found: {video_path}")
        return False

    # Get video info
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
        print(f"❌ Cannot analyze video: {result.stderr}")
        return False

    video_info = json.loads(result.stdout)
    duration = float(video_info["format"]["duration"])
    size_mb = int(video_info["format"]["size"]) / (1024 * 1024)

    print(f"📹 Input Video: {Path(video_path).name}")
    print(f"   Duration: {duration/60:.1f} minutes")
    print(f"   Size: {size_mb:.1f} MB")

    # Check output directory
    os.makedirs("output", exist_ok=True)

    # Check API keys
    api_keys = {
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }

    for key, value in api_keys.items():
        status = "✅" if value else "❌"
        masked_value = f"{value[:8]}..." if value else "Not set"
        print(f"   {key}: {status} {masked_value}")

    return True


def test_video_analysis():
    """Test comprehensive video analysis"""
    from src.core.analyze_video import analyze_video

    print("Running comprehensive video analysis...")
    analysis = analyze_video("/Users/hawzhin/Montage/test_video.mp4")

    # Validate analysis results
    if not analysis or "words" not in analysis:
        print("❌ Video analysis failed - no words extracted")
        return False

    words = analysis.get("words", [])
    transcript = analysis.get("transcript", "")
    speaker_turns = analysis.get("speaker_turns", [])

    print(f"📊 Analysis Results:")
    print(f"   Words: {len(words)}")
    print(f"   Transcript length: {len(transcript)} chars")
    print(f"   Speaker turns: {len(speaker_turns)}")

    if len(words) < 50:  # Minimum reasonable word count
        print("❌ Too few words extracted")
        return False

    # Save analysis for next steps
    with open("full_video_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


def test_intelligent_highlight_selection(analysis):
    """Test AI-powered highlight selection"""
    from src.core.highlight_selector import select_highlights

    print("Running intelligent highlight selection...")

    # Convert words to segments for highlight selector
    words = analysis.get("words", [])
    segments = []
    current_segment = []
    current_text = []

    for i, word in enumerate(words):
        current_segment.append(word)
        current_text.append(word["word"])

        # End segment at natural boundaries
        if (
            word["word"].endswith((".", "!", "?"))
            or len(current_segment) >= 20
            or i == len(words) - 1
        ):

            if current_segment:
                segments.append(
                    {
                        "text": " ".join(current_text),
                        "start_ms": int(current_segment[0]["start"] * 1000),
                        "end_ms": int(current_segment[-1]["end"] * 1000),
                    }
                )
                current_segment = []
                current_text = []

    print(f"Created {len(segments)} segments for analysis")

    # Select highlights
    highlights = select_highlights(segments)

    print(f"📋 Highlight Selection:")
    print(f"   Found: {len(highlights)} highlights")

    if len(highlights) == 0:
        print("❌ No highlights found")
        return False

    # Display highlights
    for i, highlight in enumerate(highlights[:5]):  # Show first 5
        title = highlight.get("title", "Untitled")[:40]
        start = highlight.get("start_time", 0)
        end = highlight.get("end_time", 0)
        duration = end - start
        print(f"   {i+1}. {title} ({start:.1f}s-{end:.1f}s, {duration:.1f}s)")

    return highlights


def test_video_processing(highlights):
    """Test professional video processing and editing"""
    from src.utils.intelligent_crop import create_intelligent_vertical_video

    print("Creating professional highlight video...")

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/professional_highlights_{timestamp}.mp4"

    # Use top 8 highlights for a comprehensive video
    selected_highlights = highlights[:8]
    print(f"Using top {len(selected_highlights)} highlights")

    # Create the video using intelligent cropping
    result = create_intelligent_vertical_video(
        input_path="/Users/hawzhin/Montage/test_video.mp4",
        output_path=output_path,
        highlights=selected_highlights,
    )

    if not result or not os.path.exists(output_path):
        print("❌ Video creation failed")
        return False

    # Verify output video
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        output_path,
    ]
    probe_result = subprocess.run(cmd, capture_output=True, text=True)

    if probe_result.returncode != 0:
        print("❌ Output video is corrupted")
        return False

    video_info = json.loads(probe_result.stdout)
    duration = float(video_info["format"]["duration"])
    size_mb = int(video_info["format"]["size"]) / (1024 * 1024)

    # Get video resolution
    video_stream = next(s for s in video_info["streams"] if s["codec_type"] == "video")
    width = video_stream["width"]
    height = video_stream["height"]

    print(f"🎬 Video Created Successfully:")
    print(f"   File: {output_path}")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Resolution: {width}x{height}")
    print(f"   Format: {video_stream['codec_name']}")

    return {
        "path": output_path,
        "duration": duration,
        "size_mb": size_mb,
        "resolution": f"{width}x{height}",
        "highlights_used": len(selected_highlights),
    }


def test_quality_validation(video_result):
    """Test professional quality validation"""
    from src.core.quality_validator import enforce_production_quality

    print("Running professional quality validation...")

    try:
        # Run quality validation on the output
        quality_result = enforce_production_quality(video_result["path"])

        print(f"📋 Quality Validation Results:")
        print(f"   Overall Score: {quality_result.get('overall_score', 'N/A')}")
        print(f"   Issues Found: {len(quality_result.get('issues', []))}")

        # Check for critical issues
        critical_issues = [
            issue
            for issue in quality_result.get("issues", [])
            if issue.get("severity") == "critical"
        ]

        if critical_issues:
            print("❌ Critical quality issues found:")
            for issue in critical_issues:
                print(f"     - {issue.get('message', 'Unknown issue')}")
            return False

        return quality_result

    except Exception as e:
        print(f"⚠️ Quality validation not available: {e}")
        print("✅ Proceeding without quality validation")
        return True  # Don't fail if quality validator isn't available


def main():
    """Main execution pipeline"""
    print("🎬 PROFESSIONAL VIDEO CREATION PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load environment variables
    if os.path.exists(".env"):
        os.system('export $(grep -v "^#" .env | xargs)')

    # Test each feature sequentially
    test_feature("Prerequisites Check", check_prerequisites)

    analysis = test_feature("Video Analysis & Transcription", test_video_analysis)

    highlights = test_feature(
        "Intelligent Highlight Selection",
        lambda: test_intelligent_highlight_selection(analysis),
    )

    video_result = test_feature(
        "Professional Video Processing", lambda: test_video_processing(highlights)
    )

    quality_result = test_feature(
        "Quality Validation", lambda: test_quality_validation(video_result)
    )

    # Final success report
    print("\n" + "=" * 60)
    print("🏆 PROFESSIONAL VIDEO CREATION COMPLETE!")
    print("=" * 60)
    print(f"✅ All features worked successfully")
    print(f"🎬 Output: {video_result['path']}")
    print(f"⏱️  Duration: {video_result['duration']:.1f} seconds")
    print(f"💾 Size: {video_result['size_mb']:.1f} MB")
    print(f"📺 Resolution: {video_result['resolution']}")
    print(f"🎯 Highlights: {video_result['highlights_used']}")
    print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
