#!/usr/bin/env python3
"""
Create professional story-driven video using ONLY working features
No external API calls - pure local intelligence
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import re

# Disable all external APIs to ensure local-only processing
os.environ["GEMINI_API_KEY"] = ""
os.environ["DEEPGRAM_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""


def analyze_full_video():
    """Analyze the full 43-minute video with local tools only"""
    print("🎯 Analyzing full video with LOCAL intelligence...")

    from src.core.analyze_video import analyze_video

    start_time = time.time()
    analysis = analyze_video("/Users/hawzhin/Montage/test_video.mp4")
    duration = time.time() - start_time

    if not analysis or "words" not in analysis:
        print("❌ Video analysis failed")
        return None

    words = analysis.get("words", [])
    transcript = analysis.get("transcript", "")
    speaker_turns = analysis.get("speaker_turns", [])

    print(f"✅ Analysis Complete ({duration:.1f}s):")
    print(f"   📝 Words: {len(words)}")
    print(f"   📖 Transcript: {len(transcript)} characters")
    print(f"   🗣️ Speaker turns: {len(speaker_turns)}")
    print(
        f"   ⏱️ Video duration: {words[-1]['end']/60:.1f} minutes"
        if words
        else "Unknown"
    )

    # Save full analysis
    with open("full_video_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


def create_intelligent_story_structure(analysis):
    """Create a compelling story structure using local intelligence"""
    print("\n📚 Creating intelligent story structure...")

    words = analysis.get("words", [])
    transcript = analysis.get("transcript", "")
    speaker_turns = analysis.get("speaker_turns", [])

    # Story keywords categorized by narrative importance
    story_keywords = {
        "introduction": [
            "welcome",
            "introduce",
            "professor",
            "doctor",
            "meet",
            "hello",
            "today",
        ],
        "authority": [
            "expert",
            "renowned",
            "scientist",
            "researcher",
            "harvard",
            "mit",
            "professor",
        ],
        "credibility": [
            "study",
            "research",
            "data",
            "evidence",
            "published",
            "peer",
            "reviewed",
        ],
        "intrigue": [
            "breakthrough",
            "discovery",
            "secret",
            "revealed",
            "amazing",
            "incredible",
        ],
        "problem": [
            "problem",
            "issue",
            "challenge",
            "difficulty",
            "struggle",
            "aging",
            "death",
        ],
        "solution": [
            "solution",
            "answer",
            "method",
            "technique",
            "breakthrough",
            "discovery",
        ],
        "proof": [
            "proof",
            "evidence",
            "data",
            "results",
            "study",
            "experiment",
            "research",
        ],
        "impact": [
            "important",
            "significant",
            "crucial",
            "key",
            "essential",
            "revolutionary",
        ],
        "story": [
            "story",
            "example",
            "case",
            "patient",
            "person",
            "experience",
            "happened",
        ],
        "emotion": [
            "exciting",
            "amazing",
            "incredible",
            "shocking",
            "surprising",
            "wonderful",
        ],
        "conclusion": [
            "conclusion",
            "summary",
            "finally",
            "end",
            "remember",
            "takeaway",
        ],
    }

    # Analyze transcript for story elements
    story_segments = []
    current_segment = []
    current_text = []
    segment_scores = {}

    print("   🔍 Scanning for story elements...")

    for i, word in enumerate(words):
        current_segment.append(word)
        current_text.append(word["word"])

        # Create segments at natural boundaries
        is_boundary = (
            word["word"].endswith((".", "!", "?"))
            or len(current_segment) >= 25  # Longer segments for better context
            or i == len(words) - 1
        )

        if is_boundary and len(current_segment) >= 8:  # Minimum segment length
            segment_text = " ".join(current_text).lower()
            segment_start = current_segment[0]["start"]
            segment_end = current_segment[-1]["end"]
            segment_duration = segment_end - segment_start

            # Only consider segments of reasonable length
            if 3 <= segment_duration <= 30:
                # Calculate story relevance score
                story_score = 0
                story_elements = []

                for category, keywords in story_keywords.items():
                    matches = sum(1 for keyword in keywords if keyword in segment_text)
                    if matches > 0:
                        story_score += matches * (
                            2
                            if category in ["introduction", "authority", "intrigue"]
                            else 1
                        )
                        story_elements.append(category)

                # Bonus for complete sentences
                if word["word"].endswith((".", "!", "?")):
                    story_score += 2

                # Bonus for speaker changes (dialogue/interaction)
                speaker = current_segment[0].get("speaker", "unknown")
                if len([s for s in story_segments if s.get("speaker") != speaker]) > 0:
                    story_score += 1

                if story_score > 0:
                    story_segments.append(
                        {
                            "text": " ".join(current_text),
                            "start_time": segment_start,
                            "end_time": segment_end,
                            "duration": segment_duration,
                            "score": story_score,
                            "elements": story_elements,
                            "speaker": speaker,
                        }
                    )

            current_segment = []
            current_text = []

    # Sort by story relevance
    story_segments.sort(key=lambda x: x["score"], reverse=True)

    print(f"   📊 Found {len(story_segments)} story-relevant segments")

    # Select best segments for a compelling narrative
    selected_segments = []
    total_duration = 0
    target_duration = 90  # 1.5 minutes for a compelling story

    # Ensure narrative flow: intro -> content -> conclusion
    intro_segments = [
        s for s in story_segments if "introduction" in s.get("elements", [])
    ]
    content_segments = [s for s in story_segments if s.get("score", 0) >= 3]
    conclusion_segments = [
        s for s in story_segments if "conclusion" in s.get("elements", [])
    ]

    print("   🎭 Building narrative structure...")

    # Add introduction (if available)
    if intro_segments and total_duration < target_duration:
        segment = intro_segments[0]
        selected_segments.append(segment)
        total_duration += segment["duration"]
        print(
            f"      🚀 Opening: {segment['text'][:50]}... ({segment['duration']:.1f}s)"
        )

    # Add high-value content segments
    for segment in content_segments[:6]:  # Up to 6 content segments
        if total_duration + segment["duration"] <= target_duration:
            if segment not in selected_segments:
                selected_segments.append(segment)
                total_duration += segment["duration"]
                elements_str = ", ".join(segment.get("elements", []))
                print(
                    f"      💡 Content: {segment['text'][:50]}... ({segment['duration']:.1f}s) [{elements_str}]"
                )

    # Sort selected segments by time order for coherent flow
    selected_segments.sort(key=lambda x: x["start_time"])

    print(
        f"   ✅ Story structure: {len(selected_segments)} segments, {total_duration:.1f}s total"
    )

    return selected_segments


def create_professional_video(story_segments):
    """Create professional video with story-driven editing"""
    print(f"\n🎬 Creating professional story-driven video...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"output/smart_story_{timestamp}.mp4"

    if not story_segments:
        print("❌ No story segments to process")
        return None

    print(f"   🎯 Processing {len(story_segments)} story segments...")

    # Create individual clips with professional transitions
    clip_files = []

    for i, segment in enumerate(story_segments):
        clip_file = f"temp_story_clip_{i}.mp4"

        # Add small padding for smoother transitions
        start_time = max(0, segment["start_time"] - 0.2)
        end_time = segment["end_time"] + 0.2
        duration = end_time - start_time

        # Create clip with fade transitions
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            "/Users/hawzhin/Montage/test_video.mp4",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-vf",
            "fade=in:0:12,fade=out:st={0}:d=12".format(max(0, duration * 24 - 12)),
            "-af",
            "afade=in:st=0:d=0.5,afade=out:st={0}:d=0.5".format(max(0, duration - 0.5)),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            clip_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(clip_file):
            clip_files.append(clip_file)
            print(f"      ✅ Clip {i+1}: {segment['text'][:40]}... ({duration:.1f}s)")
        else:
            print(f"      ❌ Failed clip {i+1}: {result.stderr[:100]}")

    if not clip_files:
        print("❌ No clips created successfully")
        return None

    # Create concatenation file
    concat_file = f"story_concat_{timestamp}.txt"
    with open(concat_file, "w") as f:
        for clip_file in clip_files:
            f.write(f"file '{clip_file}'\n")

    print(f"   🔗 Concatenating {len(clip_files)} clips...")

    # Final video assembly with professional settings
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
        "scale=1080:-2,pad=1080:1080:(ow-iw)/2:(oh-ih)/2",  # Square format for social media
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",  # Web optimization
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

    if result.returncode == 0 and os.path.exists(output_path):
        # Get final video stats
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

        if probe_result.returncode == 0:
            video_info = json.loads(probe_result.stdout)
            duration = float(video_info["format"]["duration"])
            size_mb = int(video_info["format"]["size"]) / (1024 * 1024)

            video_stream = next(
                s for s in video_info["streams"] if s["codec_type"] == "video"
            )
            width = video_stream["width"]
            height = video_stream["height"]

            return {
                "path": output_path,
                "duration": duration,
                "size_mb": size_mb,
                "resolution": f"{width}x{height}",
                "segments": len(clip_files),
                "format": "Square (optimized for social media)",
            }

    print(f"❌ Video creation failed: {result.stderr}")
    return None


def main():
    """Main execution"""
    print("🎭 SMART STORY-DRIVEN VIDEO CREATION")
    print("=" * 60)
    print("Using LOCAL intelligence only - no external API dependencies")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure output directory
    os.makedirs("output", exist_ok=True)

    try:
        # Step 1: Analyze full video
        analysis = analyze_full_video()
        if not analysis:
            print("🛑 STOPPING - Video analysis failed")
            sys.exit(1)

        # Step 2: Create intelligent story structure
        story_segments = create_intelligent_story_structure(analysis)
        if not story_segments:
            print("🛑 STOPPING - Story structure creation failed")
            sys.exit(1)

        # Step 3: Create professional video
        video_result = create_professional_video(story_segments)
        if not video_result:
            print("🛑 STOPPING - Video creation failed")
            sys.exit(1)

        # Success!
        print("\n" + "=" * 60)
        print("🏆 SMART STORY VIDEO CREATION COMPLETE!")
        print("=" * 60)
        print(f"🎬 Output: {video_result['path']}")
        print(f"⏱️  Duration: {video_result['duration']:.1f} seconds")
        print(f"💾 Size: {video_result['size_mb']:.1f} MB")
        print(f"📺 Resolution: {video_result['resolution']}")
        print(f"📚 Story segments: {video_result['segments']}")
        print(f"🎨 Format: {video_result['format']}")
        print(f"✨ Intelligence: 100% local processing")
        print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(
            f"\n🎉 SUCCESS! Professional story-driven video created using only working features."
        )

    except Exception as e:
        print(f"\n🛑 CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
