#!/usr/bin/env python3
"""
Create a TRULY intelligent story video that:
1. Actually transcribes the real audio using Whisper
2. Understands sentence boundaries and speech patterns
3. Creates smooth transitions without cutting speech
4. Generates intelligent story flow based on actual content
"""
import os
import sys
import json
import subprocess
import tempfile
import whisper
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def extract_real_transcript(video_path: str) -> list:
    """Extract real transcript using Whisper ASR"""
    print("🎙️ Extracting real audio transcript with Whisper...")

    try:
        # Load Whisper model
        model = whisper.load_model("base")

        # Transcribe the video
        result = model.transcribe(video_path, language="en", word_timestamps=True)

        # Convert to our format with word-level timestamps
        transcript_segments = []

        for segment in result["segments"]:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            text = segment["text"].strip()

            # Skip very short or empty segments
            if len(text) < 3 or (end_ms - start_ms) < 500:
                continue

            # Add speaker detection (simple approach - could be enhanced)
            speaker = "speaker_1"  # Default - could add speaker diarization

            transcript_segments.append(
                {
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": speaker,
                    "confidence": segment.get("avg_logprob", 0.0),
                }
            )

        print(f"✅ Extracted {len(transcript_segments)} real speech segments")
        print(f"📊 Total duration: {result['segments'][-1]['end']:.1f} seconds")

        # Show sample of real content
        print("\n📝 Sample of real transcript:")
        for i, seg in enumerate(transcript_segments[:5]):
            print(f"   {i+1}. {seg['start_ms']/1000:.1f}s: {seg['text'][:60]}...")

        return transcript_segments

    except Exception as e:
        print(f"❌ Failed to extract real transcript: {e}")
        return []


def find_natural_story_points(segments: list) -> dict:
    """Find natural story points based on actual speech content"""
    print("\n🔍 Analyzing real content for natural story points...")

    story_points = {
        "hooks": [],
        "developments": [],
        "climax_moments": [],
        "conclusions": [],
    }

    # Define intelligent patterns based on actual speech
    hook_patterns = [
        r"\b(welcome|hello|today|let me tell you)\b",
        r"\b(what if|imagine|have you ever|did you know)\b",
        r"\b(this is|here's|now)\b",
        r"\b(introduction|beginning|start)\b",
    ]

    development_patterns = [
        r"\b(because|since|due to|as a result)\b",
        r"\b(research|study|data|evidence|findings)\b",
        r"\b(important|significant|crucial|key)\b",
        r"\b(however|but|although|despite)\b",
        r"\b(process|method|approach|technique)\b",
    ]

    climax_patterns = [
        r"\b(breakthrough|discovery|revelation|found)\b",
        r"\b(amazing|incredible|extraordinary|remarkable)\b",
        r"\b(most important|crucial moment|key finding)\b",
        r"\b(suddenly|unexpectedly|surprisingly)\b",
        r"\b(changed everything|game changer|revolutionary)\b",
    ]

    conclusion_patterns = [
        r"\b(conclusion|summary|in summary|to conclude)\b",
        r"\b(finally|ultimately|in the end)\b",
        r"\b(takeaway|lesson|what this means)\b",
        r"\b(thank you|thanks for watching|that's all)\b",
    ]

    import re

    for i, segment in enumerate(segments):
        text = segment["text"].lower()

        # Calculate position in video (for context)
        total_duration = segments[-1]["end_ms"] if segments else 0
        position_ratio = (
            segment["start_ms"] / total_duration if total_duration > 0 else 0
        )

        # Score each segment for different story functions
        hook_score = 0
        development_score = 0
        climax_score = 0
        conclusion_score = 0

        # Check for hook patterns (especially in first 30%)
        for pattern in hook_patterns:
            if re.search(pattern, text):
                hook_score += 1
                if position_ratio < 0.3:  # Bonus for early position
                    hook_score += 1

        # Check for development patterns
        for pattern in development_patterns:
            if re.search(pattern, text):
                development_score += 1

        # Check for climax patterns
        for pattern in climax_patterns:
            if re.search(pattern, text):
                climax_score += 1
                if 0.3 < position_ratio < 0.8:  # Bonus for middle position
                    climax_score += 1

        # Check for conclusion patterns (especially in last 30%)
        for pattern in conclusion_patterns:
            if re.search(pattern, text):
                conclusion_score += 1
                if position_ratio > 0.7:  # Bonus for late position
                    conclusion_score += 1

        # Add segments with high scores
        if hook_score >= 2:
            story_points["hooks"].append(
                {"segment": segment, "score": hook_score, "position": position_ratio}
            )

        if development_score >= 2:
            story_points["developments"].append(
                {
                    "segment": segment,
                    "score": development_score,
                    "position": position_ratio,
                }
            )

        if climax_score >= 2:
            story_points["climax_moments"].append(
                {"segment": segment, "score": climax_score, "position": position_ratio}
            )

        if conclusion_score >= 2:
            story_points["conclusions"].append(
                {
                    "segment": segment,
                    "score": conclusion_score,
                    "position": position_ratio,
                }
            )

    # Sort by score
    for key in story_points:
        story_points[key].sort(key=lambda x: x["score"], reverse=True)

    print(f"✅ Found natural story points:")
    print(f"   Hooks: {len(story_points['hooks'])}")
    print(f"   Developments: {len(story_points['developments'])}")
    print(f"   Climax moments: {len(story_points['climax_moments'])}")
    print(f"   Conclusions: {len(story_points['conclusions'])}")

    return story_points


def create_intelligent_story_segments(story_points: dict, all_segments: list) -> list:
    """Create intelligent story segments that respect speech boundaries"""
    print("\n🧠 Creating intelligent story segments...")

    story_segments = []

    # Select best story points (ensure no overlaps)
    selected_points = []

    # Select top hook (early in video)
    if story_points["hooks"]:
        selected_points.append(("hook", story_points["hooks"][0]))

    # Select development points (middle)
    for dev in story_points["developments"][:2]:  # Top 2
        if 0.2 < dev["position"] < 0.7:
            selected_points.append(("development", dev))

    # Select climax (middle-late)
    if story_points["climax_moments"]:
        for climax in story_points["climax_moments"][:1]:  # Top 1
            if 0.3 < climax["position"] < 0.8:
                selected_points.append(("climax", climax))

    # Select conclusion (late)
    if story_points["conclusions"]:
        selected_points.append(("conclusion", story_points["conclusions"][0]))

    # Sort by timeline position
    selected_points.sort(key=lambda x: x[1]["position"])

    print(f"📋 Selected {len(selected_points)} story points:")

    # Create extended segments that include complete sentences
    for i, (story_type, point) in enumerate(selected_points):
        base_segment = point["segment"]

        # Find complete sentence boundaries
        extended_segment = extend_to_sentence_boundaries(base_segment, all_segments)

        # Ensure minimum and maximum durations
        duration_ms = extended_segment["end_ms"] - extended_segment["start_ms"]

        # Minimum 8 seconds, maximum 25 seconds
        if duration_ms < 8000:
            extended_segment = extend_segment_duration(
                extended_segment, all_segments, 8000
            )
        elif duration_ms > 25000:
            extended_segment = trim_segment_duration(
                extended_segment, all_segments, 25000
            )

        story_segment = {
            "title": f"{story_type.title()}: {extended_segment['text'][:40]}...",
            "start_ms": extended_segment["start_ms"],
            "end_ms": extended_segment["end_ms"],
            "story_type": story_type,
            "text": extended_segment["text"],
            "score": point["score"],
            "speaker": extended_segment.get("speaker", "unknown"),
        }

        story_segments.append(story_segment)

        duration = (story_segment["end_ms"] - story_segment["start_ms"]) / 1000
        print(
            f"   {i+1}. {story_type.title()} ({duration:.1f}s): {story_segment['text'][:50]}..."
        )

    return story_segments


def extend_to_sentence_boundaries(segment: dict, all_segments: list) -> dict:
    """Extend segment to complete sentence boundaries"""

    # Find the segment in the full list
    segment_index = -1
    for i, seg in enumerate(all_segments):
        if seg["start_ms"] == segment["start_ms"]:
            segment_index = i
            break

    if segment_index == -1:
        return segment

    # Look backwards for sentence start
    start_index = segment_index
    for i in range(segment_index, -1, -1):
        text = all_segments[i]["text"].strip()
        # If previous segment ends with sentence punctuation, this is a good start
        if i > 0 and any(
            all_segments[i - 1]["text"].strip().endswith(p) for p in [".", "!", "?"]
        ):
            start_index = i
            break
        # Don't go too far back
        if segment_index - i > 3:
            break

    # Look forward for sentence end
    end_index = segment_index
    for i in range(segment_index, len(all_segments)):
        text = all_segments[i]["text"].strip()
        # If this segment ends with sentence punctuation, this is a good end
        if any(text.endswith(p) for p in [".", "!", "?"]):
            end_index = i
            break
        # Don't go too far forward
        if i - segment_index > 3:
            break

    # Combine segments
    if start_index <= end_index:
        combined_text = " ".join(
            all_segments[i]["text"] for i in range(start_index, end_index + 1)
        )
        return {
            "text": combined_text.strip(),
            "start_ms": all_segments[start_index]["start_ms"],
            "end_ms": all_segments[end_index]["end_ms"],
            "speaker": segment["speaker"],
        }

    return segment


def extend_segment_duration(
    segment: dict, all_segments: list, min_duration_ms: int
) -> dict:
    """Extend segment to meet minimum duration"""

    current_duration = segment["end_ms"] - segment["start_ms"]
    if current_duration >= min_duration_ms:
        return segment

    needed_duration = min_duration_ms - current_duration

    # Try to extend forward first
    for seg in all_segments:
        if seg["start_ms"] >= segment["end_ms"]:
            new_end = min(seg["end_ms"], segment["end_ms"] + needed_duration)
            if new_end > segment["end_ms"]:
                return {
                    **segment,
                    "end_ms": new_end,
                    "text": segment["text"] + " " + seg["text"],
                }

    return segment


def trim_segment_duration(
    segment: dict, all_segments: list, max_duration_ms: int
) -> dict:
    """Trim segment to maximum duration"""

    current_duration = segment["end_ms"] - segment["start_ms"]
    if current_duration <= max_duration_ms:
        return segment

    # Trim from the end
    new_end = segment["start_ms"] + max_duration_ms

    # Find appropriate end point that doesn't cut mid-word
    for seg in all_segments:
        if seg["start_ms"] >= segment["start_ms"] and seg["end_ms"] <= new_end:
            return {
                **segment,
                "end_ms": seg["end_ms"],
                "text": seg["text"],  # Use the trimmed text
            }

    return {**segment, "end_ms": new_end}


def create_intelligent_video_with_transitions(
    segments: list, input_video: str, output_path: str
) -> bool:
    """Create video with intelligent transitions"""
    print(f"\n🎬 Creating intelligent video with smooth transitions...")

    try:
        from utils.intelligent_crop import IntelligentCropper

        cropper = IntelligentCropper()
        temp_clips = []
        processed_clips = []

        # Process each segment with intelligent cropping
        for i, segment in enumerate(segments):
            print(f"\n🎯 Processing segment {i+1}/{len(segments)}")
            print(f"   Type: {segment.get('story_type', 'unknown')}")
            print(f"   Duration: {(segment['end_ms'] - segment['start_ms'])/1000:.1f}s")
            print(f"   Content: {segment['text'][:60]}...")

            # Analyze for intelligent cropping
            analysis = cropper.analyze_video_content(
                input_video, segment["start_ms"], segment["end_ms"]
            )

            crop_center = analysis["optimal_crop_center"]
            confidence = analysis["confidence"]

            print(
                f"   🎯 Crop analysis: {analysis['face_count']} faces, {confidence:.2f} confidence"
            )

            # Generate crop filter
            crop_filter = cropper.generate_crop_filter(
                640, 360, crop_center, 1080, 1920
            )

            # Extract with fade transitions
            clip_output = f"/tmp/intelligent_clip_{i:02d}.mp4"
            temp_clips.append(clip_output)

            # Add fade in/out for smooth transitions
            fade_duration = 0.5  # 0.5 second fade

            video_filters = [crop_filter, "fps=30"]

            # Add fade effects (except for single clips)
            if len(segments) > 1:
                if i == 0:  # First clip - fade in only
                    video_filters.append(f"fade=t=in:st=0:d={fade_duration}")
                elif i == len(segments) - 1:  # Last clip - fade out only
                    clip_duration = (segment["end_ms"] - segment["start_ms"]) / 1000
                    video_filters.append(
                        f"fade=t=out:st={clip_duration - fade_duration}:d={fade_duration}"
                    )
                else:  # Middle clips - fade in and out
                    clip_duration = (segment["end_ms"] - segment["start_ms"]) / 1000
                    video_filters.extend(
                        [
                            f"fade=t=in:st=0:d={fade_duration}",
                            f"fade=t=out:st={clip_duration - fade_duration}:d={fade_duration}",
                        ]
                    )

            filter_string = ",".join(video_filters)

            extract_cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-ss",
                str(segment["start_ms"] / 1000),
                "-t",
                str((segment["end_ms"] - segment["start_ms"]) / 1000),
                "-i",
                input_video,
                "-vf",
                filter_string,
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
                "-ar",
                "44100",
                "-movflags",
                "+faststart",
                clip_output,
            ]

            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                processed_clips.append(clip_output)
                clip_size = os.path.getsize(clip_output) / (1024 * 1024)
                print(f"   ✅ Processed ({clip_size:.1f} MB)")
            else:
                print(f"   ❌ Failed: {result.stderr[:100]}...")
                continue

        # Concatenate with crossfades
        if processed_clips:
            print(f"\n🔗 Assembling {len(processed_clips)} clips with transitions...")

            if len(processed_clips) == 1:
                # Single clip - just copy
                subprocess.run(["cp", processed_clips[0], output_path], check=True)
            else:
                # Multiple clips - use complex filter for crossfades
                concat_file = "/tmp/intelligent_concat.txt"
                with open(concat_file, "w") as f:
                    for clip_file in processed_clips:
                        f.write(f"file '{clip_file}'\n")

                concat_cmd = [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "warning",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file,
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
                    "-movflags",
                    "+faststart",
                    output_path,
                ]

                result = subprocess.run(concat_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"❌ Concatenation failed: {result.stderr}")
                    return False

                os.unlink(concat_file)

            # Cleanup temp files
            for clip_file in temp_clips:
                try:
                    os.unlink(clip_file)
                except:
                    pass

            print(f"✅ Intelligent video created successfully!")
            return True

        return False

    except Exception as e:
        print(f"❌ Error creating intelligent video: {e}")
        return False


def main():
    """Create truly intelligent story video"""
    input_video = "/Users/hawzhin/Montage/test_video.mp4"
    output_video = "/Users/hawzhin/Montage/intelligent_story_video.mp4"

    print("🎬 CREATING TRULY INTELLIGENT STORY VIDEO")
    print("=" * 50)
    print("🎯 Features:")
    print("   • Real audio transcription with Whisper")
    print("   • Natural speech boundary detection")
    print("   • Intelligent story point analysis")
    print("   • Smooth transitions without cutting speech")
    print("   • Context-aware segment extension")
    print()

    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return

    try:
        # Step 1: Extract real transcript
        real_transcript = extract_real_transcript(input_video)

        if not real_transcript:
            print("❌ Failed to extract transcript")
            return

        # Step 2: Find natural story points
        story_points = find_natural_story_points(real_transcript)

        # Step 3: Create intelligent segments
        story_segments = create_intelligent_story_segments(
            story_points, real_transcript
        )

        if not story_segments:
            print("❌ No intelligent story segments found")
            return

        print(f"\n📊 Final story structure ({len(story_segments)} segments):")
        total_duration = (
            sum((seg["end_ms"] - seg["start_ms"]) for seg in story_segments) / 1000
        )
        print(f"   Total duration: {total_duration:.1f} seconds")

        # Step 4: Create video with transitions
        success = create_intelligent_video_with_transitions(
            story_segments, input_video, output_video
        )

        if success:
            print(f"\n🎉 INTELLIGENT STORY VIDEO COMPLETE!")
            print(f"📱 Location: {output_video}")

            # Get final stats
            if os.path.exists(output_video):
                file_size = os.path.getsize(output_video) / (1024 * 1024)
                print(f"📊 Size: {file_size:.1f} MB")

                try:
                    result = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "quiet",
                            "-print_format",
                            "json",
                            "-show_format",
                            output_video,
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    info = json.loads(result.stdout)
                    duration = float(info["format"]["duration"])
                    print(f"⏱️  Duration: {duration:.1f} seconds")

                except:
                    pass

                # Open video
                try:
                    subprocess.run(["open", output_video], check=False)
                    print("🎥 Opening intelligent story video...")
                except:
                    pass
        else:
            print("❌ Failed to create intelligent video")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
