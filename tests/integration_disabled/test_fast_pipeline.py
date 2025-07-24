#!/usr/bin/env python3
"""Fast test that skips slow API calls"""

import os
import json

# Disable Gemini API to test without it
os.environ["GEMINI_API_KEY"] = ""

from src.core.analyze_video import analyze_video
from src.core.highlight_selector import select_highlights
from src.utils.intelligent_crop import create_intelligent_vertical_video

print("Testing fast pipeline (local only)...")

try:
    # Use existing analysis if available
    if os.path.exists("test_analysis.json"):
        print("Using cached analysis...")
        with open("test_analysis.json", "r") as f:
            analysis = json.load(f)
        print(f"Loaded analysis with {len(analysis.get('words', []))} words")
    else:
        print("Running fresh analysis...")
        analysis = analyze_video("test_video_1min.mp4")

    # Convert to transcript segments
    words = analysis.get("words", [])
    transcript_segments = []

    if words:
        current_segment = []
        current_text = []

        for i, word in enumerate(words):
            current_segment.append(word)
            current_text.append(word["word"])

            if (
                word["word"].endswith((".", "!", "?"))
                or len(current_segment) >= 10
                or i == len(words) - 1
            ):

                if current_segment:
                    transcript_segments.append(
                        {
                            "text": " ".join(current_text),
                            "start_ms": int(current_segment[0]["start"] * 1000),
                            "end_ms": int(current_segment[-1]["end"] * 1000),
                        }
                    )
                    current_segment = []
                    current_text = []

        print(f"Converted to {len(transcript_segments)} segments")

        # Get highlights (should use local scoring only without Gemini)
        print("Getting highlights with local scoring...")
        highlights = select_highlights(transcript_segments)

        print(f"Found {len(highlights)} highlights")

        if highlights:
            print("\nHighlights found:")
            for i, h in enumerate(highlights):
                print(
                    f"  {i+1}. {h.get('title', 'No title')} ({h.get('start_time', 0):.1f}s - {h.get('end_time', 0):.1f}s)"
                )

            # Test creating a simple highlight video (without intelligent crop)
            print(f"\nCreating simple highlight video with {len(highlights)} clips...")

            # Create a simple list for video creation
            simple_highlights = []
            for h in highlights:
                simple_highlights.append(
                    {
                        "start_time": h.get("start_time", 0),
                        "end_time": h.get("end_time", 0),
                        "title": h.get("title", "Highlight"),
                    }
                )

            # Use ffmpeg directly for a simple concatenation
            import subprocess
            import tempfile

            # Create individual clips
            clip_files = []
            for i, highlight in enumerate(simple_highlights):
                clip_file = f"temp_clip_{i}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        "test_video_1min.mp4",
                        "-ss",
                        str(highlight["start_time"]),
                        "-t",
                        str(highlight["end_time"] - highlight["start_time"]),
                        "-c",
                        "copy",
                        clip_file,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                clip_files.append(clip_file)

            # Create concat file
            concat_file = "concat_list.txt"
            with open(concat_file, "w") as f:
                for clip_file in clip_files:
                    f.write(f"file '{clip_file}'\n")

            # Concatenate clips
            output_file = "output/test_highlights.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file,
                    "-c",
                    "copy",
                    output_file,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Clean up
            for clip_file in clip_files:
                os.unlink(clip_file)
            os.unlink(concat_file)

            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"✅ Output video created: {output_file} ({size_mb:.1f}MB)")
            else:
                print("❌ Failed to create output video")
        else:
            print("No highlights found")

    print("\n✅ Fast pipeline test completed successfully!")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
