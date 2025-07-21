#!/usr/bin/env python3
"""Simple test of the video processing pipeline"""

import json
from src.core.analyze_video import analyze_video
from src.core.highlight_selector import select_highlights
from src.utils.intelligent_crop import create_intelligent_vertical_video

print("Testing simple pipeline...")

# Analyze the 1-minute video
print("\n1. Analyzing video...")
try:
    analysis = analyze_video("test_video_1min.mp4")
    print(f"Analysis complete! Found {len(analysis.get('words', []))} words")

    # Save analysis for inspection
    with open("test_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("Analysis saved to test_analysis.json")

    # Convert words to transcript segments format expected by highlight selector
    words = analysis.get("words", [])
    transcript_segments = []
    if words:
        # Group words into sentences or segments
        current_segment = []
        current_text = []

        for i, word in enumerate(words):
            current_segment.append(word)
            current_text.append(word["word"])

            # End segment at sentence boundaries or every 10 words
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

        # Extract highlights using the highlight selector
        print("\n2. Selecting highlights...")
        highlights = select_highlights(transcript_segments)
        print(f"Found {len(highlights)} highlights")
    else:
        print("No words found for highlight selection")
        highlights = []
    if highlights:
        print(f"\nFirst highlight: {highlights[0].get('title', 'No title')}")
        print(
            f"Time: {highlights[0].get('start_time', 0):.1f}s - {highlights[0].get('end_time', 0):.1f}s"
        )

    # Create vertical video
    if highlights:
        print("\n3. Creating vertical video...")
        result = create_intelligent_vertical_video(
            input_path="test_video_1min.mp4",
            output_path="output/test_vertical_output.mp4",
            highlights=highlights[:3],  # Use first 3 highlights
        )
        print(f"Video created: {result}")
    else:
        print("No highlights found to create video")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
