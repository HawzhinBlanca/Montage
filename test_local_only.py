#!/usr/bin/env python3
"""Test pipeline with local processing only"""

import os
import json
import subprocess

# Disable all external APIs
os.environ["GEMINI_API_KEY"] = ""
os.environ["DEEPGRAM_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

print("Testing LOCAL-ONLY pipeline...")

try:
    # Load cached analysis
    if os.path.exists("test_analysis.json"):
        print("Using cached whisper analysis...")
        with open("test_analysis.json", "r") as f:
            analysis = json.load(f)
        print(f"Loaded: {len(analysis.get('words', []))} words")

        # Extract first few meaningful segments manually
        words = analysis.get("words", [])

        # Find segments with interesting keywords
        interesting_segments = []
        current_words = []
        current_start = 0

        keywords = [
            "professor",
            "david",
            "sinclair",
            "geneticist",
            "longevity",
            "scientist",
            "exciting",
        ]

        for i, word in enumerate(words):
            current_words.append(word)

            # Check if we hit a keyword or sentence boundary
            word_text = word["word"].lower().strip(".,!?")
            is_interesting = any(keyword in word_text for keyword in keywords)
            is_sentence_end = word["word"].endswith((".", "!", "?"))

            if (is_interesting or is_sentence_end or len(current_words) >= 15) and len(
                current_words
            ) >= 5:
                segment_text = " ".join(w["word"] for w in current_words)
                segment_start = current_words[0]["start"]
                segment_end = current_words[-1]["end"]

                if segment_end - segment_start >= 3:  # At least 3 seconds
                    interesting_segments.append(
                        {
                            "start_time": segment_start,
                            "end_time": segment_end,
                            "text": segment_text.strip(),
                            "title": (
                                segment_text.strip()[:35] + "..."
                                if len(segment_text) > 35
                                else segment_text.strip()
                            ),
                        }
                    )

                current_words = []

        print(f"Found {len(interesting_segments)} interesting segments:")
        for i, seg in enumerate(interesting_segments):
            print(
                f"  {i+1}. {seg['title']} ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s)"
            )

        if interesting_segments:
            print(
                f"\nCreating highlight video with {len(interesting_segments)} segments..."
            )

            # Create clips using ffmpeg
            clip_files = []
            for i, segment in enumerate(interesting_segments[:3]):  # Use first 3
                clip_file = f"temp_highlight_{i}.mp4"
                duration = segment["end_time"] - segment["start_time"]

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "test_video_1min.mp4",
                    "-ss",
                    str(segment["start_time"]),
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    clip_file,
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    clip_files.append(clip_file)
                    print(f"  ✅ Created clip {i+1}: {duration:.1f}s")
                else:
                    print(f"  ❌ Failed to create clip {i+1}: {result.stderr}")

            if clip_files:
                # Create concat file
                concat_file = "local_concat.txt"
                with open(concat_file, "w") as f:
                    for clip_file in clip_files:
                        f.write(f"file '{clip_file}'\n")

                # Concatenate clips
                output_file = "output/local_highlights.mp4"
                os.makedirs("output", exist_ok=True)

                cmd = [
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
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Clean up temp files
                for clip_file in clip_files:
                    os.unlink(clip_file)
                os.unlink(concat_file)

                if result.returncode == 0 and os.path.exists(output_file):
                    size_mb = os.path.getsize(output_file) / (1024 * 1024)
                    duration_cmd = [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "csv=p=0",
                        output_file,
                    ]
                    duration_result = subprocess.run(
                        duration_cmd, capture_output=True, text=True
                    )
                    duration = (
                        float(duration_result.stdout.strip())
                        if duration_result.stdout.strip()
                        else 0
                    )

                    print(f"\n🎬 SUCCESS! Created highlight video:")
                    print(f"   File: {output_file}")
                    print(f"   Size: {size_mb:.1f} MB")
                    print(f"   Duration: {duration:.1f} seconds")
                    print(f"   Clips: {len(clip_files)}")

                    print(
                        f"\n✅ LOCAL PIPELINE WORKS! Video processing completed successfully."
                    )
                else:
                    print(f"❌ Failed to create final video: {result.stderr}")
            else:
                print("❌ No clips were created successfully")
        else:
            print("❌ No interesting segments found")
    else:
        print("❌ No cached analysis found. Run full analysis first.")

except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
