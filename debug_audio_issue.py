#!/usr/bin/env python3
"""Debug the audio cutting issue"""

import subprocess
import json

# Test the exact FFmpeg command I used for a short clip
# Using the 3.6s clip that was created (Clip 5)

# From the log: "We can do better. And so I said, all rig..." (3.6s)
# This was around segments that scored high for "intrigue, emotion"

print("🔍 DEBUGGING AUDIO CUTTING ISSUE")
print("=" * 50)

# Let's recreate the exact clip that likely has the audio issue
test_start = 20.0  # Example timing
test_duration = 3.6  # The short clip duration that had issues

# Add the padding I used
padded_start = max(0, test_start - 0.2)
padded_end = test_start + test_duration + 0.2
padded_duration = padded_end - padded_start

print(f"Original clip: {test_duration}s")
print(f"With padding: {padded_duration}s (start: {padded_start}s)")

# The EXACT command I used (with the bug)
cmd = [
    "ffmpeg",
    "-y",
    "-i",
    "/Users/hawzhin/Montage/test_video.mp4",
    "-ss",
    str(padded_start),
    "-t",
    str(padded_duration),
    "-vf",
    f"fade=in:0:12,fade=out:st={max(0, padded_duration*24-12)}:d=12",
    "-af",
    f"afade=in:st=0:d=0.5,afade=out:st={max(0, padded_duration-0.5)}:d=0.5",
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
    "debug_clip.mp4",
]

print(f"\n🛠️ FFmpeg Audio Fade Settings:")
print(f"   Fade IN:  starts at 0s, duration 0.5s")
print(f"   Fade OUT: starts at {max(0, padded_duration-0.5):.1f}s, duration 0.5s")
print(f"   Clear audio: {max(0, padded_duration-1.0):.1f}s")

if padded_duration <= 1.0:
    print("   ❌ CRITICAL BUG: Fades overlap! No clear audio!")
elif padded_duration <= 2.0:
    print("   ⚠️  WARNING: Very short clear audio period")
else:
    print("   ✅ Audio timing should be OK")

# Run the command
print(f"\n🔧 Testing FFmpeg command...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Clip created successfully")

    # Check the audio
    probe_cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_streams",
        "-select_streams",
        "a",
        "debug_clip.mp4",
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

    if "duration=" in probe_result.stdout:
        duration_line = [
            line for line in probe_result.stdout.split("\n") if "duration=" in line
        ][0]
        actual_duration = duration_line.split("=")[1]
        print(f"📊 Actual audio duration: {actual_duration}s")

    print(f"\n📝 Check if audio plays properly:")
    print(f"   ffplay -autoexit debug_clip.mp4")

else:
    print(f"❌ FFmpeg failed: {result.stderr}")

print(f"\n🔍 POTENTIAL BUGS IDENTIFIED:")
print(f"1. Audio fade timing - short clips get over-faded")
print(f"2. Concatenation might not handle fades properly")
print(f"3. Aspect ratio scaling creates distortion")
print(f"4. No audio level normalization between clips")
