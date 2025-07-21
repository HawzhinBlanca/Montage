#!/usr/bin/env python3
"""Create a test video with actual speech using text-to-speech"""
import subprocess
import os

# Create output directory
os.makedirs("tests/data", exist_ok=True)

# Professional speech content for testing
speech_text = """Welcome to this amazing demonstration. Today, I'm going to share something incredible with you. 
This is the moment you've been waiting for. Here's the key insight that will transform your perspective. 
First, let me tell you a story that changed everything. It started with a simple observation.
Then came the breakthrough. This discovery revolutionized how we think about success.
Remember this crucial point: persistence combined with innovation leads to extraordinary results.
In conclusion, the future belongs to those who dare to dream big and take action."""

# Create audio file
audio_file = "/tmp/test_speech.aiff"
subprocess.run(["say", "-v", "Alex", "-r", "180", "-o", audio_file, speech_text])

# Convert to video with dynamic visuals
output_file = "tests/data/speech_test.mp4"
cmd = [
    "ffmpeg",
    "-y",
    "-f",
    "lavfi",
    "-i",
    "color=c=black:s=1920x1080:d=30:r=30",
    "-i",
    audio_file,
    "-filter_complex",
    "[0:v]drawtext=fontfile=/System/Library/Fonts/Helvetica.ttc:text='Professional Video Test':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2[v]",
    "-map",
    "[v]",
    "-map",
    "1:a",
    "-c:v",
    "libx264",
    "-preset",
    "fast",
    "-c:a",
    "aac",
    "-b:a",
    "192k",
    "-shortest",
    output_file,
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ Created professional test video: {output_file}")
    size = os.path.getsize(output_file) / 1024 / 1024
    print(f"   Size: {size:.2f} MB")
else:
    print(f"❌ Failed to create video: {result.stderr}")

# Clean up
if os.path.exists(audio_file):
    os.unlink(audio_file)
