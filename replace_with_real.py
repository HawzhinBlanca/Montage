#!/usr/bin/env python3
"""
Replace mock analyze_video with REAL implementation
This will make Montage use actual APIs instead of mock data
"""
import shutil
import os

print("🔄 Replacing mock analyze_video with REAL implementation...")

# Backup original
if os.path.exists("montage/core/analyze_video.py"):
    shutil.copy("montage/core/analyze_video.py", "montage/core/analyze_video_mock.py")
    print("✅ Backed up original to analyze_video_mock.py")

# Copy real implementation
shutil.copy("montage/core/analyze_video_real.py", "montage/core/analyze_video.py")
print("✅ Replaced with REAL implementation")

print("\n🎉 Done! Montage will now use:")
print("  - REAL Whisper transcription")
print("  - REAL AI highlight selection")
print("  - REAL speaker diarization")
print("  - REAL creative titles")
print("\nRun: python run_real_pipeline.py")