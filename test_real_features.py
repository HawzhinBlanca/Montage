#!/usr/bin/env python3
"""
Test REAL features using existing Montage structure
"""
import os
import sys
import json

# Setup environment
os.environ["JWT_SECRET_KEY"] = "FZbEvojxK-bNZIE6a6BuA_hzKkuYItSyVtxsC6d3z78"

from dotenv import load_dotenv
load_dotenv()

# Replace analyze_video with real implementation
import shutil
print("🔄 Activating REAL implementations...")

# Backup and replace
shutil.copy("montage/core/analyze_video.py", "montage/core/analyze_video_original.py")
shutil.copy("montage/core/analyze_video_real.py", "montage/core/analyze_video.py")

# Now import and test
from montage.core.analyze_video import analyze_video

print("\n🧪 Testing REAL Montage features...")
print("=" * 50)

# Check APIs
print("✅ API Status:")
for key in ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "HUGGINGFACE_TOKEN"]:
    status = "✓ Loaded" if os.getenv(key) else "✗ Missing"
    print(f"  {key}: {status}")

print("\n🎬 Running REAL analysis on test_video.mp4...")
try:
    result = analyze_video("test_video.mp4")
    
    print("\n📊 REAL Results:")
    print(f"  Transcript segments: {len(result.get('transcript', []))}")
    print(f"  Words detected: {len(result.get('words', []))}")
    print(f"  Speakers found: {len(result.get('speaker_turns', []))}")
    print(f"  AI Highlights: {len(result.get('highlights', []))}")
    
    if result.get('highlights'):
        print("\n🎯 AI-Selected Highlights:")
        for i, h in enumerate(result['highlights'][:3]):
            print(f"\n  Highlight {i+1}:")
            print(f"    Time: {h['start']:.1f}s - {h['end']:.1f}s")
            print(f"    Title: {h.get('title', 'N/A')}")
            print(f"    Score: {h.get('score', 0):.2f}")
            print(f"    Hashtags: {', '.join(h.get('hashtags', []))}")
            print(f"    Emojis: {' '.join(h.get('emojis', []))}")
    
    # Save results
    with open('real_analysis_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\n💾 Full results saved to: real_analysis_results.json")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Restore original
    print("\n🔄 Restoring original analyze_video...")
    shutil.copy("montage/core/analyze_video_original.py", "montage/core/analyze_video.py")
    os.remove("montage/core/analyze_video_original.py")