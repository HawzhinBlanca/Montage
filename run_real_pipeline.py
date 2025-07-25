#!/usr/bin/env python3
"""
Run REAL Montage Pipeline with ALL Working Features
No mock data - uses actual Whisper, AI APIs, etc.
"""
import os
import sys
import asyncio
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the REAL implementations
from montage.core.intelligent_pipeline import IntelligentVideoPipeline


async def run_real_montage():
    """Run Montage with REAL features - no mocks!"""
    print("🚀 REAL Montage Pipeline - No Mock Data!")
    print("=" * 50)
    
    # Verify APIs are loaded
    print("✅ API Keys Loaded:")
    print(f"  - Anthropic: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
    print(f"  - Gemini: {'✓' if os.getenv('GEMINI_API_KEY') else '✗'}")
    print(f"  - OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
    print(f"  - Deepgram: {'✓' if os.getenv('DEEPGRAM_API_KEY') else '✗'}")
    print(f"  - HuggingFace: {'✓' if os.getenv('HUGGINGFACE_TOKEN') else '✗'}")
    print()
    
    # Input/output
    video_path = "test_video.mp4"
    output_dir = "real_output"
    
    print(f"📹 Input: {video_path}")
    print(f"📁 Output: {output_dir}")
    print()
    
    # Create pipeline with REAL components
    print("🔧 Initializing REAL components...")
    pipeline = IntelligentVideoPipeline(enable_metrics=True)
    
    # Configure for maximum features
    config = {
        "target_clips": 3,
        "clip_duration": 60,
        "platforms": ["tiktok", "instagram", "youtube"],
        "style": "viral",
        "enable_all_features": True
    }
    
    print("🎬 Processing with REAL features:")
    print("  1. Whisper transcription (not mock)")
    print("  2. PyAnnote speaker diarization")
    print("  3. Claude/Gemini AI highlights")
    print("  4. Story beat detection")
    print("  5. Smart face cropping")
    print("  6. Animated captions")
    print("  7. EBU R128 audio")
    print("  8. AI titles & hashtags")
    print("  9. Emoji overlays")
    print("  10. Metrics monitoring")
    print()
    
    # Run the REAL pipeline
    try:
        results = await pipeline.process_video(
            input_video=video_path,
            output_dir=output_dir,
            config=config
        )
        
        if results["success"]:
            print("\n✅ SUCCESS! Real clips generated:")
            for clip in results["clips"]:
                print(f"\n📹 {clip['filename']}")
                print(f"   Duration: {clip['duration']:.1f}s")
                print(f"   Platform content generated: {bool(clip.get('platform_content'))}")
                print(f"   Emojis used: {' '.join(clip.get('emojis_used', []))}")
                
            print(f"\n📊 Metrics available at: http://localhost:8000/metrics/proc_mem")
        else:
            print(f"\n❌ Failed: {results.get('errors')}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure JWT key is set
    if not os.getenv("JWT_SECRET_KEY"):
        os.environ["JWT_SECRET_KEY"] = "FZbEvojxK-bNZIE6a6BuA_hzKkuYItSyVtxsC6d3z78"
    
    # Run the REAL pipeline
    asyncio.run(run_real_montage())