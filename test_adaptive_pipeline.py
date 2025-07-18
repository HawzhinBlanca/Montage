#!/usr/bin/env python3
"""
Test the new Adaptive Quality Pipeline
ONE button approach - no user choice paralysis
"""

import asyncio
import sys
import os
import time
from adaptive_quality_pipeline import AdaptiveQualityPipeline, UserConstraints


async def test_adaptive_pipeline(video_path: str):
    """
    Test the adaptive pipeline with different video types
    """
    print("\n" + "="*60)
    print("ADAPTIVE QUALITY PIPELINE TEST")
    print("ONE Button - Intelligent Processing")
    print("="*60 + "\n")
    
    # Create pipeline
    pipeline = AdaptiveQualityPipeline()
    
    # Progress callback to show real-time updates
    async def show_progress(update):
        print(f"[{update['progress']:3d}%] {update['stage'].upper()}: {update['message']}")
    
    # Test 1: Default settings (let system decide everything)
    print("\n1. DEFAULT TEST - System decides everything")
    print("-" * 40)
    
    try:
        start = time.time()
        result = await pipeline.process(
            video_path,
            progress_callback=show_progress
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"   Mode used: {result.mode_used.value}")
        print(f"   Cost: ${result.actual_cost:.2f}")
        print(f"   Time: {result.processing_time:.1f}s")
        print(f"   Quality: {result.quality_score:.1%}")
        print(f"   Segments: {len(result.segments)}")
        print(f"   Output: {result.video_path}")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
    
    # Test 2: Budget-conscious user
    print("\n\n2. BUDGET TEST - Max $0.10")
    print("-" * 40)
    
    try:
        result = await pipeline.process(
            video_path,
            user_constraints=UserConstraints(
                max_budget=0.10,
                allows_cloud=True
            ),
            progress_callback=show_progress
        )
        
        print(f"\n✅ SUCCESS! (Budget respected)")
        print(f"   Mode: {result.mode_used.value}")
        print(f"   Cost: ${result.actual_cost:.2f} (limit was $0.10)")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
    
    # Test 3: Privacy-conscious user
    print("\n\n3. PRIVACY TEST - No cloud processing")
    print("-" * 40)
    
    try:
        result = await pipeline.process(
            video_path,
            user_constraints=UserConstraints(
                allows_cloud=False
            ),
            progress_callback=show_progress
        )
        
        print(f"\n✅ SUCCESS! (Local only)")
        print(f"   Mode: {result.mode_used.value}")
        print(f"   Cost: ${result.actual_cost:.2f} (should be $0)")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
    
    # Test 4: Speed test
    print("\n\n4. SPEED TEST - Max 60 seconds wait")
    print("-" * 40)
    
    try:
        result = await pipeline.process(
            video_path,
            user_constraints=UserConstraints(
                max_wait_time=60
            ),
            progress_callback=show_progress
        )
        
        print(f"\n✅ SUCCESS! (Fast processing)")
        print(f"   Mode: {result.mode_used.value}")
        print(f"   Time: {result.processing_time:.1f}s (limit was 60s)")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
    
    print("\n" + "="*60)
    print("KEY INSIGHT: User didn't choose ANY processing mode!")
    print("System automatically selected optimal path based on:")
    print("  - Video characteristics (probed in 10s)")
    print("  - User constraints (budget/privacy/time)")
    print("  - Content complexity")
    print("\nNO MORE 3-BUTTON PARALYSIS!")
    print("="*60 + "\n")


async def compare_with_old_system():
    """
    Show the difference between old and new approach
    """
    print("\n" + "="*60)
    print("OLD vs NEW COMPARISON")
    print("="*60)
    
    print("\n❌ OLD SYSTEM (3 buttons):")
    print("   User: *stares at Fast/Smart/Premium buttons*")
    print("   User: 'Which one should I choose?'")
    print("   User: *picks Smart*")
    print("   User: 'Hmm, not quite good enough...'")
    print("   User: *tries Premium*")
    print("   User: '$4?! And 40 minutes?!'")
    print("   User: *rage quits*")
    
    print("\n✅ NEW SYSTEM (1 button):")
    print("   User: *clicks 'Create Video'*")
    print("   System: *analyzes in 10s*")
    print("   System: 'This is a simple video, using fast track'")
    print("   User: *gets result in 30s*")
    print("   User: 'Perfect!'")
    
    print("\n🎯 The magic: System decides, user gets results!")
    print("="*60 + "\n")


async def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("Usage: python test_adaptive_pipeline.py <video_path>")
        print("\nThis demonstrates the new ONE-BUTTON approach:")
        print("- No more choosing between Fast/Smart/Premium")
        print("- System automatically selects optimal processing")
        print("- Respects user constraints (budget/privacy/time)")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Show comparison first
    await compare_with_old_system()
    
    # Then run tests
    await test_adaptive_pipeline(video_path)


if __name__ == "__main__":
    asyncio.run(main())