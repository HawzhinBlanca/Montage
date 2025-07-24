#!/usr/bin/env python3
"""
Simplified pipeline test with working Gemini API
"""
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test imports
try:
    from core.highlight_selector import analyze_story_structure, sort_by_narrative_flow
    from core.cost import get_current_cost, check_budget
    import utils.secret_loader

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_story_ai():
    """Test story AI with sample content"""
    print("\n🧠 Testing Story AI with Gemini...")

    # Sample transcript segments for testing
    sample_segments = [
        {
            "text": "What if I told you there's a secret to success that most people never discover?",
            "start_ms": 0,
            "end_ms": 5000,
            "speaker": "host",
        },
        {
            "text": "The breakthrough came when I realized that persistence alone wasn't enough.",
            "start_ms": 15000,
            "end_ms": 22000,
            "speaker": "host",
        },
        {
            "text": "This moment changed everything - the data showed a 300% improvement.",
            "start_ms": 45000,
            "end_ms": 52000,
            "speaker": "expert",
        },
        {
            "text": "So the key takeaway is: combine smart strategy with relentless execution.",
            "start_ms": 75000,
            "end_ms": 82000,
            "speaker": "host",
        },
    ]

    try:
        # Test story structure analysis
        result = analyze_story_structure(sample_segments, "test_job")

        print(f"📊 Story analysis result:")
        print(f"   Theme: {result.get('theme', 'N/A')}")
        print(f"   Narrative type: {result.get('narrative_type', 'N/A')}")
        print(f"   Story beats: {len(result.get('story_beats', []))}")

        # Test highlight generation
        if "story_beats" in result:
            print(f"   Raw story beats: {result['story_beats']}")

            # Convert to highlights format first
            from core.highlight_selector import convert_story_beats_to_clips

            highlights = convert_story_beats_to_clips(result, sample_segments)

            if highlights:
                sorted_highlights = sort_by_narrative_flow(highlights)
                print(f"   Generated highlights: {len(sorted_highlights)}")

                for i, highlight in enumerate(sorted_highlights[:3]):
                    print(
                        f"   [{i+1}] {highlight.get('start_ms', 0)/1000:.1f}s-{highlight.get('end_ms', 0)/1000:.1f}s: {highlight.get('title', 'N/A')}"
                    )
            else:
                print("   ❌ No highlights generated from story beats")

        # Check cost tracking
        current_cost = get_current_cost()
        within_budget, remaining = check_budget()
        print(f"💰 Cost tracking: ${current_cost:.5f} used, ${remaining:.2f} remaining")

        return True

    except Exception as e:
        print(f"❌ Story AI test failed: {e}")
        return False


def test_video_file_exists():
    """Test that video file exists"""
    video_path = "/Users/hawzhin/Montage/test_video.mp4"

    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"✅ Test video found: {file_size:.1f} MB")
        return True
    else:
        print(f"❌ Test video not found at {video_path}")
        return False


def main():
    """Run all tests"""
    print("🚀 Montage Pipeline Reality Check")
    print("=" * 40)

    # Load environment - already loaded by config

    # Test 1: Video file
    test1 = test_video_file_exists()

    # Test 2: Story AI
    test2 = test_story_ai()

    # Results
    print("\n📋 Test Results:")
    print(f"   Video file: {'✅' if test1 else '❌'}")
    print(f"   Story AI:   {'✅' if test2 else '❌'}")

    if test1 and test2:
        print("\n🎉 All core systems working! Ready for full pipeline.")
        return True
    else:
        print("\n⚠️  Some systems need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
