#!/usr/bin/env python3
"""
Complete feature demonstration on test_video.mp4
Shows all implemented capabilities working on real content
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def get_video_info(video_path):
    """Get basic video information"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "video"), None
        )
        audio_stream = next(
            (s for s in data["streams"] if s["codec_type"] == "audio"), None
        )

        info = {
            "duration": float(data["format"]["duration"]),
            "size_mb": int(data["format"]["size"]) / (1024 * 1024),
            "video": {
                "codec": video_stream["codec_name"] if video_stream else "none",
                "width": int(video_stream["width"]) if video_stream else 0,
                "height": int(video_stream["height"]) if video_stream else 0,
                "fps": eval(video_stream["r_frame_rate"]) if video_stream else 0,
            },
            "audio": {
                "codec": audio_stream["codec_name"] if audio_stream else "none",
                "sample_rate": int(audio_stream["sample_rate"]) if audio_stream else 0,
                "channels": int(audio_stream["channels"]) if audio_stream else 0,
            },
        }
        return info
    except Exception as e:
        return {"error": str(e)}


def demo_intelligent_cropping(video_path):
    """Demonstrate intelligent video cropping"""
    print("🎯 INTELLIGENT VIDEO CROPPING")
    print("-" * 40)

    try:
        from utils.intelligent_crop import IntelligentCropper

        cropper = IntelligentCropper()

        # Analyze different time segments
        segments = [
            (0, 5000),  # First 5 seconds
            (15000, 20000),  # Middle segment
            (30000, 35000),  # Later segment
        ]

        for i, (start_ms, end_ms) in enumerate(segments):
            print(f"  Segment {i+1} ({start_ms/1000:.1f}s-{end_ms/1000:.1f}s):")

            analysis = cropper.analyze_video_content(video_path, start_ms, end_ms)

            print(f"    📊 Faces detected: {analysis.get('face_count', 0)}")
            print(f"    🎯 Crop center: {analysis.get('optimal_crop_center', 'N/A')}")
            print(f"    📐 Crop dimensions: {analysis.get('crop_dimensions', 'N/A')}")
            print(f"    🔥 Motion level: {analysis.get('motion_level', 0):.2f}")
            print(f"    ⚡ Energy score: {analysis.get('energy_score', 0):.2f}")
            print()

        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def demo_story_ai_analysis(video_path):
    """Demonstrate AI story analysis with Gemini"""
    print("🧠 AI STORY ANALYSIS (Gemini 2.5)")
    print("-" * 40)

    try:
        from core.highlight_selector import analyze_story_structure

        # Create sample transcript segments (in real pipeline, these come from Whisper)
        sample_transcript = [
            {
                "text": "Welcome to this amazing journey of discovery",
                "start_ms": 0,
                "end_ms": 3000,
                "speaker": "host",
            },
            {
                "text": "Today we're going to explore something incredible",
                "start_ms": 3000,
                "end_ms": 6000,
                "speaker": "host",
            },
            {
                "text": "But first, let me ask you a question",
                "start_ms": 8000,
                "end_ms": 11000,
                "speaker": "host",
            },
            {
                "text": "What if everything you knew was wrong?",
                "start_ms": 11000,
                "end_ms": 14000,
                "speaker": "host",
            },
            {
                "text": "The research shows something fascinating",
                "start_ms": 20000,
                "end_ms": 23000,
                "speaker": "expert",
            },
            {
                "text": "This breakthrough changes everything",
                "start_ms": 35000,
                "end_ms": 38000,
                "speaker": "expert",
            },
            {
                "text": "The results were beyond our expectations",
                "start_ms": 40000,
                "end_ms": 43000,
                "speaker": "expert",
            },
            {
                "text": "So what does this mean for you?",
                "start_ms": 50000,
                "end_ms": 53000,
                "speaker": "host",
            },
            {
                "text": "The key takeaway is clear",
                "start_ms": 55000,
                "end_ms": 58000,
                "speaker": "host",
            },
        ]

        result = analyze_story_structure(sample_transcript, "test_demo")

        print(f"  📖 Story Theme: {result.get('theme', 'Not detected')}")
        print(f"  🎭 Narrative Type: {result.get('narrative_type', 'Not detected')}")
        print(f"  🎯 Target Emotion: {result.get('target_emotion', 'Not detected')}")
        print(f"  📊 Story Beats: {len(result.get('story_beats', []))}")
        print()

        # Show story beats
        beats = result.get("story_beats", [])
        for i, beat in enumerate(beats):
            print(f"  Beat {i+1}: {beat.get('beat_type', 'unknown').title()}")
            print(f"    Purpose: {beat.get('purpose', 'N/A')}")
            print(f"    Function: {beat.get('narrative_function', 'N/A')}")
            print(f"    Trigger: {beat.get('emotional_trigger', 'N/A')}")
            print(f"    Segments: {beat.get('target_segment_ids', [])}")
            print()

        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def demo_emotion_analysis():
    """Demonstrate emotion and energy analysis"""
    print("😊 EMOTION & ENERGY ANALYSIS")
    print("-" * 40)

    try:
        from core.emotion_analyzer import EmotionAnalyzer

        analyzer = EmotionAnalyzer()

        # Sample transcript for emotion analysis
        transcript = [
            {
                "text": "This is absolutely amazing and incredible",
                "start_ms": 0,
                "end_ms": 3000,
                "speaker": "host",
                "energy_level": 0.8,
            },
            {
                "text": "I'm curious about what happens next",
                "start_ms": 5000,
                "end_ms": 8000,
                "speaker": "guest",
                "energy_level": 0.6,
            },
            {
                "text": "The breakthrough was shocking and unexpected",
                "start_ms": 15000,
                "end_ms": 18000,
                "speaker": "expert",
                "energy_level": 0.9,
            },
            {
                "text": "This research proves our hypothesis",
                "start_ms": 25000,
                "end_ms": 28000,
                "speaker": "expert",
                "energy_level": 0.7,
            },
            {
                "text": "I feel relieved and much better now",
                "start_ms": 35000,
                "end_ms": 38000,
                "speaker": "host",
                "energy_level": 0.5,
            },
        ]

        # Mock audio energy data
        audio_energy = [0.4, 0.6, 0.8, 0.7, 0.9, 0.6, 0.5, 0.7, 0.8, 0.4] * 4

        analysis = analyzer.analyze_emotions_and_energy(transcript, audio_energy)

        print(
            f"  📊 Emotional Timeline: {len(analysis.get('emotion_timeline', []))} segments"
        )
        print(
            f"  ⚡ Energy Patterns: {len(analysis.get('energy_patterns', []))} patterns"
        )
        print(f"  🔥 Emotional Peaks: {len(analysis.get('emotional_peaks', []))} peaks")
        print(
            f"  🎯 Engagement Peaks: {len(analysis.get('engagement_peaks', []))} moments"
        )
        print()

        # Show emotional profile
        profile = analysis.get("overall_emotional_profile", {})
        print(f"  🎭 Dominant Emotion: {profile.get('dominant_emotion', 'neutral')}")
        print(f"  🌈 Emotional Diversity: {profile.get('emotional_diversity', 0):.2f}")
        print(f"  📈 Engagement Level: {profile.get('engagement_level', 0):.2f}")
        print()

        # Show top engagement moments
        peaks = analysis.get("engagement_peaks", [])[:3]
        for i, peak in enumerate(peaks):
            print(
                f"  Peak {i+1}: {peak.get('engagement_score', 0):.2f} at {peak.get('start_ms', 0)/1000:.1f}s"
            )
            print(f"    Emotion: {peak.get('primary_emotion', 'unknown')}")
            print(f"    Type: {peak.get('engagement_type', 'general')}")

        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def demo_narrative_detection():
    """Demonstrate narrative beat detection"""
    print("📚 NARRATIVE BEAT DETECTION")
    print("-" * 40)

    try:
        from core.narrative_detector import NarrativeDetector

        detector = NarrativeDetector()

        # Sample transcript with narrative elements
        transcript = [
            {
                "text": "What if I told you there's a secret most people never discover?",
                "start_ms": 0,
                "end_ms": 4000,
                "speaker": "host",
            },
            {
                "text": "This journey began three years ago",
                "start_ms": 5000,
                "end_ms": 8000,
                "speaker": "host",
            },
            {
                "text": "The biggest challenge we faced was finding the answer",
                "start_ms": 20000,
                "end_ms": 24000,
                "speaker": "expert",
            },
            {
                "text": "Then we had a breakthrough moment that changed everything",
                "start_ms": 35000,
                "end_ms": 39000,
                "speaker": "expert",
            },
            {
                "text": "In conclusion, the key takeaway is clear",
                "start_ms": 50000,
                "end_ms": 54000,
                "speaker": "host",
            },
            {
                "text": "Now you can take action and transform your life",
                "start_ms": 55000,
                "end_ms": 59000,
                "speaker": "host",
            },
        ]

        analysis = detector.detect_narrative_beats(transcript)

        hooks = analysis.get("hooks", [])
        climaxes = analysis.get("climaxes", [])
        resolutions = analysis.get("resolutions", [])

        print(f"  🎣 Hooks: {len(hooks)} detected")
        for hook in hooks[:2]:
            print(f"    • {hook.get('text', '')[:50]}...")
            print(
                f"      Score: {hook.get('hook_score', 0):.2f} | Type: {hook.get('dominant_hook_type', 'unknown')}"
            )

        print(f"  🎯 Climaxes: {len(climaxes)} detected")
        for climax in climaxes[:2]:
            print(f"    • {climax.get('text', '')[:50]}...")
            print(
                f"      Score: {climax.get('climax_score', 0):.2f} | Type: {climax.get('dominant_climax_type', 'unknown')}"
            )

        print(f"  ✅ Resolutions: {len(resolutions)} detected")
        for resolution in resolutions[:2]:
            print(f"    • {resolution.get('text', '')[:50]}...")
            print(
                f"      Score: {resolution.get('resolution_score', 0):.2f} | Type: {resolution.get('dominant_resolution_type', 'unknown')}"
            )

        print()
        print(f"  📊 Narrative Coherence: {analysis.get('narrative_coherence', 0):.2f}")
        print(
            f"  📈 Story Completeness: {analysis.get('story_arc_completeness', 0):.2f}"
        )

        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def demo_speaker_analysis():
    """Demonstrate speaker/character analysis"""
    print("👥 SPEAKER/CHARACTER ANALYSIS")
    print("-" * 40)

    try:
        from core.speaker_analysis import SpeakerAnalyzer

        analyzer = SpeakerAnalyzer()

        # Sample multi-speaker transcript
        transcript = [
            {
                "text": "Welcome everyone to today's show",
                "start_ms": 0,
                "end_ms": 3000,
                "speaker": "host",
                "energy_level": 0.7,
            },
            {
                "text": "Thank you for having me, great question",
                "start_ms": 5000,
                "end_ms": 8000,
                "speaker": "guest",
                "energy_level": 0.6,
            },
            {
                "text": "The research shows fascinating results",
                "start_ms": 10000,
                "end_ms": 14000,
                "speaker": "expert",
                "energy_level": 0.8,
            },
            {
                "text": "Let me ask you about that discovery",
                "start_ms": 16000,
                "end_ms": 19000,
                "speaker": "host",
                "energy_level": 0.7,
            },
            {
                "text": "In my experience, this changes everything",
                "start_ms": 21000,
                "end_ms": 25000,
                "speaker": "guest",
                "energy_level": 0.9,
            },
            {
                "text": "Data suggests we're seeing a breakthrough",
                "start_ms": 27000,
                "end_ms": 31000,
                "speaker": "expert",
                "energy_level": 0.8,
            },
        ]

        analysis = analyzer.analyze_speakers(transcript)

        speakers = analysis.get("speakers", {})
        roles = analysis.get("character_roles", {})
        anchors = analysis.get("narrative_anchors", [])

        print(f"  👤 Speakers: {len(speakers)} identified")
        for speaker_id, profile in speakers.items():
            role = roles.get(speaker_id, "unknown")
            is_anchor = speaker_id in anchors

            print(f"    {speaker_id}: {role} {'⭐' if is_anchor else ''}")
            print(
                f"      Words: {profile.get('total_words', 0)} | Time: {profile.get('speaking_time', 0)/1000:.1f}s"
            )
            print(
                f"      Energy: {profile.get('avg_energy', 0):.2f} | Dominance: {profile.get('dominance', 0):.2f}"
            )
            print(
                f"      Consistency: {profile.get('consistency', 0):.2f} | Sentiment: {profile.get('sentiment', 'neutral')}"
            )

        print()
        print(f"  📊 Continuity Score: {analysis.get('continuity_score', 0):.2f}")
        print(f"  ⭐ Narrative Anchors: {len(anchors)} speakers")

        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def demo_cost_tracking():
    """Demonstrate budget and cost tracking"""
    print("💰 BUDGET & COST TRACKING")
    print("-" * 40)

    try:
        from core.cost import get_current_cost, check_budget, _BUDGET_HARD_CAP

        current = get_current_cost()
        within_budget, remaining = check_budget()

        print(f"  💳 Current Spend: ${current:.5f}")
        print(f"  💸 Hard Cap: ${float(_BUDGET_HARD_CAP):.2f}")
        print(f"  💰 Remaining: ${remaining:.5f}")
        print(f"  ✅ Within Budget: {within_budget}")

        # Show cost per service
        print()
        print("  📊 Cost per Service:")
        print("    • Gemini 2.5 Flash: $0.00001 per call")
        print("    • Whisper (local): $0.00000 (free)")
        print("    • Face Detection: $0.00000 (local)")
        print("    • Video Processing: $0.00000 (local)")

        return True
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False


def main():
    """Run complete feature demonstration"""
    video_path = "/Users/hawzhin/Montage/test_video.mp4"

    print("🎬 MONTAGE COMPLETE FEATURE DEMO")
    print("=" * 50)
    print(f"📹 Video: {video_path}")

    # Get video info
    info = get_video_info(video_path)
    if "error" not in info:
        print(f"📊 Duration: {info['duration']:.1f}s | Size: {info['size_mb']:.1f}MB")
        print(
            f"📐 Resolution: {info['video']['width']}x{info['video']['height']} @ {info['video']['fps']:.1f}fps"
        )
        print(
            f"🔊 Audio: {info['audio']['codec']} {info['audio']['sample_rate']}Hz {info['audio']['channels']}ch"
        )
    else:
        print(f"❌ Video info error: {info['error']}")

    print("\n" + "=" * 50)

    # Run all feature demonstrations
    features = [
        ("Video Info", lambda: True),  # Already shown above
        ("Intelligent Cropping", lambda: demo_intelligent_cropping(video_path)),
        ("AI Story Analysis", lambda: demo_story_ai_analysis(video_path)),
        ("Emotion Analysis", demo_emotion_analysis),
        ("Narrative Detection", demo_narrative_detection),
        ("Speaker Analysis", demo_speaker_analysis),
        ("Cost Tracking", demo_cost_tracking),
    ]

    results = {}

    for feature_name, demo_func in features[1:]:  # Skip video info
        print(f"\n{demo_func()}")
        try:
            success = demo_func()
            results[feature_name] = "✅" if success else "❌"
        except Exception as e:
            results[feature_name] = f"❌ {str(e)[:30]}..."
        print()

    # Summary
    print("🏆 FEATURE SUMMARY")
    print("-" * 30)
    for feature, status in results.items():
        print(f"  {status} {feature}")

    working_count = sum(1 for status in results.values() if status == "✅")
    total_count = len(results)

    print(f"\n📊 Overall: {working_count}/{total_count} features working")
    if working_count == total_count:
        print("🎉 All systems operational!")
    else:
        print("⚠️  Some features need attention")


if __name__ == "__main__":
    main()
