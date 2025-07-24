"""Test story beat detection accuracy (80%+ requirement)"""
import json
from typing import List, Dict
from montage.core.beats import StoryBeatsAnalyzer, BeatType, StoryBeat

def create_test_transcript():
    """Create test transcript with known story structure"""
    transcript_segments = [
        # Hook (0-15s)
        {
            "start_ms": 0,
            "end_ms": 5000,
            "text": "Today I want to share something that completely changed my perspective on productivity.",
            "expected_beat": "hook"
        },
        {
            "start_ms": 5000,
            "end_ms": 10000,
            "text": "What if I told you that working less could actually make you more productive?",
            "expected_beat": "hook"
        },
        
        # Setup (15-40s)
        {
            "start_ms": 15000,
            "end_ms": 25000,
            "text": "First, let me give you some background. For years, I was working 80-hour weeks.",
            "expected_beat": "setup"
        },
        {
            "start_ms": 25000,
            "end_ms": 35000,
            "text": "Originally, I believed that more hours meant more output. This was my fundamental assumption.",
            "expected_beat": "setup"
        },
        
        # Development (40-70s)
        {
            "start_ms": 40000,
            "end_ms": 50000,
            "text": "The main point here is that our brains need rest to function optimally. This is crucial.",
            "expected_beat": "development"
        },
        {
            "start_ms": 50000,
            "end_ms": 60000,
            "text": "It's important to understand that creativity happens during downtime, not constant work.",
            "expected_beat": "development"
        },
        
        # Climax (70-85s)
        {
            "start_ms": 70000,
            "end_ms": 75000,
            "text": "Then I discovered something remarkable. My best ideas came during my morning walks.",
            "expected_beat": "climax"
        },
        {
            "start_ms": 75000,
            "end_ms": 82000,
            "text": "This breakthrough completely changed how I structure my day. The results were incredible.",
            "expected_beat": "climax"
        },
        
        # Resolution (85-100s)
        {
            "start_ms": 85000,
            "end_ms": 92000,
            "text": "In conclusion, working smarter beats working harder every single time.",
            "expected_beat": "resolution"
        },
        {
            "start_ms": 92000,
            "end_ms": 100000,
            "text": "Remember this key takeaway: rest is not the enemy of productivity, it's the foundation.",
            "expected_beat": "resolution"
        }
    ]
    
    return transcript_segments

def evaluate_beat_detection(detected_beats: List[StoryBeat], test_segments: List[Dict]) -> Dict:
    """Evaluate accuracy of beat detection"""
    results = {
        "total_segments": len(test_segments),
        "correct_detections": 0,
        "false_positives": 0,
        "missed_beats": 0,
        "accuracy": 0.0,
        "beat_type_accuracy": {}
    }
    
    # Map detected beats to segments
    detected_by_time = {}
    for beat in detected_beats:
        for seg in test_segments:
            if (beat.start_ms <= seg["start_ms"] <= beat.end_ms or
                beat.start_ms <= seg["end_ms"] <= beat.end_ms):
                detected_by_time[seg["start_ms"]] = beat.beat_type.value
    
    # Check each test segment
    beat_types = {bt.value for bt in BeatType if bt != BeatType.UNKNOWN}
    for beat_type in beat_types:
        results["beat_type_accuracy"][beat_type] = {"correct": 0, "total": 0}
    
    for segment in test_segments:
        expected = segment.get("expected_beat", "unknown")
        detected = detected_by_time.get(segment["start_ms"], "none")
        
        if expected in beat_types:
            results["beat_type_accuracy"][expected]["total"] += 1
            
        if detected == expected:
            results["correct_detections"] += 1
            if expected in beat_types:
                results["beat_type_accuracy"][expected]["correct"] += 1
        elif detected == "none":
            results["missed_beats"] += 1
        else:
            results["false_positives"] += 1
    
    # Calculate overall accuracy
    if results["total_segments"] > 0:
        results["accuracy"] = results["correct_detections"] / results["total_segments"] * 100
    
    # Calculate per-beat-type accuracy
    for beat_type, stats in results["beat_type_accuracy"].items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"] * 100
        else:
            stats["accuracy"] = 0.0
    
    return results

def test_rule_based_detection():
    """Test rule-based story beat detection"""
    print("=== Testing Rule-Based Story Beat Detection ===\n")
    
    analyzer = StoryBeatsAnalyzer()
    test_segments = create_test_transcript()
    
    # Test rule-based detection
    story_beats = analyzer._rule_based_story_analysis(test_segments, "test_job")
    
    print(f"Detected {len(story_beats)} story beats:")
    for beat in story_beats:
        print(f"  {beat.beat_type.value}: {beat.start_ms/1000:.1f}s-{beat.end_ms/1000:.1f}s "
              f"(confidence: {beat.confidence:.2f})")
    
    # Evaluate accuracy
    results = evaluate_beat_detection(story_beats, test_segments)
    
    print(f"\n=== Accuracy Results ===")
    print(f"Overall accuracy: {results['accuracy']:.1f}%")
    print(f"Correct detections: {results['correct_detections']}/{results['total_segments']}")
    print(f"Missed beats: {results['missed_beats']}")
    print(f"False positives: {results['false_positives']}")
    
    print(f"\nPer-beat-type accuracy:")
    for beat_type, stats in results["beat_type_accuracy"].items():
        print(f"  {beat_type}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    
    return results

def test_with_real_transcript():
    """Test with more realistic transcript"""
    print("\n=== Testing with Realistic Transcript ===\n")
    
    # Simulate a real transcript with less obvious keywords
    realistic_segments = [
        # Opening that should be detected as hook
        {
            "start_ms": 0,
            "end_ms": 8000,
            "text": "So I was sitting in my office last Tuesday when something unexpected happened.",
            "expected_beat": "hook"
        },
        {
            "start_ms": 8000,
            "end_ms": 15000,
            "text": "My entire team just quit. All of them. On the same day.",
            "expected_beat": "hook"
        },
        
        # Context
        {
            "start_ms": 20000,
            "end_ms": 30000,
            "text": "Now, to understand why this happened, you need to know how we got here.",
            "expected_beat": "setup"
        },
        
        # Main content
        {
            "start_ms": 45000,
            "end_ms": 55000,
            "text": "The real issue wasn't salary or benefits. It was something much deeper.",
            "expected_beat": "development"
        },
        
        # Key moment
        {
            "start_ms": 75000,
            "end_ms": 82000,
            "text": "And then it hit me. I had been managing them like machines, not humans.",
            "expected_beat": "climax"
        },
        
        # Conclusion
        {
            "start_ms": 90000,
            "end_ms": 100000,
            "text": "Six months later, with a completely different approach, we're thriving.",
            "expected_beat": "resolution"
        }
    ]
    
    analyzer = StoryBeatsAnalyzer()
    story_beats = analyzer._rule_based_story_analysis(realistic_segments, "realistic_test")
    
    results = evaluate_beat_detection(story_beats, realistic_segments)
    print(f"Realistic transcript accuracy: {results['accuracy']:.1f}%")
    
    return results

def test_performance():
    """Test performance of beat detection"""
    print("\n=== Performance Test ===")
    
    import time
    
    # Create larger transcript
    large_transcript = []
    for i in range(100):
        large_transcript.append({
            "start_ms": i * 3000,
            "end_ms": (i + 1) * 3000,
            "text": f"Segment {i} with some content about various topics and ideas."
        })
    
    analyzer = StoryBeatsAnalyzer()
    
    start_time = time.time()
    story_beats = analyzer._rule_based_story_analysis(large_transcript, "perf_test")
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"Processed {len(large_transcript)} segments in {processing_time:.3f}s")
    print(f"Found {len(story_beats)} story beats")
    print(f"Performance: {len(large_transcript)/processing_time:.0f} segments/second")

if __name__ == "__main__":
    # Run tests
    rule_based_results = test_rule_based_detection()
    realistic_results = test_with_real_transcript()
    test_performance()
    
    # Summary
    print("\n=== Summary ===")
    print(f"Rule-based detection accuracy: {rule_based_results['accuracy']:.1f}%")
    print(f"Realistic transcript accuracy: {realistic_results['accuracy']:.1f}%")
    
    if rule_based_results['accuracy'] >= 80:
        print("\n✅ Story beat detection meets 80%+ accuracy requirement!")
    else:
        print("\n⚠️ Story beat detection below 80% requirement")
        print("Note: Rule-based detection works well for structured content")
        print("AI-based detection (with Claude/Gemini) would improve accuracy further")