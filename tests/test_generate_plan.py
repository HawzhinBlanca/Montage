"""
Unit tests for generate_plan() function
"""
import pytest
from montage.core.plan import generate_plan, Segment

def test_generate_plan():
    """Test basic plan generation from segments and scores"""
    segs = [Segment(0, 0, 1), Segment(1, 1, 2)]
    scores = {0: 0.1, 1: 0.9}
    plan = generate_plan(segs, scores, "/tmp/foo.mp4")
    
    assert plan["source"] == "/tmp/foo.mp4"
    assert len(plan["actions"]) == 2
    assert plan["actions"][0]["score"] == 0.9  # Higher score should be first
    assert plan["actions"][1]["score"] == 0.1

def test_generate_plan_sorts_by_score():
    """Test that segments are sorted by score in descending order"""
    segs = [
        Segment(0, 0, 1),
        Segment(1, 1, 2), 
        Segment(2, 2, 3)
    ]
    scores = {0: 0.5, 1: 0.9, 2: 0.2}
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    # Should be ordered by score: 1(0.9), 0(0.5), 2(0.2)
    assert plan["actions"][0]["score"] == 0.9
    assert plan["actions"][1]["score"] == 0.5
    assert plan["actions"][2]["score"] == 0.2

def test_generate_plan_limits_to_five():
    """Test that only top 5 segments are included"""
    segs = [Segment(i, i, i+1) for i in range(10)]
    scores = {i: i * 0.1 for i in range(10)}  # 0.0 to 0.9
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    assert len(plan["actions"]) == 5
    # Should have top 5 scores: 0.9, 0.8, 0.7, 0.6, 0.5
    expected_scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    actual_scores = [action["score"] for action in plan["actions"]]
    assert actual_scores == expected_scores

def test_generate_plan_converts_times_to_ms():
    """Test that start_time and end_time are converted to milliseconds"""
    segs = [Segment(0, 1.5, 2.7)]  # 1.5s to 2.7s
    scores = {0: 0.8}
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    action = plan["actions"][0]
    assert action["start_ms"] == 1500  # 1.5s = 1500ms
    assert action["end_ms"] == 2700    # 2.7s = 2700ms

def test_generate_plan_handles_missing_scores():
    """Test that segments with missing scores are handled gracefully"""
    segs = [Segment(0, 0, 1), Segment(1, 1, 2)]
    scores = {0: 0.7}  # Missing score for segment 1
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    # Segment with missing score should get 0 and be sorted last
    assert len(plan["actions"]) == 2
    assert plan["actions"][0]["score"] == 0.7
    assert plan["actions"][1]["score"] == 0.0

def test_generate_plan_has_required_fields():
    """Test that generated plan has all required schema fields"""
    segs = [Segment(0, 0, 1)]
    scores = {0: 0.5}
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    # Check required top-level fields
    assert "version" in plan
    assert "source" in plan
    assert "actions" in plan
    assert "render" in plan
    
    # Check render fields
    render = plan["render"]
    assert "format" in render
    assert "codec" in render
    assert "crf" in render

def test_generate_plan_action_has_required_fields():
    """Test that generated actions have required fields"""
    segs = [Segment(0, 0, 1)]
    scores = {0: 0.5}
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    action = plan["actions"][0]
    assert action["type"] == "cut"
    assert "start_ms" in action
    assert "end_ms" in action
    assert "score" in action

def test_generate_plan_empty_segments():
    """Test generate_plan with empty segments list"""
    segs = []
    scores = {}
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    assert plan["source"] == "/tmp/test.mp4"
    assert len(plan["actions"]) == 0
    assert "version" in plan
    assert "render" in plan

def test_generate_plan_rounds_scores():
    """Test that scores are rounded to 3 decimal places"""
    segs = [Segment(0, 0, 1)]
    scores = {0: 0.123456789}
    plan = generate_plan(segs, scores, "/tmp/test.mp4")
    
    assert plan["actions"][0]["score"] == 0.123