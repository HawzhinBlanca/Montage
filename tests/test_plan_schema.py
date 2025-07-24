"""
Unit tests for plan schema validation
"""
import pytest
import json
import jsonschema
from montage.core.plan import validate_plan

@pytest.fixture
def valid_plan():
    return {
        "version": "1.0",
        "source": "file:///tmp/example.mp4",
        "actions": [{"type": "cut", "start_ms": 0, "end_ms": 1000, "score": 0.5}],
        "render": {"format": "9:16", "codec": "h264", "crf": 18}
    }

def test_valid_plan(valid_plan):
    """Test that a valid plan passes validation"""
    validate_plan(valid_plan)  # Should not raise

@pytest.mark.parametrize("missing", ["version", "source", "actions", "render"])
def test_missing_required(missing, valid_plan):
    """Test that missing required fields raise ValidationError"""
    plan = valid_plan.copy()
    plan.pop(missing)
    with pytest.raises(jsonschema.ValidationError):
        validate_plan(plan)

def test_invalid_action_type(valid_plan):
    """Test that invalid action types raise ValidationError"""
    plan = valid_plan.copy()
    plan["actions"][0]["type"] = "invalid_type"
    with pytest.raises(jsonschema.ValidationError):
        validate_plan(plan)

def test_negative_times(valid_plan):
    """Test that negative times raise ValidationError"""
    plan = valid_plan.copy()
    plan["actions"][0]["start_ms"] = -1
    with pytest.raises(jsonschema.ValidationError):
        validate_plan(plan)

def test_score_out_of_range(valid_plan):
    """Test that scores outside 0-1 range raise ValidationError"""
    plan = valid_plan.copy()
    plan["actions"][0]["score"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        validate_plan(plan)

def test_minimal_valid_plan():
    """Test minimal valid plan with only required fields"""
    minimal_plan = {
        "version": "1.0",
        "source": "file:///tmp/test.mp4",
        "actions": [{"type": "cut"}],
        "render": {"format": "16:9", "codec": "h265", "crf": 23}
    }
    validate_plan(minimal_plan)  # Should not raise

def test_transition_action_type(valid_plan):
    """Test that transition action type is valid"""
    plan = valid_plan.copy()
    plan["actions"][0]["type"] = "transition"
    validate_plan(plan)  # Should not raise

def test_empty_actions_array(valid_plan):
    """Test that empty actions array is valid"""
    plan = valid_plan.copy()
    plan["actions"] = []
    validate_plan(plan)  # Should not raise

def test_multiple_actions(valid_plan):
    """Test plan with multiple actions"""
    plan = valid_plan.copy()
    plan["actions"].append({
        "type": "transition",
        "start_ms": 1000,
        "end_ms": 2000,
        "score": 0.8,
        "style": "fade"
    })
    validate_plan(plan)  # Should not raise