"""
Plan generation and validation for Montage video processing
"""
import json
import jsonschema
from pathlib import Path
from typing import List, Dict, Any

# Load schema
_schema_path = Path(__file__).parent / "plan_schema.json"
_schema = json.loads(_schema_path.read_text())

def validate_plan(plan: dict) -> None:
    """Validate a plan against the JSON schema"""
    jsonschema.validate(plan, _schema)

class Segment:
    """Simple segment class for plan generation"""
    def __init__(self, id: int, start_time: float, end_time: float):
        self.id = id
        self.start_time = start_time
        self.end_time = end_time

def generate_plan(segments: List[Segment], scores: Dict[int, float], source: str) -> dict:
    """Produce a plan dict from scored segments."""
    # Sort segments by score and take top 5
    top = sorted(segments, key=lambda s: scores.get(s.id, 0), reverse=True)[:5]
    
    actions = []
    for s in top:
        actions.append({
            "type": "cut",
            "start_ms": int(s.start_time * 1000),
            "end_ms": int(s.end_time * 1000),
            "score": round(scores.get(s.id, 0), 3)
        })
    
    plan = {
        "version": "1.0",
        "source": source,
        "actions": actions,
        "render": {"format": "9:16", "codec": "h264", "crf": 18}
    }
    
    # Validate before returning
    validate_plan(plan)
    return plan

def generate_plan_from_highlights(highlights: List[Dict[str, Any]], source: str) -> dict:
    """Generate a plan from highlight segments (used by existing pipeline)"""
    actions = []
    
    for highlight in highlights:
        actions.append({
            "type": "cut",
            "start_ms": highlight.get("start_ms", 0),
            "end_ms": highlight.get("end_ms", 1000),
            "score": round(highlight.get("score", 0.5), 3)
        })
    
    plan = {
        "version": "1.0",
        "source": source,
        "actions": actions,
        "render": {"format": "9:16", "codec": "h264", "crf": 18}
    }
    
    # Validate before returning
    validate_plan(plan)
    return plan