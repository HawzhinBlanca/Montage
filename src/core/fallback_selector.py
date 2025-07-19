#!/usr/bin/env python3
"""
Fallback highlight selector for when APIs are down
"""
from typing import List, Dict, Any
import random


def analyze_with_fallback(text: str, word_timings: List[Dict]) -> Dict[str, Any]:
    """Fallback analysis when APIs are unavailable"""

    # Simple keyword-based scoring
    keywords = [
        "important",
        "key",
        "significant",
        "remember",
        "note",
        "first",
        "second",
        "third",
        "finally",
        "conclusion",
        "amazing",
        "interesting",
        "crucial",
        "essential",
    ]

    # Score segments based on keywords
    segments = []
    current_segment = []

    for i, word in enumerate(word_timings):
        current_segment.append(word)

        # Create segment every 20 words or at sentence end
        if (i + 1) % 20 == 0 or word["word"].endswith("."):
            if current_segment:
                segment_text = " ".join([w["word"] for w in current_segment])
                score = sum(
                    1 for keyword in keywords if keyword in segment_text.lower()
                )

                segments.append(
                    {
                        "start": current_segment[0]["start"],
                        "end": current_segment[-1]["end"],
                        "text": segment_text,
                        "score": score + random.random(),  # Add some randomness
                    }
                )
                current_segment = []

    # Sort by score and take top 3-5
    segments.sort(key=lambda x: x["score"], reverse=True)
    top_segments = segments[: min(5, len(segments))]

    # Format as highlights
    highlights = []
    for i, seg in enumerate(top_segments):
        highlights.append(
            {
                "slug": f"highlight-{i+1}",
                "title": seg["text"][:40] + "...",
                "start_ms": int(seg["start"] * 1000),
                "end_ms": int(seg["end"] * 1000),
                "score": seg["score"],
            }
        )

    return {"highlights": highlights, "cost": 0}  # Free fallback


# Save this as a backup
if __name__ == "__main__":
    print("Fallback highlight selector ready")
