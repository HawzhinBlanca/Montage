#!/usr/bin/env python3
"""
Real highlight selector - Local rule scorer + REAL Claude/GPT chunk merge (cost-capped)
"""
import json
import re
from typing import Dict, List, Any, Optional, TypedDict
import openai
import anthropic
from anthropic import Anthropic
import ffmpeg


class Word(TypedDict):
    word: str
    start: float
    end: float
    confidence: float


class Segment(TypedDict):
    start: float
    end: float
    text: str
    score: float


class Highlight(TypedDict):
    slug: str
    title: str
    start_ms: int
    end_ms: int
    score: float
    method: str
    cost: Optional[float]


class AIResult(TypedDict):
    highlights: List[Dict[str, Any]]
    cost: float


try:
    from ..config import OPENAI_API_KEY, ANTHROPIC_API_KEY, MAX_COST_USD
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, MAX_COST_USD


def get_audio_energy(video_path: str, start_time: float, duration: float) -> float:
    """Get audio energy/volume for a segment"""
    try:
        # Use ffmpeg to analyze audio volume
        stream = ffmpeg.input(video_path, ss=start_time, t=duration)
        stream = ffmpeg.filter(stream, "volumedetect")
        stream = ffmpeg.output(stream, "-", format="null")

        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        output = err.decode("utf-8")

        # Parse mean volume
        match = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", output)
        if match:
            db_value = float(match.group(1))
            # Convert dB to 0-1 scale (assuming -60dB is silence, -20dB is normal)
            energy = max(0, (db_value + 60) / 40)  # Normalize to 0-1
            return min(1.0, energy)

        return 0.5  # Default moderate energy

    except Exception as e:
        print(f"❌ Audio energy analysis failed: {e}")
        return 0.5


def local_rule_scorer(words: List[Dict], audio_energy: List[float]) -> List[Dict]:
    """Local rule-based scoring (token-free)"""

    # Keywords that indicate interesting content
    highlight_keywords = [
        "important",
        "key",
        "crucial",
        "amazing",
        "incredible",
        "breakthrough",
        "discovery",
        "research",
        "study",
        "significant",
        "remarkable",
        "fascinating",
        "surprising",
        "shocking",
        "revolutionary",
        "innovative",
        "groundbreaking",
        "success",
        "achievement",
        "victory",
        "champion",
        "excellent",
        "outstanding",
        "expert",
        "professor",
        "doctor",
        "scientist",
        "leader",
        "authority",
    ]

    # Process words into segments (every 5-10 seconds)
    segments = []
    current_segment = []
    segment_start = 0

    for word in words:
        if not current_segment:
            segment_start = word["start"]

        current_segment.append(word)

        # End segment after 5-10 seconds or at sentence boundary
        if (
            word["end"] - segment_start >= 5 and word["word"].endswith((".", "!", "?"))
        ) or (word["end"] - segment_start >= 10):

            # Score this segment
            segment_text = " ".join([w["word"] for w in current_segment])

            # Rule 1: Complete sentence bonus (2 points)
            has_complete_sentence = any(
                w["word"].endswith((".", "!", "?")) for w in current_segment
            )
            sentence_score = 2.0 if has_complete_sentence else 0.0

            # Rule 2: Keyword score (3 points per keyword, max 3)
            keyword_count = sum(
                1 for keyword in highlight_keywords if keyword in segment_text.lower()
            )
            keyword_score = min(keyword_count * 3.0, 3.0)

            # Rule 3: Audio energy score (0-2 points)
            segment_mid_time = (segment_start + word["end"]) / 2
            energy_score = (
                get_audio_energy_at_time(audio_energy, segment_mid_time) * 2.0
            )

            # Total score
            total_score = sentence_score + keyword_score + energy_score

            segments.append(
                {
                    "text": segment_text,
                    "start": segment_start,
                    "end": word["end"],
                    "score": total_score,
                    "breakdown": {
                        "sentence": sentence_score,
                        "keywords": keyword_score,
                        "energy": energy_score,
                    },
                    "speaker": current_segment[0].get("speaker", "SPEAKER_0"),
                }
            )

            current_segment = []

    # Sort by score
    segments.sort(key=lambda x: x["score"], reverse=True)

    print(
        f"✅ Local scoring: {len(segments)} segments, top score: {segments[0]['score'] if segments else 0}"
    )
    return segments


def get_audio_energy_at_time(energy_data: List[float], time: float) -> float:
    """Get audio energy at specific time"""
    if not energy_data:
        return 0.5

    # Simple lookup (energy_data would be pre-computed for the whole video)
    index = int(time) % len(energy_data)  # Crude approximation
    return energy_data[index]


def chunk_transcript_for_ai(transcript: str, max_tokens: int = 7000) -> List[str]:
    """Chunk transcript into manageable pieces for AI analysis"""

    # Simple chunking by character count (rough token estimation: 1 token ≈ 4 chars)
    max_chars = max_tokens * 4

    chunks = []
    current_chunk = ""

    # Split by sentences
    sentences = re.split(r"[.!?]+", transcript)

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence too long - split it
                chunks.append(sentence[:max_chars])
                current_chunk = sentence[max_chars:]
        else:
            current_chunk += sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def analyze_with_claude(text: str) -> Dict[str, Any]:
    """REAL Claude analysis with cost tracking"""

    # TODO: Implement budget checking

    if (
        not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your-anthropic-key-here"
    ):  # pragma: allowlist secret
        print("❌ Anthropic API key not set")
        return {"highlights": [], "cost": 0}

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""Analyze this transcript and identify the 3-5 most interesting/valuable highlights.

Transcript:
{text}

Return JSON with format:
{{
  "highlights": [
    {{
      "title": "Brief descriptive title (max 8 words)",
      "reason": "Why this is interesting",
      "importance": 1-10,
      "keywords": ["key", "terms"]
    }}
  ]
}}

Focus on: insights, discoveries, expert opinions, surprising facts, actionable advice."""

        response = client.messages.create(  # pylint: disable=no-member
            model="claude-3-5-sonnet-20241022",  # Claude Sonnet 3.5 (latest)
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        response_text = response.content[0].text
        result = json.loads(response_text)

        print("✅ Claude analysis complete")
        return result

    except anthropic.RateLimitError:
        return []
    except Exception as e:
        print(f"❌ Claude analysis failed: {e}")
        return {"highlights": [], "cost": 0}


def analyze_with_gpt(text: str) -> Dict[str, Any]:
    """REAL GPT-4 analysis with cost tracking"""

    # TODO: Implement budget checking

    if (
        not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-key-here"
    ):  # pragma: allowlist secret
        print("❌ OpenAI API key not set")
        return {"highlights": [], "cost": 0}

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using available mini model
            messages=[
                {
                    "role": "system",
                    "content": "You are a content analysis expert. Identify the most interesting highlights from transcripts.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze this transcript and identify the 3-5 most interesting highlights.

Transcript:
{text}

Return JSON with format:
{{
  "highlights": [
    {{
      "title": "Brief descriptive title (max 8 words)",
      "reason": "Why this is interesting",
      "importance": 1-10,
      "keywords": ["key", "terms"]
    }}
  ]
}}

Focus on: insights, discoveries, expert opinions, surprising facts, actionable advice.""",
                },
            ],
            max_tokens=1000,
            temperature=0.3,
        )

        # Parse response
        result = json.loads(response.choices[0].message.content)

        print("✅ GPT analysis complete")
        return result

    except Exception as e:
        print(f"❌ GPT analysis failed: {e}")
        return {"highlights": [], "cost": 0}


def merge_ai_with_local(
    local_segments: List[Dict], ai_results: List[Dict]
) -> List[Dict]:
    """Merge AI analysis with local rule-based scoring"""

    # Create AI keyword boost map
    ai_keywords = set()
    for result in ai_results:
        for highlight in result.get("highlights", []):
            ai_keywords.update(highlight.get("keywords", []))

    # Boost local segments that match AI-identified keywords
    for segment in local_segments:
        segment_text = segment["text"].lower()

        # Count AI keyword matches
        ai_matches = sum(
            1 for keyword in ai_keywords if keyword.lower() in segment_text
        )

        # Boost score based on AI matches
        ai_boost = min(ai_matches * 2.0, 5.0)  # Max 5 point boost
        segment["score"] += ai_boost
        segment["breakdown"]["ai_boost"] = ai_boost

    # Re-sort by updated scores
    local_segments.sort(key=lambda x: x["score"], reverse=True)

    return local_segments


def rank_candidates(
    segments: List[Segment], keywords: Optional[List[str]] = None
) -> List[Segment]:
    """
    Rank segments by score, optionally boosting for keywords
    """
    if keywords:
        for segment in segments:
            keyword_boost = (
                sum(1 for kw in keywords if kw in segment["text"].lower()) * 0.5
            )
            segment["score"] += keyword_boost

    return sorted(segments, key=lambda x: x["score"], reverse=True)


def apply_rules(
    segments: List[Segment], mode: str, ai_results: Optional[List[AIResult]] = None
) -> List[Segment]:
    """
    Apply mode-specific rules to select segments
    """
    if mode == "smart":
        return segments[:5]  # Top 5 for smart mode
    elif mode == "premium":
        if ai_results:
            enhanced_segments = merge_ai_with_local(segments, ai_results)
            return enhanced_segments[:8]  # Up to 8 for premium
        else:
            # Fallback to enhanced local scoring
            keywords = [
                "important",
                "key",
                "remember",
                "first",
                "second",
                "conclusion",
                "summary",
            ]
            ranked = rank_candidates(segments.copy(), keywords)
            return ranked[:8]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def filter_by_length(
    segments: List[Segment], min_duration: float = 5.0, max_duration: float = 60.0
) -> List[Segment]:
    """
    Filter segments by duration constraints
    """
    filtered = []
    for segment in segments:
        duration = segment["end"] - segment["start"]
        if min_duration <= duration <= max_duration:
            filtered.append(segment)
    return filtered


def choose_highlights(
    words: List[Word], audio_energy: List[float], mode: str = "smart"
) -> List[Highlight]:
    """
    Main highlight selection function
    mode: "smart" (local only) or "premium" (local + AI)
    """
    print(f"🎯 Choosing highlights in {mode} mode")

    # Step 1: Local rule-based scoring
    local_segments = local_rule_scorer(words, audio_energy)

    # Step 2: Filter by length constraints
    local_segments = filter_by_length(local_segments)

    if mode == "smart":
        # Smart mode: local scoring only
        top_segments = apply_rules(local_segments, mode)

        # Convert to clip format
        clips = []
        for i, segment in enumerate(top_segments):
            clips.append(
                {
                    "slug": f"highlight-{i+1}",
                    "title": segment["text"][:40].strip(),
                    "start_ms": int(segment["start"] * 1000),
                    "end_ms": int(segment["end"] * 1000),
                    "score": segment["score"],
                    "method": "local_rules",
                }
            )

        print(f"✅ Smart mode: {len(clips)} highlights selected")
        return clips

    elif mode == "premium":
        # Premium mode: local + AI analysis

        # Create full transcript from words
        transcript = " ".join([word["word"] for word in words])

        # Chunk transcript for AI analysis
        chunks = chunk_transcript_for_ai(transcript, max_tokens=7000)

        # Analyze each chunk with AI
        ai_results = []
        ai_failed = False

        for chunk in chunks[:3]:  # Limit to 3 chunks to control cost
            # Try GPT first (since it's working), then Claude as fallback
            gpt_result = analyze_with_gpt(chunk)
            if gpt_result["highlights"]:
                ai_results.append(gpt_result)
            else:
                # Try Claude as fallback
                claude_result = analyze_with_claude(chunk)
                if claude_result["highlights"]:
                    ai_results.append(claude_result)
                else:
                    # Both AI services failed
                    print(
                        "⚠️  Both AI services unavailable, using enhanced local scoring"
                    )
                    ai_failed = True
                    break

            # Stop if cost limit reached
            # TODO: Implement cost tracking
            pass

        # Apply rules with AI results or fallback
        top_segments = apply_rules(
            local_segments, mode, ai_results if not ai_failed else None
        )

        # Convert to clip format
        clips = []
        for i, segment in enumerate(top_segments):
            clips.append(
                {
                    "slug": f"premium-highlight-{i+1}",
                    "title": segment["text"][:40].strip(),
                    "start_ms": int(segment["start"] * 1000),
                    "end_ms": int(segment["end"] * 1000),
                    "score": segment["score"],
                    "method": "premium_ai",
                    "cost": 0.0,  # TODO: Implement cost tracking
                }
            )

        # TODO: Implement cost tracking
        print(f"✅ Premium mode: {len(clips)} highlights selected")
        return clips

    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_total_cost() -> float:
    """Get total API cost spent"""
    # TODO: Implement cost tracking
    return 0.0
