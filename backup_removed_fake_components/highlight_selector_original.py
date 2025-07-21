#!/usr/bin/env python3
"""
Real highlight selector - Local rule scorer + REAL Claude/GPT chunk merge (cost-capped)
"""
import json
import re
from typing import Dict, List, Any, Optional, TypedDict
from decimal import Decimal
import logging
import openai
import ffmpeg

logger = logging.getLogger(__name__)


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
    from ..config import OPENAI_API_KEY, MAX_COST_USD
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from config import OPENAI_API_KEY, MAX_COST_USD


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


try:
    from ..core.cost import priced, check_budget
except ImportError:
    from core.cost import priced, check_budget


@priced("gemini.2-5-flash", Decimal("0.00001"))  # Gemini is much cheaper
def analyze_story_structure(
    transcript_segments: List[Dict], job_id: str = "default"
) -> Dict[str, Any]:
    """Advanced story analysis with timestamp mapping and narrative arc creation using Gemini 2.5"""

    # Check budget before proceeding
    can_afford, remaining = check_budget(0.05)
    if not can_afford:
        logger.warning(
            f"Budget insufficient for Gemini analysis: ${remaining} remaining"
        )
        return {"highlights": [], "cost": 0}

    try:
        import google.generativeai as genai
        import os

        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Gemini API key not set")
            return {"highlights": [], "cost": 0}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Prepare time-indexed transcript for analysis
        time_indexed_content = []
        for i, segment in enumerate(transcript_segments):
            time_indexed_content.append(
                {
                    "segment_id": i,
                    "start_ms": segment.get("start_ms", 0),
                    "end_ms": segment.get("end_ms", 0),
                    "text": segment.get("text", ""),
                    "speaker": segment.get("speaker", "unknown"),
                    "energy_level": segment.get("energy_level", 0.5),
                }
            )

        # Create focused story analysis prompt
        segments_text = "\n".join(
            [
                f"[{seg['start_ms']//1000}s-{seg['end_ms']//1000}s] Speaker {seg['speaker']}: {seg['text']}"
                for seg in time_indexed_content[:50]  # Limit for token budget
            ]
        )

        prompt = f"""You are a professional video editor creating a compelling story from this transcript.

TRANSCRIPT WITH TIMESTAMPS:
{segments_text}

TASK: Create a narrative arc for a viral vertical video (60-90 seconds max).

REQUIREMENTS:
1. Find the BEST story that flows chronologically (respect timeline order)
2. Identify clear narrative beats: HOOK → DEVELOPMENT → CLIMAX → RESOLUTION
3. Ensure speaker continuity (don't jump between speakers randomly)
4. Map each story beat to specific timestamp ranges

STRICT JSON FORMAT:
{{
  "story_analysis": {{
    "main_theme": "One sentence story summary",
    "narrative_type": "reveal/journey/transformation/conflict/explanation",
    "target_emotion": "curiosity/inspiration/shock/empowerment",
    "key_speakers": ["Speaker_0", "Speaker_1"],
    "optimal_duration": "60-90 seconds"
  }},
  "story_beats": [
    {{
      "beat_type": "hook",
      "purpose": "grab attention in first 3 seconds",
      "target_segment_ids": [0, 1],
      "narrative_function": "why this matters now",
      "emotional_trigger": "curiosity/urgency/surprise"
    }},
    {{
      "beat_type": "development", 
      "purpose": "build tension/explain context",
      "target_segment_ids": [15, 22, 28],
      "narrative_function": "what's the problem/opportunity",
      "emotional_trigger": "investment/concern"
    }},
    {{
      "beat_type": "climax",
      "purpose": "peak revelation/turning point", 
      "target_segment_ids": [45],
      "narrative_function": "the key insight/breakthrough",
      "emotional_trigger": "aha moment/shock/inspiration"
    }},
    {{
      "beat_type": "resolution",
      "purpose": "satisfying conclusion/call-to-action",
      "target_segment_ids": [48, 49],
      "narrative_function": "what this means for viewer",
      "emotional_trigger": "empowerment/motivation"
    }}
  ],
  "continuity_rules": {{
    "maintain_speaker_context": true,
    "chronological_flow": true,
    "smooth_transitions": true
  }}
}}

Prioritize: surprising revelations, transformative moments, relatable content, actionable insights.

Return ONLY the JSON, no other text."""

        response = model.generate_content(prompt)

        # Parse response
        response_text = response.text.strip()

        # Extract JSON from response (sometimes has extra text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            response_text = response_text[json_start:json_end]

        result = json.loads(response_text)

        # Convert story beats to actual highlight clips
        highlights = convert_story_beats_to_clips(result, transcript_segments)

        print(
            f"✅ Gemini story analysis complete: {len(highlights)} narrative beats identified"
        )
        return {
            "story_analysis": result.get("story_analysis", {}),
            "highlights": highlights,
            "story_beats": result.get("story_beats", []),
            "continuity_rules": result.get("continuity_rules", {}),
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response as JSON: {e}")
        print(f"Raw response: {response_text[:500]}...")
        return {"highlights": [], "cost": 0}
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return {"highlights": [], "cost": 0}


def convert_story_beats_to_clips(
    story_result: Dict, transcript_segments: List[Dict]
) -> List[Dict]:
    """Convert AI story beats into actual video clips with timestamps"""

    highlights = []
    story_beats = story_result.get("story_beats", [])
    story_analysis = story_result.get("story_analysis", {})

    for beat in story_beats:
        target_segment_ids = beat.get("target_segment_ids", [])

        if not target_segment_ids:
            continue

        # Get segments for this story beat
        beat_segments = []
        for seg_id in target_segment_ids:
            if seg_id < len(transcript_segments):
                beat_segments.append(transcript_segments[seg_id])

        if not beat_segments:
            continue

        # Create clip spanning all segments in this beat
        start_ms = min(seg.get("start_ms", 0) for seg in beat_segments)
        end_ms = max(seg.get("end_ms", 0) for seg in beat_segments)

        # Ensure reasonable clip length (10-30 seconds)
        duration_ms = end_ms - start_ms
        if duration_ms < 5000:  # Less than 5 seconds, extend
            end_ms = start_ms + 8000
        elif duration_ms > 30000:  # More than 30 seconds, trim
            end_ms = start_ms + 25000

        # Combine text from all segments
        combined_text = " ".join(seg.get("text", "") for seg in beat_segments)

        # Create title from first few words
        words = combined_text.split()[:8]
        title = " ".join(words)

        # Calculate narrative importance score
        role_scores = {
            "hook": 10.0,  # Always include hook
            "climax": 9.5,  # Peak moment
            "resolution": 8.5,  # Strong ending
            "development": 7.0,  # Supporting content
        }

        base_score = role_scores.get(beat.get("beat_type", "development"), 5.0)
        importance = beat.get("importance", 5)
        final_score = base_score + (importance / 10.0)

        highlight = {
            "slug": f"story-{beat.get('beat_type', 'unknown')}-{len(highlights)}",
            "title": title,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "score": final_score,
            "method": "story_ai",
            # Story metadata
            "narrative_role": beat.get("beat_type", "development"),
            "narrative_function": beat.get("narrative_function", ""),
            "emotional_impact": beat.get("emotional_trigger", "neutral"),
            "story_theme": story_analysis.get("main_theme", ""),
            "target_emotion": story_analysis.get("target_emotion", ""),
            # Technical metadata
            "segment_ids": target_segment_ids,
            "speaker": (
                beat_segments[0].get("speaker", "unknown")
                if beat_segments
                else "unknown"
            ),
            "energy_level": (
                sum(seg.get("energy_level", 0.5) for seg in beat_segments)
                / len(beat_segments)
                if beat_segments
                else 0.5
            ),
        }

        highlights.append(highlight)

    # Sort by intelligent narrative flow that respects both story structure and time
    return sort_by_narrative_flow(highlights)


def sort_by_narrative_flow(highlights: List[Dict]) -> List[Dict]:
    """Sort highlights by intelligent narrative flow balancing story structure and chronology"""

    if not highlights:
        return highlights

    # Group by narrative role
    story_groups = {"hook": [], "development": [], "climax": [], "resolution": []}

    for highlight in highlights:
        role = highlight.get("narrative_role", "development")
        if role in story_groups:
            story_groups[role].append(highlight)
        else:
            story_groups["development"].append(highlight)

    # Sort each group chronologically
    for role in story_groups:
        story_groups[role].sort(key=lambda x: x["start_ms"])

    # Intelligent narrative ordering
    final_order = []

    # 1. Hook: Use earliest hook (natural opening)
    if story_groups["hook"]:
        final_order.extend(story_groups["hook"][:1])  # Only first hook

    # 2. Development: All development beats in chronological order
    final_order.extend(story_groups["development"])

    # 3. Climax: Use the most impactful climax (highest score)
    if story_groups["climax"]:
        # Select best climax by score, but maintain temporal context
        best_climax = max(story_groups["climax"], key=lambda x: x["score"])

        # Ensure climax comes after development contextually
        if final_order:
            last_dev_time = final_order[-1]["start_ms"]
            # If climax is too early, include some development context
            if best_climax["start_ms"] < last_dev_time:
                # Find development pieces that provide context for this climax
                context_dev = [
                    d
                    for d in story_groups["development"]
                    if d["start_ms"] < best_climax["start_ms"] and d not in final_order
                ]
                if context_dev:
                    # Add most relevant context
                    context_dev.sort(
                        key=lambda x: abs(x["start_ms"] - best_climax["start_ms"])
                    )
                    final_order.extend(context_dev[:1])

        final_order.append(best_climax)

    # 4. Resolution: Use latest resolution (natural ending)
    if story_groups["resolution"]:
        final_order.extend(story_groups["resolution"][-1:])  # Only last resolution

    # Final pass: Ensure no jarring time jumps
    final_order = smooth_temporal_transitions(final_order)

    return final_order


def smooth_temporal_transitions(highlights: List[Dict]) -> List[Dict]:
    """Smooth out jarring temporal transitions while maintaining story flow"""

    if len(highlights) <= 1:
        return highlights

    smoothed = [highlights[0]]  # Start with first clip

    for i in range(1, len(highlights)):
        current = highlights[i]
        previous = smoothed[-1]

        time_gap = abs(current["start_ms"] - previous["end_ms"])

        # If there's a large time jump (> 2 minutes), check if we need bridge content
        if time_gap > 120000:  # 2 minutes
            # Look for bridge content between these clips
            bridge_candidates = [
                h
                for h in highlights
                if h not in smoothed
                and h != current
                and previous["end_ms"] < h["start_ms"] < current["start_ms"]
            ]

            if bridge_candidates:
                # Add best bridge (highest score, closest to timeline)
                bridge = min(
                    bridge_candidates,
                    key=lambda x: abs(x["start_ms"] - previous["end_ms"]),
                )

                # Only add bridge if it doesn't break narrative flow
                if bridge.get("narrative_role", "development") in [
                    "development",
                    "hook",
                ]:
                    smoothed.append(bridge)

        smoothed.append(current)

    return smoothed


@priced("openai.gpt-4", Decimal("0.0001"))  # $0.01 per 100 tokens
def analyze_with_gpt(text: str, job_id: str = "default") -> Dict[str, Any]:
    """REAL GPT-4 analysis with cost tracking"""

    # Check budget before proceeding
    can_afford, remaining = check_budget(0.03)  # Estimate max cost
    if not can_afford:
        logger.warning(f"Budget insufficient for GPT analysis: ${remaining} remaining")
        return {"highlights": [], "cost": 0}

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
                    "content": """You are an expert video editor and storyteller specializing in creating viral social media content.
Your task is to identify moments that form a compelling narrative for vertical video format (Instagram Reels, TikTok, YouTube Shorts).
Focus on: hooks, emotional peaks, plot twists, revelations, and memorable soundbites that drive engagement.""",
                },
                {
                    "role": "user",
                    "content": f"""Analyze this transcript and extract 3-8 highlights that create a compelling story arc for a vertical video.

Transcript:
{text}

Create a narrative structure with:
1. Hook - Opening that grabs attention (first 3 seconds crucial)
2. Development - Building tension or interest
3. Climax - Peak moment or revelation
4. Resolution - Satisfying conclusion or call-to-action

Return JSON:
{{
  "story_theme": "One compelling sentence describing the narrative",
  "target_emotion": "primary emotion to evoke (curiosity/inspiration/surprise/empowerment)",
  "hook_text": "Exact opening line that will grab viewers",
  "highlights": [
    {{
      "title": "Punchy 5-8 word title",
      "reason": "Why this moment drives the story forward",
      "narrative_role": "hook/development/climax/resolution",
      "emotional_impact": "specific emotion this evokes",
      "importance": 1-10,
      "keywords": ["viral", "potential", "words"],
      "timestamp_hint": "approximate position in content"
    }}
  ],
  "viral_potential": 1-10,
  "suggested_music_mood": "upbeat/dramatic/inspirational/mysterious"
}}

Prioritize: surprising revelations, transformative moments, relatable struggles, actionable insights, emotional peaks.""",
                },
            ],
            max_tokens=1000,
            temperature=0.3,
        )

        # Parse response
        content = response.choices[0].message.content

        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        print("✅ GPT analysis complete")
        return result

    except Exception as e:
        print(f"❌ GPT analysis failed: {e}")
        return {"highlights": [], "cost": 0}


def merge_ai_with_local(
    local_segments: List[Dict], ai_results: List[Dict]
) -> List[Dict]:
    """Merge AI story analysis with local rule-based scoring"""

    # Handle new story-structured AI results
    story_data = {}
    ai_highlights = []

    # Extract highlights from AI results
    for result in ai_results:
        if isinstance(result, dict):
            # Check if it's our new story format
            if "story_theme" in result:
                story_data = result
                ai_highlights.extend(result.get("highlights", []))
            else:
                # Old format
                ai_highlights.extend(result.get("highlights", []))

    # Create narrative role priority
    narrative_priority = {"hook": 4, "development": 2, "climax": 3, "resolution": 1}

    # Create keyword boost map including narrative keywords
    ai_keywords = set()
    narrative_segments = []

    for highlight in ai_highlights:
        # Add keywords
        ai_keywords.update(highlight.get("keywords", []))

        # Try to match with local segments
        best_match = None
        best_score = 0

        for segment in local_segments:
            segment_text = segment["text"].lower()

            # Check keyword matches
            keyword_matches = sum(
                1 for kw in highlight.get("keywords", []) if kw.lower() in segment_text
            )

            # Check title similarity
            title_match = 0
            if highlight.get("title", "").lower() in segment_text:
                title_match = 3

            match_score = keyword_matches + title_match

            if match_score > best_score:
                best_score = match_score
                best_match = segment

        if best_match and best_score > 0:
            # Enhance the segment with AI insights
            enhanced = best_match.copy()
            enhanced["ai_title"] = highlight.get("title", "")
            enhanced["narrative_role"] = highlight.get("narrative_role", "development")
            enhanced["emotional_impact"] = highlight.get("emotional_impact", "")
            enhanced["ai_importance"] = highlight.get("importance", 5)

            # Apply narrative boost
            role_boost = narrative_priority.get(enhanced["narrative_role"], 1)
            enhanced["score"] += role_boost * 2
            enhanced["breakdown"]["narrative_boost"] = role_boost * 2
            enhanced["breakdown"]["ai_importance"] = enhanced["ai_importance"]

            narrative_segments.append(enhanced)

    # Also boost remaining segments by keyword matches
    for segment in local_segments:
        if not any(
            ns.get("start_ms") == segment.get("start_ms") for ns in narrative_segments
        ):
            segment_text = segment["text"].lower()

            # Count AI keyword matches
            ai_matches = sum(
                1 for keyword in ai_keywords if keyword.lower() in segment_text
            )

            # Boost score based on AI matches
            ai_boost = min(ai_matches * 1.5, 4.0)
            segment["score"] += ai_boost
            segment["breakdown"]["ai_boost"] = ai_boost

    # Combine narrative segments with remaining high-scoring segments
    all_segments = narrative_segments + [
        s
        for s in local_segments
        if not any(ns.get("start_ms") == s.get("start_ms") for ns in narrative_segments)
    ]

    # Sort by score but ensure hook comes first if present
    all_segments.sort(
        key=lambda x: (
            x.get("narrative_role") != "hook",  # Hook first
            -x["score"],  # Then by score
        )
    )

    # Add story metadata to all selected segments
    if story_data:
        for segment in all_segments[:8]:  # Top 8
            segment["story_theme"] = story_data.get("story_theme", "")
            segment["viral_potential"] = story_data.get("viral_potential", 5)
            segment["suggested_music"] = story_data.get("suggested_music_mood", "")

    return all_segments


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
    words: List[Word],
    audio_energy: List[float],
    mode: str = "smart",
    job_id: str = "default",
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
        # Premium mode: Story-driven AI analysis

        # Convert words to structured segments for story analysis
        transcript_segments = []
        current_segment = None
        segment_duration = 10.0  # 10-second segments for analysis

        for word in words:
            segment_start = int(word["start"] // segment_duration) * segment_duration

            if (
                current_segment is None
                or current_segment["start_time"] != segment_start
            ):
                # Start new segment
                if current_segment:
                    transcript_segments.append(current_segment)

                current_segment = {
                    "start_time": segment_start,
                    "start_ms": int(segment_start * 1000),
                    "end_ms": int((segment_start + segment_duration) * 1000),
                    "text": word["word"],
                    "speaker": getattr(word, "speaker", "Speaker_0"),
                    "words": [word],
                }
            else:
                # Add to current segment
                current_segment["text"] += " " + word["word"]
                current_segment["end_ms"] = int(word["end"] * 1000)
                current_segment["words"].append(word)

        # Add final segment
        if current_segment:
            transcript_segments.append(current_segment)

        # Add energy levels to segments
        for segment in transcript_segments:
            start_sec = segment["start_ms"] // 1000
            end_sec = segment["end_ms"] // 1000

            if start_sec < len(audio_energy) and end_sec <= len(audio_energy):
                segment_energy = (
                    audio_energy[start_sec:end_sec]
                    if end_sec > start_sec
                    else [audio_energy[start_sec]]
                )
                segment["energy_level"] = (
                    sum(segment_energy) / len(segment_energy) if segment_energy else 0.5
                )
            else:
                segment["energy_level"] = 0.5

        # Perform speaker analysis for continuity
        print("👥 Analyzing speakers and character continuity...")
        from .speaker_analysis import (
            SpeakerAnalyzer,
            enhance_highlights_with_speaker_continuity,
        )

        speaker_analyzer = SpeakerAnalyzer()
        speaker_analysis = speaker_analyzer.analyze_speakers(transcript_segments)

        print(f"   Found {len(speaker_analysis['speakers'])} speakers")
        print(f"   Narrative anchors: {speaker_analysis['narrative_anchors']}")
        print(f"   Continuity score: {speaker_analysis['continuity_score']:.2f}")

        # Perform emotion and energy analysis
        print("💝 Analyzing emotions and energy patterns...")
        from .emotion_analyzer import (
            EmotionAnalyzer,
            enhance_highlights_with_emotion_analysis,
        )

        emotion_analyzer = EmotionAnalyzer()
        emotion_analysis = emotion_analyzer.analyze_emotions_and_energy(
            transcript_segments, audio_energy
        )

        print(f"   Found {len(emotion_analysis['engagement_peaks'])} engagement peaks")
        print(f"   Emotional arcs detected: {len(emotion_analysis['emotional_arcs'])}")
        print(
            f"   Dominant emotion: {emotion_analysis['overall_emotional_profile']['dominant_emotion']}"
        )
        print(
            f"   Engagement level: {emotion_analysis['overall_emotional_profile']['engagement_level']:.2f}"
        )

        # Perform narrative beat detection
        print("🎭 Detecting narrative beats (hooks, climaxes, resolutions)...")
        from .narrative_detector import (
            NarrativeDetector,
            enhance_highlights_with_narrative_beats,
        )

        narrative_detector = NarrativeDetector()
        narrative_detection = narrative_detector.detect_narrative_beats(
            transcript_segments, emotion_analysis, speaker_analysis
        )

        print(f"   Hooks detected: {len(narrative_detection['hooks'])}")
        print(f"   Climaxes detected: {len(narrative_detection['climaxes'])}")
        print(f"   Resolutions detected: {len(narrative_detection['resolutions'])}")
        print(
            f"   Narrative coherence: {narrative_detection['narrative_coherence']:.2f}"
        )
        print(
            f"   Story completeness: {narrative_detection['story_arc_completeness']:.2f}"
        )

        # Perform story analysis
        print("🎬 Analyzing narrative structure with AI...")
        story_result = analyze_story_structure(transcript_segments, job_id)

        story_highlights = story_result.get("highlights", [])

        if story_highlights:
            # Story AI succeeded - use narrative-driven highlights
            print(
                f"✅ Story analysis complete: {len(story_highlights)} narrative beats identified"
            )

            # Enhance with speaker continuity analysis
            print("🔗 Enhancing highlights with speaker continuity...")
            story_highlights = enhance_highlights_with_speaker_continuity(
                story_highlights, speaker_analysis
            )

            # Enhance with emotion and energy analysis
            print("💫 Enhancing highlights with emotional intelligence...")
            story_highlights = enhance_highlights_with_emotion_analysis(
                story_highlights, emotion_analysis
            )

            # Enhance with narrative beat detection
            print("🎭 Enhancing highlights with narrative beats...")
            story_highlights = enhance_highlights_with_narrative_beats(
                story_highlights, narrative_detection
            )

            # Add story metadata to clips
            story_analysis = story_result.get("story_analysis", {})

            for highlight in story_highlights:
                highlight.update(
                    {
                        "story_theme": story_analysis.get("main_theme", ""),
                        "target_emotion": story_analysis.get("target_emotion", ""),
                        "narrative_type": story_analysis.get("narrative_type", ""),
                        "speaker_analysis": {
                            "total_speakers": len(speaker_analysis["speakers"]),
                            "narrative_anchors": speaker_analysis["narrative_anchors"],
                            "continuity_score": speaker_analysis["continuity_score"],
                        },
                        "emotion_analysis": {
                            "engagement_peaks": len(
                                emotion_analysis["engagement_peaks"]
                            ),
                            "emotional_arcs": len(emotion_analysis["emotional_arcs"]),
                            "dominant_emotion": emotion_analysis[
                                "overall_emotional_profile"
                            ]["dominant_emotion"],
                            "engagement_level": emotion_analysis[
                                "overall_emotional_profile"
                            ]["engagement_level"],
                        },
                        "narrative_detection": {
                            "hooks_detected": len(narrative_detection["hooks"]),
                            "climaxes_detected": len(narrative_detection["climaxes"]),
                            "resolutions_detected": len(
                                narrative_detection["resolutions"]
                            ),
                            "narrative_coherence": narrative_detection[
                                "narrative_coherence"
                            ],
                            "story_completeness": narrative_detection[
                                "story_arc_completeness"
                            ],
                        },
                        "cost": get_total_cost(),
                    }
                )

            # Re-sort after speaker enhancement (scores may have changed)
            story_highlights = sort_by_narrative_flow(story_highlights)

            total_cost = get_total_cost()
            print(
                f"✅ Premium mode: {len(story_highlights)} story-driven highlights with speaker continuity (cost: ${total_cost:.4f})"
            )
            return story_highlights

        else:
            # Story AI failed - fallback to enhanced local scoring
            print("⚠️  Story AI unavailable, using enhanced local scoring")

            # Apply enhanced rules
            top_segments = apply_rules(local_segments, mode, None)

            # Convert to clip format
            clips = []
            for i, segment in enumerate(top_segments):
                clips.append(
                    {
                        "slug": f"fallback-highlight-{i+1}",
                        "title": segment["text"][:40].strip(),
                        "start_ms": int(segment["start"] * 1000),
                        "end_ms": int(segment["end"] * 1000),
                        "score": segment["score"],
                        "method": "local_fallback",
                        "narrative_role": "development",  # Default role
                        "emotional_impact": "neutral",
                        "cost": get_total_cost(),
                    }
                )

            total_cost = get_total_cost()
            print(
                f"✅ Fallback mode: {len(clips)} highlights selected (cost: ${total_cost:.4f})"
            )
        return clips

    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_total_cost() -> float:
    """Get total API cost spent"""
    try:
        from ..core.cost import get_current_cost
    except ImportError:
        from core.cost import get_current_cost
    return get_current_cost()
