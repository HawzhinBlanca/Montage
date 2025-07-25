#!/usr/bin/env python3
"""
Clean highlight selector with ONLY real functionality
Removed all fake AI components: emotion_analyzer, narrative_detector, speaker_analysis
Kept only: real Gemini API integration + practical local scoring
"""
import logging
import re
from typing import List, Dict, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

# Import real working modules only
try:
    from .cost import priced
    from .analyze_video import final_text_cleanup
    from .api_wrappers import gemini_service, graceful_api_call
    from .rate_limiter import Priority

    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

    # Fallback for testing
    def priced(service: str, per_unit_usd: Decimal):
        def decorator(func):
            return func

        return decorator

    def final_text_cleanup(text: str) -> str:
        """Fallback deduplication for testing"""
        words = text.split()
        cleaned = []
        prev_word = None
        for word in words:
            clean_word = word.lower().strip(".,!?;:")
            prev_clean = prev_word.lower().strip(".,!?;:") if prev_word else None
            if not (prev_clean and clean_word == prev_clean):
                cleaned.append(word)
            prev_word = word
        return " ".join(cleaned)


# Real Gemini API integration (keep this - it actually works)
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini not available - using local scoring only")


def analyze_highlights(
    transcript_segments: List[Dict], audio_energy: List[float], job_id: str = "default"
) -> List[Dict]:
    """
    Analyze highlights using ONLY real working functionality
    Removed all fake AI components
    """
    logger.info("🎯 Starting REAL highlight analysis (no fake AI)")

    if not transcript_segments:
        logger.error("No transcript segments provided")
        return []

    # Calculate total video duration for audio energy indexing
    video_duration_ms = 0
    if transcript_segments:
        video_duration_ms = max(seg.get("end_ms", 0) for seg in transcript_segments)
    video_duration_seconds = video_duration_ms / 1000.0

    # Step 1: Real local scoring (this actually works)
    logger.info("📊 Performing real local content scoring...")
    local_highlights = local_rule_scoring(transcript_segments, audio_energy, job_id)

    if not local_highlights:
        logger.warning("No highlights found with local scoring")
        return []

    logger.info(f"✅ Local scoring found {len(local_highlights)} potential highlights")

    # Step 2: Real Gemini API analysis (if available)
    if GEMINI_AVAILABLE:
        try:
            logger.info("🧠 Enhancing with real Gemini AI analysis...")
            story_result = analyze_story_structure(transcript_segments, job_id)

            if story_result and story_result.get("story_beats"):
                # Convert story beats to highlights
                story_highlights = convert_story_beats_to_clips(
                    story_result,
                    transcript_segments,
                    audio_energy,
                    video_duration_seconds,
                )
                logger.info(
                    f"✅ Gemini generated {len(story_highlights)} story-based highlights"
                )

                # Combine local and AI results
                combined_highlights = combine_local_and_ai_highlights(
                    local_highlights, story_highlights
                )
                logger.info(
                    f"🔗 Combined into {len(combined_highlights)} final highlights"
                )

                return sort_by_narrative_flow(combined_highlights)
            else:
                logger.info(
                    "📊 Gemini analysis returned no results, using local scoring"
                )

        except Exception as e:
            logger.warning(
                f"Gemini analysis failed: {e}, falling back to local scoring"
            )

    # Step 3: Return local results if AI not available
    logger.info("📊 Using local scoring results only")
    return sort_by_timeline(local_highlights)


def local_rule_scoring(
    transcript_segments: List[Dict], audio_energy: List[float], job_id: str
) -> List[Dict]:
    """
    REAL local scoring based on practical criteria
    No fake AI - just working keyword and audio analysis
    """
    logger.info("⚙️ Running real local rule-based scoring...")

    # Calculate total video duration for proper audio energy indexing
    video_duration_ms = 0
    if transcript_segments:
        video_duration_ms = max(seg.get("end_ms", 0) for seg in transcript_segments)
    video_duration_seconds = video_duration_ms / 1000.0

    # Real highlight keywords (practical content indicators)
    highlight_keywords = [
        "important",
        "key",
        "crucial",
        "significant",
        "amazing",
        "incredible",
        "breakthrough",
        "discovery",
        "found",
        "revealed",
        "secret",
        "truth",
        "research",
        "study",
        "data",
        "evidence",
        "proof",
        "shows",
        "demonstrates",
        "question",
        "answer",
        "problem",
        "solution",
        "method",
        "technique",
        "example",
        "case",
        "story",
        "experience",
        "lesson",
        "insight",
    ]

    # Process transcript into segments
    words = []
    for segment in transcript_segments:
        text = segment.get("text", "")
        start_ms = segment.get("start_ms", 0)
        end_ms = segment.get("end_ms", 0)

        # Split into words with timing
        word_count = len(text.split())
        if word_count > 0:
            word_duration = (end_ms - start_ms) / word_count
            for i, word in enumerate(text.split()):
                word_start = start_ms + (i * word_duration)
                word_end = word_start + word_duration
                words.append(
                    {
                        "word": word,
                        "start": word_start / 1000,  # Convert to seconds
                        "end": word_end / 1000,
                    }
                )

    if not words:
        logger.warning("No words extracted from transcript")
        return []

    # Process words into highlight segments
    segments = []
    current_segment = []
    segment_start = 0

    for word in words:
        if not current_segment:
            segment_start = word["start"]

        current_segment.append(word)

        # End segment after 5-15 seconds or at sentence boundary
        segment_duration = word["end"] - segment_start
        is_sentence_end = word["word"].endswith((".", "!", "?"))

        if (segment_duration >= 5 and is_sentence_end) or segment_duration >= 15:
            # Score this segment using REAL criteria
            # Deduplicate words before joining
            words_list = []
            prev_word = None
            for w in current_segment:
                word_text = w["word"].strip()
                if prev_word is None or word_text.lower() != prev_word.lower():
                    words_list.append(word_text)
                prev_word = word_text
            segment_text = final_text_cleanup(" ".join(words_list))

            # Rule 1: Complete sentence bonus (practical criteria)
            sentence_score = 2.0 if is_sentence_end else 0.0

            # Rule 2: Keyword matching (real content analysis)
            keyword_count = sum(
                1
                for keyword in highlight_keywords
                if keyword.lower() in segment_text.lower()
            )
            keyword_score = min(keyword_count * 3.0, 9.0)  # Max 9 points

            # Rule 3: Audio energy (real signal processing) - FIXED INDEXING
            segment_mid_time = (segment_start + word["end"]) / 2

            # Initialize default values
            energy_score = 1.0
            actual_energy = 0.5

            if audio_energy and video_duration_seconds > 0:
                # Properly map time to audio sample index using actual video duration
                # audio_energy samples are spread across the entire video duration
                energy_index = min(
                    int(
                        (segment_mid_time / video_duration_seconds) * len(audio_energy)
                    ),
                    len(audio_energy) - 1,
                )

                energy_score = audio_energy[energy_index] * 2.0
                actual_energy = audio_energy[energy_index]

            # Rule 4: Duration bonus (practical timing)
            duration_score = min(
                segment_duration / 10.0, 2.0
            )  # Prefer 10+ second segments

            # Total score
            total_score = sentence_score + keyword_score + energy_score + duration_score

            # Only keep segments with reasonable scores
            if total_score >= 4.0:  # Minimum threshold
                highlight = {
                    "start_ms": int(segment_start * 1000),
                    "end_ms": int(word["end"] * 1000),
                    "text": segment_text,
                    "title": _generate_title(segment_text),
                    "score": total_score,
                    "sentence_complete": is_sentence_end,
                    "keyword_count": keyword_count,
                    "duration": segment_duration,
                    "audio_energy": actual_energy,
                }
                segments.append(highlight)
                logger.debug(
                    f"Segment scored {total_score:.1f}: {segment_text[:50]}..."
                )

            # Reset for next segment
            current_segment = []

    # Sort by score and return top segments
    segments.sort(key=lambda x: x["score"], reverse=True)
    top_segments = segments[:8]  # Top 8 segments

    logger.info(f"✅ Real local scoring generated {len(top_segments)} highlights")
    return top_segments


def _generate_title(text: str) -> str:
    """Generate a simple title from text (max 40 characters)"""
    # Take first few words, clean up duplicates
    words = text.strip().split()[:6]
    title = final_text_cleanup(" ".join(words))

    # Truncate if too long
    if len(title) > 37:
        title = title[:37] + "..."

    return title


@priced("gemini.2-5-flash", Decimal("0.00001"))
def analyze_story_structure(
    transcript_segments: List[Dict], job_id: str = "default"
) -> Dict[str, Any]:
    """
    Rate-limited Gemini API integration for story analysis
    Uses rate limiting and graceful degradation
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available")
        return {}

    # Use rate-limited service if available
    if RATE_LIMITING_AVAILABLE:
        try:
            return graceful_api_call(
                service_func=gemini_service.analyze_story_structure,
                fallback_func=_analyze_story_structure_direct,
                service_name="gemini",
                transcript_segments=transcript_segments,
                job_id=job_id,
            )
        except Exception as e:
            logger.warning(f"Rate-limited Gemini failed, using direct: {e}")
            # Fall through to direct call

    # Fallback to direct API call
    return _analyze_story_structure_direct(transcript_segments, job_id)


def _analyze_story_structure_direct(
    transcript_segments: List[Dict], job_id: str = "default"
) -> Dict[str, Any]:
    """
    Direct Gemini API call for story analysis (fallback)
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available")
        return {}

    try:
        from ..utils.secret_loader import get as get_secret

        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")

        if not api_key:
            logger.error("No Gemini API key available")
            return {}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        # Prepare transcript text
        transcript_text = "\n".join(
            [
                f"{seg.get('start_ms', 0)/1000:.1f}s: {seg.get('text', '')}"
                for seg in transcript_segments[:20]  # Limit for API
            ]
        )

        # Real prompt for actual story analysis
        prompt = f"""Analyze this video transcript and identify 3-5 story beats for highlights.

Transcript:
{transcript_text}

Return JSON with this structure:
{{
  "story_beats": [
    {{
      "beat_type": "hook|development|climax|resolution",
      "purpose": "brief description",
      "target_segment_ids": [list of segment indices],
      "narrative_function": "what this accomplishes",
      "emotional_trigger": "target emotion"
    }}
  ]
}}

Focus on moments with:
- Clear narrative purpose (introductions, revelations, conclusions)
- Emotional engagement (excitement, curiosity, surprise)
- Information density (key facts, insights, examples)
- Natural speech boundaries

Respond only with valid JSON."""

        response = model.generate_content(prompt)

        if response and response.text:
            # Try to parse JSON response
            try:
                import json

                # Clean response text
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                result = json.loads(response_text)
                logger.info(
                    f"✅ Gemini story analysis complete: {len(result.get('story_beats', []))} narrative beats identified"
                )
                return result

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                return {}
        else:
            logger.error("Empty response from Gemini")
            return {}

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {}


def convert_story_beats_to_clips(
    story_result: Dict,
    transcript_segments: List[Dict],
    audio_energy: List[float],
    video_duration_seconds: float,
) -> List[Dict]:
    """Convert AI story beats into video clips with timestamps"""
    highlights = []
    story_beats = story_result.get("story_beats", [])

    for beat in story_beats:
        target_segment_ids = beat.get("target_segment_ids", [])

        if not target_segment_ids:
            continue

        # Get segments for this story beat
        beat_segments = []
        for seg_id in target_segment_ids:
            if 0 <= seg_id < len(transcript_segments):
                beat_segments.append(transcript_segments[seg_id])

        if not beat_segments:
            continue

        # Create clip spanning all segments in this beat
        start_ms = min(seg.get("start_ms", 0) for seg in beat_segments)
        end_ms = max(seg.get("end_ms", 0) for seg in beat_segments)

        # Ensure reasonable clip length (8-25 seconds)
        duration_ms = end_ms - start_ms
        if duration_ms < 8000:  # Less than 8 seconds, extend
            end_ms = start_ms + 10000
        elif duration_ms > 25000:  # More than 25 seconds, trim
            end_ms = start_ms + 25000

        # Combine text from all segments with final deduplication
        all_text_parts = []
        for seg in beat_segments:
            all_text_parts.append(seg.get("text", ""))

        combined_text = final_text_cleanup(" ".join(all_text_parts))

        # Calculate audio energy for this highlight
        segment_mid_time = (start_ms + end_ms) / 2000.0  # Convert to seconds
        actual_energy = 0.5  # Default

        if audio_energy and video_duration_seconds > 0:
            energy_index = min(
                int((segment_mid_time / video_duration_seconds) * len(audio_energy)),
                len(audio_energy) - 1,
            )
            actual_energy = audio_energy[energy_index]

        highlight = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": combined_text,
            "title": _generate_title(combined_text),
            "score": 10.0,  # High score for AI-selected content
            "beat_type": beat.get("beat_type", "unknown"),
            "purpose": beat.get("purpose", ""),
            "narrative_function": beat.get("narrative_function", ""),
            "source": "gemini_ai",
            "audio_energy": actual_energy,
        }

        highlights.append(highlight)

    return highlights


def combine_local_and_ai_highlights(
    local_highlights: List[Dict], ai_highlights: List[Dict]
) -> List[Dict]:
    """Combine local and AI highlights, preferring AI when available"""

    # Start with AI highlights (they're more intelligent)
    combined = ai_highlights.copy()

    # Add local highlights that don't overlap with AI highlights
    for local_highlight in local_highlights:
        overlaps = False

        for ai_highlight in ai_highlights:
            if _segments_overlap(local_highlight, ai_highlight):
                overlaps = True
                break

        if not overlaps:
            # Boost local highlights that complement AI
            local_highlight["score"] = local_highlight.get("score", 0) + 2.0
            combined.append(local_highlight)

    # Sort by score and limit to reasonable number
    combined.sort(key=lambda x: x.get("score", 0), reverse=True)
    return combined[:6]  # Top 6 highlights


def _segments_overlap(seg1: Dict, seg2: Dict) -> bool:
    """Check if two segments overlap in time"""
    return not (
        seg1["end_ms"] <= seg2["start_ms"] or seg2["end_ms"] <= seg1["start_ms"]
    )


def sort_by_narrative_flow(highlights: List[Dict]) -> List[Dict]:
    """Sort highlights by narrative flow (chronological with story logic)"""

    # Group by beat type for intelligent ordering
    hooks = [h for h in highlights if h.get("beat_type") == "hook"]
    developments = [h for h in highlights if h.get("beat_type") == "development"]
    climaxes = [h for h in highlights if h.get("beat_type") == "climax"]
    resolutions = [h for h in highlights if h.get("beat_type") == "resolution"]
    others = [
        h
        for h in highlights
        if h.get("beat_type") not in ["hook", "development", "climax", "resolution"]
    ]

    # Sort each group chronologically
    for group in [hooks, developments, climaxes, resolutions, others]:
        group.sort(key=lambda x: x.get("start_ms", 0))

    # Combine in narrative order: hooks first, then developments, climaxes, resolutions
    narrative_order = hooks + developments + climaxes + resolutions + others

    # If no story beats detected, fall back to chronological
    if not any([hooks, developments, climaxes, resolutions]):
        narrative_order = sorted(highlights, key=lambda x: x.get("start_ms", 0))

    return narrative_order


def sort_by_timeline(highlights: List[Dict]) -> List[Dict]:
    """Simple chronological sort"""
    return sorted(highlights, key=lambda x: x.get("start_ms", 0))


# Main entry point
def select_highlights(
    transcript_segments: List[Dict],
    audio_energy: List[float] = None,
    job_id: str = "default",
) -> List[Dict]:
    """
    Main highlight selection using only REAL functionality
    """
    if audio_energy is None:
        audio_energy = [0.5] * 100  # Default audio energy

    return analyze_highlights(transcript_segments, audio_energy, job_id)
