#!/usr/bin/env python3
"""
Clean highlight selector with ONLY real functionality
Removed all fake AI components: emotion_analyzer, narrative_detector, speaker_analysis
Kept only: real Gemini API integration + practical local scoring
"""
import logging
from decimal import Decimal
from typing import Any, Dict, List

from ..settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Import real working modules only
try:
    # T4: Import Gemma scene ranker
    from ..providers.gemma_scene_ranker import rank_scenes
    from .analyze_video import final_text_cleanup
    from .api_wrappers import gemini_service, graceful_api_call
    from .cost import priced
    from .rate_limiter import Priority

    RATE_LIMITING_AVAILABLE = True
    GEMMA_AVAILABLE = True
except ImportError as e:
    RATE_LIMITING_AVAILABLE = False
    GEMMA_AVAILABLE = False
    logger.warning(f"Import failed: {e}")

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

    # Step 2: Local Ollama/Gemma3 analysis (preferred) or Gemini API fallback
    try:
        # First try local Ollama/Gemma3 for maximum speed
        from ..providers.gemma_scene_ranker import rank_scenes, OLLAMA_AVAILABLE
        
        if OLLAMA_AVAILABLE and settings.features.prefer_local_models:
            logger.info("🚀 Using local Gemma3 for ultra-fast AI analysis...")
            gemma_rankings = rank_scenes(local_highlights)
            
            # Apply Gemma3 rankings to highlights
            for ranking in gemma_rankings:
                highlight_id = ranking.get("id", 0)
                if highlight_id < len(local_highlights):
                    # Boost score based on Gemma3 importance rating
                    importance = ranking.get("importance", 0.5)
                    local_highlights[highlight_id]["score"] *= (1 + importance)
                    local_highlights[highlight_id]["ai_reason"] = ranking.get("reason", "AI selected")
            
            logger.info(f"✅ Local Gemma3 enhanced {len(local_highlights)} highlights")
            # Sort by enhanced scores and return - skip external API calls
            final_highlights = sorted(local_highlights, key=lambda x: x.get("score", 0), reverse=True)
            return final_highlights[:8]  # Return top 8 highlights
            
        elif GEMINI_AVAILABLE:
            try:
                logger.info("🧠 Enhancing with external Gemini API analysis...")
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

                    # T4: Rank scenes with Gemma if available
                    if GEMMA_AVAILABLE:
                        try:
                            logger.info("🎯 Ranking scenes with Gemma...")
                            scenes_for_ranking = [
                                {"id": i, "text": h.get("text", ""), "score": h.get("score", 0)}
                                for i, h in enumerate(combined_highlights)
                            ]
                            ranked_scenes = rank_scenes(scenes_for_ranking)

                            # Reorder highlights based on Gemma ranking
                            reordered_highlights = []
                            for ranked in ranked_scenes:
                                idx = ranked["id"]
                                if 0 <= idx < len(combined_highlights):
                                    highlight = combined_highlights[idx].copy()
                                    highlight["gemma_importance"] = ranked.get("importance", 0)
                                    reordered_highlights.append(highlight)

                            combined_highlights = reordered_highlights
                            logger.info(f"✅ Gemma ranked {len(combined_highlights)} highlights")
                        except Exception as e:
                            logger.warning(f"Gemma ranking failed: {e}, using original order")
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
    except Exception as e:
        logger.warning(f"AI analysis failed: {e}, using local scoring only")

    # Step 3: Return local results if AI not available
    logger.info("📊 Using local scoring results only")
    return sort_by_timeline(local_highlights)


def analyze_highlights_premium(
    transcript_segments: List[Dict], audio_energy: List[float], job_id: str = "default"
) -> List[Dict]:
    """
    Premium highlight analysis with enhanced AI features and story beats
    """
    logger.info("🌟 Starting PREMIUM highlight analysis with story beats")

    if not transcript_segments:
        logger.error("No transcript segments provided")
        return []

    # Step 1: Enhanced local scoring for premium
    logger.info("📊 Running enhanced local scoring for premium mode...")
    local_highlights = local_rule_scoring(transcript_segments, audio_energy, job_id)

    if not local_highlights:
        logger.warning("No highlights found with enhanced local scoring")
        return []

    # Step 2: Story beats analysis (premium feature)
    story_highlights = []
    try:
        from .beats import analyze_story_beats
        logger.info("📚 Running story beats analysis...")
        story_highlights = analyze_story_beats(transcript_segments, job_id, use_ai=True)
        logger.info(f"✅ Story beats generated {len(story_highlights)} narrative highlights")
    except Exception as e:
        logger.warning(f"Story beats analysis failed: {e}")

    # Step 3: Real Gemini API analysis (premium feature)
    gemini_highlights = []
    if GEMINI_AVAILABLE:
        try:
            logger.info("🧠 Running premium Gemini AI analysis...")
            story_result = analyze_story_structure(transcript_segments, job_id)

            if story_result and story_result.get("story_beats"):
                video_duration_ms = max(seg.get("end_ms", 0) for seg in transcript_segments)
                video_duration_seconds = video_duration_ms / 1000.0
                
                gemini_highlights = convert_story_beats_to_clips(
                    story_result,
                    transcript_segments,
                    audio_energy,
                    video_duration_seconds,
                )
                logger.info(f"✅ Gemini generated {len(gemini_highlights)} AI highlights")
        except Exception as e:
            logger.warning(f"Gemini premium analysis failed: {e}")

    # Step 4: Combine all highlight sources for premium
    all_highlights = local_highlights + story_highlights + gemini_highlights
    
    if not all_highlights:
        logger.warning("No highlights generated from any premium source")
        return []

    # Step 5: Premium merging and ranking
    from .highlight_merger import merge_highlights
    
    premium_highlights = merge_highlights(
        all_highlights,
        strategy="premium_quality",  # Use premium merging strategy
        max_highlights=8  # Allow more highlights in premium mode
    )

    logger.info(f"🌟 Premium analysis complete: {len(premium_highlights)} final highlights")
    return premium_highlights


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

    # Using content-based analysis (no primitive keyword matching)
    logger.info("Using intelligent content analysis (speech patterns, information density, structure)...")

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

        # End segment after 3-15 seconds or at sentence boundary (lowered for shorter videos)
        segment_duration = word["end"] - segment_start
        is_sentence_end = word["word"].endswith((".", "!", "?"))

        if (segment_duration >= 3 and is_sentence_end) or segment_duration >= 15:
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

            # Rule 2: Content quality analysis (no primitive keyword matching)
            # Score based on speech patterns, completeness, and information density
            word_count = len(segment_text.split())
            content_score = 0.0
            
            # Complete thoughts bonus (10-50 words is optimal for highlights)
            if 10 <= word_count <= 50:
                content_score += 4.0
            elif 5 <= word_count <= 60:
                content_score += 2.0
                
            # Sentence structure bonus (proper grammar indicates good content)
            if any(segment_text.count(punct) >= 1 for punct in ['.', '!', '?']):
                content_score += 2.0
                
            # Information density (longer words often indicate more informative content)
            avg_word_length = sum(len(word) for word in segment_text.split()) / max(word_count, 1)
            if avg_word_length >= 5:  # Technical/informative content
                content_score += 3.0
            elif avg_word_length >= 4:
                content_score += 1.0

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
            total_score = sentence_score + content_score + energy_score + duration_score

            # Only keep segments with reasonable scores
            # Lower threshold for shorter videos
            if total_score >= 4.0:  # Require meaningful content
                highlight = {
                    "start_ms": int(segment_start * 1000),
                    "end_ms": int(word["end"] * 1000),
                    "text": segment_text,
                    "title": _generate_title(segment_text),
                    "score": total_score,
                    "sentence_complete": is_sentence_end,
                    "content_quality": content_score,
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
    """Combine local and AI highlights using P1-04 ROVER merge algorithm"""
    from .highlight_merger import merge_highlights

    # Tag highlights with source information
    for highlight in local_highlights:
        highlight["source"] = "local"
    for highlight in ai_highlights:
        highlight["source"] = "ai"

    # Combine all highlights
    all_highlights = local_highlights + ai_highlights

    if not all_highlights:
        return []

    # P1-04: Use ROVER O(n log n) merge algorithm
    logger.info(f"Using ROVER merge: {len(local_highlights)} local + {len(ai_highlights)} AI highlights")

    # Apply intelligent merging with balanced strategy
    merged_highlights = merge_highlights(
        all_highlights,
        strategy="balanced",
        max_highlights=6
    )

    logger.info(f"ROVER merge result: {len(merged_highlights)} final highlights")
    return merged_highlights


def _segments_overlap(seg1: Dict, seg2: Dict) -> bool:
    """Check if two segments overlap in time"""
    return not (
        seg1["end_ms"] <= seg2["start_ms"] or seg2["end_ms"] <= seg1["start_ms"]
    )


def sort_by_narrative_flow(highlights: List[Dict]) -> List[Dict]:
    """Sort highlights by narrative flow using P1-04 ROVER algorithm"""
    from .highlight_merger import merge_highlights

    # P1-04: Use ROVER merge algorithm with narrative flow strategy
    return merge_highlights(
        highlights,
        strategy="narrative_flow",
        max_highlights=len(highlights)  # Don't limit, just sort
    )


def sort_by_timeline(highlights: List[Dict]) -> List[Dict]:
    """Simple chronological sort using P1-04 ROVER algorithm"""
    from .highlight_merger import merge_highlights

    # P1-04: Use ROVER merge algorithm with timeline-first strategy
    return merge_highlights(
        highlights,
        strategy="timeline_first",
        max_highlights=len(highlights)  # Don't limit, just sort
    )


# Main entry point
def select_highlights(
    transcript_segments: List[Dict],
    audio_energy: List[float] = None,
    mode: str = "smart",
    job_id: str = "default",
) -> List[Dict]:
    """
    Main highlight selection using only REAL functionality
    """
    if audio_energy is None:
        audio_energy = [0.5] * 100  # Default audio energy

    # Use mode to determine processing intensity
    if mode == "premium":
        # Premium mode: Use full AI analysis with story beats
        logger.info("Using PREMIUM mode: Full AI analysis with story beats")
        return analyze_highlights_premium(transcript_segments, audio_energy, job_id)
    else:
        # Smart mode: Standard AI analysis  
        logger.info("Using SMART mode: Standard AI analysis")
        return analyze_highlights(transcript_segments, audio_energy, job_id)
