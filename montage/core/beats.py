#!/usr/bin/env python3
"""
Story Beats Analysis Module - Tasks.md Step 2B
Narrative structure detection and story flow optimization for video highlights

This module provides:
1. Story beat detection using AI analysis
2. Narrative flow optimization
3. Integration with A/B testing pipeline
4. Metrics tracking for story beat effectiveness
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Import AI services for story analysis
try:
    from .api_wrappers import anthropic_service, gemini_service, graceful_api_call
    from .rate_limiter import Priority
    AI_SERVICES_AVAILABLE = True
except ImportError:
    AI_SERVICES_AVAILABLE = False
    logger.warning("AI services not available - using fallback story analysis")

# Import metrics for tracking
try:
    from .metrics import metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class BeatType(Enum):
    """Narrative beat types for story structure"""
    HOOK = "hook"           # Opening attention grabber
    SETUP = "setup"         # Context and background
    INCITING_INCIDENT = "inciting_incident"  # Problem/opportunity introduction
    DEVELOPMENT = "development"    # Main content development
    CLIMAX = "climax"       # Peak intensity/revelation
    RESOLUTION = "resolution"      # Conclusion/takeaway
    UNKNOWN = "unknown"     # Unclassified


@dataclass
class StoryBeat:
    """Represents a narrative beat in the video"""
    beat_type: BeatType
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    narrative_function: str
    segment_indices: List[int]

    @property
    def duration_sec(self) -> float:
        return (self.end_ms - self.start_ms) / 1000.0

    def to_highlight_dict(self) -> Dict[str, Any]:
        """Convert to highlight format for video processing"""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "title": self._generate_title(),
            "score": self.confidence * 10,  # Scale to highlight score range
            "beat_type": self.beat_type.value,
            "narrative_function": self.narrative_function,
            "duration": self.duration_sec,
            "is_story_beat": True
        }

    def _generate_title(self) -> str:
        """Generate title based on beat type and content"""
        type_prefixes = {
            BeatType.HOOK: "Opening",
            BeatType.SETUP: "Context",
            BeatType.INCITING_INCIDENT: "Problem",
            BeatType.DEVELOPMENT: "Main Point",
            BeatType.CLIMAX: "Key Insight",
            BeatType.RESOLUTION: "Takeaway"
        }

        prefix = type_prefixes.get(self.beat_type, "Segment")
        content_words = self.text.split()[:4]  # First 4 words
        title = f"{prefix}: {' '.join(content_words)}"

        return title[:40] + "..." if len(title) > 40 else title


class StoryBeatsAnalyzer:
    """Main story beats analysis engine"""

    def __init__(self):
        self.ai_available = AI_SERVICES_AVAILABLE

    def analyze_story_beats(
        self,
        transcript_segments: List[Dict],
        job_id: str = "default",
        use_ai: bool = True
    ) -> List[StoryBeat]:
        """
        Analyze transcript segments to identify story beats
        Tasks.md Step 2B: Main story beats detection function
        
        Args:
            transcript_segments: List of transcript segments with text and timing
            job_id: Job identifier for tracking
            use_ai: Whether to use AI analysis (for A/B testing)
        
        Returns:
            List of StoryBeat objects ordered by narrative importance
        """
        logger.info(f"Analyzing story beats for job {job_id}, AI enabled: {use_ai}")
        start_time = time.time()

        if use_ai and self.ai_available:
            story_beats = self._ai_story_analysis(transcript_segments, job_id)
        else:
            story_beats = self._rule_based_story_analysis(transcript_segments, job_id)

        # Track metrics
        analysis_time = time.time() - start_time
        if METRICS_AVAILABLE:
            metrics.processing_duration.labels(stage="story_beats").observe(analysis_time)
            metrics.segments_detected.observe(len(story_beats))

        logger.info(f"Story beats analysis completed: {len(story_beats)} beats in {analysis_time:.2f}s")
        return story_beats

    def _ai_story_analysis(self, transcript_segments: List[Dict], job_id: str) -> List[StoryBeat]:
        """AI-powered story beats analysis using Gemini/Anthropic"""
        logger.info("Running AI-powered story beats analysis")

        try:
            # Prepare transcript for AI analysis
            transcript_text = self._prepare_transcript_for_ai(transcript_segments)

            # Use Anthropic Claude for story analysis (more cost-effective than Gemini for this task)
            story_result = self._analyze_with_claude(transcript_text, job_id)

            if not story_result or not story_result.get("story_beats"):
                logger.warning("AI analysis returned no story beats, falling back to rule-based")
                return self._rule_based_story_analysis(transcript_segments, job_id)

            # Convert AI results to StoryBeat objects
            story_beats = self._convert_ai_results_to_beats(
                story_result, transcript_segments
            )

            logger.info(f"AI analysis identified {len(story_beats)} story beats")
            return story_beats

        except Exception as e:
            logger.error(f"AI story analysis failed: {e}")
            return self._rule_based_story_analysis(transcript_segments, job_id)

    def _rule_based_story_analysis(self, transcript_segments: List[Dict], job_id: str) -> List[StoryBeat]:
        """Rule-based story beats detection as fallback"""
        logger.info("Running rule-based story beats analysis")

        story_beats = []
        total_duration_ms = max(seg.get("end_ms", 0) for seg in transcript_segments) if transcript_segments else 0

        # Rule-based beat detection using content patterns and positioning
        for i, segment in enumerate(transcript_segments):
            text = segment.get("text", "").lower()
            start_ms = segment.get("start_ms", 0)
            end_ms = segment.get("end_ms", 0)
            position_ratio = start_ms / total_duration_ms if total_duration_ms > 0 else 0

            beat_type = BeatType.UNKNOWN
            confidence = 0.3  # Default confidence for rule-based
            narrative_function = "content segment"

            # Hook detection (first 20% of video)
            if position_ratio < 0.2:
                if any(word in text for word in ["today", "let me", "imagine", "what if", "question"]):
                    beat_type = BeatType.HOOK
                    confidence = 0.7
                    narrative_function = "opening hook to capture attention"

            # Setup/Context (20%-40%)
            elif 0.2 <= position_ratio < 0.4:
                if any(word in text for word in ["background", "context", "first", "originally", "before"]):
                    beat_type = BeatType.SETUP
                    confidence = 0.6
                    narrative_function = "provide context and setup"

            # Development (40%-70%)
            elif 0.4 <= position_ratio < 0.7:
                if any(word in text for word in ["important", "key", "main", "significant", "because"]):
                    beat_type = BeatType.DEVELOPMENT
                    confidence = 0.6
                    narrative_function = "develop main points"

            # Climax (70%-85%)
            elif 0.7 <= position_ratio < 0.85:
                if any(word in text for word in ["discovered", "found", "realized", "breakthrough", "crucial"]):
                    beat_type = BeatType.CLIMAX
                    confidence = 0.8
                    narrative_function = "key insight or revelation"

            # Resolution (85%-100%)
            elif position_ratio >= 0.85:
                if any(word in text for word in ["conclusion", "summary", "takeaway", "remember", "final"]):
                    beat_type = BeatType.RESOLUTION
                    confidence = 0.7
                    narrative_function = "conclusion and takeaway"

            # Only create beats for segments with reasonable confidence
            if confidence > 0.4:
                story_beat = StoryBeat(
                    beat_type=beat_type,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=segment.get("text", ""),
                    confidence=confidence,
                    narrative_function=narrative_function,
                    segment_indices=[i]
                )
                story_beats.append(story_beat)

        # Ensure we have a reasonable distribution of beats
        story_beats = self._balance_beat_selection(story_beats)

        logger.info(f"Rule-based analysis identified {len(story_beats)} story beats")
        return story_beats

    def _prepare_transcript_for_ai(self, transcript_segments: List[Dict]) -> str:
        """Prepare transcript text for AI analysis with segment markers"""
        segments_text = []

        for i, segment in enumerate(transcript_segments[:30]):  # Limit for API costs
            text = segment.get("text", "").strip()
            start_sec = segment.get("start_ms", 0) / 1000

            segments_text.append(f"[{i:02d}|{start_sec:.1f}s] {text}")

        return "\n".join(segments_text)

    def _analyze_with_claude(self, transcript_text: str, job_id: str) -> Dict[str, Any]:
        """Use Claude API for story structure analysis"""
        prompt = f"""Analyze this video transcript and identify 3-6 key story beats that would make compelling highlights.

Transcript with segment IDs and timestamps:
{transcript_text}

Return JSON with this exact structure:
{{
  "story_beats": [
    {{
      "beat_type": "hook|setup|inciting_incident|development|climax|resolution",
      "purpose": "brief description of narrative function",
      "target_segment_ids": [list of segment indices from transcript],
      "confidence": 0.8,
      "narrative_function": "what this accomplishes in the story"
    }}
  ]
}}

Focus on:
1. Natural story progression (hook → development → climax → resolution)
2. High-impact moments that would engage viewers
3. Complete thoughts and clear segment boundaries
4. Segments between 10-30 seconds long

Return only JSON, no other text."""

        try:
            if AI_SERVICES_AVAILABLE:
                result = graceful_api_call(
                    anthropic_service.generate_highlights,
                    service_name="anthropic",
                    transcript_data={"transcript": transcript_text},
                    job_id=job_id
                )
                return result
            else:
                return {}
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return {}

    def _convert_ai_results_to_beats(
        self, story_result: Dict, transcript_segments: List[Dict]
    ) -> List[StoryBeat]:
        """Convert AI analysis results to StoryBeat objects"""
        story_beats = []

        for beat_data in story_result.get("story_beats", []):
            try:
                # Get beat type
                beat_type_str = beat_data.get("beat_type", "unknown").lower()
                beat_type = BeatType.UNKNOWN

                for bt in BeatType:
                    if bt.value == beat_type_str:
                        beat_type = bt
                        break

                # Get target segments
                target_segment_ids = beat_data.get("target_segment_ids", [])
                if not target_segment_ids:
                    continue

                # Calculate timing from target segments
                target_segments = [
                    transcript_segments[i] for i in target_segment_ids
                    if 0 <= i < len(transcript_segments)
                ]

                if not target_segments:
                    continue

                start_ms = min(seg.get("start_ms", 0) for seg in target_segments)
                end_ms = max(seg.get("end_ms", 0) for seg in target_segments)
                text = " ".join(seg.get("text", "") for seg in target_segments)

                story_beat = StoryBeat(
                    beat_type=beat_type,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=beat_data.get("confidence", 0.7),
                    narrative_function=beat_data.get("narrative_function", "story segment"),
                    segment_indices=target_segment_ids
                )

                story_beats.append(story_beat)

            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Failed to process story beat: {e}")
                continue

        return story_beats

    def _balance_beat_selection(self, story_beats: List[StoryBeat]) -> List[StoryBeat]:
        """Ensure balanced selection of beat types"""
        if len(story_beats) <= 6:
            return story_beats

        # Group by beat type
        beats_by_type = {}
        for beat in story_beats:
            if beat.beat_type not in beats_by_type:
                beats_by_type[beat.beat_type] = []
            beats_by_type[beat.beat_type].append(beat)

        # Select top beats from each type
        balanced_beats = []
        for beat_type in BeatType:
            if beat_type in beats_by_type:
                # Sort by confidence and take top 1-2
                type_beats = sorted(beats_by_type[beat_type], key=lambda x: x.confidence, reverse=True)
                balanced_beats.extend(type_beats[:2])

        # Sort by chronological order
        balanced_beats.sort(key=lambda x: x.start_ms)

        return balanced_beats[:6]  # Max 6 beats

    def optimize_narrative_flow(self, highlights: List[Dict]) -> List[Dict]:
        """
        Optimize highlight order for narrative flow
        Tasks.md Step 2B: Story beat integration with highlight pipeline
        """
        logger.info("Optimizing highlights for narrative flow")

        # Separate story beats from regular highlights
        story_beats = [h for h in highlights if h.get("is_story_beat", False)]
        regular_highlights = [h for h in highlights if not h.get("is_story_beat", False)]

        if not story_beats:
            # No story beats, return chronological order
            return sorted(highlights, key=lambda x: x.get("start_ms", 0))

        # Sort story beats by narrative order
        beat_order = ["hook", "setup", "inciting_incident", "development", "climax", "resolution"]

        def beat_sort_key(highlight):
            beat_type = highlight.get("beat_type", "unknown")
            try:
                return beat_order.index(beat_type)
            except ValueError:
                return len(beat_order)  # Put unknown types at end

        sorted_beats = sorted(story_beats, key=beat_sort_key)

        # Interweave with regular highlights in chronological gaps
        final_highlights = []
        last_beat_end = 0

        for beat in sorted_beats:
            # Add regular highlights that occur before this beat
            gap_highlights = [
                h for h in regular_highlights
                if last_beat_end <= h.get("start_ms", 0) < beat.get("start_ms", 0)
            ]
            gap_highlights.sort(key=lambda x: x.get("start_ms", 0))
            final_highlights.extend(gap_highlights[:2])  # Max 2 per gap

            # Add the story beat
            final_highlights.append(beat)
            last_beat_end = beat.get("end_ms", 0)

        # Add any remaining regular highlights
        remaining_highlights = [
            h for h in regular_highlights
            if h.get("start_ms", 0) >= last_beat_end
        ]
        remaining_highlights.sort(key=lambda x: x.get("start_ms", 0))
        final_highlights.extend(remaining_highlights[:2])

        logger.info(f"Narrative flow optimization: {len(story_beats)} beats + {len(regular_highlights)} regular → {len(final_highlights)} final")

        return final_highlights


# Global instance for easy access
story_beats_analyzer = StoryBeatsAnalyzer()


def analyze_story_beats(
    transcript_segments: List[Dict],
    job_id: str = "default",
    use_ai: bool = True
) -> List[Dict]:
    """
    Main entry point for story beats analysis - Tasks.md Step 2B
    
    Args:
        transcript_segments: Video transcript segments
        job_id: Job identifier
        use_ai: Whether to use AI analysis (for A/B testing)
        
    Returns:
        List of highlight dictionaries with story beat information
    """
    analyzer = story_beats_analyzer
    story_beats = analyzer.analyze_story_beats(transcript_segments, job_id, use_ai)

    # Convert to highlight format
    highlights = [beat.to_highlight_dict() for beat in story_beats]

    # Track story beats metrics
    if METRICS_AVAILABLE:
        try:
            for beat in story_beats:
                metrics.highlight_scores.observe(beat.confidence)

            # Track beat type distribution
            beat_types = [beat.beat_type.value for beat in story_beats]
            beat_type_counts = {bt: beat_types.count(bt) for bt in set(beat_types)}
            logger.info(f"Story beats distribution: {beat_type_counts}")

        except Exception as e:
            logger.debug(f"Failed to track story beats metrics: {e}")

    return highlights


def optimize_highlights_with_story_beats(
    highlights: List[Dict],
    transcript_segments: List[Dict],
    job_id: str = "default",
    use_story_beats: bool = True
) -> List[Dict]:
    """
    Enhance existing highlights with story beats analysis - Tasks.md Step 2B
    
    Args:
        highlights: Existing highlights from other algorithms
        transcript_segments: Original transcript data
        job_id: Job identifier
        use_story_beats: Whether to enhance with story beats (for A/B testing)
        
    Returns:
        Enhanced and reordered highlights optimized for narrative flow
    """
    if not use_story_beats:
        # Control variant: just return chronologically sorted highlights
        return sorted(highlights, key=lambda x: x.get("start_ms", 0))

    # Experimental variant: enhance with story beats
    logger.info(f"Enhancing {len(highlights)} highlights with story beats analysis")

    # Generate story beats
    story_beats_highlights = analyze_story_beats(transcript_segments, job_id, use_ai=True)

    # Combine with existing highlights (avoiding duplicates)
    combined_highlights = highlights.copy()

    for story_highlight in story_beats_highlights:
        # Check for overlap with existing highlights
        has_overlap = False
        for existing in highlights:
            if (abs(story_highlight.get("start_ms", 0) - existing.get("start_ms", 0)) < 5000):  # 5 second overlap threshold
                has_overlap = True
                # Merge story beat info into existing highlight
                existing["is_story_beat"] = True
                existing["beat_type"] = story_highlight.get("beat_type")
                existing["narrative_function"] = story_highlight.get("narrative_function")
                break

        if not has_overlap:
            combined_highlights.append(story_highlight)

    # Optimize narrative flow
    analyzer = story_beats_analyzer
    optimized_highlights = analyzer.optimize_narrative_flow(combined_highlights)

    logger.info(f"Story beats enhancement complete: {len(highlights)} → {len(optimized_highlights)} highlights")

    return optimized_highlights
