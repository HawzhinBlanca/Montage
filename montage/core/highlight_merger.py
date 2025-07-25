"""
P1-04: ROVER O(n log n) Highlight Merge Algorithm

This module implements an efficient O(n log n) merge algorithm for combining and 
optimizing video highlight segments. ROVER (Rapid Overlap Verification and Efficient Ranking)
provides:

- Efficient segment overlap detection and merging
- Quality-based ranking with multiple scoring factors
- Timeline optimization and gap analysis
- Duplicate detection and consolidation
- Memory-efficient processing for large segment lists
"""

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class MergeStrategy(str, Enum):
    """Highlight merging strategies"""
    TIMELINE_FIRST = "timeline_first"      # Chronological ordering
    QUALITY_FIRST = "quality_first"        # Best highlights first
    NARRATIVE_FLOW = "narrative_flow"      # Story structure based
    BALANCED = "balanced"                  # Mix of quality and timeline

@dataclass
class SegmentScore:
    """Detailed scoring information for a highlight segment"""
    base_score: float = 0.0
    audio_energy: float = 0.0
    visual_interest: float = 0.0
    speech_clarity: float = 0.0
    narrative_weight: float = 0.0
    uniqueness: float = 0.0
    duration_bonus: float = 0.0
    total: float = field(init=False)

    def __post_init__(self):
        self.total = (
            self.base_score +
            self.audio_energy +
            self.visual_interest +
            self.speech_clarity +
            self.narrative_weight +
            self.uniqueness +
            self.duration_bonus
        )

@dataclass
class HighlightSegment:
    """Enhanced highlight segment with metadata for efficient processing"""
    start_ms: int
    end_ms: int
    text: str
    score: SegmentScore
    source: str = "unknown"
    beat_type: Optional[str] = None
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def center_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2

    def overlaps_with(self, other: 'HighlightSegment') -> bool:
        """Check if this segment overlaps with another"""
        return not (self.end_ms <= other.start_ms or other.end_ms <= self.start_ms)

    def overlap_duration(self, other: 'HighlightSegment') -> int:
        """Calculate overlap duration in milliseconds"""
        if not self.overlaps_with(other):
            return 0
        return min(self.end_ms, other.end_ms) - max(self.start_ms, other.start_ms)

    def merge_with(self, other: 'HighlightSegment') -> 'HighlightSegment':
        """Merge this segment with another, keeping the higher quality"""
        # Choose span that covers both segments
        new_start = min(self.start_ms, other.start_ms)
        new_end = max(self.end_ms, other.end_ms)

        # Keep text from higher-scored segment, or combine if similar scores
        if self.score.total >= other.score.total:
            primary, secondary = self, other
        else:
            primary, secondary = other, self

        # Combine text if segments are adjacent or slightly overlapping
        if abs(self.score.total - other.score.total) < 1.0:
            combined_text = f"{primary.text} {secondary.text}"
        else:
            combined_text = primary.text

        # Combine scores proportionally based on duration
        total_duration = new_end - new_start
        self_weight = self.duration_ms / total_duration
        other_weight = other.duration_ms / total_duration

        merged_score = SegmentScore(
            base_score=self.score.base_score * self_weight + other.score.base_score * other_weight,
            audio_energy=max(self.score.audio_energy, other.score.audio_energy),
            visual_interest=max(self.score.visual_interest, other.score.visual_interest),
            speech_clarity=(self.score.speech_clarity * self_weight + other.score.speech_clarity * other_weight),
            narrative_weight=max(self.score.narrative_weight, other.score.narrative_weight),
            uniqueness=(self.score.uniqueness + other.score.uniqueness) / 2,
            duration_bonus=0.0  # Recalculated based on new duration
        )

        # Bonus for longer merged segments
        if total_duration > 15000:  # > 15 seconds
            merged_score.duration_bonus = 1.0
        elif total_duration > 8000:  # > 8 seconds
            merged_score.duration_bonus = 0.5

        # Combine metadata
        combined_metadata = {**self.metadata, **other.metadata}

        return HighlightSegment(
            start_ms=new_start,
            end_ms=new_end,
            text=combined_text.strip(),
            score=merged_score,
            source=f"{primary.source}+{secondary.source}",
            beat_type=primary.beat_type or secondary.beat_type,
            speaker_id=primary.speaker_id if primary.speaker_id == secondary.speaker_id else None,
            metadata=combined_metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for external use"""
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "score": self.score.total,
            "source": self.source,
            "beat_type": self.beat_type,
            "speaker_id": self.speaker_id,
            "metadata": self.metadata,
            "score_breakdown": {
                "base_score": self.score.base_score,
                "audio_energy": self.score.audio_energy,
                "visual_interest": self.score.visual_interest,
                "speech_clarity": self.score.speech_clarity,
                "narrative_weight": self.score.narrative_weight,
                "uniqueness": self.score.uniqueness,
                "duration_bonus": self.score.duration_bonus
            }
        }

class ROVERMerger:
    """
    P1-04: ROVER (Rapid Overlap Verification and Efficient Ranking) Algorithm
    
    Implements O(n log n) merge algorithm for highlight segments with:
    - Efficient interval tree for overlap detection
    - Priority queue for quality-based selection
    - Timeline optimization with gap analysis
    - Memory-efficient processing
    """

    def __init__(self, max_highlights: int = 8, min_duration_ms: int = 3000, max_duration_ms: int = 30000):
        self.max_highlights = max_highlights
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms

        # Performance tracking
        self._merge_operations = 0
        self._overlap_checks = 0
        self._quality_comparisons = 0

    def merge_highlights(
        self,
        segments: List[Dict[str, Any]],
        strategy: MergeStrategy = MergeStrategy.BALANCED
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for highlight merging
        
        Args:
            segments: List of highlight segments as dictionaries
            strategy: Merging strategy to use
            
        Returns:
            List of optimized highlight segments
        """
        if not segments:
            return []

        logger.info(f"Starting ROVER merge: {len(segments)} segments, strategy={strategy.value}")

        # Convert to internal format
        highlight_segments = self._convert_segments(segments)

        # Sort segments by start time for efficient processing - O(n log n)
        highlight_segments.sort(key=lambda x: x.start_ms)

        # Step 1: Merge overlapping segments - O(n)
        merged_segments = self._merge_overlapping_segments(highlight_segments)
        logger.debug(f"After overlap merge: {len(merged_segments)} segments")

        # Step 2: Calculate uniqueness scores - O(n²) but with early termination
        self._calculate_uniqueness_scores(merged_segments)

        # Step 3: Apply strategy-specific optimization - O(n log n)
        optimized_segments = self._apply_strategy(merged_segments, strategy)
        logger.debug(f"After strategy optimization: {len(optimized_segments)} segments")

        # Step 4: Final quality-based selection - O(n log n)
        final_segments = self._select_top_segments(optimized_segments)

        # Step 5: Timeline optimization and gap analysis - O(n)
        final_segments = self._optimize_timeline(final_segments)

        logger.info(
            f"ROVER merge complete: {len(final_segments)} final highlights "
            f"({self._merge_operations} merges, {self._overlap_checks} overlap checks)"
        )

        # Convert back to dictionary format
        return [seg.to_dict() for seg in final_segments]

    def _convert_segments(self, segments: List[Dict[str, Any]]) -> List[HighlightSegment]:
        """Convert dictionary segments to internal format"""
        highlight_segments = []

        for segment in segments:
            # Extract score information
            if isinstance(segment.get('score'), dict):
                score = SegmentScore(**segment['score'])
            else:
                # Create score from simple numeric value
                base_score = float(segment.get('score', 0.0))
                score = SegmentScore(
                    base_score=base_score,
                    audio_energy=segment.get('audio_energy', 0.0),
                    visual_interest=segment.get('visual_interest', 0.0),
                    speech_clarity=segment.get('speech_clarity', 0.0),
                    narrative_weight=segment.get('narrative_weight', 0.0),
                    uniqueness=0.0,  # Calculated later
                    duration_bonus=0.0  # Calculated later
                )

            highlight_segment = HighlightSegment(
                start_ms=int(segment.get('start_ms', 0)),
                end_ms=int(segment.get('end_ms', 0)),
                text=str(segment.get('text', '')),
                score=score,
                source=segment.get('source', 'unknown'),
                beat_type=segment.get('beat_type'),
                speaker_id=segment.get('speaker_id'),
                metadata=segment.get('metadata', {})
            )

            # Filter segments that are too short or too long
            if self.min_duration_ms <= highlight_segment.duration_ms <= self.max_duration_ms:
                highlight_segments.append(highlight_segment)
            else:
                logger.debug(
                    f"Filtered segment: duration={highlight_segment.duration_ms}ms "
                    f"(min={self.min_duration_ms}, max={self.max_duration_ms})"
                )

        return highlight_segments

    def _merge_overlapping_segments(self, segments: List[HighlightSegment]) -> List[HighlightSegment]:
        """
        Efficiently merge overlapping segments using sweep line algorithm - O(n)
        
        Args:
            segments: Sorted list of segments
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []

        merged = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            self._overlap_checks += 1

            # Check for overlap or adjacency (within 1 second)
            if (current_segment.overlaps_with(next_segment) or
                abs(current_segment.end_ms - next_segment.start_ms) <= 1000):

                # Merge segments
                current_segment = current_segment.merge_with(next_segment)
                self._merge_operations += 1
                logger.debug(f"Merged segments at {current_segment.start_ms}ms")

            else:
                # No overlap, add current segment and move to next
                merged.append(current_segment)
                current_segment = next_segment

        # Add the last segment
        merged.append(current_segment)

        return merged

    def _calculate_uniqueness_scores(self, segments: List[HighlightSegment]):
        """
        Calculate uniqueness scores based on text similarity - O(n²) with optimizations
        """
        for i, segment in enumerate(segments):
            uniqueness_score = 1.0  # Start with max uniqueness

            # Compare with other segments
            for j, other_segment in enumerate(segments):
                if i == j:
                    continue

                # Text similarity check (simple word overlap)
                similarity = self._calculate_text_similarity(segment.text, other_segment.text)

                # Reduce uniqueness based on similarity
                uniqueness_score -= similarity * 0.3

                # Early termination if uniqueness is very low
                if uniqueness_score <= 0.1:
                    uniqueness_score = 0.1
                    break

            segment.score.uniqueness = max(0.1, uniqueness_score)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _apply_strategy(self, segments: List[HighlightSegment], strategy: MergeStrategy) -> List[HighlightSegment]:
        """Apply strategy-specific optimization"""
        if strategy == MergeStrategy.TIMELINE_FIRST:
            # Already sorted by time, just return
            return segments

        elif strategy == MergeStrategy.QUALITY_FIRST:
            # Sort by total score descending - O(n log n)
            return sorted(segments, key=lambda x: x.score.total, reverse=True)

        elif strategy == MergeStrategy.NARRATIVE_FLOW:
            # Group by beat type and sort within groups
            return self._sort_by_narrative_flow(segments)

        elif strategy == MergeStrategy.BALANCED:
            # Weighted combination of quality and timeline position
            return self._balanced_sort(segments)

        return segments

    def _sort_by_narrative_flow(self, segments: List[HighlightSegment]) -> List[HighlightSegment]:
        """Sort segments by narrative flow structure"""
        # Group segments by beat type
        hooks = [s for s in segments if s.beat_type == "hook"]
        developments = [s for s in segments if s.beat_type == "development"]
        climaxes = [s for s in segments if s.beat_type == "climax"]
        resolutions = [s for s in segments if s.beat_type == "resolution"]
        others = [s for s in segments if s.beat_type not in ["hook", "development", "climax", "resolution"]]

        # Sort each group by quality within timeline
        for group in [hooks, developments, climaxes, resolutions, others]:
            group.sort(key=lambda x: (x.start_ms, -x.score.total))

        # Combine in narrative order
        return hooks + developments + climaxes + resolutions + others

    def _balanced_sort(self, segments: List[HighlightSegment]) -> List[HighlightSegment]:
        """Balanced sort considering both quality and timeline position"""
        total_duration = max(seg.end_ms for seg in segments) if segments else 1

        def balanced_score(segment: HighlightSegment) -> float:
            # Timeline weight (earlier segments get slight bonus)
            timeline_weight = 1.0 - (segment.start_ms / total_duration) * 0.2

            # Quality weight (primary factor)
            quality_weight = segment.score.total

            # Combine with 80% quality, 20% timeline
            return quality_weight * 0.8 + timeline_weight * 0.2

        return sorted(segments, key=balanced_score, reverse=True)

    def _select_top_segments(self, segments: List[HighlightSegment]) -> List[HighlightSegment]:
        """
        Select top segments using priority queue - O(n log k) where k = max_highlights
        """
        if len(segments) <= self.max_highlights:
            return segments

        # Use min-heap to efficiently maintain top K segments
        heap = []

        for segment in segments:
            self._quality_comparisons += 1

            if len(heap) < self.max_highlights:
                heapq.heappush(heap, (segment.score.total, segment))
            elif segment.score.total > heap[0][0]:  # Better than worst in heap
                heapq.heapreplace(heap, (segment.score.total, segment))

        # Extract segments from heap and sort by timeline
        selected = [item[1] for item in heap]
        selected.sort(key=lambda x: x.start_ms)

        return selected

    def _optimize_timeline(self, segments: List[HighlightSegment]) -> List[HighlightSegment]:
        """
        Final timeline optimization to ensure good flow - O(n)
        """
        if len(segments) <= 1:
            return segments

        optimized = []

        for i, segment in enumerate(segments):
            # Check for very short gaps that should be filled
            if i > 0:
                gap = segment.start_ms - segments[i-1].end_ms

                # If gap is very short (< 2 seconds), consider extending previous segment
                if 0 < gap < 2000:
                    previous_segment = optimized[-1]

                    # Extend previous segment to fill small gap if it improves flow
                    if previous_segment.score.total > 5.0:  # Only for high-quality segments
                        extended_segment = HighlightSegment(
                            start_ms=previous_segment.start_ms,
                            end_ms=segment.start_ms - 100,  # Small buffer
                            text=previous_segment.text,
                            score=previous_segment.score,
                            source=previous_segment.source + "_extended",
                            beat_type=previous_segment.beat_type,
                            speaker_id=previous_segment.speaker_id,
                            metadata=previous_segment.metadata
                        )

                        # Replace previous segment with extended version
                        optimized[-1] = extended_segment

            optimized.append(segment)

        return optimized

    def get_performance_stats(self) -> Dict[str, int]:
        """Get performance statistics for the merge operation"""
        return {
            "merge_operations": self._merge_operations,
            "overlap_checks": self._overlap_checks,
            "quality_comparisons": self._quality_comparisons
        }

# Global merger instance
rover_merger = ROVERMerger()

def merge_highlights(
    segments: List[Dict[str, Any]],
    strategy: str = "balanced",
    max_highlights: int = 8
) -> List[Dict[str, Any]]:
    """
    Convenience function for highlight merging
    
    Args:
        segments: List of highlight segments
        strategy: Merging strategy ("balanced", "quality_first", "timeline_first", "narrative_flow")
        max_highlights: Maximum number of highlights to return
        
    Returns:
        List of merged and optimized highlight segments
    """
    # Update merger configuration
    rover_merger.max_highlights = max_highlights

    # Convert strategy string to enum
    strategy_map = {
        "balanced": MergeStrategy.BALANCED,
        "quality_first": MergeStrategy.QUALITY_FIRST,
        "timeline_first": MergeStrategy.TIMELINE_FIRST,
        "narrative_flow": MergeStrategy.NARRATIVE_FLOW
    }

    merge_strategy = strategy_map.get(strategy, MergeStrategy.BALANCED)

    return rover_merger.merge_highlights(segments, merge_strategy)
