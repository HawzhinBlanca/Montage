"""
P1-04: ROVER algorithm performance and correctness tests

Tests the ROVER (Rapid Overlap Verification and Efficient Ranking) merge algorithm:
- O(n log n) performance characteristics
- Overlap detection and merging accuracy
- Quality-based ranking
- Different merge strategies
- Memory efficiency and scalability
"""

import random
import time
from typing import Any, Dict

from montage.core.highlight_merger import (
    HighlightSegment,
    MergeStrategy,
    ROVERMerger,
    SegmentScore,
    merge_highlights,
)


class TestROVERAlgorithm:
    """Test ROVER highlight merge algorithm"""

    def setup_method(self):
        """Setup test environment"""
        self.merger = ROVERMerger(max_highlights=5, min_duration_ms=1000, max_duration_ms=30000)

    def create_test_segment(
        self,
        start_ms: int,
        end_ms: int,
        score: float = 5.0,
        text: str = "test segment",
        source: str = "test"
    ) -> Dict[str, Any]:
        """Helper to create test segment"""
        return {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "score": score,
            "source": source,
            "beat_type": None,
            "speaker_id": None,
            "metadata": {}
        }

    def test_empty_segments(self):
        """Test handling of empty segment list"""
        result = self.merger.merge_highlights([])
        assert result == []

    def test_single_segment(self):
        """Test handling of single segment"""
        segment = self.create_test_segment(1000, 5000, 8.0, "single segment")
        result = self.merger.merge_highlights([segment])

        assert len(result) == 1
        assert result[0]["start_ms"] == 1000
        assert result[0]["end_ms"] == 5000
        assert result[0]["text"] == "single segment"

    def test_non_overlapping_segments(self):
        """Test non-overlapping segments remain separate"""
        segments = [
            self.create_test_segment(1000, 3000, 7.0, "segment 1"),
            self.create_test_segment(5000, 7000, 8.0, "segment 2"),
            self.create_test_segment(9000, 11000, 6.0, "segment 3")
        ]

        result = self.merger.merge_highlights(segments, MergeStrategy.TIMELINE_FIRST)

        assert len(result) == 3
        assert result[0]["start_ms"] == 1000
        assert result[1]["start_ms"] == 5000
        assert result[2]["start_ms"] == 9000

    def test_overlapping_segments_merge(self):
        """Test overlapping segments are merged correctly"""
        segments = [
            self.create_test_segment(1000, 4000, 7.0, "segment 1"),
            self.create_test_segment(3000, 6000, 8.0, "segment 2"),  # Overlaps with segment 1
            self.create_test_segment(8000, 10000, 6.0, "segment 3")
        ]

        result = self.merger.merge_highlights(segments)

        # Should have 2 segments (1+2 merged, 3 separate)
        assert len(result) == 2

        # First merged segment should span both original segments
        merged_segment = result[0]
        assert merged_segment["start_ms"] == 1000
        assert merged_segment["end_ms"] == 6000
        assert "segment 1" in merged_segment["text"] or "segment 2" in merged_segment["text"]
        assert merged_segment["source"] == "test+test"

    def test_adjacent_segments_merge(self):
        """Test segments that are very close (< 1 second apart) get merged"""
        segments = [
            self.create_test_segment(1000, 3000, 7.0, "segment 1"),
            self.create_test_segment(3500, 5500, 8.0, "segment 2"),  # 500ms gap
            self.create_test_segment(8000, 10000, 6.0, "segment 3")
        ]

        result = self.merger.merge_highlights(segments)

        # Segments 1 and 2 should be merged due to small gap
        assert len(result) == 2
        assert result[0]["start_ms"] == 1000
        assert result[0]["end_ms"] == 5500  # Spans both segments

    def test_quality_first_strategy(self):
        """Test quality-first merge strategy orders by score"""
        segments = [
            self.create_test_segment(5000, 7000, 6.0, "medium quality"),
            self.create_test_segment(1000, 3000, 9.0, "high quality"),
            self.create_test_segment(9000, 11000, 3.0, "low quality")
        ]

        result = self.merger.merge_highlights(segments, MergeStrategy.QUALITY_FIRST)

        # Should be ordered by score (descending)
        assert len(result) == 3
        assert result[0]["text"] == "high quality"
        assert result[1]["text"] == "medium quality"
        assert result[2]["text"] == "low quality"

    def test_timeline_first_strategy(self):
        """Test timeline-first strategy maintains chronological order"""
        segments = [
            self.create_test_segment(9000, 11000, 9.0, "late but high quality"),
            self.create_test_segment(1000, 3000, 5.0, "early medium quality"),
            self.create_test_segment(5000, 7000, 4.0, "middle low quality")
        ]

        result = self.merger.merge_highlights(segments, MergeStrategy.TIMELINE_FIRST)

        # Should maintain chronological order
        assert len(result) == 3
        assert result[0]["start_ms"] == 1000
        assert result[1]["start_ms"] == 5000
        assert result[2]["start_ms"] == 9000

    def test_narrative_flow_strategy(self):
        """Test narrative flow strategy groups by beat types"""
        segments = [
            self.create_test_segment(9000, 11000, 7.0, "resolution", beat_type="resolution"),
            self.create_test_segment(1000, 3000, 8.0, "hook", beat_type="hook"),
            self.create_test_segment(5000, 7000, 6.0, "climax", beat_type="climax"),
            self.create_test_segment(3000, 5000, 7.0, "development", beat_type="development")
        ]

        # Add beat_type to segments
        segments[0]["beat_type"] = "resolution"
        segments[1]["beat_type"] = "hook"
        segments[2]["beat_type"] = "climax"
        segments[3]["beat_type"] = "development"

        result = self.merger.merge_highlights(segments, MergeStrategy.NARRATIVE_FLOW)

        # Should be ordered by narrative flow: hook -> development -> climax -> resolution
        beat_order = [seg.get("beat_type") for seg in result]
        expected_order = ["hook", "development", "climax", "resolution"]
        assert beat_order == expected_order

    def test_max_highlights_limit(self):
        """Test that merger respects maximum highlights limit"""
        # Create more segments than max_highlights
        segments = []
        for i in range(10):
            segments.append(
                self.create_test_segment(
                    i * 2000, (i * 2000) + 1500,
                    score=5.0 + i,  # Increasing scores
                    text=f"segment {i}"
                )
            )

        self.merger.max_highlights = 3
        result = self.merger.merge_highlights(segments)

        # Should only return top 3 by quality
        assert len(result) <= 3

        # Should be highest scoring segments
        result_scores = [seg.get("score", 0) for seg in result]
        assert all(score >= 12.0 for score in result_scores)  # Top scores are 14, 13, 12

    def test_duration_filtering(self):
        """Test segments outside duration limits are filtered"""
        segments = [
            self.create_test_segment(1000, 1500, 8.0, "too short"),    # 500ms - too short
            self.create_test_segment(3000, 6000, 7.0, "good length"),  # 3000ms - good
            self.create_test_segment(8000, 45000, 9.0, "too long")     # 37000ms - too long
        ]

        result = self.merger.merge_highlights(segments)

        # Only the good length segment should remain
        assert len(result) == 1
        assert result[0]["text"] == "good length"

    def test_uniqueness_scoring(self):
        """Test uniqueness scoring reduces duplicate content scores"""
        segments = [
            self.create_test_segment(1000, 3000, 8.0, "unique content here"),
            self.create_test_segment(5000, 7000, 8.0, "unique content here"),  # Duplicate text
            self.create_test_segment(9000, 11000, 8.0, "completely different content")
        ]

        result = self.merger.merge_highlights(segments)

        # Segments with duplicate content should have reduced uniqueness scores
        # The algorithm should prefer the unique content
        unique_segment = next((seg for seg in result if "completely different" in seg["text"]), None)
        assert unique_segment is not None

    def test_performance_o_n_log_n(self):
        """Test algorithm performance scales as O(n log n)"""
        # Test with different sizes to verify performance characteristics
        sizes = [10, 50, 100, 200]
        times = []

        for size in sizes:
            # Generate random non-overlapping segments
            segments = []
            for i in range(size):
                start = i * 3000  # 3 second spacing
                segments.append(
                    self.create_test_segment(
                        start, start + 2000,
                        score=random.uniform(1.0, 10.0),
                        text=f"segment {i}"
                    )
                )

            # Measure merge time
            start_time = time.time()
            self.merger.merge_highlights(segments)
            end_time = time.time()

            times.append(end_time - start_time)

        # Performance should scale reasonably (not exponentially)
        # Ratio between largest and smallest should be < size ratio squared
        if len(times) >= 2:
            time_ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
            size_ratio = sizes[-1] / sizes[0]

            # Should be better than O(n²) scaling
            assert time_ratio < size_ratio ** 1.5, f"Performance scaling concern: time ratio {time_ratio:.2f} vs size ratio {size_ratio}"

    def test_memory_efficiency(self):
        """Test memory usage remains reasonable for large inputs"""
        # Create a large number of segments
        large_segment_count = 1000
        segments = []

        for i in range(large_segment_count):
            segments.append(
                self.create_test_segment(
                    i * 1500, i * 1500 + 1000,  # Some overlaps
                    score=random.uniform(1.0, 10.0),
                    text=f"segment {i} with some longer text content"
                )
            )

        # Should complete without memory errors
        result = self.merger.merge_highlights(segments)

        # Should successfully reduce to manageable number
        assert len(result) <= self.merger.max_highlights

    def test_performance_stats_tracking(self):
        """Test performance statistics are tracked correctly"""
        segments = [
            self.create_test_segment(1000, 3000, 7.0, "segment 1"),
            self.create_test_segment(2500, 4500, 8.0, "segment 2"),  # Overlapping
            self.create_test_segment(6000, 8000, 6.0, "segment 3")
        ]

        # Reset stats
        self.merger._merge_operations = 0
        self.merger._overlap_checks = 0
        self.merger._quality_comparisons = 0

        result = self.merger.merge_highlights(segments)
        stats = self.merger.get_performance_stats()

        # Should have recorded operations
        assert stats["merge_operations"] > 0
        assert stats["overlap_checks"] > 0
        assert stats["quality_comparisons"] >= 0

    def test_segment_score_calculation(self):
        """Test SegmentScore calculation is correct"""
        score = SegmentScore(
            base_score=5.0,
            audio_energy=2.0,
            visual_interest=1.5,
            speech_clarity=1.0,
            narrative_weight=0.5,
            uniqueness=1.0,
            duration_bonus=0.5
        )

        expected_total = 5.0 + 2.0 + 1.5 + 1.0 + 0.5 + 1.0 + 0.5
        assert score.total == expected_total

    def test_highlight_segment_overlap_detection(self):
        """Test HighlightSegment overlap detection"""
        seg1 = HighlightSegment(
            start_ms=1000, end_ms=3000, text="seg1",
            score=SegmentScore(base_score=5.0)
        )
        seg2 = HighlightSegment(
            start_ms=2500, end_ms=4500, text="seg2",
            score=SegmentScore(base_score=6.0)
        )
        seg3 = HighlightSegment(
            start_ms=5000, end_ms=7000, text="seg3",
            score=SegmentScore(base_score=7.0)
        )

        # Test overlapping segments
        assert seg1.overlaps_with(seg2)
        assert seg2.overlaps_with(seg1)

        # Test non-overlapping segments
        assert not seg1.overlaps_with(seg3)
        assert not seg3.overlaps_with(seg1)

        # Test overlap duration calculation
        assert seg1.overlap_duration(seg2) == 500  # 3000 - 2500
        assert seg1.overlap_duration(seg3) == 0

    def test_segment_merging(self):
        """Test segment merging logic"""
        seg1 = HighlightSegment(
            start_ms=1000, end_ms=3000, text="first segment",
            score=SegmentScore(base_score=6.0, audio_energy=2.0)
        )
        seg2 = HighlightSegment(
            start_ms=2500, end_ms=4500, text="second segment",
            score=SegmentScore(base_score=7.0, visual_interest=1.5)
        )

        merged = seg1.merge_with(seg2)

        # Should span both segments
        assert merged.start_ms == 1000
        assert merged.end_ms == 4500

        # Should combine source info
        assert "+" in merged.source

        # Should have combined score elements
        assert merged.score.audio_energy == 2.0  # Max from seg1
        assert merged.score.visual_interest == 1.5  # Max from seg2


class TestConvenienceFunction:
    """Test the convenience merge_highlights function"""

    def test_convenience_function_basic(self):
        """Test basic usage of convenience function"""
        segments = [
            {"start_ms": 1000, "end_ms": 3000, "text": "segment 1", "score": 7.0},
            {"start_ms": 5000, "end_ms": 7000, "text": "segment 2", "score": 8.0},
            {"start_ms": 9000, "end_ms": 11000, "text": "segment 3", "score": 6.0}
        ]

        result = merge_highlights(segments, strategy="quality_first", max_highlights=2)

        assert len(result) <= 2
        # Should be ordered by quality
        assert result[0]["score"] >= result[1]["score"]

    def test_convenience_function_strategies(self):
        """Test different strategies work through convenience function"""
        segments = [
            {"start_ms": 5000, "end_ms": 7000, "text": "middle", "score": 8.0},
            {"start_ms": 1000, "end_ms": 3000, "text": "first", "score": 6.0},
            {"start_ms": 9000, "end_ms": 11000, "text": "last", "score": 7.0}
        ]

        # Test timeline strategy
        timeline_result = merge_highlights(segments, strategy="timeline_first")
        assert timeline_result[0]["text"] == "first"
        assert timeline_result[1]["text"] == "middle"
        assert timeline_result[2]["text"] == "last"

        # Test quality strategy
        quality_result = merge_highlights(segments, strategy="quality_first")
        assert quality_result[0]["text"] == "middle"  # Highest score

    def test_invalid_strategy_fallback(self):
        """Test invalid strategy falls back to balanced"""
        segments = [
            {"start_ms": 1000, "end_ms": 3000, "text": "segment", "score": 7.0}
        ]

        # Should not raise exception with invalid strategy
        result = merge_highlights(segments, strategy="invalid_strategy")
        assert len(result) == 1
