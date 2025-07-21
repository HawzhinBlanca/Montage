#!/usr/bin/env python3
"""
Advanced emotion mapping and energy analysis for story beats
"""
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Advanced emotion and energy analysis for narrative beats"""

    def __init__(self):
        # Emotion lexicons for different emotional categories
        self.emotion_lexicons = {
            "excitement": {
                "words": [
                    "amazing",
                    "incredible",
                    "fantastic",
                    "wow",
                    "exciting",
                    "thrilling",
                    "awesome",
                    "brilliant",
                ],
                "patterns": [
                    r"\b(oh my god|holy\s+shit|no way|unbelievable|mind.blowing)\b"
                ],
                "intensity_multipliers": {
                    "amazing": 1.5,
                    "incredible": 2.0,
                    "fantastic": 1.8,
                },
            },
            "curiosity": {
                "words": [
                    "interesting",
                    "fascinating",
                    "intriguing",
                    "wonder",
                    "mysterious",
                    "puzzling",
                    "curious",
                ],
                "patterns": [r"\b(what if|how does|why do|tell me more|I wonder)\b"],
                "intensity_multipliers": {"fascinating": 1.8, "intriguing": 1.6},
            },
            "surprise": {
                "words": [
                    "surprised",
                    "shocked",
                    "unexpected",
                    "sudden",
                    "reveal",
                    "twist",
                    "revelation",
                ],
                "patterns": [
                    r"\b(turns out|suddenly|out of nowhere|plot twist|didn't expect)\b"
                ],
                "intensity_multipliers": {"shocked": 2.0, "revelation": 1.9},
            },
            "inspiration": {
                "words": [
                    "inspiring",
                    "motivating",
                    "uplifting",
                    "hope",
                    "dream",
                    "achieve",
                    "possible",
                    "transform",
                ],
                "patterns": [
                    r"\b(you can do|never give up|believe in|make it happen|change your life)\b"
                ],
                "intensity_multipliers": {"transform": 2.0, "inspiring": 1.7},
            },
            "concern": {
                "words": [
                    "worried",
                    "concerned",
                    "problem",
                    "issue",
                    "trouble",
                    "difficult",
                    "challenging",
                    "struggle",
                ],
                "patterns": [
                    r"\b(be careful|watch out|I'm worried|that's concerning|major problem)\b"
                ],
                "intensity_multipliers": {"struggle": 1.8, "challenging": 1.5},
            },
            "relief": {
                "words": [
                    "relief",
                    "better",
                    "solved",
                    "fixed",
                    "calm",
                    "peaceful",
                    "safe",
                    "secure",
                ],
                "patterns": [
                    r"\b(thank god|finally|at last|much better|problem solved)\b"
                ],
                "intensity_multipliers": {"relief": 1.8, "peaceful": 1.6},
            },
            "humor": {
                "words": [
                    "funny",
                    "hilarious",
                    "amusing",
                    "joke",
                    "laugh",
                    "comedy",
                    "witty",
                    "clever",
                ],
                "patterns": [r"\b(ha ha|lol|that's funny|you're kidding|good one)\b"],
                "intensity_multipliers": {"hilarious": 2.0, "witty": 1.6},
            },
            "authority": {
                "words": [
                    "research",
                    "studies",
                    "evidence",
                    "data",
                    "proven",
                    "scientific",
                    "expert",
                    "professional",
                ],
                "patterns": [
                    r"\b(research shows|studies indicate|data proves|according to experts)\b"
                ],
                "intensity_multipliers": {"proven": 1.8, "scientific": 1.7},
            },
        }

        # Energy signature patterns
        self.energy_patterns = {
            "building": {
                "description": "gradually increasing energy",
                "min_duration": 5000,
            },
            "peak": {"description": "high sustained energy", "min_duration": 3000},
            "drop": {"description": "sudden energy decrease", "min_duration": 2000},
            "plateau": {
                "description": "steady consistent energy",
                "min_duration": 8000,
            },
            "volatile": {
                "description": "rapidly changing energy",
                "min_duration": 4000,
            },
        }

    def analyze_emotions_and_energy(
        self, transcript_segments: List[Dict], audio_energy: List[float]
    ) -> Dict:
        """Comprehensive emotion and energy analysis"""

        # Analyze emotions in text
        emotion_timeline = self._analyze_emotion_timeline(transcript_segments)

        # Analyze energy patterns in audio
        energy_patterns = self._analyze_energy_patterns(
            audio_energy, transcript_segments
        )

        # Find emotional peaks and valleys
        emotional_peaks = self._find_emotional_peaks(emotion_timeline)

        # Map emotions to narrative beats
        narrative_emotions = self._map_emotions_to_narrative_beats(
            emotion_timeline, transcript_segments
        )

        # Detect emotional arcs
        emotional_arcs = self._detect_emotional_arcs(emotion_timeline)

        # Find moments of maximum engagement
        engagement_peaks = self._find_engagement_peaks(
            emotion_timeline, energy_patterns, audio_energy
        )

        return {
            "emotion_timeline": emotion_timeline,
            "energy_patterns": energy_patterns,
            "emotional_peaks": emotional_peaks,
            "narrative_emotions": narrative_emotions,
            "emotional_arcs": emotional_arcs,
            "engagement_peaks": engagement_peaks,
            "overall_emotional_profile": self._create_emotional_profile(
                emotion_timeline
            ),
        }

    def _analyze_emotion_timeline(self, segments: List[Dict]) -> List[Dict]:
        """Create timeline of emotional content"""

        emotion_timeline = []

        for segment in segments:
            text = segment.get("text", "").lower()
            start_ms = segment.get("start_ms", 0)
            end_ms = segment.get("end_ms", 0)

            # Analyze each emotion category
            segment_emotions = {}
            total_emotional_intensity = 0

            for emotion, lexicon in self.emotion_lexicons.items():
                intensity = self._calculate_emotion_intensity(text, lexicon)
                segment_emotions[emotion] = intensity
                total_emotional_intensity += intensity

            # Determine primary emotion
            primary_emotion = max(segment_emotions.items(), key=lambda x: x[1])

            # Calculate emotional valence (positive/negative)
            valence = self._calculate_emotional_valence(segment_emotions)

            # Calculate arousal (intensity level)
            arousal = min(1.0, total_emotional_intensity / 3.0)  # Normalize

            emotion_data = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": segment.get("text", ""),
                "speaker": segment.get("speaker", "unknown"),
                "emotions": segment_emotions,
                "primary_emotion": (
                    primary_emotion[0] if primary_emotion[1] > 0.1 else "neutral"
                ),
                "emotional_intensity": primary_emotion[1],
                "valence": valence,  # -1 (negative) to 1 (positive)
                "arousal": arousal,  # 0 (calm) to 1 (intense)
                "engagement_score": self._calculate_engagement_score(
                    primary_emotion[1], arousal, valence
                ),
            }

            emotion_timeline.append(emotion_data)

        return emotion_timeline

    def _calculate_emotion_intensity(self, text: str, lexicon: Dict) -> float:
        """Calculate intensity of specific emotion in text"""

        intensity = 0.0

        # Check for emotion words
        for word in lexicon["words"]:
            count = text.count(word)
            multiplier = lexicon.get("intensity_multipliers", {}).get(word, 1.0)
            intensity += count * multiplier * 0.3

        # Check for emotion patterns
        for pattern in lexicon["patterns"]:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            intensity += matches * 0.5

        return min(1.0, intensity)  # Cap at 1.0

    def _calculate_emotional_valence(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional valence (positive/negative)"""

        positive_emotions = [
            "excitement",
            "curiosity",
            "inspiration",
            "relief",
            "humor",
        ]
        negative_emotions = ["concern", "surprise"]  # Surprise can be negative
        neutral_emotions = ["authority"]

        positive_score = sum(emotions.get(em, 0) for em in positive_emotions)
        negative_score = sum(emotions.get(em, 0) for em in negative_emotions)

        if positive_score + negative_score == 0:
            return 0.0

        return (positive_score - negative_score) / (positive_score + negative_score)

    def _calculate_engagement_score(
        self, emotional_intensity: float, arousal: float, valence: float
    ) -> float:
        """Calculate overall engagement score"""

        # High arousal and moderate emotional intensity = high engagement
        # Extreme valence (very positive or very negative) also increases engagement
        valence_factor = abs(valence)  # Extreme emotions are engaging

        engagement = emotional_intensity * 0.4 + arousal * 0.4 + valence_factor * 0.2

        return min(1.0, engagement)

    def _analyze_energy_patterns(
        self, audio_energy: List[float], segments: List[Dict]
    ) -> List[Dict]:
        """Analyze energy patterns in audio"""

        if not audio_energy or len(audio_energy) < 5:
            return []

        patterns = []
        window_size = 5  # 5-second windows

        for i in range(0, len(audio_energy) - window_size, window_size):
            window = audio_energy[i : i + window_size]
            start_ms = i * 1000
            end_ms = (i + window_size) * 1000

            # Calculate energy statistics
            avg_energy = sum(window) / len(window)
            energy_std = np.std(window)
            energy_trend = self._calculate_energy_trend(window)

            # Classify pattern type
            pattern_type = self._classify_energy_pattern(window, energy_std)

            # Find corresponding text
            corresponding_text = self._find_corresponding_text(
                start_ms, end_ms, segments
            )

            pattern = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "pattern_type": pattern_type,
                "avg_energy": avg_energy,
                "energy_variance": energy_std,
                "energy_trend": energy_trend,
                "corresponding_text": corresponding_text,
                "narrative_significance": self._assess_narrative_significance(
                    pattern_type, avg_energy, corresponding_text
                ),
            }

            patterns.append(pattern)

        return patterns

    def _calculate_energy_trend(self, window: List[float]) -> str:
        """Calculate trend in energy window"""

        if len(window) < 3:
            return "stable"

        # Simple linear trend
        x = list(range(len(window)))
        slope = np.polyfit(x, window, 1)[0]

        if slope > 0.02:
            return "rising"
        elif slope < -0.02:
            return "falling"
        else:
            return "stable"

    def _classify_energy_pattern(self, window: List[float], variance: float) -> str:
        """Classify type of energy pattern"""

        avg_energy = sum(window) / len(window)

        if variance > 0.15:
            return "volatile"  # High variance
        elif avg_energy > 0.7:
            return "peak"  # High energy
        elif avg_energy < 0.3:
            return "low"  # Low energy
        else:
            trend = self._calculate_energy_trend(window)
            if trend == "rising":
                return "building"
            elif trend == "falling":
                return "drop"
            else:
                return "plateau"

    def _find_corresponding_text(
        self, start_ms: int, end_ms: int, segments: List[Dict]
    ) -> str:
        """Find text corresponding to time window"""

        relevant_segments = [
            seg
            for seg in segments
            if seg.get("start_ms", 0) < end_ms and seg.get("end_ms", 0) > start_ms
        ]

        return " ".join(seg.get("text", "") for seg in relevant_segments)

    def _assess_narrative_significance(
        self, pattern_type: str, avg_energy: float, text: str
    ) -> float:
        """Assess narrative significance of energy pattern"""

        # Base significance by pattern type
        pattern_scores = {
            "peak": 0.9,
            "building": 0.8,
            "volatile": 0.7,
            "drop": 0.6,
            "plateau": 0.4,
            "low": 0.2,
        }

        base_score = pattern_scores.get(pattern_type, 0.5)

        # Boost for high energy
        energy_boost = min(0.3, avg_energy * 0.3)

        # Boost for emotionally significant text
        text_boost = 0
        if text:
            emotional_words = [
                "breakthrough",
                "discovery",
                "amazing",
                "incredible",
                "important",
                "crucial",
            ]
            text_lower = text.lower()
            text_boost = sum(0.1 for word in emotional_words if word in text_lower)

        return min(1.0, base_score + energy_boost + text_boost)

    def _find_emotional_peaks(self, emotion_timeline: List[Dict]) -> List[Dict]:
        """Find peaks in emotional intensity"""

        if len(emotion_timeline) < 3:
            return emotion_timeline

        peaks = []

        for i in range(1, len(emotion_timeline) - 1):
            current = emotion_timeline[i]
            prev = emotion_timeline[i - 1]
            next_item = emotion_timeline[i + 1]

            current_intensity = current["emotional_intensity"]

            # Peak detection: higher than neighbors and above threshold
            if (
                current_intensity > prev["emotional_intensity"]
                and current_intensity > next_item["emotional_intensity"]
                and current_intensity > 0.5
            ):

                peak_data = current.copy()
                peak_data["peak_type"] = "emotional_peak"
                peak_data["peak_intensity"] = current_intensity
                peaks.append(peak_data)

        # Sort by intensity
        peaks.sort(key=lambda x: x["peak_intensity"], reverse=True)

        return peaks[:5]  # Top 5 peaks

    def _map_emotions_to_narrative_beats(
        self, emotion_timeline: List[Dict], segments: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Map emotions to narrative beat types"""

        narrative_mapping = {
            "hook_emotions": [],
            "development_emotions": [],
            "climax_emotions": [],
            "resolution_emotions": [],
        }

        total_duration = (
            max(seg.get("end_ms", 0) for seg in segments) if segments else 0
        )

        for emotion_data in emotion_timeline:
            start_ms = emotion_data["start_ms"]

            # Map to narrative position based on timing
            progress = start_ms / total_duration if total_duration > 0 else 0

            if progress < 0.2:  # First 20% = hook territory
                if emotion_data["emotional_intensity"] > 0.4:
                    narrative_mapping["hook_emotions"].append(emotion_data)
            elif progress < 0.7:  # Middle 50% = development
                narrative_mapping["development_emotions"].append(emotion_data)
            elif progress < 0.9:  # Next 20% = climax territory
                if emotion_data["emotional_intensity"] > 0.6:
                    narrative_mapping["climax_emotions"].append(emotion_data)
            else:  # Final 10% = resolution
                narrative_mapping["resolution_emotions"].append(emotion_data)

        return narrative_mapping

    def _detect_emotional_arcs(self, emotion_timeline: List[Dict]) -> List[Dict]:
        """Detect emotional arcs across the timeline"""

        if len(emotion_timeline) < 5:
            return []

        arcs = []

        # Sliding window for arc detection
        window_size = 5

        for i in range(len(emotion_timeline) - window_size + 1):
            window = emotion_timeline[i : i + window_size]

            # Analyze emotional trajectory
            intensities = [item["emotional_intensity"] for item in window]
            valences = [item["valence"] for item in window]

            # Detect arc type
            arc_type = self._classify_emotional_arc(intensities, valences)

            if arc_type != "flat":
                arc = {
                    "start_ms": window[0]["start_ms"],
                    "end_ms": window[-1]["end_ms"],
                    "arc_type": arc_type,
                    "intensity_range": (min(intensities), max(intensities)),
                    "valence_range": (min(valences), max(valences)),
                    "segments": window,
                    "narrative_impact": self._calculate_arc_impact(
                        intensities, valences
                    ),
                }
                arcs.append(arc)

        # Remove overlapping arcs, keep highest impact
        return self._merge_overlapping_arcs(arcs)

    def _classify_emotional_arc(
        self, intensities: List[float], valences: List[float]
    ) -> str:
        """Classify type of emotional arc"""

        intensity_trend = self._calculate_trend(intensities)
        valence_trend = self._calculate_trend(valences)

        # Arc classification
        if intensity_trend > 0.1 and valence_trend > 0.1:
            return "rising_positive"  # Building excitement
        elif intensity_trend > 0.1 and valence_trend < -0.1:
            return "rising_negative"  # Building tension
        elif intensity_trend < -0.1 and valence_trend > 0.1:
            return "resolving_positive"  # Relief/resolution
        elif intensity_trend < -0.1 and valence_trend < -0.1:
            return "declining"  # Winding down
        elif max(intensities) - min(intensities) > 0.4:
            return "volatile"  # Emotional roller coaster
        else:
            return "flat"  # Stable

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in list of values"""
        if len(values) < 2:
            return 0

        x = list(range(len(values)))
        return np.polyfit(x, values, 1)[0]

    def _calculate_arc_impact(
        self, intensities: List[float], valences: List[float]
    ) -> float:
        """Calculate narrative impact of emotional arc"""

        intensity_range = max(intensities) - min(intensities)
        valence_range = max(valences) - min(valences)
        avg_intensity = sum(intensities) / len(intensities)

        # Higher impact for more dramatic changes and higher intensity
        impact = intensity_range * 0.4 + valence_range * 0.3 + avg_intensity * 0.3
        return min(1.0, impact)

    def _merge_overlapping_arcs(self, arcs: List[Dict]) -> List[Dict]:
        """Merge overlapping emotional arcs"""

        if not arcs:
            return []

        # Sort by impact
        arcs.sort(key=lambda x: x["narrative_impact"], reverse=True)

        merged = []

        for arc in arcs:
            # Check for overlap with existing arcs
            overlaps = False
            for existing in merged:
                if (
                    arc["start_ms"] < existing["end_ms"]
                    and arc["end_ms"] > existing["start_ms"]
                ):
                    overlaps = True
                    break

            if not overlaps:
                merged.append(arc)

        return merged

    def _find_engagement_peaks(
        self,
        emotion_timeline: List[Dict],
        energy_patterns: List[Dict],
        audio_energy: List[float],
    ) -> List[Dict]:
        """Find moments of maximum audience engagement"""

        engagement_moments = []

        # Combine emotional and energy data
        for emotion_data in emotion_timeline:
            start_ms = emotion_data["start_ms"]
            end_ms = emotion_data["end_ms"]

            # Find corresponding energy pattern
            energy_data = None
            for pattern in energy_patterns:
                if pattern["start_ms"] <= start_ms and pattern["end_ms"] >= end_ms:
                    energy_data = pattern
                    break

            # Calculate combined engagement score
            emotional_engagement = emotion_data["engagement_score"]
            energy_engagement = energy_data["avg_energy"] if energy_data else 0.5
            narrative_significance = (
                energy_data.get("narrative_significance", 0.5) if energy_data else 0.5
            )

            combined_engagement = (
                emotional_engagement * 0.4
                + energy_engagement * 0.3
                + narrative_significance * 0.3
            )

            if combined_engagement > 0.7:  # High engagement threshold
                engagement_moment = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "engagement_score": combined_engagement,
                    "emotional_component": emotional_engagement,
                    "energy_component": energy_engagement,
                    "narrative_component": narrative_significance,
                    "primary_emotion": emotion_data["primary_emotion"],
                    "text": emotion_data["text"],
                    "engagement_type": self._classify_engagement_type(
                        emotion_data, energy_data
                    ),
                }
                engagement_moments.append(engagement_moment)

        # Sort by engagement score
        engagement_moments.sort(key=lambda x: x["engagement_score"], reverse=True)

        return engagement_moments[:8]  # Top 8 engagement peaks

    def _classify_engagement_type(
        self, emotion_data: Dict, energy_data: Optional[Dict]
    ) -> str:
        """Classify type of engagement moment"""

        primary_emotion = emotion_data["primary_emotion"]
        energy_pattern = (
            energy_data.get("pattern_type", "stable") if energy_data else "stable"
        )

        # Classification matrix
        if primary_emotion == "excitement" and energy_pattern in ["peak", "building"]:
            return "high_energy_excitement"
        elif primary_emotion == "surprise" and energy_pattern == "volatile":
            return "shocking_revelation"
        elif primary_emotion == "curiosity" and energy_pattern == "building":
            return "building_intrigue"
        elif primary_emotion == "inspiration" and energy_pattern in ["peak", "plateau"]:
            return "inspirational_moment"
        elif primary_emotion == "humor" and energy_pattern == "peak":
            return "comedic_peak"
        else:
            return "general_engagement"

    def _create_emotional_profile(self, emotion_timeline: List[Dict]) -> Dict:
        """Create overall emotional profile of content"""

        if not emotion_timeline:
            return {
                "dominant_emotion": "neutral",
                "emotional_diversity": 0,
                "engagement_level": 0,
            }

        # Count emotion occurrences
        emotion_counts = defaultdict(int)
        total_engagement = 0

        for item in emotion_timeline:
            emotion_counts[item["primary_emotion"]] += 1
            total_engagement += item["engagement_score"]

        # Find dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

        # Calculate emotional diversity
        unique_emotions = len([e for e in emotion_counts.values() if e > 0])
        emotional_diversity = min(1.0, unique_emotions / 6)  # Normalize

        # Average engagement
        avg_engagement = total_engagement / len(emotion_timeline)

        return {
            "dominant_emotion": dominant_emotion,
            "emotional_diversity": emotional_diversity,
            "engagement_level": avg_engagement,
            "emotion_distribution": dict(emotion_counts),
            "total_segments": len(emotion_timeline),
        }


def enhance_highlights_with_emotion_analysis(
    highlights: List[Dict], emotion_analysis: Dict
) -> List[Dict]:
    """Enhance highlights with emotional intelligence"""

    engagement_peaks = emotion_analysis.get("engagement_peaks", [])
    emotional_arcs = emotion_analysis.get("emotional_arcs", [])
    emotion_timeline = emotion_analysis.get("emotion_timeline", [])

    for highlight in highlights:
        start_ms = highlight.get("start_ms", 0)
        end_ms = highlight.get("end_ms", 0)

        # Find corresponding emotional data
        emotional_data = [
            item
            for item in emotion_timeline
            if item["start_ms"] >= start_ms and item["end_ms"] <= end_ms
        ]

        if emotional_data:
            # Calculate emotional metrics for this highlight
            avg_engagement = sum(
                item["engagement_score"] for item in emotional_data
            ) / len(emotional_data)
            dominant_emotion = max(
                emotional_data, key=lambda x: x["emotional_intensity"]
            )["primary_emotion"]

            # Check if this is an engagement peak
            is_engagement_peak = any(
                peak["start_ms"] <= start_ms and peak["end_ms"] >= end_ms
                for peak in engagement_peaks
            )

            # Check if part of emotional arc
            emotional_arc = None
            for arc in emotional_arcs:
                if arc["start_ms"] <= start_ms and arc["end_ms"] >= end_ms:
                    emotional_arc = arc
                    break

            # Update highlight with emotional intelligence
            highlight.update(
                {
                    "emotional_engagement": avg_engagement,
                    "dominant_emotion": dominant_emotion,
                    "is_engagement_peak": is_engagement_peak,
                    "emotional_arc": (
                        emotional_arc["arc_type"] if emotional_arc else None
                    ),
                    "emotional_impact": (
                        emotional_arc["narrative_impact"]
                        if emotional_arc
                        else avg_engagement
                    ),
                }
            )

            # Boost score for high engagement
            if avg_engagement > 0.7:
                engagement_boost = (avg_engagement - 0.7) * 3.0  # Scale boost
                highlight["score"] = highlight.get("score", 0) + engagement_boost
                highlight["engagement_boost"] = engagement_boost

            # Extra boost for engagement peaks
            if is_engagement_peak:
                highlight["score"] = highlight.get("score", 0) + 1.5
                highlight["peak_boost"] = 1.5

    return highlights
