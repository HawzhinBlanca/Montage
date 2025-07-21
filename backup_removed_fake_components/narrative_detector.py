#!/usr/bin/env python3
"""
Advanced hook/climax/resolution detection system for narrative structure
"""
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NarrativeDetector:
    """Advanced detection of narrative beats: hooks, climaxes, and resolutions"""

    def __init__(self):
        # Hook detection patterns
        self.hook_patterns = {
            "question_hooks": [
                r"\b(what if|imagine if|have you ever|did you know|what would happen)\b",
                r"\b(why do|how can|what makes|where does|when will)\b",
                r"\b(the question is|here's what|let me ask you)\b",
            ],
            "statement_hooks": [
                r"\b(this will change|everything you know|never seen before|incredible discovery)\b",
                r"\b(most people don't|secret to|truth about|hidden)\b",
                r"\b(breakthrough|revolutionary|game.changer|unprecedented)\b",
            ],
            "story_hooks": [
                r"\b(it all started|began when|first time|years ago)\b",
                r"\b(little did I know|never imagined|suddenly|one day)\b",
                r"\b(let me tell you|here's what happened|story begins)\b",
            ],
            "urgency_hooks": [
                r"\b(right now|happening today|urgent|critical moment)\b",
                r"\b(time is running|can't wait|immediate|this instant)\b",
                r"\b(before it's too late|limited time|act now)\b",
            ],
        }

        # Climax detection patterns
        self.climax_patterns = {
            "revelation_climax": [
                r"\b(turns out|discovered that|realized|found out|the truth is)\b",
                r"\b(breakthrough moment|aha moment|suddenly understood|it hit me)\b",
                r"\b(the answer was|key insight|revelation|epiphany)\b",
            ],
            "conflict_climax": [
                r"\b(biggest challenge|moment of truth|critical point|breaking point)\b",
                r"\b(everything changed|pivotal moment|turning point|climax)\b",
                r"\b(crisis|emergency|disaster|catastrophe|collapse)\b",
            ],
            "achievement_climax": [
                r"\b(finally achieved|reached the goal|accomplished|succeeded)\b",
                r"\b(moment of victory|breakthrough|triumph|pinnacle)\b",
                r"\b(peak performance|highest point|ultimate|maximum)\b",
            ],
            "emotional_climax": [
                r"\b(overwhelming|incredible feeling|never felt|most intense)\b",
                r"\b(tears of joy|heart racing|speechless|blown away)\b",
                r"\b(life.changing|transformative|profound impact)\b",
            ],
        }

        # Resolution detection patterns
        self.resolution_patterns = {
            "conclusion_resolution": [
                r"\b(in conclusion|to summarize|bottom line|final point)\b",
                r"\b(end result|outcome|what we learned|takeaway)\b",
                r"\b(lesson learned|key insight|main message)\b",
            ],
            "action_resolution": [
                r"\b(what you can do|next steps|take action|start today)\b",
                r"\b(here's how|follow these|try this|implement)\b",
                r"\b(call to action|your turn|now go|begin your journey)\b",
            ],
            "transformation_resolution": [
                r"\b(changed my life|never the same|transformation|evolved)\b",
                r"\b(new perspective|different person|growth|learned)\b",
                r"\b(journey continues|ongoing process|keep going)\b",
            ],
            "emotional_resolution": [
                r"\b(peace|calm|resolved|closure|satisfaction)\b",
                r"\b(feeling better|relieved|grateful|content)\b",
                r"\b(hope|optimistic|positive|bright future)\b",
            ],
        }

        # Temporal position weights (where these beats typically occur)
        self.temporal_weights = {
            "hook": {"early": 3.0, "middle": 1.0, "late": 0.5},
            "climax": {"early": 0.8, "middle": 2.5, "late": 2.0},
            "resolution": {"early": 0.5, "middle": 1.0, "late": 3.0},
        }

    def detect_narrative_beats(
        self,
        transcript_segments: List[Dict],
        emotion_analysis: Optional[Dict] = None,
        speaker_analysis: Optional[Dict] = None,
    ) -> Dict:
        """Comprehensive detection of narrative beats"""

        # Calculate temporal positions
        total_duration = (
            max(seg.get("end_ms", 0) for seg in transcript_segments)
            if transcript_segments
            else 0
        )

        # Detect hooks
        hook_candidates = self._detect_hooks(transcript_segments, total_duration)

        # Detect climaxes
        climax_candidates = self._detect_climaxes(
            transcript_segments, total_duration, emotion_analysis
        )

        # Detect resolutions
        resolution_candidates = self._detect_resolutions(
            transcript_segments, total_duration
        )

        # Cross-validate with other analyses
        if emotion_analysis:
            hook_candidates = self._enhance_with_emotion_data(
                hook_candidates, emotion_analysis, "hook"
            )
            climax_candidates = self._enhance_with_emotion_data(
                climax_candidates, emotion_analysis, "climax"
            )
            resolution_candidates = self._enhance_with_emotion_data(
                resolution_candidates, emotion_analysis, "resolution"
            )

        if speaker_analysis:
            hook_candidates = self._enhance_with_speaker_data(
                hook_candidates, speaker_analysis, "hook"
            )
            climax_candidates = self._enhance_with_speaker_data(
                climax_candidates, speaker_analysis, "climax"
            )
            resolution_candidates = self._enhance_with_speaker_data(
                resolution_candidates, speaker_analysis, "resolution"
            )

        # Select best candidates
        best_hooks = self._select_best_candidates(hook_candidates, max_count=2)
        best_climaxes = self._select_best_candidates(climax_candidates, max_count=3)
        best_resolutions = self._select_best_candidates(
            resolution_candidates, max_count=2
        )

        # Validate narrative structure
        validated_structure = self._validate_narrative_structure(
            best_hooks, best_climaxes, best_resolutions, total_duration
        )

        return {
            "hooks": validated_structure["hooks"],
            "climaxes": validated_structure["climaxes"],
            "resolutions": validated_structure["resolutions"],
            "narrative_coherence": self._calculate_narrative_coherence(
                validated_structure
            ),
            "story_arc_completeness": self._assess_story_completeness(
                validated_structure
            ),
            "recommended_structure": self._recommend_structure(validated_structure),
        }

    def _detect_hooks(self, segments: List[Dict], total_duration: int) -> List[Dict]:
        """Detect hook moments in content"""

        hook_candidates = []

        for segment in segments:
            text = segment.get("text", "").lower()
            start_ms = segment.get("start_ms", 0)
            end_ms = segment.get("end_ms", 0)

            # Calculate temporal position
            position_ratio = start_ms / total_duration if total_duration > 0 else 0

            # Score for different hook types
            hook_scores = {}
            total_hook_score = 0

            for hook_type, patterns in self.hook_patterns.items():
                type_score = 0

                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    type_score += matches * 0.3

                hook_scores[hook_type] = type_score
                total_hook_score += type_score

            # Apply temporal weighting
            if position_ratio < 0.3:  # Early in content
                temporal_weight = self.temporal_weights["hook"]["early"]
            elif position_ratio < 0.7:  # Middle
                temporal_weight = self.temporal_weights["hook"]["middle"]
            else:  # Late
                temporal_weight = self.temporal_weights["hook"]["late"]

            final_score = total_hook_score * temporal_weight

            # Additional scoring factors
            # Shorter segments often make better hooks
            if (end_ms - start_ms) < 8000:  # Less than 8 seconds
                final_score *= 1.2

            # First few segments get bonus
            if start_ms < 30000:  # First 30 seconds
                final_score *= 1.5

            if final_score > 0.5:  # Threshold for hook candidates
                hook_candidate = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": segment.get("text", ""),
                    "speaker": segment.get("speaker", "unknown"),
                    "hook_score": final_score,
                    "hook_types": hook_scores,
                    "dominant_hook_type": (
                        max(hook_scores.items(), key=lambda x: x[1])[0]
                        if hook_scores
                        else "generic"
                    ),
                    "temporal_position": position_ratio,
                    "temporal_weight": temporal_weight,
                    "beat_type": "hook",
                }
                hook_candidates.append(hook_candidate)

        return hook_candidates

    def _detect_climaxes(
        self,
        segments: List[Dict],
        total_duration: int,
        emotion_analysis: Optional[Dict] = None,
    ) -> List[Dict]:
        """Detect climactic moments in content"""

        climax_candidates = []

        # Get emotional peaks if available
        emotional_peaks = []
        if emotion_analysis:
            emotional_peaks = emotion_analysis.get("emotional_peaks", [])
            engagement_peaks = emotion_analysis.get("engagement_peaks", [])

        for segment in segments:
            text = segment.get("text", "").lower()
            start_ms = segment.get("start_ms", 0)
            end_ms = segment.get("end_ms", 0)

            # Calculate temporal position
            position_ratio = start_ms / total_duration if total_duration > 0 else 0

            # Score for different climax types
            climax_scores = {}
            total_climax_score = 0

            for climax_type, patterns in self.climax_patterns.items():
                type_score = 0

                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    type_score += matches * 0.4

                climax_scores[climax_type] = type_score
                total_climax_score += type_score

            # Apply temporal weighting
            if position_ratio < 0.3:  # Early
                temporal_weight = self.temporal_weights["climax"]["early"]
            elif position_ratio < 0.8:  # Middle-late
                temporal_weight = self.temporal_weights["climax"]["middle"]
            else:  # Very late
                temporal_weight = self.temporal_weights["climax"]["late"]

            final_score = total_climax_score * temporal_weight

            # Boost if this aligns with emotional peaks
            emotion_boost = 0
            if emotion_analysis:
                for peak in emotional_peaks:
                    if peak.get("start_ms", 0) <= start_ms <= peak.get("end_ms", 0):
                        emotion_boost = peak.get("peak_intensity", 0) * 2.0
                        break

                for peak in engagement_peaks:
                    if peak.get("start_ms", 0) <= start_ms <= peak.get("end_ms", 0):
                        emotion_boost = max(
                            emotion_boost, peak.get("engagement_score", 0) * 1.5
                        )

            final_score += emotion_boost

            # Energy level boost
            energy_level = segment.get("energy_level", 0.5)
            if energy_level > 0.7:
                final_score *= 1.3

            if final_score > 0.6:  # Threshold for climax candidates
                climax_candidate = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": segment.get("text", ""),
                    "speaker": segment.get("speaker", "unknown"),
                    "climax_score": final_score,
                    "climax_types": climax_scores,
                    "dominant_climax_type": (
                        max(climax_scores.items(), key=lambda x: x[1])[0]
                        if climax_scores
                        else "generic"
                    ),
                    "temporal_position": position_ratio,
                    "temporal_weight": temporal_weight,
                    "emotion_boost": emotion_boost,
                    "energy_level": energy_level,
                    "beat_type": "climax",
                }
                climax_candidates.append(climax_candidate)

        return climax_candidates

    def _detect_resolutions(
        self, segments: List[Dict], total_duration: int
    ) -> List[Dict]:
        """Detect resolution moments in content"""

        resolution_candidates = []

        for segment in segments:
            text = segment.get("text", "").lower()
            start_ms = segment.get("start_ms", 0)
            end_ms = segment.get("end_ms", 0)

            # Calculate temporal position
            position_ratio = start_ms / total_duration if total_duration > 0 else 0

            # Score for different resolution types
            resolution_scores = {}
            total_resolution_score = 0

            for resolution_type, patterns in self.resolution_patterns.items():
                type_score = 0

                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    type_score += matches * 0.35

                resolution_scores[resolution_type] = type_score
                total_resolution_score += type_score

            # Apply temporal weighting (resolutions favor later positions)
            if position_ratio < 0.5:  # Early
                temporal_weight = self.temporal_weights["resolution"]["early"]
            elif position_ratio < 0.8:  # Middle
                temporal_weight = self.temporal_weights["resolution"]["middle"]
            else:  # Late
                temporal_weight = self.temporal_weights["resolution"]["late"]

            final_score = total_resolution_score * temporal_weight

            # Boost for final segments
            if position_ratio > 0.85:
                final_score *= 1.4

            # Boost for conclusive language
            conclusive_words = [
                "finally",
                "conclusion",
                "end",
                "result",
                "outcome",
                "learned",
                "takeaway",
            ]
            conclusive_count = sum(1 for word in conclusive_words if word in text)
            final_score += conclusive_count * 0.2

            if final_score > 0.4:  # Threshold for resolution candidates
                resolution_candidate = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": segment.get("text", ""),
                    "speaker": segment.get("speaker", "unknown"),
                    "resolution_score": final_score,
                    "resolution_types": resolution_scores,
                    "dominant_resolution_type": (
                        max(resolution_scores.items(), key=lambda x: x[1])[0]
                        if resolution_scores
                        else "generic"
                    ),
                    "temporal_position": position_ratio,
                    "temporal_weight": temporal_weight,
                    "conclusive_words": conclusive_count,
                    "beat_type": "resolution",
                }
                resolution_candidates.append(resolution_candidate)

        return resolution_candidates

    def _enhance_with_emotion_data(
        self, candidates: List[Dict], emotion_analysis: Dict, beat_type: str
    ) -> List[Dict]:
        """Enhance narrative beat detection with emotion data"""

        emotion_timeline = emotion_analysis.get("emotion_timeline", [])

        for candidate in candidates:
            start_ms = candidate["start_ms"]
            end_ms = candidate["end_ms"]

            # Find corresponding emotional data
            corresponding_emotions = [
                item
                for item in emotion_timeline
                if item["start_ms"] >= start_ms and item["end_ms"] <= end_ms
            ]

            if corresponding_emotions:
                avg_engagement = sum(
                    item["engagement_score"] for item in corresponding_emotions
                ) / len(corresponding_emotions)
                dominant_emotion = max(
                    corresponding_emotions, key=lambda x: x["emotional_intensity"]
                )

                # Apply emotion-specific boosts based on beat type
                emotion_boost = 0

                if beat_type == "hook":
                    if dominant_emotion["primary_emotion"] in [
                        "curiosity",
                        "surprise",
                        "excitement",
                    ]:
                        emotion_boost = avg_engagement * 1.5
                elif beat_type == "climax":
                    if dominant_emotion["primary_emotion"] in [
                        "excitement",
                        "surprise",
                        "inspiration",
                    ]:
                        emotion_boost = avg_engagement * 2.0
                elif beat_type == "resolution":
                    if dominant_emotion["primary_emotion"] in [
                        "relief",
                        "inspiration",
                        "authority",
                    ]:
                        emotion_boost = avg_engagement * 1.3

                # Update candidate score
                score_key = f"{beat_type}_score"
                candidate[score_key] = candidate.get(score_key, 0) + emotion_boost
                candidate["emotion_boost"] = emotion_boost
                candidate["dominant_emotion"] = dominant_emotion["primary_emotion"]
                candidate["avg_engagement"] = avg_engagement

        return candidates

    def _enhance_with_speaker_data(
        self, candidates: List[Dict], speaker_analysis: Dict, beat_type: str
    ) -> List[Dict]:
        """Enhance narrative beat detection with speaker data"""

        speakers = speaker_analysis.get("speakers", {})
        narrative_anchors = speaker_analysis.get("narrative_anchors", [])
        character_roles = speaker_analysis.get("character_roles", {})

        for candidate in candidates:
            speaker_id = candidate.get("speaker", "unknown")

            if speaker_id in speakers:
                speaker_profile = speakers[speaker_id]
                character_role = character_roles.get(speaker_id, "unknown")

                # Apply speaker-specific boosts
                speaker_boost = 0

                # Narrative anchors get boost for all beats
                if speaker_id in narrative_anchors:
                    speaker_boost += 0.5

                # Role-specific boosts
                if beat_type == "hook" and character_role in ["host", "storyteller"]:
                    speaker_boost += 0.4
                elif beat_type == "climax" and character_role in [
                    "expert",
                    "guest",
                    "storyteller",
                ]:
                    speaker_boost += 0.6
                elif beat_type == "resolution" and character_role in [
                    "host",
                    "expert",
                    "educator",
                ]:
                    speaker_boost += 0.4

                # Consistency boost
                consistency = speaker_profile.get("consistency", 0)
                speaker_boost += consistency * 0.3

                # Update candidate score
                score_key = f"{beat_type}_score"
                candidate[score_key] = candidate.get(score_key, 0) + speaker_boost
                candidate["speaker_boost"] = speaker_boost
                candidate["character_role"] = character_role
                candidate["is_narrative_anchor"] = speaker_id in narrative_anchors

        return candidates

    def _select_best_candidates(
        self, candidates: List[Dict], max_count: int
    ) -> List[Dict]:
        """Select best candidates for each beat type"""

        if not candidates:
            return []

        # Sort by score
        beat_type = candidates[0].get("beat_type", "unknown")
        score_key = f"{beat_type}_score"

        candidates.sort(key=lambda x: x.get(score_key, 0), reverse=True)

        # Remove candidates that are too close together
        selected = []
        min_gap = 30000  # 30 seconds minimum gap

        for candidate in candidates:
            # Check if too close to already selected
            too_close = False
            for selected_item in selected:
                if abs(candidate["start_ms"] - selected_item["start_ms"]) < min_gap:
                    too_close = True
                    break

            if not too_close:
                selected.append(candidate)

                if len(selected) >= max_count:
                    break

        return selected

    def _validate_narrative_structure(
        self,
        hooks: List[Dict],
        climaxes: List[Dict],
        resolutions: List[Dict],
        total_duration: int,
    ) -> Dict:
        """Validate and adjust narrative structure for coherence"""

        # Ensure temporal ordering
        all_beats = []
        all_beats.extend([(beat, "hook") for beat in hooks])
        all_beats.extend([(beat, "climax") for beat in climaxes])
        all_beats.extend([(beat, "resolution") for beat in resolutions])

        # Sort by timestamp
        all_beats.sort(key=lambda x: x[0]["start_ms"])

        # Validate structure rules
        validated = {"hooks": [], "climaxes": [], "resolutions": []}

        for beat, beat_type in all_beats:
            timestamp = beat["start_ms"]
            position_ratio = timestamp / total_duration if total_duration > 0 else 0

            # Apply position rules
            if beat_type == "hook" and position_ratio < 0.4:  # Hooks in first 40%
                validated["hooks"].append(beat)
            elif (
                beat_type == "climax" and 0.2 < position_ratio < 0.9
            ):  # Climaxes in middle 70%
                validated["climaxes"].append(beat)
            elif (
                beat_type == "resolution" and position_ratio > 0.6
            ):  # Resolutions in last 40%
                validated["resolutions"].append(beat)

        return validated

    def _calculate_narrative_coherence(self, structure: Dict) -> float:
        """Calculate how coherent the narrative structure is"""

        hooks = structure.get("hooks", [])
        climaxes = structure.get("climaxes", [])
        resolutions = structure.get("resolutions", [])

        # Base coherence factors
        has_hook = len(hooks) > 0
        has_climax = len(climaxes) > 0
        has_resolution = len(resolutions) > 0

        base_coherence = (has_hook + has_climax + has_resolution) / 3

        # Check temporal ordering
        all_timestamps = []
        if hooks:
            all_timestamps.extend([h["start_ms"] for h in hooks])
        if climaxes:
            all_timestamps.extend([c["start_ms"] for c in climaxes])
        if resolutions:
            all_timestamps.extend([r["start_ms"] for r in resolutions])

        temporal_coherence = 1.0
        if len(all_timestamps) > 1:
            # Check if mostly in correct order
            is_ordered = all_timestamps == sorted(all_timestamps)
            temporal_coherence = 1.0 if is_ordered else 0.7

        # Quality factors
        avg_hook_score = (
            sum(h.get("hook_score", 0) for h in hooks) / len(hooks) if hooks else 0
        )
        avg_climax_score = (
            sum(c.get("climax_score", 0) for c in climaxes) / len(climaxes)
            if climaxes
            else 0
        )
        avg_resolution_score = (
            sum(r.get("resolution_score", 0) for r in resolutions) / len(resolutions)
            if resolutions
            else 0
        )

        quality_coherence = (
            avg_hook_score + avg_climax_score + avg_resolution_score
        ) / 3

        return base_coherence * 0.4 + temporal_coherence * 0.3 + quality_coherence * 0.3

    def _assess_story_completeness(self, structure: Dict) -> float:
        """Assess how complete the story arc is"""

        hooks = structure.get("hooks", [])
        climaxes = structure.get("climaxes", [])
        resolutions = structure.get("resolutions", [])

        # Count present elements
        elements = 0
        if hooks:
            elements += 1
        if climaxes:
            elements += 1
        if resolutions:
            elements += 1

        base_completeness = elements / 3

        # Bonus for multiple quality beats
        quality_bonus = 0
        if len(climaxes) >= 2:  # Multiple climaxes make for richer story
            quality_bonus += 0.1
        if len(hooks) >= 1 and len(resolutions) >= 1:  # Proper bookends
            quality_bonus += 0.1

        return min(1.0, base_completeness + quality_bonus)

    def _recommend_structure(self, structure: Dict) -> Dict:
        """Recommend improvements to narrative structure"""

        recommendations = {
            "missing_elements": [],
            "suggested_improvements": [],
            "optimal_beat_count": {"hooks": 1, "climaxes": 2, "resolutions": 1},
        }

        hooks = structure.get("hooks", [])
        climaxes = structure.get("climaxes", [])
        resolutions = structure.get("resolutions", [])

        # Check missing elements
        if not hooks:
            recommendations["missing_elements"].append("hook")
            recommendations["suggested_improvements"].append(
                "Add attention-grabbing opening"
            )

        if not climaxes:
            recommendations["missing_elements"].append("climax")
            recommendations["suggested_improvements"].append(
                "Include peak moment or revelation"
            )

        if not resolutions:
            recommendations["missing_elements"].append("resolution")
            recommendations["suggested_improvements"].append(
                "Add satisfying conclusion or call-to-action"
            )

        # Quality improvements
        if len(climaxes) > 3:
            recommendations["suggested_improvements"].append(
                "Consider reducing number of climactic moments"
            )

        if len(hooks) > 2:
            recommendations["suggested_improvements"].append(
                "Focus on single strong hook"
            )

        return recommendations


def enhance_highlights_with_narrative_beats(
    highlights: List[Dict], narrative_detection: Dict
) -> List[Dict]:
    """Enhance highlights with narrative beat information"""

    hooks = narrative_detection.get("hooks", [])
    climaxes = narrative_detection.get("climaxes", [])
    resolutions = narrative_detection.get("resolutions", [])

    for highlight in highlights:
        start_ms = highlight.get("start_ms", 0)
        end_ms = highlight.get("end_ms", 0)

        # Check if highlight contains narrative beats
        narrative_beats = []
        beat_score_boost = 0

        # Check for hooks
        for hook in hooks:
            if hook["start_ms"] >= start_ms and hook["end_ms"] <= end_ms:
                narrative_beats.append(("hook", hook.get("hook_score", 0)))
                beat_score_boost += hook.get("hook_score", 0) * 2.0

        # Check for climaxes
        for climax in climaxes:
            if climax["start_ms"] >= start_ms and climax["end_ms"] <= end_ms:
                narrative_beats.append(("climax", climax.get("climax_score", 0)))
                beat_score_boost += climax.get("climax_score", 0) * 2.5

        # Check for resolutions
        for resolution in resolutions:
            if resolution["start_ms"] >= start_ms and resolution["end_ms"] <= end_ms:
                narrative_beats.append(
                    ("resolution", resolution.get("resolution_score", 0))
                )
                beat_score_boost += resolution.get("resolution_score", 0) * 1.8

        # Update highlight
        if narrative_beats:
            highlight["narrative_beats"] = narrative_beats
            highlight["beat_score_boost"] = beat_score_boost
            highlight["score"] = highlight.get("score", 0) + beat_score_boost

            # Determine primary narrative function
            if narrative_beats:
                primary_beat = max(narrative_beats, key=lambda x: x[1])
                highlight["primary_narrative_function"] = primary_beat[0]

        # Add narrative metadata
        highlight["narrative_coherence"] = narrative_detection.get(
            "narrative_coherence", 0
        )
        highlight["story_completeness"] = narrative_detection.get(
            "story_arc_completeness", 0
        )

    return highlights
