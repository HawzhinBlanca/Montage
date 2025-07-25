#!/usr/bin/env python3
"""
Speaker/Character analysis for story continuity and narrative coherence
"""
import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class SpeakerAnalyzer:
    """Analyzes speakers/characters for story continuity"""

    def __init__(self):
        self.speaker_profiles = {}
        self.relationship_graph = defaultdict(set)
        self.dialogue_patterns = {}

    def analyze_speakers(self, transcript_segments: List[Dict]) -> Dict:
        """Comprehensive speaker analysis for story continuity"""

        # Extract speaker information
        speakers = self._identify_speakers(transcript_segments)

        # Analyze speaking patterns and relationships
        speaker_relationships = self._analyze_relationships(transcript_segments)

        # Identify character roles in narrative
        character_roles = self._identify_character_roles(transcript_segments, speakers)

        # Detect conversation flows and dynamics
        conversation_flows = self._analyze_conversation_flows(transcript_segments)

        # Find narrative anchors (key speakers)
        narrative_anchors = self._find_narrative_anchors(speakers, transcript_segments)

        return {
            "speakers": speakers,
            "relationships": speaker_relationships,
            "character_roles": character_roles,
            "conversation_flows": conversation_flows,
            "narrative_anchors": narrative_anchors,
            "continuity_score": self._calculate_continuity_score(
                speakers, conversation_flows
            ),
        }

    def _identify_speakers(self, segments: List[Dict]) -> Dict[str, Dict]:
        """Identify and profile each speaker"""

        speakers = defaultdict(
            lambda: {
                "total_words": 0,
                "segments": [],
                "speaking_time": 0,
                "energy_levels": [],
                "topics": [],
                "sentiment": "neutral",
                "role_indicators": [],
                "first_appearance": float("inf"),
                "last_appearance": 0,
            }
        )

        for segment in segments:
            speaker_id = segment.get("speaker", "unknown")
            text = segment.get("text", "")
            start_ms = segment.get("start_ms", 0)
            end_ms = segment.get("end_ms", 0)
            energy = segment.get("energy_level", 0.5)

            # Update speaker profile
            profile = speakers[speaker_id]
            profile["total_words"] += len(text.split())
            profile["segments"].append(segment)
            profile["speaking_time"] += end_ms - start_ms
            profile["energy_levels"].append(energy)
            profile["first_appearance"] = min(profile["first_appearance"], start_ms)
            profile["last_appearance"] = max(profile["last_appearance"], end_ms)

            # Analyze content
            profile["topics"].extend(self._extract_topics(text))
            profile["role_indicators"].extend(self._detect_role_indicators(text))

        # Calculate average energy and sentiment for each speaker
        for speaker_id, profile in speakers.items():
            if profile["energy_levels"]:
                profile["avg_energy"] = sum(profile["energy_levels"]) / len(
                    profile["energy_levels"]
                )
            else:
                profile["avg_energy"] = 0.5

            profile["sentiment"] = self._analyze_speaker_sentiment(profile["segments"])
            profile["dominance"] = self._calculate_speaker_dominance(profile, segments)
            profile["consistency"] = self._calculate_speaker_consistency(profile)

        return dict(speakers)

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics/themes from speaker text"""

        # Topic keywords for different domains
        topic_patterns = {
            "health": r"\b(health|diet|exercise|nutrition|medical|doctor|therapy|wellness)\b",
            "business": r"\b(business|company|market|sales|revenue|profit|strategy|growth)\b",
            "science": r"\b(research|study|data|analysis|discovery|science|experiment|findings)\b",
            "technology": r"\b(technology|digital|software|ai|algorithm|innovation|platform)\b",
            "education": r"\b(learn|teach|education|knowledge|skill|training|development)\b",
            "personal": r"\b(life|personal|experience|story|journey|challenge|success)\b",
            "finance": r"\b(money|financial|investment|cost|price|budget|economic)\b",
            "relationships": r"\b(family|friend|relationship|love|marriage|partner|community)\b",
        }

        topics = []
        text_lower = text.lower()

        for topic, pattern in topic_patterns.items():
            if re.search(pattern, text_lower):
                topics.append(topic)

        return topics

    def _detect_role_indicators(self, text: str) -> List[str]:
        """Detect speaker role indicators from their language"""

        role_patterns = {
            "expert": r"\b(research shows|studies indicate|data suggests|evidence points|scientifically)\b",
            "host": r"\b(welcome|today we|let me ask|that\'s interesting|tell us about)\b",
            "guest": r"\b(thank you for having me|great question|in my experience|what I\'ve found)\b",
            "interviewer": r"\b(how do you|what about|can you explain|would you say)\b",
            "storyteller": r"\b(once upon|happened to me|I remember when|let me tell you)\b",
            "educator": r"\b(the key is|important to understand|fundamental principle|basic concept)\b",
            "motivator": r"\b(you can do|believe in yourself|never give up|achieve your goals)\b",
        }

        roles = []
        text_lower = text.lower()

        for role, pattern in role_patterns.items():
            if re.search(pattern, text_lower):
                roles.append(role)

        return roles

    def _analyze_relationships(self, segments: List[Dict]) -> Dict:
        """Analyze relationships and interactions between speakers"""

        relationships = defaultdict(
            lambda: {
                "interaction_count": 0,
                "topics_discussed": set(),
                "energy_correlation": 0,
                "response_patterns": [],
                "relationship_type": "neutral",
            }
        )

        # Analyze speaker transitions
        for i in range(len(segments) - 1):
            current = segments[i]
            next_segment = segments[i + 1]

            current_speaker = current.get("speaker", "unknown")
            next_speaker = next_segment.get("speaker", "unknown")

            if (
                current_speaker != next_speaker
                and current_speaker != "unknown"
                and next_speaker != "unknown"
            ):
                # Record interaction
                interaction_key = tuple(sorted([current_speaker, next_speaker]))
                rel = relationships[interaction_key]

                rel["interaction_count"] += 1
                rel["topics_discussed"].update(
                    self._extract_topics(current.get("text", ""))
                )
                rel["response_patterns"].append(
                    {
                        "from_speaker": current_speaker,
                        "to_speaker": next_speaker,
                        "time_gap": next_segment.get("start_ms", 0)
                        - current.get("end_ms", 0),
                        "energy_change": next_segment.get("energy_level", 0.5)
                        - current.get("energy_level", 0.5),
                    }
                )

        # Analyze relationship types
        for interaction_key, rel_data in relationships.items():
            rel_data["relationship_type"] = self._classify_relationship(rel_data)
            rel_data["topics_discussed"] = list(rel_data["topics_discussed"])

        return dict(relationships)

    def _classify_relationship(self, rel_data: Dict) -> str:
        """Classify the type of relationship between speakers"""

        interaction_count = rel_data["interaction_count"]
        response_patterns = rel_data["response_patterns"]

        if interaction_count < 2:
            return "minimal"

        # Analyze response patterns
        avg_time_gap = (
            sum(rp["time_gap"] for rp in response_patterns) / len(response_patterns)
            if response_patterns
            else 0
        )
        energy_changes = [rp["energy_change"] for rp in response_patterns]
        avg_energy_change = (
            sum(energy_changes) / len(energy_changes) if energy_changes else 0
        )

        # Classify based on patterns
        if avg_time_gap < 1000:  # Quick responses
            if avg_energy_change > 0.1:
                return "dynamic_conversation"
            else:
                return "collaborative_discussion"
        elif interaction_count > 5:
            return "structured_interview"
        else:
            return "casual_exchange"

    def _identify_character_roles(
        self, segments: List[Dict], speakers: Dict
    ) -> Dict[str, str]:
        """Identify narrative roles of each character/speaker"""

        character_roles = {}

        for speaker_id, profile in speakers.items():
            role_indicators = Counter(profile["role_indicators"])
            total_words = profile["total_words"]
            speaking_time = profile["speaking_time"]
            dominance = profile.get("dominance", 0)

            # Determine primary role
            if role_indicators:
                primary_role = role_indicators.most_common(1)[0][0]
            else:
                # Fallback based on speaking patterns
                if dominance > 0.6:
                    primary_role = "primary_speaker"
                elif speaking_time > 60000:  # More than 1 minute
                    primary_role = "main_character"
                else:
                    primary_role = "supporting_character"

            # Refine role based on additional context
            if "host" in role_indicators or "interviewer" in role_indicators:
                if dominance > 0.4:
                    primary_role = "host"
                else:
                    primary_role = "interviewer"
            elif "expert" in role_indicators and total_words > 500:
                primary_role = "expert"
            elif "guest" in role_indicators:
                primary_role = "guest"

            character_roles[speaker_id] = primary_role

        return character_roles

    def _analyze_conversation_flows(self, segments: List[Dict]) -> List[Dict]:
        """Analyze conversation flows and dynamics"""

        flows = []
        current_flow = None

        for i, segment in enumerate(segments):
            speaker = segment.get("speaker", "unknown")

            if current_flow is None or current_flow["speakers"][-1] != speaker:
                # New flow or speaker change
                if current_flow:
                    flows.append(current_flow)

                current_flow = {
                    "start_ms": segment.get("start_ms", 0),
                    "end_ms": segment.get("end_ms", 0),
                    "speakers": [speaker],
                    "segments": [segment],
                    "topic": self._extract_primary_topic(segment.get("text", "")),
                    "energy_arc": [segment.get("energy_level", 0.5)],
                }
            else:
                # Continue current flow
                current_flow["end_ms"] = segment.get("end_ms", 0)
                current_flow["segments"].append(segment)
                current_flow["energy_arc"].append(segment.get("energy_level", 0.5))

        if current_flow:
            flows.append(current_flow)

        # Analyze flow characteristics
        for flow in flows:
            flow["duration"] = flow["end_ms"] - flow["start_ms"]
            flow["avg_energy"] = (
                sum(flow["energy_arc"]) / len(flow["energy_arc"])
                if flow["energy_arc"]
                else 0.5
            )
            flow["energy_trend"] = self._calculate_energy_trend(flow["energy_arc"])
            flow["flow_type"] = self._classify_flow_type(flow)

        return flows

    def _extract_primary_topic(self, text: str) -> str:
        """Extract the primary topic from text"""
        topics = self._extract_topics(text)
        return topics[0] if topics else "general"

    def _calculate_energy_trend(self, energy_arc: List[float]) -> str:
        """Calculate energy trend over time"""
        if len(energy_arc) < 2:
            return "stable"

        start_avg = (
            sum(energy_arc[: len(energy_arc) // 3]) / (len(energy_arc) // 3)
            if len(energy_arc) >= 3
            else energy_arc[0]
        )
        end_avg = (
            sum(energy_arc[-len(energy_arc) // 3 :]) / (len(energy_arc) // 3)
            if len(energy_arc) >= 3
            else energy_arc[-1]
        )

        diff = end_avg - start_avg

        if diff > 0.1:
            return "rising"
        elif diff < -0.1:
            return "falling"
        else:
            return "stable"

    def _classify_flow_type(self, flow: Dict) -> str:
        """Classify the type of conversation flow"""
        duration = flow["duration"]
        avg_energy = flow["avg_energy"]
        energy_trend = flow["energy_trend"]

        if duration < 10000:  # Less than 10 seconds
            return "brief_exchange"
        elif duration > 60000:  # More than 1 minute
            if avg_energy > 0.7:
                return "passionate_monologue"
            else:
                return "detailed_explanation"
        elif energy_trend == "rising":
            return "building_excitement"
        elif energy_trend == "falling":
            return "conclusion"
        else:
            return "steady_discussion"

    def _find_narrative_anchors(
        self, speakers: Dict, segments: List[Dict]
    ) -> List[str]:
        """Find speakers who serve as narrative anchors"""

        anchors = []

        for speaker_id, profile in speakers.items():
            dominance = profile.get("dominance", 0)
            consistency = profile.get("consistency", 0)
            speaking_time = profile["speaking_time"]

            # Anchor criteria
            if (
                dominance > 0.3 and consistency > 0.6 and speaking_time > 30000
            ) or dominance > 0.6:
                anchors.append(speaker_id)

        # Sort by dominance
        anchors.sort(key=lambda s: speakers[s].get("dominance", 0), reverse=True)

        return anchors[:3]  # Top 3 anchors

    def _analyze_speaker_sentiment(self, segments: List[Dict]) -> str:
        """Analyze overall sentiment of a speaker"""

        positive_words = {
            "great",
            "amazing",
            "excellent",
            "wonderful",
            "fantastic",
            "love",
            "best",
            "perfect",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "hate",
            "worst",
            "problem",
            "issue",
            "difficult",
        }

        positive_count = 0
        negative_count = 0

        for segment in segments:
            text_lower = segment.get("text", "").lower()
            positive_count += sum(1 for word in positive_words if word in text_lower)
            negative_count += sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count * 1.5:
            return "positive"
        elif negative_count > positive_count * 1.5:
            return "negative"
        else:
            return "neutral"

    def _calculate_speaker_dominance(
        self, profile: Dict, all_segments: List[Dict]
    ) -> float:
        """Calculate how dominant a speaker is in the conversation"""

        total_speaking_time = sum(
            seg.get("end_ms", 0) - seg.get("start_ms", 0) for seg in all_segments
        )
        speaker_time = profile["speaking_time"]

        if total_speaking_time == 0:
            return 0

        return speaker_time / total_speaking_time

    def _calculate_speaker_consistency(self, profile: Dict) -> float:
        """Calculate how consistent a speaker's energy and content is"""

        energy_levels = profile["energy_levels"]
        if len(energy_levels) < 2:
            return 1.0

        # Energy consistency (lower variance = higher consistency)
        avg_energy = sum(energy_levels) / len(energy_levels)
        energy_variance = sum((e - avg_energy) ** 2 for e in energy_levels) / len(
            energy_levels
        )
        energy_consistency = max(0, 1 - energy_variance)

        # Topic consistency (fewer topics = higher consistency)
        unique_topics = len(set(profile["topics"]))
        topic_consistency = max(0, 1 - (unique_topics / 10))  # Normalize to 0-1

        return (energy_consistency + topic_consistency) / 2

    def _calculate_continuity_score(self, speakers: Dict, flows: List[Dict]) -> float:
        """Calculate overall narrative continuity score"""

        if not speakers or not flows:
            return 0.0

        # Factors for continuity
        speaker_count = len(speakers)
        avg_consistency = sum(s.get("consistency", 0) for s in speakers.values()) / len(
            speakers
        )
        flow_transitions = len(
            [f for f in flows if f["duration"] > 5000]
        )  # Meaningful flows

        # Calculate score (0-1)
        speaker_score = (
            min(1.0, 2 / speaker_count) if speaker_count > 0 else 0
        )  # Fewer speakers = better continuity
        consistency_score = avg_consistency
        flow_score = min(1.0, flow_transitions / 10)  # Reasonable number of flows

        return (speaker_score + consistency_score + flow_score) / 3


def enhance_highlights_with_speaker_continuity(
    highlights: List[Dict], speaker_analysis: Dict
) -> List[Dict]:
    """Enhance highlights with speaker continuity information"""

    speakers = speaker_analysis.get("speakers", {})
    character_roles = speaker_analysis.get("character_roles", {})
    narrative_anchors = speaker_analysis.get("narrative_anchors", [])

    for highlight in highlights:
        speaker_id = highlight.get("speaker", "unknown")

        if speaker_id in speakers:
            speaker_profile = speakers[speaker_id]

            # Add speaker metadata
            highlight.update(
                {
                    "character_role": character_roles.get(speaker_id, "unknown"),
                    "speaker_dominance": speaker_profile.get("dominance", 0),
                    "speaker_consistency": speaker_profile.get("consistency", 0),
                    "is_narrative_anchor": speaker_id in narrative_anchors,
                    "speaker_sentiment": speaker_profile.get("sentiment", "neutral"),
                    "speaker_avg_energy": speaker_profile.get("avg_energy", 0.5),
                }
            )

            # Boost score for narrative anchors
            if speaker_id in narrative_anchors:
                highlight["score"] = highlight.get("score", 0) + 2.0
                highlight["continuity_boost"] = 2.0

            # Boost for consistent speakers
            consistency_boost = speaker_profile.get("consistency", 0) * 1.5
            highlight["score"] = highlight.get("score", 0) + consistency_boost
            highlight["consistency_boost"] = consistency_boost

    return highlights
