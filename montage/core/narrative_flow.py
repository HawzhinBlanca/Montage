#!/usr/bin/env python3
"""
Narrative Flow and Story Beat Detection
Reorders clips for compelling storytelling
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx

logger = logging.getLogger(__name__)


class StoryBeat(Enum):
    """Standard story beat types"""
    HOOK = "hook"
    SETUP = "setup"
    CONTEXT = "context"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    CALL_TO_ACTION = "call_to_action"


@dataclass
class NarrativeSegment:
    """Segment with narrative properties"""
    start: float
    end: float
    text: str
    beat: StoryBeat
    emotional_arc: float  # -1 (negative) to 1 (positive)
    tension_level: float  # 0 (calm) to 1 (intense)
    dependencies: List[int]  # Indices of segments this depends on


class NarrativeFlowOptimizer:
    """Optimize clip order for narrative flow"""
    
    def __init__(self):
        self.beat_sequence = [
            StoryBeat.HOOK,
            StoryBeat.SETUP,
            StoryBeat.CONTEXT,
            StoryBeat.RISING_ACTION,
            StoryBeat.CLIMAX,
            StoryBeat.FALLING_ACTION,
            StoryBeat.RESOLUTION,
            StoryBeat.CALL_TO_ACTION
        ]
        
    def optimize_flow(self, segments: List[Dict]) -> List[Dict]:
        """
        Reorder segments for optimal narrative flow
        
        Args:
            segments: List of segments with narrative metadata
            
        Returns:
            Reordered segments
        """
        # Convert to narrative segments
        narrative_segments = self._analyze_narrative(segments)
        
        # Build dependency graph
        graph = self._build_dependency_graph(narrative_segments)
        
        # Find optimal ordering
        optimal_order = self._find_optimal_order(narrative_segments, graph)
        
        # Reorder original segments
        reordered = [segments[i] for i in optimal_order]
        
        return reordered
        
    def _analyze_narrative(self, segments: List[Dict]) -> List[NarrativeSegment]:
        """Analyze narrative properties of each segment"""
        narrative_segments = []
        
        for i, segment in enumerate(segments):
            # Detect story beat
            beat = self._detect_beat(segment)
            
            # Analyze emotional content
            emotional_arc = self._analyze_emotion(segment.get("text", ""))
            
            # Calculate tension
            tension = self._calculate_tension(segment, i, len(segments))
            
            # Find dependencies
            dependencies = self._find_dependencies(segment, segments[:i])
            
            narrative_segments.append(NarrativeSegment(
                start=segment["start"],
                end=segment["end"],
                text=segment.get("text", ""),
                beat=beat,
                emotional_arc=emotional_arc,
                tension_level=tension,
                dependencies=dependencies
            ))
            
        return narrative_segments
        
    def _detect_beat(self, segment: Dict) -> StoryBeat:
        """Detect the story beat type"""
        text = segment.get("text", "").lower()
        
        # Hook detection
        hook_phrases = ["let me tell you", "you won't believe", "imagine if", "what if"]
        if any(phrase in text for phrase in hook_phrases):
            return StoryBeat.HOOK
            
        # Climax detection
        climax_phrases = ["the moment", "suddenly", "everything changed", "peaked"]
        if any(phrase in text for phrase in climax_phrases):
            return StoryBeat.CLIMAX
            
        # Resolution detection
        resolution_phrases = ["in the end", "finally", "the lesson", "learned"]
        if any(phrase in text for phrase in resolution_phrases):
            return StoryBeat.RESOLUTION
            
        # Call to action
        cta_phrases = ["subscribe", "follow", "share", "comment", "try this"]
        if any(phrase in text for phrase in cta_phrases):
            return StoryBeat.CALL_TO_ACTION
            
        # Default based on position
        position = segment.get("position", 0.5)
        if position < 0.2:
            return StoryBeat.SETUP
        elif position < 0.4:
            return StoryBeat.CONTEXT
        elif position < 0.7:
            return StoryBeat.RISING_ACTION
        else:
            return StoryBeat.FALLING_ACTION
            
    def _analyze_emotion(self, text: str) -> float:
        """Simple emotion analysis"""
        positive_words = ["happy", "amazing", "great", "wonderful", "excited", "love"]
        negative_words = ["sad", "angry", "frustrated", "disappointed", "hate", "terrible"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        return (positive_count - negative_count) / (positive_count + negative_count)
        
    def _calculate_tension(self, segment: Dict, index: int, total: int) -> float:
        """Calculate narrative tension level"""
        # Base tension on position (builds to climax)
        position = index / max(total - 1, 1)
        
        if position < 0.2:
            base_tension = 0.2
        elif position < 0.7:
            base_tension = 0.2 + (position - 0.2) * 1.6  # Build up
        else:
            base_tension = 1.0 - (position - 0.7) * 1.5  # Release
            
        # Adjust based on content
        text = segment.get("text", "").lower()
        
        tension_words = ["but", "however", "suddenly", "unexpected", "shock"]
        if any(word in text for word in tension_words):
            base_tension = min(base_tension + 0.2, 1.0)
            
        return base_tension
        
    def _find_dependencies(self, segment: Dict, previous_segments: List[Dict]) -> List[int]:
        """Find segments this one depends on"""
        dependencies = []
        text = segment.get("text", "").lower()
        
        # Look for references
        reference_words = ["this", "that", "it", "they", "earlier", "mentioned"]
        
        if any(word in text for word in reference_words):
            # Find related previous segments
            for i, prev in enumerate(previous_segments):
                prev_text = prev.get("text", "").lower()
                
                # Simple keyword matching
                keywords = set(text.split()) & set(prev_text.split())
                if len(keywords) > 3:
                    dependencies.append(i)
                    
        return dependencies
        
    def _build_dependency_graph(self, segments: List[NarrativeSegment]) -> nx.DiGraph:
        """Build directed graph of segment dependencies"""
        graph = nx.DiGraph()
        
        for i, segment in enumerate(segments):
            graph.add_node(i, segment=segment)
            
            # Add dependency edges
            for dep in segment.dependencies:
                graph.add_edge(dep, i)
                
        return graph
        
    def _find_optimal_order(self, segments: List[NarrativeSegment], graph: nx.DiGraph) -> List[int]:
        """Find optimal segment ordering"""
        # Check if we can do topological sort (no cycles)
        if nx.is_directed_acyclic_graph(graph):
            # Get all valid topological orderings
            all_orderings = list(nx.all_topological_sorts(graph))
            
            # Score each ordering
            best_score = -float('inf')
            best_order = list(range(len(segments)))
            
            for ordering in all_orderings[:100]:  # Limit for performance
                score = self._score_ordering(ordering, segments)
                if score > best_score:
                    best_score = score
                    best_order = ordering
                    
            return list(best_order)
        else:
            # Fallback: order by story beats
            return self._order_by_beats(segments)
            
    def _score_ordering(self, ordering: List[int], segments: List[NarrativeSegment]) -> float:
        """Score a segment ordering"""
        score = 0.0
        
        # Reward following ideal beat sequence
        beat_positions = {}
        for i, idx in enumerate(ordering):
            beat = segments[idx].beat
            if beat not in beat_positions:
                beat_positions[beat] = i
                
        # Check beat order
        for i, beat in enumerate(self.beat_sequence[:-1]):
            next_beat = self.beat_sequence[i + 1]
            if beat in beat_positions and next_beat in beat_positions:
                if beat_positions[beat] < beat_positions[next_beat]:
                    score += 10.0
                    
        # Reward smooth emotional transitions
        for i in range(len(ordering) - 1):
            curr = segments[ordering[i]]
            next_seg = segments[ordering[i + 1]]
            
            emotional_diff = abs(curr.emotional_arc - next_seg.emotional_arc)
            score -= emotional_diff * 2.0
            
        # Reward proper tension arc
        max_tension_pos = max(range(len(ordering)), 
                            key=lambda i: segments[ordering[i]].tension_level)
        ideal_climax_pos = len(ordering) * 0.7
        score -= abs(max_tension_pos - ideal_climax_pos) * 0.5
        
        return score
        
    def _order_by_beats(self, segments: List[NarrativeSegment]) -> List[int]:
        """Simple ordering by story beat type"""
        # Group by beat type
        beat_groups = {}
        for i, segment in enumerate(segments):
            if segment.beat not in beat_groups:
                beat_groups[segment.beat] = []
            beat_groups[segment.beat].append(i)
            
        # Order according to ideal sequence
        ordered = []
        for beat in self.beat_sequence:
            if beat in beat_groups:
                # Sort within beat by timestamp
                beat_indices = beat_groups[beat]
                beat_indices.sort(key=lambda i: segments[i].start)
                ordered.extend(beat_indices)
                
        # Add any remaining
        all_indices = set(range(len(segments)))
        remaining = all_indices - set(ordered)
        ordered.extend(sorted(remaining, key=lambda i: segments[i].start))
        
        return ordered


class EmotionalArcAnalyzer:
    """Analyze and optimize emotional arc of story"""
    
    def __init__(self):
        self.target_arc_types = {
            "man_in_hole": [0, -0.5, -0.8, -0.5, 0.3, 0.6],  # Fall then rise
            "cinderella": [0, 0.3, -0.6, -0.8, 0.5, 1.0],    # Rise-fall-rise
            "tragedy": [0, 0.3, 0.5, 0.2, -0.5, -0.8],       # Fall
            "comedy": [-0.3, -0.5, 0, 0.3, 0.6, 0.8],        # Rise
        }
        
    def optimize_emotional_arc(self, segments: List[NarrativeSegment], 
                              arc_type: str = "cinderella") -> List[int]:
        """Reorder segments to match target emotional arc"""
        target_arc = self.target_arc_types.get(arc_type, self.target_arc_types["cinderella"])
        
        # Get current emotional values
        current_emotions = [s.emotional_arc for s in segments]
        
        # Find best mapping to target arc
        from itertools import permutations
        
        # For performance, only try a subset of permutations
        if len(segments) <= 8:
            perms = permutations(range(len(segments)))
        else:
            # Use greedy approach for larger sets
            return self._greedy_arc_matching(segments, target_arc)
            
        best_order = list(range(len(segments)))
        best_score = float('inf')
        
        for perm in perms:
            score = self._arc_distance(
                [segments[i].emotional_arc for i in perm],
                target_arc
            )
            if score < best_score:
                best_score = score
                best_order = list(perm)
                
        return best_order
        
    def _arc_distance(self, emotions: List[float], target: List[float]) -> float:
        """Calculate distance between emotional arcs"""
        # Interpolate to same length
        from numpy import interp, linspace
        
        x_current = linspace(0, 1, len(emotions))
        x_target = linspace(0, 1, len(target))
        
        # Resample current to match target length
        resampled = interp(x_target, x_current, emotions)
        
        # Calculate MSE
        return sum((a - b) ** 2 for a, b in zip(resampled, target)) / len(target)
        
    def _greedy_arc_matching(self, segments: List[NarrativeSegment], 
                            target_arc: List[float]) -> List[int]:
        """Greedy approach for matching emotional arc"""
        remaining = list(range(len(segments)))
        ordered = []
        
        for target_emotion in target_arc:
            if not remaining:
                break
                
            # Find closest match
            best_idx = min(remaining, 
                         key=lambda i: abs(segments[i].emotional_arc - target_emotion))
            ordered.append(best_idx)
            remaining.remove(best_idx)
            
        # Add any remaining
        ordered.extend(remaining)
        
        return ordered