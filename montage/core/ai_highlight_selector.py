#!/usr/bin/env python3
"""
AI-Powered Highlight Selection using Claude and Gemini
Intelligent content analysis and highlight scoring
"""
import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)


@dataclass
class HighlightSegment:
    """Represents a highlight segment"""
    start: float
    end: float
    score: float
    title: str
    reason: str
    story_beat: str  # hook, context, climax, resolution
    emotions: List[str]
    keywords: List[str]


class AIHighlightSelector:
    """AI-powered highlight selection using Claude and Gemini"""
    
    def __init__(self):
        """Initialize AI models"""
        self.claude_client = None
        self.gemini_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize Claude and Gemini"""
        # Initialize Claude
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        if claude_key:
            self.claude_client = anthropic.Anthropic(api_key=claude_key)
            logger.info("Claude API initialized")
        else:
            logger.warning("ANTHROPIC_API_KEY not set")  # pragma: no cover
            
        # Initialize Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Gemini API initialized")
        else:
            logger.warning("GEMINI_API_KEY not set")  # pragma: no cover
            
    async def select_highlights(
        self,
        transcript: Dict[str, Any],
        video_duration: float,
        target_clips: int = 5
    ) -> List[HighlightSegment]:
        """
        Select highlight segments using AI analysis
        
        Args:
            transcript: Full transcript with timestamps
            video_duration: Total video duration
            target_clips: Number of highlights to generate
            
        Returns:
            List of highlight segments
        """
        # Analyze with both models for best results
        claude_analysis = await self._analyze_with_claude(transcript)
        gemini_analysis = await self._analyze_with_gemini(transcript)
        
        # Combine insights
        combined_segments = self._combine_analyses(claude_analysis, gemini_analysis)
        
        # Score and rank segments
        scored_segments = self._score_segments(combined_segments, transcript)
        
        # Select top highlights
        highlights = self._select_top_highlights(scored_segments, target_clips)
        
        return highlights
        
    async def _analyze_with_claude(self, transcript: Dict) -> Dict[str, Any]:
        """Analyze transcript with Claude"""
        if not self.claude_client:
            return {"segments": []}  # pragma: no cover
            
        prompt = f"""Analyze this transcript and identify the most engaging moments for social media clips.
        
        Transcript: {json.dumps(transcript['segments'][:50])}  # Limit for token count
        
        For each highlight, provide:
        1. Start and end timestamps
        2. A catchy title (8 words max)
        3. Story beat type (hook/context/climax/resolution)
        4. Key emotions present
        5. Viral potential score (0-1)
        
        Return as JSON with a 'highlights' array."""
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            content = response.content[0].text
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:  # pragma: no cover
            logger.error(f"Claude analysis failed: {e}")
            
        return {"segments": []}
        
    async def _analyze_with_gemini(self, transcript: Dict) -> Dict[str, Any]:
        """Analyze transcript with Gemini"""
        if not self.gemini_model:
            return {"segments": []}  # pragma: no cover
            
        prompt = f"""You are a viral content expert. Analyze this transcript and find moments that would make great social media clips.

        Transcript: {json.dumps(transcript['segments'][:50])}
        
        Identify 5-8 segments that have:
        - High emotional impact
        - Memorable quotes
        - Story turning points
        - Surprising revelations
        - Humorous moments
        
        For each segment provide:
        - start_time and end_time
        - viral_score (0-1)
        - title (catchy, under 8 words)
        - hashtags (3-5 relevant ones)
        - emoji (1-3 that capture the mood)
        
        Format as JSON."""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
                
        except Exception as e:  # pragma: no cover
            logger.error(f"Gemini analysis failed: {e}")
            
        return {"segments": []}
        
    def _combine_analyses(self, claude_analysis: Dict, gemini_analysis: Dict) -> List[Dict]:
        """Combine insights from both AI models"""
        segments = []
        
        # Process Claude results
        for highlight in claude_analysis.get("highlights", []):
            segments.append({
                "start": highlight.get("start", 0),
                "end": highlight.get("end", 0),
                "title": highlight.get("title", ""),
                "story_beat": highlight.get("story_beat", "context"),
                "emotions": highlight.get("emotions", []),
                "score": highlight.get("viral_potential", 0.5),
                "source": "claude"
            })
            
        # Process Gemini results
        for segment in gemini_analysis.get("segments", []):
            segments.append({
                "start": segment.get("start_time", 0),
                "end": segment.get("end_time", 0),
                "title": segment.get("title", ""),
                "hashtags": segment.get("hashtags", []),
                "emoji": segment.get("emoji", []),
                "score": segment.get("viral_score", 0.5),
                "source": "gemini"
            })
            
        return segments
        
    def _score_segments(self, segments: List[Dict], transcript: Dict) -> List[HighlightSegment]:
        """Score and enhance segments with additional analysis"""
        scored = []
        
        for segment in segments:
            # Calculate comprehensive score
            base_score = segment.get("score", 0.5)
            
            # Boost score based on features
            if segment.get("story_beat") == "hook":
                base_score += 0.2
            elif segment.get("story_beat") == "climax":
                base_score += 0.15
                
            # Check for engagement indicators in text
            text = self._get_segment_text(segment, transcript)
            
            if any(word in text.lower() for word in ["amazing", "incredible", "unbelievable"]):
                base_score += 0.1
                
            if "?" in text:  # Questions engage viewers
                base_score += 0.05
                
            # Create highlight segment
            highlight = HighlightSegment(
                start=segment["start"],
                end=segment["end"],
                score=min(base_score, 1.0),
                title=segment.get("title", "Highlight"),
                reason=f"High engagement potential from {segment.get('source', 'AI')}",
                story_beat=segment.get("story_beat", "context"),
                emotions=segment.get("emotions", []),
                keywords=self._extract_keywords(text)
            )
            
            scored.append(highlight)
            
        return scored
        
    def _get_segment_text(self, segment: Dict, transcript: Dict) -> str:
        """Extract text for a time segment"""
        text_parts = []
        
        for trans_segment in transcript.get("segments", []):
            if (trans_segment["start"] >= segment["start"] and 
                trans_segment["end"] <= segment["end"]):
                text_parts.append(trans_segment["text"])
                
        return " ".join(text_parts)
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key words from text"""
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter common words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Return top 5 most common
        from collections import Counter
        return [word for word, _ in Counter(keywords).most_common(5)]
        
    def _select_top_highlights(self, segments: List[HighlightSegment], target_count: int) -> List[HighlightSegment]:
        """Select the best highlights ensuring good story flow"""
        # Sort by score
        segments.sort(key=lambda x: x.score, reverse=True)
        
        # Ensure we have different story beats
        selected = []
        beats_included = set()
        
        # First pass: get best from each story beat
        for segment in segments:
            if len(selected) >= target_count:
                break
                
            if segment.story_beat not in beats_included:
                selected.append(segment)
                beats_included.add(segment.story_beat)
                
        # Second pass: fill remaining with highest scores
        for segment in segments:
            if len(selected) >= target_count:
                break
                
            if segment not in selected:
                selected.append(segment)
                
        # Sort by timestamp for natural flow
        selected.sort(key=lambda x: x.start)
        
        return selected


class StoryBeatDetector:
    """Detect narrative story beats in content"""
    
    def __init__(self):
        self.beat_patterns = {
            "hook": ["let me tell you", "you won't believe", "this changed everything", "the moment when"],
            "context": ["background", "started when", "originally", "at the time"],
            "climax": ["suddenly", "then everything changed", "the turning point", "peaked when"],
            "resolution": ["in the end", "finally", "the lesson", "what I learned"]
        }
        
    def detect_beats(self, transcript: Dict) -> List[Tuple[str, float, float]]:
        """
        Detect story beats in transcript
        
        Returns:
            List of (beat_type, start_time, end_time)
        """
        beats = []
        
        for segment in transcript.get("segments", []):
            text = segment["text"].lower()
            
            for beat_type, patterns in self.beat_patterns.items():
                if any(pattern in text for pattern in patterns):
                    beats.append((
                        beat_type,
                        segment["start"],
                        segment["end"]
                    ))
                    
        return beats