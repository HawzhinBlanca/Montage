#!/usr/bin/env python3
"""
Phase 1.2: Local highlight scorer (Smart Track)
Token-free rule: score = complete-sentence (2 pt) + keyword (3 pt) + normalized audio RMS (0 – 2 pt)
Top 5 segments → JSON {slug,start_ms,end_ms}
"""

import os
import sys
import json
import re
import logging
import time
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HighlightSegment:
    """Scored highlight segment"""
    slug: str
    start_ms: int
    end_ms: int
    text: str
    score: float
    complete_sentence_score: float
    keyword_score: float
    audio_rms_score: float
    confidence: float

class LocalHighlightScorer:
    """Smart Track - Local highlight scorer with token-free rules"""
    
    def __init__(self):
        self.highlight_keywords = [
            # Science & Research
            "scientist", "research", "study", "discovery", "breakthrough", "innovation",
            "experiment", "data", "analysis", "findings", "evidence", "conclusion",
            
            # Health & Medicine
            "health", "medicine", "doctor", "patient", "treatment", "therapy", "cure",
            "disease", "diagnosis", "symptoms", "recovery", "prevention", "wellness",
            
            # Achievement & Success
            "success", "achievement", "accomplishment", "victory", "win", "champion",
            "excellence", "outstanding", "remarkable", "incredible", "amazing", "brilliant",
            
            # Names & Titles
            "professor", "doctor", "minister", "president", "director", "expert",
            "specialist", "authority", "leader", "pioneer", "founder",
            
            # Strong Emotions
            "excited", "thrilled", "passionate", "enthusiastic", "inspiring", "motivated",
            "proud", "honored", "grateful", "delighted", "impressed", "fascinated",
            
            # Impact & Importance
            "important", "significant", "crucial", "essential", "vital", "critical",
            "revolutionary", "groundbreaking", "transformative", "game-changing",
            
            # Superlatives
            "best", "greatest", "most", "top", "leading", "premier", "ultimate",
            "exceptional", "extraordinary", "unprecedented", "unparalleled",
            
            # Book & Media
            "book", "author", "published", "written", "story", "chapter", "content",
            "podcast", "interview", "program", "show", "episode"
        ]
        
        # Sentence endings that indicate complete sentences
        self.sentence_endings = ['.', '!', '?']
        
    def is_complete_sentence(self, text: str) -> bool:
        """Check if text contains a complete sentence"""
        text = text.strip()
        
        # Split text into potential sentences by sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Check if any sentence in the text is complete
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty or very short sentences
            if len(sentence.split()) < 3:
                continue
            
            # Check for subject-verb structure (basic heuristic)
            words = sentence.lower().split()
            has_subject_verb = False
            
            # Look for pronoun + verb patterns
            pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'who', 'what', 'which']
            verbs = ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'needs', 'wants']
            
            for i, word in enumerate(words[:-1]):
                if word in pronouns and words[i+1] in verbs:
                    has_subject_verb = True
                    break
            
            # Also check for noun + verb patterns (simplified)
            if not has_subject_verb:
                for i, word in enumerate(words[:-1]):
                    if len(word) > 2 and words[i+1] in verbs:
                        has_subject_verb = True
                        break
            
            # If we found a complete sentence, return True
            if has_subject_verb:
                return True
        
        # Also check if the entire text forms a complete thought
        # Even without perfect punctuation, if it has good structure and length
        if len(text.split()) >= 5:
            words = text.lower().split()
            
            # Check for basic sentence structure
            pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'who', 'what', 'which']
            verbs = ['am', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did',
                    'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'needs', 'wants']
            
            for i, word in enumerate(words[:-1]):
                if word in pronouns and words[i+1] in verbs:
                    return True
                if len(word) > 2 and words[i+1] in verbs:
                    return True
        
        return False
    
    def count_keywords(self, text: str) -> int:
        """Count highlight keywords in text"""
        text_lower = text.lower()
        keyword_count = 0
        
        for keyword in self.highlight_keywords:
            # Count occurrences of each keyword
            keyword_count += text_lower.count(keyword.lower())
        
        return keyword_count
    
    def estimate_audio_rms(self, audio_path: str, start_time: float, duration: float) -> float:
        """Estimate audio RMS for a segment (0-2 points)"""
        try:
            # Extract audio segment
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-af', 'volumedetect',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            # Parse volume detection output
            output = result.stderr
            
            # Look for mean volume in dB
            mean_volume_match = re.search(r'mean_volume:\s*(-?\d+\.?\d*)\s*dB', output)
            if mean_volume_match:
                mean_volume_db = float(mean_volume_match.group(1))
                
                # Convert dB to normalized score (0-2)
                # -60 dB = silence (0 points)
                # -20 dB = normal speech (1 point)  
                # -10 dB = loud speech (2 points)
                if mean_volume_db <= -60:
                    return 0.0
                elif mean_volume_db >= -10:
                    return 2.0
                else:
                    # Linear interpolation between -60 and -10 dB
                    return ((mean_volume_db + 60) / 50) * 2
            
        except Exception as e:
            logger.warning(f"Could not estimate audio RMS: {e}")
        
        # Default to moderate score if analysis fails
        return 1.0
    
    def create_slug(self, text: str, max_length: int = 30) -> str:
        """Create a URL-friendly slug from text"""
        # Remove punctuation and convert to lowercase
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        
        # Replace spaces with hyphens
        slug = re.sub(r'\s+', '-', slug)
        
        # Remove multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        
        # Truncate to max length
        if len(slug) > max_length:
            slug = slug[:max_length].rstrip('-')
        
        return slug
    
    def score_segments(self, transcript_data: Dict[str, Any], audio_path: Optional[str] = None) -> List[HighlightSegment]:
        """Score all segments using token-free rules"""
        logger.info("🎯 Starting local highlight scoring...")
        
        segments = transcript_data.get('segments', [])
        scored_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            duration = end_time - start_time
            confidence = segment.get('confidence', 0)
            
            if not text or duration <= 0:
                continue
            
            # Rule 1: Complete sentence score (2 points)
            complete_sentence_score = 2.0 if self.is_complete_sentence(text) else 0.0
            
            # Rule 2: Keyword score (3 points per keyword, max 3 points)
            keyword_count = self.count_keywords(text)
            keyword_score = min(keyword_count * 3.0, 3.0)
            
            # Rule 3: Audio RMS score (0-2 points)
            if audio_path and os.path.exists(audio_path):
                audio_rms_score = self.estimate_audio_rms(audio_path, start_time, duration)
            else:
                # Default scoring based on confidence if no audio
                audio_rms_score = min(abs(confidence) * 2, 2.0) if confidence < 0 else min(confidence * 2, 2.0)
            
            # Total score
            total_score = complete_sentence_score + keyword_score + audio_rms_score
            
            # Create highlight segment
            highlight = HighlightSegment(
                slug=self.create_slug(text),
                start_ms=int(start_time * 1000),
                end_ms=int(end_time * 1000),
                text=text,
                score=total_score,
                complete_sentence_score=complete_sentence_score,
                keyword_score=keyword_score,
                audio_rms_score=audio_rms_score,
                confidence=abs(confidence) if confidence else 0.0
            )
            
            scored_segments.append(highlight)
            
            logger.info(f"Segment {i+1}: {total_score:.1f} pts (sent:{complete_sentence_score:.1f} + key:{keyword_score:.1f} + rms:{audio_rms_score:.1f})")
        
        # Sort by score (descending)
        scored_segments.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"✅ Scored {len(scored_segments)} segments")
        return scored_segments
    
    def get_top_highlights(self, scored_segments: List[HighlightSegment], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top N highlight segments in JSON format"""
        top_segments = scored_segments[:top_n]
        
        results = []
        for segment in top_segments:
            results.append({
                "slug": segment.slug,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
                "score": segment.score,
                "breakdown": {
                    "complete_sentence": segment.complete_sentence_score,
                    "keywords": segment.keyword_score,
                    "audio_rms": segment.audio_rms_score
                },
                "confidence": segment.confidence
            })
        
        return results

async def main():
    """Test local highlight scorer"""
    if len(sys.argv) < 2:
        print("Usage: python phase1_2_local_highlight_scorer.py <transcript_json_path> [audio_path]")
        return
    
    transcript_path = sys.argv[1]
    audio_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load transcript data
    if not os.path.exists(transcript_path):
        print(f"❌ Transcript file not found: {transcript_path}")
        return
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Initialize scorer
    scorer = LocalHighlightScorer()
    
    # Score segments
    scored_segments = scorer.score_segments(transcript_data, audio_path)
    
    # Get top highlights
    top_highlights = scorer.get_top_highlights(scored_segments, top_n=5)
    
    # Print results
    print("\n" + "=" * 60)
    print("🎯 LOCAL HIGHLIGHT SCORER RESULTS")
    print("=" * 60)
    
    for i, highlight in enumerate(top_highlights, 1):
        print(f"\n{i}. {highlight['slug']} ({highlight['score']:.1f} pts)")
        print(f"   📝 {highlight['text']}")
        print(f"   ⏱️  {highlight['start_ms']}ms - {highlight['end_ms']}ms")
        print(f"   📊 Breakdown: sentence:{highlight['breakdown']['complete_sentence']:.1f} + "
              f"keywords:{highlight['breakdown']['keywords']:.1f} + "
              f"audio:{highlight['breakdown']['audio_rms']:.1f}")
    
    # Save results
    output_path = "local_highlights_result.json"
    with open(output_path, 'w') as f:
        json.dump({
            "top_highlights": top_highlights,
            "total_segments": len(scored_segments),
            "processing_time": time.time(),
            "method": "local_highlight_scorer"
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())