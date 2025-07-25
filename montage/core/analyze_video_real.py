#!/usr/bin/env python3
"""
REAL analyze_video implementation - No Mock Data!
Uses actual Whisper, AI APIs, and all features
"""
import os
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

# Import REAL implementations
from .real_whisper_transcriber import UltraAccurateWhisperTranscriber
from .real_speaker_diarization import RealSpeakerDiarization
from .ai_highlight_selector import AIHighlightSelector
from .narrative_flow import NarrativeFlowOptimizer
from .creative_titles import AICreativeTitleGenerator

logger = logging.getLogger(__name__)


async def analyze_video_real(video_path: str) -> Dict[str, Any]:
    """
    REAL video analysis - no mock data!
    
    Returns:
        Complete analysis with real transcription, AI highlights, etc.
    """
    logger.info(f"Starting REAL analysis of {video_path}")
    
    try:
        # 1. REAL Whisper Transcription
        logger.info("Running REAL Whisper transcription...")
        transcriber = UltraAccurateWhisperTranscriber()
        transcript_result = await transcriber.transcribe(video_path)
        
        # 2. REAL Speaker Diarization
        logger.info("Running REAL speaker diarization...")
        diarizer = RealSpeakerDiarization()
        speaker_segments = await diarizer.diarize(video_path)
        
        # 3. REAL AI Highlight Selection
        logger.info("Running REAL AI highlight selection...")
        highlight_selector = AIHighlightSelector()
        highlights = await highlight_selector.select_highlights(
            transcript_result,
            video_duration=_get_video_duration(video_path),
            target_clips=3
        )
        
        # 4. REAL Narrative Flow Optimization
        logger.info("Optimizing narrative flow...")
        flow_optimizer = NarrativeFlowOptimizer()
        
        # Convert highlights to segments
        segments = [
            {
                "start": h.start,
                "end": h.end,
                "text": h.title,
                "score": h.score,
                "story_beat": h.story_beat
            }
            for h in highlights
        ]
        
        optimized_segments = flow_optimizer.optimize_flow(segments)
        
        # 5. Generate Creative Titles
        logger.info("Generating AI titles...")
        title_generator = AICreativeTitleGenerator()
        
        titles = []
        for seg in optimized_segments[:3]:
            content = await title_generator.generate_creative_content(
                {"text": seg["text"]},
                platform="tiktok"
            )
            titles.append({
                "title": content.title,
                "hashtags": content.hashtags,
                "emojis": content.emojis
            })
        
        # Build complete result
        result = {
            "highlights": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "score": seg["score"],
                    "title": titles[i]["title"] if i < len(titles) else seg["text"],
                    "hashtags": titles[i]["hashtags"] if i < len(titles) else [],
                    "emojis": titles[i]["emojis"] if i < len(titles) else []
                }
                for i, seg in enumerate(optimized_segments)
            ],
            "transcript": transcript_result.get("segments", []),
            "words": _extract_words(transcript_result),
            "speaker_turns": [
                {
                    "speaker": s["speaker"],
                    "start": s["start"],
                    "end": s["end"]
                }
                for s in speaker_segments
            ],
            "analysis": {
                "total_words": len(_extract_words(transcript_result)),
                "total_speakers": len(set(s["speaker"] for s in speaker_segments)),
                "ai_model": "claude+gemini",
                "transcription_model": "whisper-large-v3"
            }
        }
        
        logger.info(f"REAL analysis complete: {len(result['highlights'])} highlights found")
        return result
        
    except Exception as e:
        logger.error(f"REAL analysis failed: {e}")
        # Return minimal valid structure on error
        return {
            "highlights": [],
            "transcript": [],
            "words": [],
            "speaker_turns": [],
            "analysis": {"error": str(e)}
        }


def _get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe"""
    import subprocess
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return 300.0  # Default 5 minutes


def _extract_words(transcript: Dict) -> List[Dict]:
    """Extract all words from transcript"""
    words = []
    
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"]
            })
            
    return words


# Make it synchronous for compatibility
def analyze_video(video_path: str) -> Dict[str, Any]:
    """Synchronous wrapper for real analysis"""
    return asyncio.run(analyze_video_real(video_path))