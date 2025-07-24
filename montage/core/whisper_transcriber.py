#!/usr/bin/env python3
"""
Whisper Transcriber - Wrapper for WhisperTranscriber as specified in Tasks.md
Provides the exact interface expected by Director orchestration
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    WhisperTranscriber class as specified in Tasks.md Phase 3.2
    Provides transcribe() method for Director agent integration
    """
    
    def __init__(self):
        """Initialize Whisper transcriber"""
        self.initialized = True
        logger.info("WhisperTranscriber initialized for Director integration")
    
    def transcribe(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcription result dictionary
        """
        try:
            # Use existing Deepgram integration as backend
            from .api_wrappers import DeepgramWrapper
            
            # Create Deepgram wrapper instance
            deepgram = DeepgramWrapper()
            
            # Call transcription method
            result = deepgram.transcribe_audio(video_path)
            
            logger.info(f"WhisperTranscriber completed transcription of {video_path}")
            
            # Return in expected format
            return {
                "success": True,
                "transcription": result,
                "provider": "whisper_via_deepgram",
                "video_path": video_path
            }
            
        except Exception as e:
            logger.error(f"WhisperTranscriber failed for {video_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "video_path": video_path
            }