#!/usr/bin/env python3
"""
Real speaker diarization using PyAnnote
Replaces fake alternating speaker IDs with actual speaker detection
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import PyAnnote
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("PyAnnote not available - speaker diarization will fallback to VAD")

from ..settings import get_settings

settings = get_settings()


class RealDiarization:
    """Real speaker diarization using PyAnnote"""
    
    def __init__(self):
        """Initialize PyAnnote pipeline"""
        if not PYANNOTE_AVAILABLE:
            raise ImportError("PyAnnote is not installed. Install with: pip install pyannote.audio")
        
        # Get HuggingFace token from settings
        hf_token = None
        if hasattr(settings, 'api_keys') and hasattr(settings.api_keys, 'huggingface_token'):
            token_obj = settings.api_keys.huggingface_token
            if token_obj and hasattr(token_obj, 'get_secret_value'):
                hf_token = token_obj.get_secret_value()
        
        if not hf_token:
            logger.warning("No HuggingFace token found - using offline model if available")
        
        try:
            # Initialize PyAnnote pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization@2.1",
                use_auth_token=hf_token
            )
            logger.info("PyAnnote speaker diarization pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (if known)
            
        Returns:
            List of speaker segments with start/end times
        """
        segments = []
        
        try:
            # Run diarization
            if num_speakers:
                diarization = self.pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarization = self.pipeline(audio_path)
            
            # Convert to our format
            speaker_mapping = {}
            speaker_counter = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Map PyAnnote speaker labels to consistent IDs
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"SPEAKER_{speaker_counter:02d}"
                    speaker_counter += 1
                
                segments.append({
                    "speaker": speaker_mapping[speaker],
                    "start": float(turn.start),
                    "end": float(turn.end)
                })
            
            logger.info(f"Diarization complete: {len(segments)} segments, {len(speaker_mapping)} speakers")
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise


def get_diarizer() -> Optional[RealDiarization]:
    """
    Get diarizer instance if available
    
    Returns:
        RealDiarization instance or None if not available
    """
    if not PYANNOTE_AVAILABLE:
        return None
    
    try:
        return RealDiarization()
    except Exception as e:
        logger.warning(f"Could not initialize diarizer: {e}")
        return None