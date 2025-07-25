#!/usr/bin/env python3
"""
Speaker Diarization using pyannote-audio
Identifies and segments different speakers in audio
"""
import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    import torch
    PYANNOTE_AVAILABLE = True
except (ImportError, OSError) as e:
    PYANNOTE_AVAILABLE = False
    logger.warning(f"pyannote-audio not available: {e}")


class SpeakerDiarizer:
    """Speaker diarization using pyannote-audio"""
    
    def __init__(self, auth_token: Optional[str] = None):
        """
        Initialize speaker diarizer
        
        Args:
            auth_token: HuggingFace auth token for model access
        """
        self.auth_token = auth_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.pipeline = None
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
        
    def _load_pipeline(self):
        """Lazy load the diarization pipeline"""
        if self.pipeline is None:
            if not PYANNOTE_AVAILABLE:
                raise ImportError("pyannote-audio is required for speaker diarization")
                
            if not self.auth_token:
                raise ValueError("HuggingFace auth token required. Set HUGGINGFACE_TOKEN env var")
                
            # Load pretrained pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
            
            # Send to GPU if available
            if self.device == "cuda":
                self.pipeline.to(torch.device("cuda"))
                
            logger.info(f"Loaded speaker diarization pipeline on {self.device}")
    
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (if known)
            
        Returns:
            List of speaker segments with format:
            [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.5,
                    "end": 10.2
                },
                ...
            ]
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Load pipeline if needed
        self._load_pipeline()
        
        # Run diarization
        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
            
        diarization = self.pipeline(audio_path, **params)
        
        # Convert to list format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
            
        return segments
    
    def diarize_video(self, video_path: str, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on video file
        
        Args:
            video_path: Path to video file
            num_speakers: Optional number of speakers (if known)
            
        Returns:
            List of speaker segments
        """
        # Extract audio from video
        audio_path = self._extract_audio(video_path)
        
        try:
            return self.diarize(audio_path, num_speakers)
        finally:
            # Cleanup
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video as WAV"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name
            
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_path, "-y"
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_path
    
    def merge_with_transcript(self, diarization: List[Dict], transcript: List[Dict]) -> List[Dict]:
        """
        Merge diarization results with transcript segments
        
        Args:
            diarization: Speaker diarization segments
            transcript: Whisper transcript segments
            
        Returns:
            Transcript segments with speaker labels added
        """
        # Create a sorted list of speaker segments
        speaker_segments = sorted(diarization, key=lambda x: x["start"])
        
        # For each transcript segment, find the overlapping speaker
        merged = []
        for segment in transcript:
            seg_start = segment["start"]
            seg_end = segment["end"]
            seg_mid = (seg_start + seg_end) / 2
            
            # Find speaker at segment midpoint
            speaker = None
            for spk_seg in speaker_segments:
                if spk_seg["start"] <= seg_mid <= spk_seg["end"]:
                    speaker = spk_seg["speaker"]
                    break
                    
            # Add speaker to segment
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = speaker or "UNKNOWN"
            merged.append(segment_with_speaker)
            
        return merged