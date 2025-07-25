#!/usr/bin/env python3
"""
Real Whisper Transcription Implementation
Uses faster-whisper for efficient speech-to-text
"""
import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

try:
    import faster_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not available. Install with: pip install faster-whisper")


class RealWhisperTranscriber:
    """Real Whisper transcription using faster-whisper"""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (auto, cpu, cuda)
        """
        self.model_size = model_size
        self.device = device if device != "auto" else self._detect_device()
        self.model = None
        
        # Check for model path override
        self.model_path = os.environ.get("WHISPER_MODEL_PATH")
        
    def _detect_device(self) -> str:
        """Detect best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
        
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            if not WHISPER_AVAILABLE:
                raise ImportError("faster-whisper is required for transcription")
                
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            # Use custom model path if provided
            if self.model_path and os.path.exists(self.model_path):
                model_path = self.model_path
            else:
                model_path = self.model_size
                
            self.model = faster_whisper.WhisperModel(
                model_path,
                device=self.device,
                compute_type=compute_type
            )
            logger.info(f"Loaded Whisper model: {self.model_size} on {self.device}")
    
    def transcribe(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe audio from video file
        
        Returns:
            {
                "segments": [...],
                "text": "full transcript",
                "language": "en",
                "duration": 123.45
            }
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        # Extract audio
        audio_path = self._extract_audio(video_path)
        
        try:
            # Load model if needed
            self._load_model()
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=5,
                best_of=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )
            
            # Format results
            result_segments = []
            full_text = []
            
            for segment in segments:
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    seg_dict["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                    
                result_segments.append(seg_dict)
                full_text.append(segment.text.strip())
                
            return {
                "segments": result_segments,
                "text": " ".join(full_text),
                "language": info.language,
                "duration": info.duration or 0
            }
            
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