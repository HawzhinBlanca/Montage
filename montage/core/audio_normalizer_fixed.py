#!/usr/bin/env python3
"""
EBU R128 Audio Normalization
Two-pass loudness normalization for broadcast standards
"""
import subprocess
import json
import logging
import tempfile
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoudnessStats:
    """Audio loudness measurements"""
    input_i: float          # Integrated loudness (LUFS)
    input_tp: float         # True peak (dBTP)
    input_lra: float        # Loudness range (LU)
    input_thresh: float     # Threshold
    target_offset: float    # Offset to target


@dataclass
class NormalizationTarget:
    """Target loudness parameters"""
    integrated: float = -23.0    # EBU R128 standard
    true_peak: float = -1.0      # Maximum true peak
    lra: float = 7.0            # Target loudness range


class EBUAudioNormalizer:
    """EBU R128 compliant audio normalization"""
    
    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()
        
    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable"""
        try:
            result = subprocess.run(
                ["which", "ffmpeg"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "ffmpeg"  # Assume in PATH
        
    def normalize(
        self,
        input_path: str,
        output_path: str,
        target: Optional[NormalizationTarget] = None
    ) -> Dict[str, Any]:
        """
        Perform two-pass EBU R128 normalization
        
        Args:
            input_path: Input audio/video file
            output_path: Output file path
            target: Target loudness parameters
            
        Returns:
            Normalization results and statistics
        """
        if target is None:
            target = NormalizationTarget()
            
        try:
            # First pass: analyze loudness
            logger.info("Pass 1: Analyzing audio loudness...")
            stats = self._analyze_loudness(input_path)
            
            # Second pass: apply normalization
            logger.info("Pass 2: Applying normalization...")
            result = self._apply_normalization(
                input_path, output_path, stats, target
            )
            
            # Verify results
            logger.info("Verifying normalized audio...")
            final_stats = self._analyze_loudness(output_path)
            
            return {
                "success": True,
                "input_loudness": stats.input_i,
                "output_loudness": final_stats.input_i,
                "true_peak": final_stats.input_tp,
                "loudness_range": final_stats.input_lra,
                "normalization_gain": stats.target_offset
            }
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _analyze_loudness(self, input_path: str) -> LoudnessStats:
        """First pass: analyze loudness with loudnorm filter"""
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-af", "loudnorm=print_format=json",
            "-f", "null",
            "-"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        # Parse loudnorm output
        output_lines = result.stderr.split('\n')
        json_start = False
        json_data = []
        
        for line in output_lines:
            if '{' in line:
                json_start = True
            if json_start:
                json_data.append(line)
            if '}' in line and json_start:
                break
                
        json_str = '\n'.join(json_data)
        
        try:
            data = json.loads(json_str)
            
            return LoudnessStats(
                input_i=float(data["input_i"]),
                input_tp=float(data["input_tp"]),
                input_lra=float(data["input_lra"]),
                input_thresh=float(data["input_thresh"]),
                target_offset=float(data["target_offset"])
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to regex parsing
            import re
            
            stats = LoudnessStats(
                input_i=-23.0,
                input_tp=-1.0,
                input_lra=7.0,
                input_thresh=-33.0,
                target_offset=0.0
            )
            
            # Try to extract values
            for line in result.stderr.split('\n'):
                if "input_i" in line:
                    match = re.search(r"input_i\s*:\s*([-\d.]+)", line)
                    if match:
                        stats.input_i = float(match.group(1))
                elif "input_tp" in line:
                    match = re.search(r"input_tp\s*:\s*([-\d.]+)", line)
                    if match:
                        stats.input_tp = float(match.group(1))
                        
            return stats
            
    def _apply_normalization(
        self,
        input_path: str,
        output_path: str,
        stats: LoudnessStats,
        target: NormalizationTarget
    ) -> Dict[str, Any]:
        """Second pass: apply normalization with measured values"""
        # Build loudnorm filter with linear normalization
        audio_filter = (
            f"loudnorm="
            f"I={target.integrated}:"
            f"TP={target.true_peak}:"
            f"LRA={target.lra}:"
            f"measured_I={stats.input_i}:"
            f"measured_TP={stats.input_tp}:"
            f"measured_LRA={stats.input_lra}:"
            f"measured_thresh={stats.input_thresh}:"
            f"offset={stats.target_offset}:"
            f"linear=true:"
            f"print_format=summary"
        )
        
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",  # Copy video stream
            "-c:a", "aac",   # Re-encode audio
            "-b:a", "192k",  # Audio bitrate
            output_path,
            "-y"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
        return {"filter_applied": audio_filter}
        

class AudioEnhancer:
    """Additional audio enhancements"""
    
    def __init__(self):
        self.ffmpeg_path = "ffmpeg"
        
    def enhance_audio(
        self,
        input_path: str,
        output_path: str,
        enhancements: Dict[str, bool]
    ) -> bool:
        """
        Apply audio enhancements
        
        Args:
            enhancements: Dict of enhancement options
                - noise_reduction: Remove background noise
                - compression: Apply dynamic range compression
                - eq: Apply EQ for voice clarity
        """
        filters = []
        
        if enhancements.get("noise_reduction"):
            # High-pass filter to remove low frequency noise
            filters.append("highpass=f=80")
            
        if enhancements.get("compression"):
            # Gentle compression for consistent levels
            filters.append("acompressor=threshold=-20dB:ratio=4:attack=5:release=50")
            
        if enhancements.get("eq"):
            # EQ for voice clarity
            filters.append("equalizer=f=3000:t=h:width=200:g=3")
            
        if not filters:
            # No enhancements requested
            return False
            
        audio_filter = ",".join(filters)
        
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
            "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio enhancement failed: {e}")
            return False
            

class SpeechOptimizer:
    """Optimize audio specifically for speech content"""
    
    def __init__(self):
        self.speech_eq = [
            # Frequency, bandwidth, gain
            (200, 100, -2),    # Reduce muddiness
            (800, 200, 1),     # Boost warmth
            (2500, 500, 2),    # Boost presence
            (5000, 1000, 1),   # Boost clarity
            (8000, 2000, -1)   # Reduce harshness
        ]
        
    def optimize_for_speech(self, input_path: str, output_path: str) -> bool:
        """Optimize audio for speech clarity"""
        # Build EQ filter chain
        eq_filters = []
        for freq, width, gain in self.speech_eq:
            eq_filters.append(f"equalizer=f={freq}:t=h:width={width}:g={gain}")
            
        # Add other speech optimizations
        filters = [
            "highpass=f=80",  # Remove rumble
            "lowpass=f=12000",  # Remove high frequency noise
            *eq_filters,
            "loudnorm=I=-16:TP=-1.5:LRA=11",  # Louder for mobile
            "alimiter=limit=0.95"  # Prevent clipping
        ]
        
        audio_filter = ",".join(filters)
        
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-af", audio_filter,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
            "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Speech optimization failed: {e}")
            return False