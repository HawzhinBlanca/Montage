#!/usr/bin/env python3
"""
FFMPEGEditor - Core FFmpeg-based video editing functionality
Provides low-level video processing operations
"""
import logging
import subprocess
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FFMPEGEditor:
    """FFmpeg-based video editor for processing clips"""
    
    def __init__(self, source_path: str = None):
        """
        Initialize FFmpeg editor
        
        Args:
            source_path: Path to source video file
        """
        self.source_path = source_path
        if source_path and not Path(source_path).exists():
            raise FileNotFoundError(f"Source video not found: {source_path}")
    
    def process(self, clips: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
        """
        Process video clips and create output
        
        Args:
            clips: List of clip definitions with start/end times
            output_path: Path for output video
            
        Returns:
            Processing result with success status and metadata
        """
        try:
            if not clips:
                raise ValueError("No clips provided for processing")
            
            # Create concat list file
            concat_file = self._create_concat_file(clips)
            
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c:v", "h264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",
                output_path
            ]
            
            # Execute FFmpeg
            logger.info(f"Processing {len(clips)} clips to: {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            # Clean up temp file
            os.unlink(concat_file)
            
            # Verify output
            if not Path(output_path).exists():
                raise FileNotFoundError(f"Output file not created: {output_path}")
            
            return {
                "success": True,
                "output_path": output_path,
                "clips_processed": len(clips),
                "file_size": Path(output_path).stat().st_size
            }
            
        except Exception as e:
            logger.error(f"FFmpeg processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "clips_processed": 0
            }
    
    def _create_concat_file(self, clips: List[Dict[str, Any]]) -> str:
        """Create concat demuxer file for FFmpeg"""
        import tempfile
        
        concat_content = []
        
        for clip in clips:
            # Extract clip using FFmpeg
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", start_time + 1)
            duration = end_time - start_time
            
            # Create temp file for this clip
            clip_file = tempfile.mktemp(suffix=".mp4")
            
            cmd = [
                "ffmpeg",
                "-i", self.source_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c", "copy",
                "-y",
                clip_file
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            concat_content.append(f"file '{clip_file}'")
        
        # Write concat file
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, 'w') as f:
            f.write('\n'.join(concat_content))
        
        return concat_file
    
    def extract_clip(self, start_time: float, end_time: float, output_path: str) -> bool:
        """Extract a single clip from video"""
        try:
            duration = end_time - start_time
            
            cmd = [
                "ffmpeg",
                "-i", self.source_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "h264",
                "-preset", "fast",
                "-c:a", "aac",
                "-y",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Clip extraction failed: {e}")
            return False
    
    def get_video_info(self) -> Dict[str, Any]:
        """Get video metadata using FFprobe"""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                self.source_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
        
        return {}