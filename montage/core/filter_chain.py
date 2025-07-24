"""
FilterChain for FFmpeg filters - Tasks.md Step 1.3
Provides face_crop(box) integration for concat.py
"""
from typing import Optional, Tuple


class FilterChain:
    """FFmpeg filter chain builder for video processing"""

    @staticmethod
    def face_crop(face_box: Optional[Tuple[int, int, int, int]]) -> str:
        """
        Generate FFmpeg filter for face-aware cropping - Tasks.md Step 1.3
        
        Args:
            face_box: (x, y, w, h) face bounding box or None for center crop
            
        Returns:
            FFmpeg filter string for cropping
        """
        if face_box is None:
            # Fallback to center crop for 9:16 vertical format
            return "crop=1080:1920:(iw-1080)/2:(ih-1920)/2,pad=1080:1920:(1080-iw)/2:(1920-ih)/2"

        x, y, w, h = face_box

        # Create face-aware crop filter
        # Crop to face region, then scale/pad to 9:16 format
        crop_filter = f"crop={w}:{h}:{x}:{y}"

        # Scale to fit 9:16 maintaining aspect ratio
        scale_filter = "scale=1080:1920:force_original_aspect_ratio=decrease"

        # Pad to exact 9:16 dimensions
        pad_filter = "pad=1080:1920:(1080-iw)/2:(1920-ih)/2"

        return f"{crop_filter},{scale_filter},{pad_filter}"

    @staticmethod
    def audio_normalize() -> str:
        """Audio normalization filter"""
        return "loudnorm=I=-16:LRA=11:TP=-1.5"

    @staticmethod
    def video_stabilize() -> str:
        """Video stabilization filter"""
        return "vidstabdetect=shakiness=10:accuracy=10:result=/tmp/transforms.trf,vidstabtransform=input=/tmp/transforms.trf:zoom=0:smoothing=10"

    @staticmethod
    def color_enhance() -> str:
        """Color enhancement filter"""
        return "eq=contrast=1.1:brightness=0.05:saturation=1.2"
