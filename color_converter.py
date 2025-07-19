"""Color space validation and conversion for video processing"""

import logging
import subprocess
import json
import os
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from config import Config
from metrics import metrics, track_processing_stage

logger = logging.getLogger(__name__)


class ColorConversionError(Exception):
    """Color conversion error"""
    pass


@dataclass
class ColorSpaceInfo:
    """Video color space information"""
    color_space: str          # Color space (e.g., bt709, bt2020nc)
    color_primaries: str      # Color primaries (e.g., bt709, bt2020)
    color_transfer: str       # Transfer characteristics (e.g., bt709, smpte2084)
    color_range: str          # Color range (tv/pc)
    is_hdr: bool             # Whether this is HDR content
    
    @classmethod
    def from_ffprobe(cls, stream_data: Dict[str, Any]) -> 'ColorSpaceInfo':
        """Create from FFprobe stream data"""
        color_space = stream_data.get('color_space', 'unknown')
        color_primaries = stream_data.get('color_primaries', 'unknown') 
        color_transfer = stream_data.get('color_transfer', 'unknown')
        color_range = stream_data.get('color_range', 'tv')
        
        # HDR detection
        hdr_spaces = ['bt2020nc', 'bt2020c', 'smpte2084', 'arib-std-b67']
        hdr_transfers = ['smpte2084', 'arib-std-b67', 'smpte428']
        
        is_hdr = (
            color_space in hdr_spaces or 
            color_transfer in hdr_transfers or
            color_primaries == 'bt2020'
        )
        
        return cls(
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            color_range=color_range,
            is_hdr=is_hdr
        )


class ColorSpaceConverter:
    """Handles color space validation and conversion to BT.709"""
    
    def __init__(self):
        self.ffmpeg_path = Config.FFMPEG_PATH
        self.ffprobe_path = Config.FFPROBE_PATH
        
    def analyze_color_space(self, input_file: str) -> ColorSpaceInfo:
        """Analyze video color space information"""
        cmd = [
            self.ffprobe_path,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_streams',
            '-of', 'json',
            input_file
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            data = json.loads(result.stdout)
            
            if not data.get('streams'):
                raise ColorConversionError("No video stream found")
            
            stream = data['streams'][0]
            color_info = ColorSpaceInfo.from_ffprobe(stream)
            
            logger.info(f"Color space analysis:")
            logger.info(f"  Space: {color_info.color_space}")
            logger.info(f"  Primaries: {color_info.color_primaries}")
            logger.info(f"  Transfer: {color_info.color_transfer}")
            logger.info(f"  Range: {color_info.color_range}")
            logger.info(f"  HDR: {color_info.is_hdr}")
            
            return color_info
            
        except subprocess.CalledProcessError as e:
            raise ColorConversionError(f"Failed to analyze color space: {e.stderr}")
        except json.JSONDecodeError as e:
            raise ColorConversionError(f"Failed to parse ffprobe output: {e}")
    
    def validate_sdr_input(self, input_file: str) -> Tuple[bool, str]:
        """
        Validate that input is SDR (not HDR).
        
        Returns:
            (is_valid, error_message)
        """
        try:
            color_info = self.analyze_color_space(input_file)
            
            if color_info.is_hdr:
                return False, "HDR input not supported"
            
            return True, ""
            
        except Exception as e:
            return False, f"Color space validation failed: {str(e)}"
    
    def build_color_conversion_filter(self, source_color_info: Optional[ColorSpaceInfo] = None) -> str:
        """
        Build zscale filter for BT.709 conversion.
        
        This ensures all output is correctly flagged as BT.709.
        """
        # Full conversion pipeline to BT.709
        filter_parts = []
        
        # Convert to linear light
        filter_parts.append("zscale=t=linear:npl=100")
        
        # Convert to RGB float for processing
        filter_parts.append("format=gbrpf32le")
        
        # Convert to BT.709 primaries and transfer
        filter_parts.append("zscale=p=bt709:t=bt709:m=bt709:r=tv")
        
        # Convert back to YUV420 for encoding
        filter_parts.append("format=yuv420p")
        
        return ",".join(filter_parts)
    
    @track_processing_stage('color_conversion')
    def convert_to_bt709(self, input_file: str, output_file: str,
                         video_codec: str = "libx264",
                         audio_codec: str = "copy",
                         additional_filters: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert video to BT.709 color space.
        
        Args:
            input_file: Input video path
            output_file: Output video path
            video_codec: Video codec for encoding
            audio_codec: Audio codec (default: copy)
            additional_filters: Additional video filters to apply
            
        Returns:
            Conversion results and verification
        """
        # First validate input
        is_valid, error_msg = self.validate_sdr_input(input_file)
        if not is_valid:
            raise ColorConversionError(error_msg)
        
        # Analyze source color space
        source_info = self.analyze_color_space(input_file)
        
        # Build color conversion filter
        color_filter = self.build_color_conversion_filter(source_info)
        
        # Combine with additional filters if provided
        if additional_filters:
            full_filter = f"{additional_filters},{color_filter}"
        else:
            full_filter = color_filter
        
        # Build conversion command
        cmd = [
            self.ffmpeg_path,
            '-i', input_file,
            '-vf', full_filter,
            '-c:v', video_codec,
            '-preset', 'fast',
            '-crf', '18',  # Good quality
            '-c:a', audio_codec,
            '-colorspace', 'bt709',
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-movflags', '+faststart',
            '-y',
            output_file
        ]
        
        logger.info(f"Converting to BT.709 color space...")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Verify output
            output_info = self.analyze_color_space(output_file)
            
            result = {
                'source_color_space': source_info.color_space,
                'source_primaries': source_info.color_primaries,
                'output_color_space': output_info.color_space,
                'output_primaries': output_info.color_primaries,
                'conversion_successful': output_info.color_primaries == 'bt709'
            }
            
            if not result['conversion_successful']:
                logger.warning(f"Color conversion may have failed. Output primaries: {output_info.color_primaries}")
            else:
                logger.info("Successfully converted to BT.709")
            
            return result
            
        except subprocess.CalledProcessError as e:
            raise ColorConversionError(f"Color conversion failed: {e.stderr}")
    
    def get_encoding_color_params(self) -> Dict[str, str]:
        """
        Get FFmpeg parameters to ensure BT.709 output.
        
        Returns dict of parameter: value for FFmpeg command.
        """
        return {
            '-colorspace': 'bt709',
            '-color_primaries': 'bt709', 
            '-color_trc': 'bt709',
            '-color_range': 'tv'  # Broadcast safe
        }
    
    def build_safe_encoding_command(self, input_file: str, output_file: str,
                                   video_filters: Optional[str] = None,
                                   video_codec: str = "libx264",
                                   audio_codec: str = "aac") -> List[str]:
        """
        Build FFmpeg command with proper color space handling.
        
        This ensures output is always BT.709 regardless of input.
        """
        cmd = [self.ffmpeg_path, '-i', input_file]
        
        # Add video filters including color conversion
        if video_filters:
            # Append color conversion to existing filters
            full_filter = f"{video_filters},{self.build_color_conversion_filter()}"
        else:
            full_filter = self.build_color_conversion_filter()
        
        cmd.extend(['-vf', full_filter])
        
        # Video encoding
        cmd.extend([
            '-c:v', video_codec,
            '-preset', 'fast',
            '-crf', '18'
        ])
        
        # Add color space parameters
        color_params = self.get_encoding_color_params()
        for param, value in color_params.items():
            cmd.extend([param, value])
        
        # Audio encoding
        cmd.extend(['-c:a', audio_codec])
        
        # Output options
        cmd.extend([
            '-movflags', '+faststart',
            '-y',
            output_file
        ])
        
        return cmd


# Integration functions

def ensure_bt709_output(input_file: str, output_file: str, 
                       preserve_filters: Optional[str] = None) -> Dict[str, Any]:
    """
    Ensure video output is BT.709 color space.
    
    This is a convenience function for pipeline integration.
    """
    converter = ColorSpaceConverter()
    
    # Validate SDR input
    is_valid, error_msg = converter.validate_sdr_input(input_file)
    if not is_valid:
        raise ColorConversionError(error_msg)
    
    # Convert with preservation of other filters
    return converter.convert_to_bt709(
        input_file,
        output_file,
        additional_filters=preserve_filters
    )


def get_safe_color_filter() -> str:
    """Get the color conversion filter string for manual integration"""
    converter = ColorSpaceConverter()
    return converter.build_color_conversion_filter()


# Example integration with video editor
class ColorSafeVideoEditor:
    """Example video editor with color space safety"""
    
    def __init__(self):
        self.color_converter = ColorSpaceConverter()
        
    def edit_with_color_safety(self, input_file: str, output_file: str,
                              edit_filters: str = "") -> Dict[str, Any]:
        """Edit video ensuring BT.709 output"""
        
        # Validate input is SDR
        is_valid, error = self.color_converter.validate_sdr_input(input_file)
        if not is_valid:
            raise ColorConversionError(error)
        
        # Build safe encoding command
        cmd = self.color_converter.build_safe_encoding_command(
            input_file,
            output_file,
            video_filters=edit_filters
        )
        
        # Execute
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Verify output
        output_info = self.color_converter.analyze_color_space(output_file)
        
        return {
            'output_primaries': output_info.color_primaries,
            'is_bt709': output_info.color_primaries == 'bt709'
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    converter = ColorSpaceConverter()
    
    # Analyze input
    info = converter.analyze_color_space("input.mp4")
    print(f"Input color space: {info.color_space}")
    print(f"Is HDR: {info.is_hdr}")
    
    # Convert to BT.709
    if not info.is_hdr:
        result = converter.convert_to_bt709("input.mp4", "output_bt709.mp4")
        print(f"Conversion successful: {result['conversion_successful']}")