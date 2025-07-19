"""Video file validation and pre-flight checks"""

import json
import logging
import subprocess
import os
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

try:
    from ..config import get
except ImportError:
    from src.config import get
from ..core.db import Database

logger = logging.getLogger(__name__)


class VideoValidationError(Exception):
    """Base exception for video validation errors"""

    pass


class CorruptedVideoError(VideoValidationError):
    """Raised when video file is corrupted"""

    pass


class UnsupportedFormatError(VideoValidationError):
    """Raised when video format is not supported"""

    pass


@dataclass
class VideoMetadata:
    """Video file metadata"""

    duration: float
    codec: str
    resolution: str
    fps: float
    bitrate: int
    color_space: str
    color_primaries: str
    color_transfer: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    file_size: int = 0
    container_format: str = ""


class VideoValidator:
    """Validates video files and extracts metadata"""

    def __init__(self):
        self.ffprobe_path = get("FFPROBE_PATH", "ffprobe")
        self.db = Database()

    def validate_file(
        self, file_path: str
    ) -> Tuple[bool, VideoMetadata, Optional[str]]:
        """
        Validate video file and extract metadata

        Returns:
            (is_valid, metadata, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, None, "File does not exist"

            # Check file is readable
            if not os.access(file_path, os.R_OK):
                return False, None, "File is not readable"

            # Get file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, None, "File is empty"

            # Run ffprobe
            probe_data = self._run_ffprobe(file_path)

            # Parse metadata
            metadata = self._parse_probe_data(probe_data, file_size)

            # Validate metadata
            validation_errors = self._validate_metadata(metadata)
            if validation_errors:
                return False, metadata, "; ".join(validation_errors)

            return True, metadata, None

        except CorruptedVideoError as e:
            return False, None, str(e)
        except UnsupportedFormatError as e:
            return False, None, str(e)
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False, None, f"Validation failed: {str(e)}"

    def _run_ffprobe(self, file_path: str) -> Dict[str, Any]:
        """Run ffprobe and return parsed JSON output"""
        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-read_intervals",
            "%+#2",  # Read first 2 packets to check file integrity
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            file_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Check for common corruption indicators in stderr
            if result.stderr:
                stderr_lower = result.stderr.lower()
                if "moov atom not found" in stderr_lower:
                    raise CorruptedVideoError("File is corrupted: moov atom not found")
                elif "invalid data" in stderr_lower:
                    raise CorruptedVideoError("File is corrupted: invalid data found")
                elif "could not find codec parameters" in stderr_lower:
                    raise CorruptedVideoError(
                        "File is corrupted: could not find codec parameters"
                    )

            if result.returncode != 0:
                raise VideoValidationError(
                    f"ffprobe failed with code {result.returncode}: {result.stderr}"
                )

            # Parse JSON output
            try:
                probe_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                raise CorruptedVideoError("Failed to parse ffprobe output")

            # Verify we have required data
            if "format" not in probe_data:
                raise CorruptedVideoError("No format information found")

            if "streams" not in probe_data or not probe_data["streams"]:
                raise CorruptedVideoError("No streams found in file")

            return probe_data

        except subprocess.TimeoutExpired:
            raise VideoValidationError(
                "ffprobe timed out - file may be corrupted or too large"
            )
        except subprocess.SubprocessError as e:
            raise VideoValidationError(f"Failed to run ffprobe: {e}")

    def _parse_probe_data(
        self, probe_data: Dict[str, Any], file_size: int
    ) -> VideoMetadata:
        """Parse ffprobe output into VideoMetadata"""
        format_info = probe_data["format"]
        streams = probe_data["streams"]

        # Find video stream
        video_stream = None
        audio_stream = None

        for stream in streams:
            if stream["codec_type"] == "video" and video_stream is None:
                video_stream = stream
            elif stream["codec_type"] == "audio" and audio_stream is None:
                audio_stream = stream

        if not video_stream:
            raise UnsupportedFormatError("No video stream found")

        # Extract video metadata
        duration = float(format_info.get("duration", 0))

        # Get resolution
        width = video_stream.get("width", 0)
        height = video_stream.get("height", 0)
        resolution = f"{width}x{height}"

        # Get FPS
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            fps_parts = fps_str.split("/")
            fps = (
                float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 0
            )
        except Exception:
            fps = 0

        # Get color space information
        color_space = video_stream.get("color_space", "unknown")
        color_primaries = video_stream.get("color_primaries", "unknown")
        color_transfer = video_stream.get("color_transfer", "unknown")

        # Get audio information
        has_audio = audio_stream is not None
        audio_codec = audio_stream.get("codec_name") if audio_stream else None
        audio_channels = audio_stream.get("channels") if audio_stream else None
        audio_sample_rate = (
            int(audio_stream.get("sample_rate", 0)) if audio_stream else None
        )

        metadata = VideoMetadata(
            duration=duration,
            codec=video_stream.get("codec_name", "unknown"),
            resolution=resolution,
            fps=fps,
            bitrate=int(format_info.get("bit_rate", 0)),
            color_space=color_space,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            file_size=file_size,
            container_format=format_info.get("format_name", "unknown"),
        )

        return metadata

    def _validate_metadata(self, metadata: VideoMetadata) -> List[str]:
        """Validate metadata and return list of errors"""
        errors = []

        # Check duration
        if metadata.duration <= 0:
            errors.append("Invalid duration")
        elif metadata.duration > 7200:  # 2 hours
            errors.append("Video duration exceeds 2 hours limit")

        # Check resolution
        width, height = metadata.resolution.split("x")
        if int(width) == 0 or int(height) == 0:
            errors.append("Invalid resolution")
        elif int(width) > 3840 or int(height) > 2160:  # 4K limit
            errors.append("Resolution exceeds 4K limit")

        # Check FPS
        if metadata.fps <= 0:
            errors.append("Invalid frame rate")
        elif metadata.fps > 120:
            errors.append("Frame rate exceeds 120fps limit")

        # Check codec
        supported_codecs = ["h264", "hevc", "vp9", "av1", "mpeg4", "mpeg2video"]
        if metadata.codec not in supported_codecs:
            errors.append(f"Unsupported video codec: {metadata.codec}")

        # Check for HDR (as per requirements, HDR is not supported)
        hdr_color_spaces = ["bt2020nc", "bt2020c", "smpte2084", "arib-std-b67"]
        hdr_transfers = ["smpte2084", "arib-std-b67", "smpte428"]

        if (
            metadata.color_space in hdr_color_spaces
            or metadata.color_transfer in hdr_transfers
        ):
            errors.append("HDR input not supported")

        # Check audio if present
        if metadata.has_audio:
            supported_audio_codecs = [
                "aac",
                "mp3",
                "opus",
                "vorbis",
                "pcm_s16le",
                "pcm_s24le",
            ]
            if metadata.audio_codec not in supported_audio_codecs:
                errors.append(f"Unsupported audio codec: {metadata.audio_codec}")

        return errors

    def calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            raise VideoValidationError(f"Failed to calculate file hash: {e}")

    def validate_and_store(self, job_id: str, file_path: str) -> bool:
        """Validate video and store metadata in database"""
        try:
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_path)

            # Update job with hash
            self.db.update("video_job", {"src_hash": file_hash}, {"id": job_id})

            # Validate file
            is_valid, metadata, error_message = self.validate_file(file_path)

            if not is_valid:
                # Update job as failed
                self.db.update(
                    "video_job",
                    {
                        "status": "failed",
                        "error_message": f"Pre-flight check failed: {error_message}",
                    },
                    {"id": job_id},
                )

                logger.error(f"Validation failed for job {job_id}: {error_message}")
                return False

            # Store metadata in database
            self.db.update(
                "video_job",
                {
                    "duration": metadata.duration,
                    "codec": metadata.codec,
                    "color_space": metadata.color_space,
                    "status": "validated",
                },
                {"id": job_id},
            )

            logger.info(f"Validation successful for job {job_id}")
            logger.info(f"  Duration: {metadata.duration}s")
            logger.info(f"  Codec: {metadata.codec}")
            logger.info(f"  Resolution: {metadata.resolution}")
            logger.info(f"  FPS: {metadata.fps}")
            logger.info(f"  Color space: {metadata.color_space}")

            return True

        except Exception as e:
            # Update job as failed
            self.db.update(
                "video_job",
                {
                    "status": "failed",
                    "error_message": f"Pre-flight check error: {str(e)}",
                },
                {"id": job_id},
            )

            logger.error(f"Validation error for job {job_id}: {e}")
            return False


# Convenience function for integration
def perform_preflight_check(job_id: str, input_path: str) -> Dict[str, Any]:
    """Perform pre-flight check for SmartVideoEditor integration"""
    validator = VideoValidator()

    if validator.validate_and_store(job_id, input_path):
        # Get the stored metadata
        job = validator.db.find_one("video_job", {"id": job_id})

        return {
            "valid": True,
            "duration": job["duration"],
            "codec": job["codec"],
            "color_space": job["color_space"],
        }
    else:
        # Get error message
        job = validator.db.find_one("video_job", {"id": job_id})

        return {"valid": False, "error": job.get("error_message", "Unknown error")}


if __name__ == "__main__":
    # Test validation
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_validator.py <video_file>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    validator = VideoValidator()
    is_valid, metadata, error = validator.validate_file(sys.argv[1])

    if is_valid:
        print("✅ Video is valid!")
        print(f"Duration: {metadata.duration}s")
        print(f"Codec: {metadata.codec}")
        print(f"Resolution: {metadata.resolution}")
        print(f"FPS: {metadata.fps}")
        print(f"Color space: {metadata.color_space}")
    else:
        print(f"❌ Video validation failed: {error}")
