# Video Validation Guide

## Overview

The video validation module performs pre-flight checks on input videos to ensure they can be processed successfully. It detects corruption, validates formats, and enforces processing limits.

## Key Features

- **Corruption Detection**: Detects common video corruptions (moov atom errors, invalid data)
- **Format Validation**: Ensures video uses supported codecs and formats
- **HDR Detection**: Rejects HDR videos as per requirements
- **Metadata Extraction**: Extracts and stores video properties for later use
- **SHA256 Hashing**: Creates unique identifiers for caching

## Validation Criteria

### Supported Formats

**Video Codecs:**
- H.264 (h264)
- H.265/HEVC (hevc)
- VP9 (vp9)
- AV1 (av1)
- MPEG-4 (mpeg4)
- MPEG-2 (mpeg2video)

**Audio Codecs:**
- AAC (aac)
- MP3 (mp3)
- Opus (opus)
- Vorbis (vorbis)
- PCM (pcm_s16le, pcm_s24le)

### Limits

- **Duration**: Maximum 2 hours (7200 seconds)
- **Resolution**: Maximum 4K (3840x2160)
- **Frame Rate**: Maximum 120 fps
- **Color Space**: SDR only (BT.709), HDR not supported

### Corruption Checks

The validator detects:
- Missing moov atom (common MP4 corruption)
- Invalid codec parameters
- Unreadable or empty files
- Missing video streams

## Usage

### Basic Validation

```python
from video_validator import VideoValidator

validator = VideoValidator()

# Validate a file
is_valid, metadata, error = validator.validate_file('/path/to/video.mp4')

if is_valid:
    print(f"Duration: {metadata.duration}s")
    print(f"Codec: {metadata.codec}")
    print(f"Resolution: {metadata.resolution}")
    print(f"Color space: {metadata.color_space}")
else:
    print(f"Validation failed: {error}")
```

### Integration with Job Processing

```python
from video_validator import perform_preflight_check

# In SmartVideoEditor
def validate_input(self, job_id: str, input_path: str) -> dict:
    """Validate input video file"""
    
    result = perform_preflight_check(job_id, input_path)
    
    if not result['valid']:
        # Job status already updated to 'failed' in database
        raise VideoValidationError(result['error'])
    
    # Continue with processing
    return result
```

### Complete Example

```python
from video_validator import VideoValidator
from db import Database

# Initialize
validator = VideoValidator()
db = Database()

# Create job
job_id = db.insert('video_job', {
    'src_hash': 'pending',
    'status': 'queued',
    'input_path': '/videos/input.mp4'
})

# Validate and store metadata
if validator.validate_and_store(job_id, '/videos/input.mp4'):
    print("✅ Video passed pre-flight checks")
    
    # Get stored metadata
    job = db.find_one('video_job', {'id': job_id})
    print(f"Duration: {job['duration']}s")
    print(f"Codec: {job['codec']}")
    print(f"Color space: {job['color_space']}")
else:
    # Get error details
    job = db.find_one('video_job', {'id': job_id})
    print(f"❌ Validation failed: {job['error_message']}")
```

## Metadata Structure

The `VideoMetadata` dataclass contains:

```python
@dataclass
class VideoMetadata:
    duration: float          # Video duration in seconds
    codec: str              # Video codec name
    resolution: str         # Format: "widthxheight"
    fps: float              # Frames per second
    bitrate: int            # Bitrate in bits/second
    color_space: str        # Color space (e.g., "bt709")
    color_primaries: str    # Color primaries
    color_transfer: str     # Transfer characteristics
    has_audio: bool         # Whether audio stream exists
    audio_codec: Optional[str]      # Audio codec if present
    audio_channels: Optional[int]   # Number of audio channels
    audio_sample_rate: Optional[int]  # Audio sample rate in Hz
    file_size: int          # File size in bytes
    container_format: str   # Container format
```

## Error Handling

The module uses specific exceptions:

- `VideoValidationError`: Base exception for all validation errors
- `CorruptedVideoError`: File is corrupted or unreadable
- `UnsupportedFormatError`: Format not supported for processing

Example error handling:

```python
try:
    is_valid, metadata, error = validator.validate_file(input_path)
except CorruptedVideoError as e:
    logger.error(f"Corrupted video: {e}")
    # Handle corruption specifically
except VideoValidationError as e:
    logger.error(f"Validation error: {e}")
    # Handle general validation errors
```

## FFprobe Integration

The validator uses ffprobe with specific parameters:

```bash
ffprobe -v error -read_intervals "%+#2" -show_format -show_streams -of json [input]
```

- `-read_intervals "%+#2"`: Reads first 2 packets to check integrity
- `-v error`: Only show errors in output
- `-of json`: Output in JSON format for easy parsing

## Testing

Run validation tests:

```bash
# All validation tests
pytest tests/test_video_validator.py -v

# Test specific validation scenarios
pytest tests/test_video_validator.py::TestVideoValidator::test_detect_hdr_video -v
```

## Command Line Usage

The module can be run directly for quick validation:

```bash
python video_validator.py /path/to/video.mp4
```

Output:
```
✅ Video is valid!
Duration: 300.5s
Codec: h264
Resolution: 1920x1080
FPS: 30.0
Color space: bt709
```

Or for invalid files:
```
❌ Video validation failed: HDR input not supported
```