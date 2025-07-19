# Test Video Files Report

Generated test videos for comprehensive edge case testing.

**Summary:**
- Created: 14 videos
- Failed: 2 videos
- Special files: 4

## Video Files

| File | Duration | Resolution | Bitrate | Description |
|------|----------|------------|---------|-------------|
| short.mp4 | 15s | 1920x1080 | 2M | Short valid video for basic testing (2.0 MB) |
| medium.mp4 | 120s | 1920x1080 | 4M | Medium length video for processing tests (17.1 MB) |
| long.mp4 | 900s | 1920x1080 | 2M | Long video for stress testing (128.3 MB) |
| vertical.mp4 | 60s | 720x1280 | 2M | Vertical video for smart crop testing (8.1 MB) |
| square.mp4 | 45s | 1080x1080 | 2M | Square aspect ratio video (7.2 MB) |
| ultrawide.mp4 | 30s | 3440x1440 | 8M | Ultrawide aspect ratio video (9.3 MB) |
| lowres.mp4 | 30s | 320x240 | 500k | Low resolution video (2.2 MB) |
| 4k.mp4 | 30s | 3840x2160 | 20M | 4K resolution video (8.8 MB) |
| highfps.mp4 | 20s | 1920x1080 | 8M | High frame rate video (60fps) (6.2 MB) |
| lowfps.mp4 | 30s | 1920x1080 | 1M | Low frame rate video (15fps) (3.0 MB) |
| silent.mp4 | 30s | 1920x1080 | 2M | Video with silent audio track (3.7 MB) |
| highfreq_audio.mp4 | 30s | 1920x1080 | 2M | Video with high frequency audio (4.2 MB) |
| minimal.mp4 | 1s | 1920x1080 | 1M | Minimal duration video (1 second) (0.1 MB) |
| odd_dimensions.mp4 | 30s | 1001x751 | 2M | Video with odd dimensions (5.4 MB) |

## Special Files

- **corrupt.mp4**: Corrupted video file for error testing (5012 bytes)
- **empty.mp4**: Empty file (0 bytes)
- **invalid.txt**: Text file with .mp4-like name (39 bytes)
- **huge_metadata.mp4**: Video with excessive metadata (1040001 bytes)

## Usage

These test files are used by:
- `edge_path_coverage.py` for comprehensive testing
- Individual test modules for specific scenarios
- Performance and stress testing
- Error handling validation
