#!/usr/bin/env python3
"""
Create comprehensive test video files for edge case testing
"""

import os
import subprocess
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def create_test_video(
    output_path: str,
    duration: int = 30,
    width: int = 1920,
    height: int = 1080,
    bitrate: str = "1M",
    fps: int = 30,
    audio_freq: int = 1000,
    pattern: str = "testsrc",
):
    """Create a test video file using FFmpeg"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite existing files
            "-f",
            "lavfi",
            "-i",
            f"{pattern}=duration={duration}:size={width}x{height}:rate={fps}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={audio_freq}:duration={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-b:v",
            bitrate,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",  # Optimize for streaming
            output_path,
        ]

        logger.info(
            f"Creating video: {output_path} ({duration}s, {width}x{height}, {bitrate})"
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False

        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"Created: {output_path} ({file_size:.1f} MB)")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout creating video: {output_path}")
        return False
    except Exception as e:
        logger.error(f"Error creating video {output_path}: {e}")
        return False


def create_all_test_videos():
    """Create comprehensive set of test videos"""
    base_dir = Path(__file__).parent

    test_videos = {
        # Valid test videos
        "short.mp4": {
            "duration": 15,
            "width": 1920,
            "height": 1080,
            "bitrate": "2M",
            "description": "Short valid video for basic testing",
        },
        "medium.mp4": {
            "duration": 120,  # 2 minutes
            "width": 1920,
            "height": 1080,
            "bitrate": "4M",
            "description": "Medium length video for processing tests",
        },
        "long.mp4": {
            "duration": 900,  # 15 minutes (reduced from 3 hours for practicality)
            "width": 1920,
            "height": 1080,
            "bitrate": "2M",
            "description": "Long video for stress testing",
        },
        # Different aspect ratios
        "vertical.mp4": {
            "duration": 60,
            "width": 720,
            "height": 1280,  # 9:16 vertical
            "bitrate": "2M",
            "description": "Vertical video for smart crop testing",
        },
        "square.mp4": {
            "duration": 45,
            "width": 1080,
            "height": 1080,  # 1:1 square
            "bitrate": "2M",
            "description": "Square aspect ratio video",
        },
        "ultrawide.mp4": {
            "duration": 30,
            "width": 3440,
            "height": 1440,  # 21:9 ultrawide
            "bitrate": "8M",
            "description": "Ultrawide aspect ratio video",
        },
        # Different resolutions
        "lowres.mp4": {
            "duration": 30,
            "width": 320,
            "height": 240,  # Very low resolution
            "bitrate": "500k",
            "description": "Low resolution video",
        },
        "4k.mp4": {
            "duration": 30,
            "width": 3840,
            "height": 2160,  # 4K resolution
            "bitrate": "20M",
            "description": "4K resolution video",
        },
        # Different frame rates
        "highfps.mp4": {
            "duration": 20,
            "width": 1920,
            "height": 1080,
            "bitrate": "8M",
            "fps": 60,
            "description": "High frame rate video (60fps)",
        },
        "lowfps.mp4": {
            "duration": 30,
            "width": 1920,
            "height": 1080,
            "bitrate": "1M",
            "fps": 15,
            "description": "Low frame rate video (15fps)",
        },
        # Different patterns and content types
        "colortest.mp4": {
            "duration": 30,
            "width": 1920,
            "height": 1080,
            "bitrate": "4M",
            "pattern": "color=red:duration=10,color=green:duration=10,color=blue:duration=10",
            "description": "Color test pattern video",
        },
        "noisetest.mp4": {
            "duration": 20,
            "width": 1920,
            "height": 1080,
            "bitrate": "6M",
            "pattern": "geq=random(1)*255:128:128",  # Noise pattern
            "description": "Noise pattern video for motion detection testing",
        },
        # Different audio characteristics
        "silent.mp4": {
            "duration": 30,
            "width": 1920,
            "height": 1080,
            "bitrate": "2M",
            "audio_freq": 0,  # Will create silent audio
            "description": "Video with silent audio track",
        },
        "highfreq_audio.mp4": {
            "duration": 30,
            "width": 1920,
            "height": 1080,
            "bitrate": "2M",
            "audio_freq": 5000,  # High frequency audio
            "description": "Video with high frequency audio",
        },
        # Edge case videos
        "minimal.mp4": {
            "duration": 1,  # Very short
            "width": 1920,
            "height": 1080,
            "bitrate": "1M",
            "description": "Minimal duration video (1 second)",
        },
        "odd_dimensions.mp4": {
            "duration": 30,
            "width": 1001,  # Odd width
            "height": 751,  # Odd height
            "bitrate": "2M",
            "description": "Video with odd dimensions",
        },
    }

    # Special files that need custom creation
    special_files = {
        "corrupt.mp4": "Corrupted video file for error testing",
        "empty.mp4": "Empty file",
        "invalid.txt": "Text file with .mp4-like name",
        "huge_metadata.mp4": "Video with excessive metadata",
    }

    logger.info("Creating test video files...")

    created_count = 0
    failed_count = 0

    # Create regular test videos
    for filename, config in test_videos.items():
        output_path = str(base_dir / filename)

        # Skip if file already exists and is valid
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info(f"Skipping existing file: {filename}")
            continue

        success = create_test_video(
            output_path=output_path,
            duration=config["duration"],
            width=config["width"],
            height=config["height"],
            bitrate=config["bitrate"],
            fps=config.get("fps", 30),
            audio_freq=config.get("audio_freq", 1000),
            pattern=config.get("pattern", "testsrc"),
        )

        if success:
            created_count += 1
            # Write description file
            desc_path = output_path.replace(".mp4", ".txt")
            with open(desc_path, "w") as f:
                f.write(config["description"])
        else:
            failed_count += 1

    # Create special files
    logger.info("Creating special test files...")

    # Corrupted video file
    corrupt_path = base_dir / "corrupt.mp4"
    with open(corrupt_path, "wb") as f:
        # Write some bytes that look like a video header but are corrupted
        f.write(b"\x00\x00\x00\x20ftypmp42")  # Partial MP4 header
        f.write(b"This is definitely not a valid video file content!" * 100)
    logger.info(f"Created corrupted file: {corrupt_path}")

    # Empty file
    empty_path = base_dir / "empty.mp4"
    open(empty_path, "w").close()
    logger.info(f"Created empty file: {empty_path}")

    # Text file with video extension
    invalid_path = base_dir / "invalid.txt"
    with open(invalid_path, "w") as f:
        f.write("This is a plain text file, not a video!")
    logger.info(f"Created invalid file: {invalid_path}")

    # Video with huge metadata (if possible)
    huge_meta_path = base_dir / "huge_metadata.mp4"
    try:
        # Create a normal video first
        if create_test_video(str(huge_meta_path), duration=10):
            # Try to add huge metadata using FFmpeg
            temp_path = str(huge_meta_path).replace(".mp4", "_temp.mp4")
            metadata_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(huge_meta_path),
                "-c",
                "copy",
                "-metadata",
                f'description={"x" * 50000}',  # 50KB of metadata
                "-metadata",
                f'comment={"y" * 50000}',
                temp_path,
            ]

            result = subprocess.run(metadata_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                os.replace(temp_path, huge_meta_path)
                logger.info(f"Created huge metadata file: {huge_meta_path}")
            else:
                logger.warning(f"Failed to add huge metadata: {result.stderr}")
    except Exception as e:
        logger.error(f"Failed to create huge metadata file: {e}")

    # Create a playlist file for batch testing
    playlist_path = base_dir / "test_playlist.txt"
    with open(playlist_path, "w") as f:
        f.write("# Test video playlist\n")
        for filename in test_videos.keys():
            file_path = base_dir / filename
            if os.path.exists(file_path):
                f.write(f"{filename}\n")

    # Create a comprehensive test report
    report_path = base_dir / "test_videos_report.md"
    with open(report_path, "w") as f:
        f.write("# Test Video Files Report\n\n")
        f.write("Generated test videos for comprehensive edge case testing.\n\n")
        f.write("**Summary:**\n")
        f.write(f"- Created: {created_count} videos\n")
        f.write(f"- Failed: {failed_count} videos\n")
        f.write(f"- Special files: {len(special_files)}\n\n")

        f.write("## Video Files\n\n")
        f.write("| File | Duration | Resolution | Bitrate | Description |\n")
        f.write("|------|----------|------------|---------|-------------|\n")

        for filename, config in test_videos.items():
            file_path = base_dir / filename
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                f.write(
                    f"| {filename} | {config['duration']}s | "
                    f"{config['width']}x{config['height']} | {config['bitrate']} | "
                    f"{config['description']} ({size_mb:.1f} MB) |\n"
                )

        f.write("\n## Special Files\n\n")
        for filename, description in special_files.items():
            file_path = base_dir / filename
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                f.write(f"- **{filename}**: {description} ({size_bytes} bytes)\n")

        f.write("\n## Usage\n\n")
        f.write("These test files are used by:\n")
        f.write("- `edge_path_coverage.py` for comprehensive testing\n")
        f.write("- Individual test modules for specific scenarios\n")
        f.write("- Performance and stress testing\n")
        f.write("- Error handling validation\n")

    logger.info("Test video creation complete!")
    logger.info(f"Created: {created_count} videos, Failed: {failed_count}")
    logger.info(f"Report saved to: {report_path}")

    return created_count, failed_count


def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check if FFmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg not found! Please install FFmpeg to create test videos.")
        logger.info("On macOS: brew install ffmpeg")
        logger.info("On Ubuntu: sudo apt-get install ffmpeg")
        sys.exit(1)

    created, failed = create_all_test_videos()

    if failed > 0:
        logger.warning(f"Some video creation failed ({failed} failures)")
        sys.exit(1)
    else:
        logger.info("All test videos created successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
