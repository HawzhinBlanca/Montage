#!/usr/bin/env python3
"""Quick start script for video processing pipeline"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_video_editor import SmartVideoEditor
from db import Database, test_connection
from checkpoint import CheckpointManager
from metrics import metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check system prerequisites"""
    logger.info("Checking prerequisites...")
    
    # Check database connection
    if not test_connection():
        logger.error("❌ Database connection failed")
        logger.info("Please ensure PostgreSQL is running and configured in .env")
        return False
    logger.info("✅ Database connection successful")
    
    # Check Redis connection
    try:
        cm = CheckpointManager()
        if not cm.health_check():
            raise Exception("Redis health check failed")
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.info("Please ensure Redis is running")
        return False
    
    # Check FFmpeg
    if not os.path.exists(os.environ.get('FFMPEG_PATH', 'ffmpeg')):
        import subprocess
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("✅ FFmpeg found")
        except:
            logger.error("❌ FFmpeg not found")
            logger.info("Please install FFmpeg with zscale support")
            return False
    else:
        logger.info("✅ FFmpeg found")
    
    return True


def create_sample_highlights(duration: float) -> List[Dict[str, Any]]:
    """Create sample highlight segments"""
    highlights = []
    
    # Create 3-5 highlights distributed through the video
    num_highlights = min(5, max(3, int(duration / 120)))  # One per 2 minutes
    
    segment_duration = 30  # 30-second segments
    gap = (duration - (num_highlights * segment_duration)) / (num_highlights + 1)
    
    current_time = gap
    for i in range(num_highlights):
        highlights.append({
            'start_time': current_time,
            'end_time': current_time + segment_duration,
            'score': 0.8 + (i * 0.05),  # Increasing scores
            'transition': 'fade',
            'transition_duration': 0.5
        })
        current_time += segment_duration + gap
    
    return highlights


def process_video(input_path: str, output_dir: str = "./output"):
    """Process a video with automatic highlight detection"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize editor
    logger.info("Initializing video editor...")
    editor = SmartVideoEditor()
    
    # Create job in database
    db = Database()
    job_id = db.insert('video_job', {
        'src_hash': 'quickstart_demo',
        'status': 'queued',
        'input_path': input_path
    })
    logger.info(f"Created job: {job_id}")
    
    # Get video duration (simplified - in production use video_validator)
    import subprocess
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ], capture_output=True, text=True)
    
    duration = float(result.stdout.strip())
    logger.info(f"Video duration: {duration:.1f} seconds")
    
    # Create sample highlights
    highlights = create_sample_highlights(duration)
    logger.info(f"Created {len(highlights)} highlight segments")
    
    for i, h in enumerate(highlights):
        logger.info(f"  Segment {i+1}: {h['start_time']:.1f}s - {h['end_time']:.1f}s")
    
    # Process video
    logger.info("Starting video processing...")
    try:
        result = editor.process_job(job_id, input_path, highlights)
        
        logger.info("✅ Processing completed successfully!")
        logger.info(f"Output: {result['output_path']}")
        logger.info(f"Audio loudness: {result['audio_loudness']} LUFS")
        logger.info(f"Color space: {result['color_space']}")
        logger.info(f"Segments processed: {result['segments_processed']}")
        
        # Show metrics URL
        logger.info(f"\nView metrics at: http://localhost:9099/metrics")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Quick start for video processing pipeline'
    )
    parser.add_argument(
        'input_video',
        help='Path to input video file'
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip prerequisite checks'
    )
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            logger.error("Prerequisites check failed. Please fix issues and try again.")
            sys.exit(1)
    
    # Check input file
    if not os.path.exists(args.input_video):
        logger.error(f"Input file not found: {args.input_video}")
        sys.exit(1)
    
    # Process video
    try:
        process_video(args.input_video, args.output_dir)
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()