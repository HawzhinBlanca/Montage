#!/usr/bin/env python3
"""
Montage - AI-Powered Video Processing Pipeline
Clean entry point for the professional video processing system
"""

import sys
import os
import subprocess
import argparse
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Montage pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Montage - AI-powered video highlight extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with AI analysis (default)
  python run_montage.py video.mp4
  
  # Create vertical format for social media
  python run_montage.py video.mp4 --vertical
  
  # Use premium AI mode
  python run_montage.py video.mp4 --mode premium
  
  # Show video info only
  python run_montage.py video.mp4 --info
        """
    )
    
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--mode', choices=['smart', 'premium'], default='smart',
                       help='Processing mode (smart=efficient, premium=highest quality)')
    parser.add_argument('--vertical', action='store_true',
                       help='Output vertical format (1080x1920) for social media')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--info', action='store_true',
                       help='Show video info only')
    parser.add_argument('--no-server', action='store_true',
                       help='Skip MCP server (use direct FFmpeg)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video):
        logger.error(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Prepare command
    cmd = [
        sys.executable, '-m', 'src.run_pipeline',
        args.video,
        '--mode', args.mode
    ]
    
    if args.info:
        cmd.append('--info')
    
    if args.no_server:
        cmd.append('--no-server')
    
    # Note: vertical format is handled in the pipeline based on MCP request
    # This is a placeholder for future direct CLI support
    
    logger.info(f"🎬 Processing video: {args.video}")
    logger.info(f"   Mode: {args.mode}")
    if args.vertical:
        logger.info(f"   Format: Vertical (1080x1920)")
    
    try:
        # Run the pipeline
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            logger.info("✅ Video processing complete!")
            
            # Show output location
            if not args.info:
                output_dir = os.path.join(project_root, 'output')
                logger.info(f"📁 Check output directory: {output_dir}")
        else:
            logger.error("❌ Video processing failed")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()