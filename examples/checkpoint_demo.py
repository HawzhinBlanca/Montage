#!/usr/bin/env python3
"""Demonstration of checkpoint system for crash recovery"""

import sys
import time
import logging
import random
from checkpoint import CheckpointManager, SmartVideoEditorCheckpoint
from db import Database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_video_processing_with_checkpoints(job_id: str, simulate_crash_at: str = None):
    """Simulate video processing pipeline with checkpoint support"""
    
    checkpoint_mgr = CheckpointManager()
    editor_checkpoint = SmartVideoEditorCheckpoint(checkpoint_mgr)
    db = Database()
    
    # Check for existing job
    existing_job = db.find_one('video_job', {'id': job_id})
    if not existing_job:
        # Create new job
        db.insert('video_job', {
            'id': job_id,
            'src_hash': 'demo_hash_' + job_id[:8],
            'status': 'queued',
            'input_path': '/demo/input.mp4'
        })
        logger.info(f"Created new job: {job_id}")
    else:
        logger.info(f"Found existing job: {job_id} (status: {existing_job['status']})")
    
    # Check for resume point
    resume_info = editor_checkpoint.get_resume_point(job_id)
    
    if resume_info:
        logger.info(f"🔄 RESUMING from checkpoint!")
        logger.info(f"  Last completed stage: {resume_info['last_completed_stage']}")
        logger.info(f"  Resuming from: {resume_info['resume_from_stage']}")
        start_stage = resume_info['resume_from_stage']
    else:
        logger.info("📝 Starting fresh processing")
        start_stage = 'validation'
    
    # Define processing stages
    stages = [
        'validation',
        'analysis',
        'transcription',
        'highlight_detection',
        'editing',
        'audio_normalization',
        'color_correction',
        'export'
    ]
    
    # Start from appropriate stage
    start_index = stages.index(start_stage)
    
    for stage in stages[start_index:]:
        # Check if should skip
        if editor_checkpoint.should_skip_stage(job_id, stage):
            logger.info(f"⏭️  Skipping completed stage: {stage}")
            continue
        
        # Simulate crash
        if simulate_crash_at == stage:
            logger.error(f"💥 SIMULATING CRASH at stage: {stage}")
            db.update('video_job', {
                'status': f'crashed_at_{stage}',
                'error_message': 'Simulated crash for demo'
            }, {'id': job_id})
            sys.exit(1)
        
        # Process stage
        logger.info(f"▶️  Processing stage: {stage}")
        db.update('video_job', {'status': f'processing_{stage}'}, {'id': job_id})
        
        # Simulate work
        time.sleep(random.uniform(0.5, 1.5))
        
        # Generate stage results
        stage_results = generate_stage_results(stage)
        
        # Save checkpoint
        editor_checkpoint.save_stage_data(job_id, stage, **stage_results)
        logger.info(f"✅ Completed stage: {stage}")
        
        # Update database
        db.update('video_job', {'status': f'completed_{stage}'}, {'id': job_id})
    
    # Mark job as completed
    db.update('video_job', {
        'status': 'completed',
        'completed_at': 'NOW()'
    }, {'id': job_id})
    
    logger.info(f"🎉 Job {job_id} completed successfully!")
    
    # Show final progress
    progress = checkpoint_mgr.get_job_progress(job_id)
    logger.info(f"Final progress: {len(progress['completed_stages'])} stages completed")
    
    # Clean up checkpoints
    logger.info("Cleaning up checkpoints...")
    checkpoint_mgr.delete_job_checkpoints(job_id)


def generate_stage_results(stage: str) -> dict:
    """Generate mock results for each stage"""
    results = {
        'validation': {
            'valid': True,
            'duration': 300.5,
            'codec': 'h264',
            'resolution': '1920x1080',
            'fps': 30
        },
        'analysis': {
            'segments_found': 15,
            'total_score': 0.85,
            'peak_moments': [30, 60, 120, 180, 240],
            'processing_time': 12.3
        },
        'transcription': {
            'word_count': 1500,
            'language': 'en',
            'confidence': 0.92,
            'provider': 'whisper'
        },
        'highlight_detection': {
            'highlights': [
                {'start': 30, 'end': 60, 'score': 0.9},
                {'start': 120, 'end': 150, 'score': 0.85},
                {'start': 200, 'end': 230, 'score': 0.88}
            ],
            'total_highlights': 3
        },
        'editing': {
            'cuts_made': 12,
            'transitions_added': 11,
            'output_duration': 180.2
        },
        'audio_normalization': {
            'loudness_lufs': -16.0,
            'peak_db': -1.0,
            'lra': 7.5,
            'normalization_applied': True
        },
        'color_correction': {
            'color_space': 'bt709',
            'adjustments': {
                'brightness': 0,
                'contrast': 1.05,
                'saturation': 1.02
            }
        },
        'export': {
            'output_path': '/demo/output.mp4',
            'file_size_mb': 450.2,
            'encoding_time': 45.6,
            'bitrate': '8000k'
        }
    }
    
    return results.get(stage, {'completed': True})


def main():
    """Demo script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Checkpoint system demonstration')
    parser.add_argument('--job-id', default='demo-job-001', help='Job ID to process')
    parser.add_argument('--crash-at', help='Simulate crash at specified stage')
    parser.add_argument('--show-progress', action='store_true', help='Show job progress and exit')
    
    args = parser.parse_args()
    
    if args.show_progress:
        checkpoint_mgr = CheckpointManager()
        progress = checkpoint_mgr.get_job_progress(args.job_id)
        
        print(f"\nJob Progress for {args.job_id}:")
        print(f"Completed stages: {', '.join(progress['completed_stages'])}")
        print(f"Last stage: {progress.get('last_stage', 'None')}")
        
        if progress['checkpoints']:
            print("\nCheckpoint details:")
            for cp in progress['checkpoints']:
                print(f"  - {cp['stage']}: TTL {cp['ttl_seconds']}s")
    else:
        logger.info(f"Starting demo with job_id: {args.job_id}")
        if args.crash_at:
            logger.warning(f"Will simulate crash at stage: {args.crash_at}")
        
        simulate_video_processing_with_checkpoints(args.job_id, args.crash_at)


if __name__ == "__main__":
    main()