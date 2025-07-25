"""
Celery tasks for video processing pipeline
Tasks.md Step 1: Real task implementations
"""
from celery import Task
from .celery_app import app
import os
import sys
import time
import logging

# Add parent directory to path for imports

logger = logging.getLogger(__name__)

@app.task(bind=True, name='montage.process_video')
def process_video(self: Task, video_path: str, job_id: str, options: dict = None):
    """
    Main video processing task
    """
    try:
        logger.info(f"Processing video {video_path} for job {job_id}")
        
        # Import here to avoid circular imports
        from montage.core.analyze_video import analyze_video
        from montage.core.highlight_selector import select_highlights
        
        # Update task state
        self.update_state(state='ANALYZING', meta={'current': 1, 'total': 3})
        
        # Analyze video
        analysis = analyze_video(video_path, 
                               use_premium=options.get('use_premium', False),
                               speech_density_variant=options.get('variant', 'control'))
        
        # Update state
        self.update_state(state='SELECTING', meta={'current': 2, 'total': 3})
        
        # Select highlights
        highlights = analysis.get('highlights', [])
        
        # Update state
        self.update_state(state='COMPLETE', meta={'current': 3, 'total': 3})
        
        return {
            'job_id': job_id,
            'status': 'success',
            'highlights': highlights,
            'transcript_words': len(analysis.get('words', [])),
            'cost': analysis.get('cost', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return {
            'job_id': job_id,
            'status': 'error',
            'error': str(e)
        }

@app.task(name='montage.health_check')
def health_check():
    """Health check task"""
    return {'status': 'healthy', 'timestamp': time.time()}