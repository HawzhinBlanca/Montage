#!/usr/bin/env python3
"""Demonstration of metrics instrumentation"""

import time
import random
import threading
import logging
from metrics import (
    metrics,
    track_processing_stage,
    track_api_cost,
    with_job_tracking
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoProcessingDemo:
    """Demo video processing with metrics"""
    
    def __init__(self):
        # Start metrics server
        metrics.start_http_server(port=9099)
        logger.info("Metrics server started at http://localhost:9099/metrics")
    
    @with_job_tracking
    def process_job(self, job_id: str):
        """Process a complete video job"""
        logger.info(f"Starting job {job_id}")
        
        try:
            # Validation
            video_duration = self.validate_video(job_id)
            
            # Analysis
            segments = self.analyze_video(job_id, video_duration)
            
            # Transcription
            self.transcribe_audio(job_id, video_duration)
            
            # Editing
            self.edit_video(job_id, segments)
            
            # Audio normalization
            self.normalize_audio(job_id)
            
            logger.info(f"Completed job {job_id}")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            raise
    
    @track_processing_stage('validation')
    def validate_video(self, job_id: str, video_duration: float = None):
        """Validate video file"""
        logger.info(f"  Validating video for job {job_id}")
        
        # Simulate validation time
        time.sleep(random.uniform(0.5, 1.5))
        
        # Return mock duration
        duration = random.uniform(60, 600)  # 1-10 minutes
        return duration
    
    @track_processing_stage('analysis')
    def analyze_video(self, job_id: str, video_duration: float):
        """Analyze video content"""
        logger.info(f"  Analyzing video for job {job_id}")
        
        # Simulate analysis time (proportional to video duration)
        processing_time = video_duration * random.uniform(0.1, 0.3)
        time.sleep(min(processing_time, 5))  # Cap at 5 seconds for demo
        
        # Generate mock segments
        num_segments = random.randint(5, 30)
        metrics.track_segments_detected(num_segments)
        
        segments = []
        for i in range(num_segments):
            score = random.uniform(0.5, 0.95)
            metrics.track_highlight_score(score)
            segments.append({
                'start': i * 30,
                'end': (i + 1) * 30,
                'score': score
            })
        
        logger.info(f"    Found {num_segments} segments")
        return segments
    
    @track_api_cost('whisper', lambda result: 0.006 * result['duration'] / 60)
    def transcribe_audio(self, job_id: str, video_duration: float):
        """Transcribe audio track"""
        logger.info(f"  Transcribing audio for job {job_id}")
        
        # Simulate API call
        time.sleep(random.uniform(1, 3))
        
        # Simulate occasional API errors
        if random.random() < 0.1:
            metrics.track_error('transcription', 'APIError')
            raise Exception("Simulated API error")
        
        return {'duration': video_duration, 'words': int(video_duration * 2.5)}
    
    @track_processing_stage('editing')
    def edit_video(self, job_id: str, segments: list):
        """Edit video based on segments"""
        logger.info(f"  Editing video for job {job_id}")
        
        # Track FFmpeg process
        metrics.set_ffmpeg_processes(metrics.ffmpeg_processes._value.get() + 1)
        
        try:
            # Simulate editing time
            time.sleep(random.uniform(2, 5))
        finally:
            # Always decrement FFmpeg counter
            metrics.set_ffmpeg_processes(max(0, metrics.ffmpeg_processes._value.get() - 1))
    
    @track_processing_stage('audio_normalization')
    def normalize_audio(self, job_id: str):
        """Normalize audio levels"""
        logger.info(f"  Normalizing audio for job {job_id}")
        
        # Simulate normalization
        time.sleep(random.uniform(0.5, 1.5))
        
        # Track audio metrics
        loudness_spread = random.uniform(0.5, 2.0)
        metrics.track_audio_spread(loudness_spread)
        
        adjustment_db = random.uniform(-5, 5)
        metrics.track_audio_normalization(adjustment_db)
        
        logger.info(f"    Loudness spread: {loudness_spread:.1f} LU")
        logger.info(f"    Adjustment: {adjustment_db:+.1f} dB")
    
    def simulate_workload(self, num_jobs: int = 10):
        """Simulate processing multiple jobs"""
        logger.info(f"\nSimulating {num_jobs} video processing jobs...")
        
        # Update system metrics periodically
        def update_system():
            while self._running:
                # Simulate connection counts
                metrics.set_database_connections(random.randint(5, 15))
                metrics.set_redis_connections(random.randint(3, 8))
                time.sleep(5)
        
        self._running = True
        system_thread = threading.Thread(target=update_system)
        system_thread.start()
        
        try:
            for i in range(num_jobs):
                job_id = f"demo-job-{i:03d}"
                
                # Queue the job
                metrics.increment_job('queued')
                
                # Simulate queue delay
                time.sleep(random.uniform(0.5, 2))
                
                # Process job
                try:
                    self.process_job(job_id)
                    
                    # Track cost
                    total_cost = random.uniform(0.10, 0.50)
                    metrics.set_budget_remaining(job_id, 5.00 - total_cost)
                    
                except Exception as e:
                    logger.error(f"Job {job_id} failed: {e}")
                
                # Small delay between jobs
                time.sleep(random.uniform(1, 3))
        
        finally:
            self._running = False
            system_thread.join()
        
        logger.info("\nSimulation complete!")
        self.print_summary()
    
    def print_summary(self):
        """Print metrics summary"""
        logger.info("\n=== Metrics Summary ===")
        
        # Get some metric values
        jobs_completed = metrics.jobs_total.labels(status='completed')._value.get()
        jobs_failed = metrics.jobs_total.labels(status='failed')._value.get()
        total_cost = sum(
            metrics.cost_usd_total.labels(api_name='whisper', job_id=f'demo-job-{i:03d}')._value.get()
            for i in range(100)
        )
        
        logger.info(f"Jobs completed: {jobs_completed}")
        logger.info(f"Jobs failed: {jobs_failed}")
        logger.info(f"Total API cost: ${total_cost:.2f}")
        logger.info(f"Success rate: {jobs_completed / (jobs_completed + jobs_failed) * 100:.1f}%")
        
        logger.info("\nMetrics available at: http://localhost:9099/metrics")


def main():
    """Run the demo"""
    demo = VideoProcessingDemo()
    
    try:
        # Run simulation
        demo.simulate_workload(num_jobs=5)
        
        logger.info("\nPress Ctrl+C to stop the metrics server...")
        
        # Keep server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\nShutting down...")


if __name__ == "__main__":
    main()