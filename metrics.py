"""Prometheus metrics instrumentation for video processing pipeline"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    start_http_server, generate_latest,
    REGISTRY, CollectorRegistry
)
from config import Config

logger = logging.getLogger(__name__)


class MetricsManager:
    """Manages Prometheus metrics for the video processing pipeline"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Job metrics
        self.jobs_total = Counter(
            'video_processing_jobs_total',
            'Total number of video processing jobs',
            ['status']  # queued, processing, completed, failed
        )
        
        self.jobs_in_progress = Gauge(
            'video_processing_jobs_in_progress',
            'Number of jobs currently being processed'
        )
        
        # Cost metrics
        self.cost_usd_total = Counter(
            'video_processing_cost_usd_total',
            'Total cost in USD for API calls',
            ['api_name', 'job_id']
        )
        
        self.cost_per_job = Summary(
            'video_processing_cost_per_job_usd',
            'Cost distribution per job in USD'
        )
        
        # Performance metrics
        self.processing_duration = Histogram(
            'video_processing_duration_seconds',
            'Time spent processing videos',
            ['stage'],  # validation, analysis, transcription, etc.
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
        )
        
        self.processing_ratio = Histogram(
            'video_processing_ratio',
            'Ratio of processing time to video duration',
            ['stage'],
            buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
        )
        
        # Audio quality metrics
        self.audio_loudness_spread = Histogram(
            'video_processing_audio_loudness_spread_lu',
            'Audio loudness spread in LU units',
            buckets=(0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
        )
        
        self.audio_normalization_adjustments = Histogram(
            'video_processing_audio_normalization_db',
            'Audio normalization adjustments in dB',
            buckets=(-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20)
        )
        
        # Video quality metrics
        self.segments_detected = Histogram(
            'video_processing_segments_detected',
            'Number of segments detected per video',
            buckets=(1, 5, 10, 20, 30, 50, 100, 200)
        )
        
        self.highlight_scores = Histogram(
            'video_processing_highlight_scores',
            'Distribution of highlight scores',
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
        )
        
        # Error metrics
        self.errors_total = Counter(
            'video_processing_errors_total',
            'Total number of errors',
            ['stage', 'error_type']
        )
        
        # System metrics
        self.ffmpeg_processes = Gauge(
            'video_processing_ffmpeg_processes',
            'Number of active FFmpeg processes'
        )
        
        self.database_connections = Gauge(
            'video_processing_database_connections',
            'Number of active database connections'
        )
        
        self.redis_connections = Gauge(
            'video_processing_redis_connections',
            'Number of active Redis connections'
        )
        
        # API rate limiting
        self.api_calls_total = Counter(
            'video_processing_api_calls_total',
            'Total number of API calls',
            ['api_name', 'status']  # success, rate_limited, error
        )
        
        self.api_rate_limit_backoff = Histogram(
            'video_processing_api_rate_limit_backoff_seconds',
            'Time spent backing off due to rate limits',
            ['api_name'],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
        )
        
        # Budget tracking
        self.budget_remaining = Gauge(
            'video_processing_budget_remaining_usd',
            'Remaining budget in USD per job'
        )
        
        self.budget_exceeded_total = Counter(
            'video_processing_budget_exceeded_total',
            'Number of times budget was exceeded'
        )
        
        # Info metrics
        self.build_info = Info(
            'video_processing_build',
            'Build information'
        )
        self.build_info.info({
            'version': '1.0.0',
            'python_version': '3.8+'
        })
        
        self._initialized = True
        self._server_started = False
        
        logger.info("Metrics manager initialized")
    
    def start_http_server(self, port: Optional[int] = None):
        """Start Prometheus HTTP server"""
        if self._server_started:
            logger.warning("Metrics server already started")
            return
        
        port = port or Config.PROMETHEUS_PORT
        
        try:
            start_http_server(port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def increment_job(self, status: str):
        """Increment job counter"""
        self.jobs_total.labels(status=status).inc()
    
    def track_cost(self, api_name: str, cost_usd: float, job_id: str = "unknown"):
        """Track API cost"""
        self.cost_usd_total.labels(api_name=api_name, job_id=job_id).inc(cost_usd)
        self.cost_per_job.observe(cost_usd)
    
    def track_audio_spread(self, spread_lu: float):
        """Track audio loudness spread"""
        self.audio_loudness_spread.observe(spread_lu)
    
    @contextmanager
    def track_processing_time(self, stage: str, video_duration: Optional[float] = None):
        """Context manager to track processing time"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.processing_duration.labels(stage=stage).observe(duration)
            
            if video_duration and video_duration > 0:
                ratio = duration / video_duration
                self.processing_ratio.labels(stage=stage).observe(ratio)
    
    @contextmanager
    def track_job_progress(self):
        """Context manager to track jobs in progress"""
        self.jobs_in_progress.inc()
        try:
            yield
        finally:
            self.jobs_in_progress.dec()
    
    def track_error(self, stage: str, error_type: str):
        """Track errors"""
        self.errors_total.labels(stage=stage, error_type=error_type).inc()
    
    def set_budget_remaining(self, job_id: str, remaining_usd: float):
        """Set remaining budget for a job"""
        self.budget_remaining.set(remaining_usd)
        
        if remaining_usd <= 0:
            self.budget_exceeded_total.inc()
    
    def track_api_call(self, api_name: str, status: str):
        """Track API call"""
        self.api_calls_total.labels(api_name=api_name, status=status).inc()
    
    def track_rate_limit_backoff(self, api_name: str, backoff_seconds: float):
        """Track rate limit backoff time"""
        self.api_rate_limit_backoff.labels(api_name=api_name).observe(backoff_seconds)
    
    def set_ffmpeg_processes(self, count: int):
        """Set number of active FFmpeg processes"""
        self.ffmpeg_processes.set(count)
    
    def set_database_connections(self, active: int):
        """Set number of active database connections"""
        self.database_connections.set(active)
    
    def set_redis_connections(self, active: int):
        """Set number of active Redis connections"""
        self.redis_connections.set(active)
    
    def track_segments_detected(self, count: int):
        """Track number of segments detected"""
        self.segments_detected.observe(count)
    
    def track_highlight_score(self, score: float):
        """Track highlight score distribution"""
        self.highlight_scores.observe(score)
    
    def track_audio_normalization(self, adjustment_db: float):
        """Track audio normalization adjustments"""
        self.audio_normalization_adjustments.observe(adjustment_db)


# Singleton instance
metrics = MetricsManager()


# Decorators for easy instrumentation

def track_processing_stage(stage: str):
    """Decorator to track processing time for a stage"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract video duration from arguments
            video_duration = kwargs.get('video_duration')
            
            with metrics.track_processing_time(stage, video_duration):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error_type = type(e).__name__
                    metrics.track_error(stage, error_type)
                    raise
        return wrapper
    return decorator


def track_api_cost(api_name: str, cost_calculator: Callable[[Any], float]):
    """Decorator to track API costs"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            job_id = kwargs.get('job_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate cost based on result
                cost = cost_calculator(result)
                metrics.track_cost(api_name, cost, job_id)
                
                # Track successful API call
                metrics.track_api_call(api_name, 'success')
                
                return result
            except Exception as e:
                # Track failed API call
                if 'rate limit' in str(e).lower():
                    metrics.track_api_call(api_name, 'rate_limited')
                else:
                    metrics.track_api_call(api_name, 'error')
                raise
        return wrapper
    return decorator


def with_job_tracking(func):
    """Decorator to track job execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with metrics.track_job_progress():
            metrics.increment_job('processing')
            
            try:
                result = func(*args, **kwargs)
                metrics.increment_job('completed')
                return result
            except Exception as e:
                metrics.increment_job('failed')
                raise
    return wrapper


# Helper functions for common metric updates

def update_system_metrics(db_pool=None, redis_client=None):
    """Update system resource metrics"""
    if db_pool:
        # Assuming the pool has a method to get connection count
        active_connections = getattr(db_pool.pool, '_used', set())
        metrics.set_database_connections(len(active_connections))
    
    if redis_client:
        try:
            info = redis_client.info()
            connected_clients = info.get('connected_clients', 0)
            metrics.set_redis_connections(connected_clients)
        except:
            pass


def track_ffmpeg_process_start():
    """Track FFmpeg process start"""
    current = metrics.ffmpeg_processes._value.get()
    metrics.set_ffmpeg_processes(current + 1)


def track_ffmpeg_process_end():
    """Track FFmpeg process end"""
    current = metrics.ffmpeg_processes._value.get()
    metrics.set_ffmpeg_processes(max(0, current - 1))


# Example usage for SmartVideoEditor integration
class MetricsExample:
    """Example of how to integrate metrics into SmartVideoEditor"""
    
    @track_processing_stage('validation')
    def validate_input(self, job_id: str, input_path: str, video_duration: float = None):
        """Example validation with metrics"""
        # Validation logic here
        return {'valid': True}
    
    @track_processing_stage('analysis')
    def analyze_video(self, job_id: str, video_duration: float):
        """Example analysis with metrics"""
        # Analysis logic here
        segments = 15
        metrics.track_segments_detected(segments)
        
        # Track highlight scores
        for score in [0.8, 0.9, 0.75, 0.85]:
            metrics.track_highlight_score(score)
        
        return {'segments': segments}
    
    @track_api_cost('openai', lambda result: 0.02 * len(result.get('tokens', [])) / 1000)
    def call_openai_api(self, job_id: str, prompt: str):
        """Example API call with cost tracking"""
        # API call logic here
        return {'tokens': [1, 2, 3], 'response': 'test'}


if __name__ == "__main__":
    # Start metrics server
    logging.basicConfig(level=logging.INFO)
    
    metrics.start_http_server()
    
    logger.info("Metrics server running. Access metrics at http://localhost:9099/metrics")
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down metrics server")