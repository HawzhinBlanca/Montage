#!/usr/bin/env python3
"""
Comprehensive metrics and progress tracking system
Real-time monitoring with Prometheus instrumentation
"""

import time
import threading
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Any, Optional, Callable

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    start_http_server,
    CollectorRegistry,
    REGISTRY,
)

logger = logging.getLogger(__name__)


class MetricsManager:
    """Singleton metrics manager with comprehensive tracking"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._server_started = False

        # Job tracking metrics
        self.jobs_total = Counter(
            "montage_jobs_total",
            "Total number of jobs by status",
            ["status"],  # queued, processing, completed, failed
        )

        self.jobs_in_progress = Gauge(
            "montage_jobs_in_progress", "Number of jobs currently being processed"
        )

        # Processing time metrics
        self.processing_duration = Histogram(
            "montage_processing_duration_seconds",
            "Processing time by stage",
            ["stage"],  # validation, transcription, analysis, editing
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        self.processing_ratio = Histogram(
            "montage_processing_ratio",
            "Processing time to video duration ratio by stage",
            ["stage"],
            buckets=[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        )

        # API cost tracking
        self.cost_usd_total = Counter(
            "montage_api_cost_usd_total",
            "Total API costs in USD",
            ["api_name", "job_id"],
        )

        self.api_calls_total = Counter(
            "montage_api_calls_total",
            "Total API calls by service and status",
            ["api_name", "status"],  # success, error, rate_limited
        )

        # Budget tracking
        self.budget_remaining = Gauge(
            "montage_budget_remaining_usd", "Remaining budget in USD"
        )

        self.budget_exceeded_total = Counter(
            "montage_budget_exceeded_total", "Number of times budget was exceeded"
        )

        # Error tracking
        self.errors_total = Counter(
            "montage_errors_total",
            "Total errors by stage and type",
            ["stage", "error_type"],
        )

        # Audio processing metrics
        self.audio_loudness_spread = Histogram(
            "montage_audio_loudness_spread_lu",
            "Audio loudness spread in LU (loudness units)",
            buckets=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        )

        self.audio_normalization_adjustments = Histogram(
            "montage_audio_normalization_db",
            "Audio normalization adjustments in dB",
            buckets=[-10, -5, -2, -1, 0, 1, 2, 5, 10],
        )

        # Segment detection metrics
        self.segments_detected = Histogram(
            "montage_segments_detected",
            "Number of segments detected per video",
            buckets=[1, 3, 5, 10, 15, 20, 30, 50],
        )

        self.highlight_scores = Histogram(
            "montage_highlight_scores",
            "Distribution of highlight scores",
            buckets=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
        )

        # System resource metrics
        self.ffmpeg_processes = Gauge(
            "montage_ffmpeg_processes", "Number of active FFmpeg processes"
        )

        self.database_connections = Gauge(
            "montage_database_connections", "Number of active database connections"
        )

        self.redis_connections = Gauge(
            "montage_redis_connections", "Number of active Redis connections"
        )

        logger.info("Metrics manager initialized")

    def start_http_server(self, port: int = 9090):
        """Start Prometheus HTTP metrics server"""
        if not self._server_started:
            start_http_server(port)
            self._server_started = True
            logger.info(f"Metrics server started on port {port}")

    # Job tracking methods
    def increment_job(self, status: str):
        """Increment job counter for given status"""
        self.jobs_total.labels(status=status).inc()

    @contextmanager
    def track_job_progress(self):
        """Context manager to track job in progress"""
        self.jobs_in_progress.inc()
        try:
            yield
        finally:
            self.jobs_in_progress.dec()

    # Processing time tracking
    @contextmanager
    def track_processing_time(self, stage: str, video_duration: float = None):
        """Context manager to track processing time and calculate ratios"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.processing_duration.labels(stage=stage).observe(elapsed)

            if video_duration and video_duration > 0:
                ratio = elapsed / video_duration
                self.processing_ratio.labels(stage=stage).observe(ratio)

    # Cost tracking methods
    def track_cost(self, api_name: str, cost_usd: float, job_id: str):
        """Track API cost"""
        self.cost_usd_total.labels(api_name=api_name, job_id=job_id).inc(cost_usd)

    def track_api_call(self, api_name: str, status: str):
        """Track API call by status"""
        self.api_calls_total.labels(api_name=api_name, status=status).inc()

    def set_budget_remaining(self, job_id: str, amount_usd: float):
        """Set remaining budget amount"""
        self.budget_remaining.set(amount_usd)
        if amount_usd <= 0:
            self.budget_exceeded_total.inc()

    # Error tracking
    def track_error(self, stage: str, error_type: str):
        """Track error by stage and type"""
        self.errors_total.labels(stage=stage, error_type=error_type).inc()

    # Audio metrics
    def track_audio_spread(self, loudness_spread_lu: float):
        """Track audio loudness spread"""
        self.audio_loudness_spread.observe(loudness_spread_lu)

    def track_audio_normalization(self, adjustment_db: float):
        """Track audio normalization adjustment"""
        self.audio_normalization_adjustments.observe(adjustment_db)

    # Segment metrics
    def track_segments_detected(self, count: int):
        """Track number of segments detected"""
        self.segments_detected.observe(count)

    def track_highlight_score(self, score: float):
        """Track highlight score"""
        self.highlight_scores.observe(score)

    # System resource metrics
    def set_ffmpeg_processes(self, count: int):
        """Set number of active FFmpeg processes"""
        self.ffmpeg_processes.set(count)

    def set_database_connections(self, count: int):
        """Set number of database connections"""
        self.database_connections.set(count)

    def set_redis_connections(self, count: int):
        """Set number of Redis connections"""
        self.redis_connections.set(count)


# Global metrics instance
metrics = MetricsManager()


# Decorator functions
def track_processing_stage(stage: str):
    """Decorator to track processing stage with metrics"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract video_duration from kwargs if available
            video_duration = kwargs.get("video_duration")

            try:
                with metrics.track_processing_time(stage, video_duration):
                    result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Track error
                error_type = type(e).__name__
                metrics.track_error(stage, error_type)
                raise

        return wrapper

    return decorator


def track_api_cost(api_name: str, cost_extractor: Callable):
    """Decorator to track API costs"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            job_id = kwargs.get("job_id", "unknown")

            try:
                result = func(*args, **kwargs)

                # Extract cost from result
                cost = cost_extractor(result)
                if cost > 0:
                    metrics.track_cost(api_name, cost, job_id)

                # Track successful API call
                metrics.track_api_call(api_name, "success")

                return result

            except Exception as e:
                # Determine error type for tracking
                error_str = str(e).lower()
                if "rate limit" in error_str or "quota" in error_str:
                    status = "rate_limited"
                else:
                    status = "error"

                metrics.track_api_call(api_name, status)
                raise

        return wrapper

    return decorator


def with_job_tracking(func):
    """Decorator to track complete job lifecycle"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Mark job as processing
        metrics.increment_job("processing")

        try:
            with metrics.track_job_progress():
                result = func(*args, **kwargs)

            # Mark job as completed
            metrics.increment_job("completed")
            return result

        except Exception as e:
            # Mark job as failed
            metrics.increment_job("failed")
            raise

    return wrapper


# FFmpeg process tracking helpers
def track_ffmpeg_process_start():
    """Track start of FFmpeg process"""
    current = metrics.ffmpeg_processes._value.get()
    metrics.set_ffmpeg_processes(current + 1)


def track_ffmpeg_process_end():
    """Track end of FFmpeg process"""
    current = metrics.ffmpeg_processes._value.get()
    metrics.set_ffmpeg_processes(max(0, current - 1))


# System metrics update helper
def update_system_metrics(db_pool=None, redis_client=None):
    """Update system metrics from external sources"""
    try:
        if db_pool and hasattr(db_pool, "pool") and hasattr(db_pool.pool, "_used"):
            db_connections = len(db_pool.pool._used)
            metrics.set_database_connections(db_connections)

        if redis_client:
            info = redis_client.info()
            redis_connections = info.get("connected_clients", 0)
            metrics.set_redis_connections(redis_connections)

    except Exception as e:
        logger.debug(f"Failed to update system metrics: {e}")


# Progress tracking context managers
@contextmanager
def track_progress(stage: str, total_items: int = None):
    """Context manager for detailed progress tracking"""
    start_time = time.time()
    logger.info(f"Starting {stage}...")

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed {stage} in {elapsed:.1f}s")


if __name__ == "__main__":
    # Test metrics
    logger.basicConfig(level=logging.INFO)

    # Start server
    metrics.start_http_server(9090)

    # Test some metrics
    with metrics.track_job_progress():
        metrics.increment_job("processing")

        with metrics.track_processing_time("test", 60):
            time.sleep(0.1)

        metrics.track_cost("test_api", 0.05, "test-job")
        metrics.increment_job("completed")

    logger.info(
        "Metrics test complete. Server running on http://localhost:9090/metrics"
    )
