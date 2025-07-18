"""Test metrics instrumentation"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from prometheus_client import REGISTRY
from metrics import (
    MetricsManager,
    metrics,
    track_processing_stage,
    track_api_cost,
    with_job_tracking,
    update_system_metrics,
    track_ffmpeg_process_start,
    track_ffmpeg_process_end
)


class TestMetricsManager:
    """Test metrics manager functionality"""
    
    def test_singleton_instance(self):
        """Test metrics manager is a singleton"""
        m1 = MetricsManager()
        m2 = MetricsManager()
        assert m1 is m2
        assert m1 is metrics
    
    def test_increment_job(self):
        """Test job counter increments"""
        # Get initial value
        before_queued = metrics.jobs_total.labels(status='queued')._value.get()
        before_completed = metrics.jobs_total.labels(status='completed')._value.get()
        
        # Increment counters
        metrics.increment_job('queued')
        metrics.increment_job('completed')
        
        # Check values increased
        after_queued = metrics.jobs_total.labels(status='queued')._value.get()
        after_completed = metrics.jobs_total.labels(status='completed')._value.get()
        
        assert after_queued == before_queued + 1
        assert after_completed == before_completed + 1
    
    def test_track_cost(self):
        """Test cost tracking"""
        # Track some costs
        metrics.track_cost('openai', 0.05, 'job-123')
        metrics.track_cost('openai', 0.03, 'job-123')
        metrics.track_cost('whisper', 0.02, 'job-456')
        
        # Check counter increased
        openai_total = metrics.cost_usd_total.labels(api_name='openai', job_id='job-123')._value.get()
        whisper_total = metrics.cost_usd_total.labels(api_name='whisper', job_id='job-456')._value.get()
        
        assert openai_total >= 0.08  # Should be 0.08
        assert whisper_total >= 0.02  # Should be 0.02
    
    def test_track_processing_time(self):
        """Test processing time tracking"""
        # Track processing time
        with metrics.track_processing_time('validation', video_duration=60.0):
            time.sleep(0.1)  # Simulate work
        
        # Check histogram was updated
        validation_count = metrics.processing_duration.labels(stage='validation')._count.get()
        assert validation_count > 0
        
        # Check ratio was calculated
        ratio_count = metrics.processing_ratio.labels(stage='validation')._count.get()
        assert ratio_count > 0
    
    def test_track_job_progress(self):
        """Test job progress tracking"""
        # Check initial value
        initial = metrics.jobs_in_progress._value.get()
        
        # Track job progress
        with metrics.track_job_progress():
            # Check gauge increased
            during = metrics.jobs_in_progress._value.get()
            assert during == initial + 1
        
        # Check gauge decreased after context
        after = metrics.jobs_in_progress._value.get()
        assert after == initial
    
    def test_track_error(self):
        """Test error tracking"""
        # Track some errors
        metrics.track_error('transcription', 'APIError')
        metrics.track_error('transcription', 'APIError')
        metrics.track_error('validation', 'CorruptedVideoError')
        
        # Check counters
        api_errors = metrics.errors_total.labels(stage='transcription', error_type='APIError')._value.get()
        corruption_errors = metrics.errors_total.labels(stage='validation', error_type='CorruptedVideoError')._value.get()
        
        assert api_errors >= 2
        assert corruption_errors >= 1
    
    def test_budget_tracking(self):
        """Test budget tracking"""
        # Set budget remaining
        metrics.set_budget_remaining('job-123', 3.50)
        assert metrics.budget_remaining._value.get() == 3.50
        
        # Track budget exceeded
        before_exceeded = metrics.budget_exceeded_total._value.get()
        metrics.set_budget_remaining('job-456', 0)
        after_exceeded = metrics.budget_exceeded_total._value.get()
        
        assert after_exceeded == before_exceeded + 1
    
    def test_audio_metrics(self):
        """Test audio quality metrics"""
        # Track audio spread
        metrics.track_audio_spread(1.2)
        metrics.track_audio_spread(0.8)
        metrics.track_audio_spread(1.5)
        
        # Check histogram
        count = metrics.audio_loudness_spread._count.get()
        assert count >= 3
        
        # Track normalization
        metrics.track_audio_normalization(-3.5)
        norm_count = metrics.audio_normalization_adjustments._count.get()
        assert norm_count >= 1
    
    def test_segment_metrics(self):
        """Test segment detection metrics"""
        # Track segments
        metrics.track_segments_detected(15)
        metrics.track_segments_detected(8)
        
        count = metrics.segments_detected._count.get()
        assert count >= 2
        
        # Track highlight scores
        for score in [0.75, 0.85, 0.92]:
            metrics.track_highlight_score(score)
        
        score_count = metrics.highlight_scores._count.get()
        assert score_count >= 3
    
    def test_system_metrics(self):
        """Test system resource metrics"""
        # Set FFmpeg processes
        metrics.set_ffmpeg_processes(3)
        assert metrics.ffmpeg_processes._value.get() == 3
        
        # Track FFmpeg process lifecycle
        track_ffmpeg_process_start()
        assert metrics.ffmpeg_processes._value.get() == 4
        
        track_ffmpeg_process_end()
        assert metrics.ffmpeg_processes._value.get() == 3
        
        # Set connection metrics
        metrics.set_database_connections(10)
        assert metrics.database_connections._value.get() == 10
        
        metrics.set_redis_connections(5)
        assert metrics.redis_connections._value.get() == 5


class TestDecorators:
    """Test metric decorators"""
    
    def test_track_processing_stage_decorator(self):
        """Test processing stage tracking decorator"""
        
        @track_processing_stage('test_stage')
        def process_video(video_duration=100):
            time.sleep(0.05)
            return "processed"
        
        # Call decorated function
        result = process_video(video_duration=100)
        assert result == "processed"
        
        # Check metrics were recorded
        count = metrics.processing_duration.labels(stage='test_stage')._count.get()
        assert count > 0
        
        ratio_count = metrics.processing_ratio.labels(stage='test_stage')._count.get()
        assert ratio_count > 0
    
    def test_track_processing_stage_with_error(self):
        """Test processing stage tracking with error"""
        
        @track_processing_stage('error_stage')
        def failing_process():
            raise ValueError("Test error")
        
        # Call should raise but still track error
        with pytest.raises(ValueError):
            failing_process()
        
        # Check error was tracked
        error_count = metrics.errors_total.labels(stage='error_stage', error_type='ValueError')._value.get()
        assert error_count >= 1
    
    def test_track_api_cost_decorator(self):
        """Test API cost tracking decorator"""
        
        @track_api_cost('test_api', lambda result: result['cost'])
        def call_api(job_id='test-job'):
            return {'cost': 0.15, 'data': 'response'}
        
        # Call decorated function
        result = call_api(job_id='job-789')
        assert result['cost'] == 0.15
        
        # Check cost was tracked
        cost = metrics.cost_usd_total.labels(api_name='test_api', job_id='job-789')._value.get()
        assert cost >= 0.15
        
        # Check API call was tracked
        api_calls = metrics.api_calls_total.labels(api_name='test_api', status='success')._value.get()
        assert api_calls >= 1
    
    def test_track_api_cost_with_rate_limit(self):
        """Test API cost tracking with rate limit error"""
        
        @track_api_cost('rate_limited_api', lambda result: 0)
        def rate_limited_call(job_id='test'):
            raise Exception("Rate limit exceeded")
        
        # Call should raise
        with pytest.raises(Exception):
            rate_limited_call()
        
        # Check rate limit was tracked
        rate_limited = metrics.api_calls_total.labels(
            api_name='rate_limited_api', 
            status='rate_limited'
        )._value.get()
        assert rate_limited >= 1
    
    def test_with_job_tracking_decorator(self):
        """Test job tracking decorator"""
        
        @with_job_tracking
        def process_job():
            return "completed"
        
        # Get initial counts
        before_processing = metrics.jobs_total.labels(status='processing')._value.get()
        before_completed = metrics.jobs_total.labels(status='completed')._value.get()
        
        # Call decorated function
        result = process_job()
        assert result == "completed"
        
        # Check counters increased
        after_processing = metrics.jobs_total.labels(status='processing')._value.get()
        after_completed = metrics.jobs_total.labels(status='completed')._value.get()
        
        assert after_processing == before_processing + 1
        assert after_completed == before_completed + 1
    
    def test_with_job_tracking_failure(self):
        """Test job tracking with failure"""
        
        @with_job_tracking
        def failing_job():
            raise RuntimeError("Job failed")
        
        # Get initial count
        before_failed = metrics.jobs_total.labels(status='failed')._value.get()
        
        # Call should raise
        with pytest.raises(RuntimeError):
            failing_job()
        
        # Check failure was tracked
        after_failed = metrics.jobs_total.labels(status='failed')._value.get()
        assert after_failed == before_failed + 1


class TestMetricsIntegration:
    """Test metrics integration scenarios"""
    
    def test_concurrent_metric_updates(self):
        """Test thread-safe metric updates"""
        errors = []
        
        def update_metrics(thread_id):
            try:
                for i in range(100):
                    metrics.increment_job('processing')
                    metrics.track_cost('api', 0.01, f'job-{thread_id}')
                    
                    with metrics.track_processing_time('stage', 60):
                        time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent updates
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_metrics, args=(i,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
    
    @patch('prometheus_client.start_http_server')
    def test_start_http_server(self, mock_start_server):
        """Test HTTP server startup"""
        # Reset server started flag for test
        metrics._server_started = False
        
        # Start server
        metrics.start_http_server(9099)
        
        # Check server was started
        mock_start_server.assert_called_once_with(9099)
        assert metrics._server_started
        
        # Second call should not start again
        metrics.start_http_server(9099)
        assert mock_start_server.call_count == 1
    
    def test_update_system_metrics(self):
        """Test system metrics update helper"""
        # Mock database pool
        mock_pool = Mock()
        mock_pool.pool._used = {1, 2, 3}  # 3 active connections
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.info.return_value = {'connected_clients': 7}
        
        # Update metrics
        update_system_metrics(db_pool=mock_pool, redis_client=mock_redis)
        
        # Check values
        assert metrics.database_connections._value.get() == 3
        assert metrics.redis_connections._value.get() == 7


class TestMetricsExample:
    """Test the example integration class"""
    
    def test_example_usage(self):
        """Test example metrics integration"""
        from metrics import MetricsExample
        
        example = MetricsExample()
        
        # Test validation with metrics
        result = example.validate_input('job-123', '/path/to/video.mp4', video_duration=120)
        assert result['valid']
        
        # Check metrics were recorded
        validation_count = metrics.processing_duration.labels(stage='validation')._count.get()
        assert validation_count > 0
        
        # Test analysis with metrics
        result = example.analyze_video('job-123', video_duration=120)
        assert result['segments'] == 15
        
        # Check segment metrics
        segments_count = metrics.segments_detected._count.get()
        assert segments_count > 0
        
        # Test API call with cost
        result = example.call_openai_api('job-123', 'test prompt')
        
        # Check cost was tracked (0.02 * 3 / 1000 = 0.00006)
        cost = metrics.cost_usd_total.labels(api_name='openai', job_id='job-123')._value.get()
        assert cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])