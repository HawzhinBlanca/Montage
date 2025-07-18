# Metrics Instrumentation Guide

## Overview

The metrics module provides comprehensive Prometheus instrumentation for monitoring the video processing pipeline. It tracks costs, performance, quality, and system health metrics.

## Key Metrics

### Job Metrics
- `video_processing_jobs_total`: Total jobs by status (queued, processing, completed, failed)
- `video_processing_jobs_in_progress`: Currently processing jobs (gauge)

### Cost Metrics
- `video_processing_cost_usd_total`: Total API costs by API name and job ID
- `video_processing_cost_per_job_usd`: Cost distribution per job (summary)
- `video_processing_budget_remaining_usd`: Remaining budget per job (gauge)
- `video_processing_budget_exceeded_total`: Times budget was exceeded

### Performance Metrics
- `video_processing_duration_seconds`: Processing time by stage (histogram)
- `video_processing_ratio`: Processing time to video duration ratio (histogram)

### Quality Metrics
- `video_processing_audio_loudness_spread_lu`: Audio loudness uniformity (histogram)
- `video_processing_audio_normalization_db`: Audio adjustments applied (histogram)
- `video_processing_segments_detected`: Number of segments found (histogram)
- `video_processing_highlight_scores`: Distribution of highlight quality scores (histogram)

### System Metrics
- `video_processing_ffmpeg_processes`: Active FFmpeg processes (gauge)
- `video_processing_database_connections`: Active DB connections (gauge)
- `video_processing_redis_connections`: Active Redis connections (gauge)

### Error Metrics
- `video_processing_errors_total`: Errors by stage and type
- `video_processing_api_calls_total`: API calls by name and status

## Usage

### Basic Setup

```python
from metrics import metrics

# Start Prometheus HTTP server
metrics.start_http_server(port=9099)

# Metrics are now available at http://localhost:9099/metrics
```

### Job Tracking

```python
from metrics import metrics, with_job_tracking

# Track job lifecycle
metrics.increment_job('queued')

# Using decorator
@with_job_tracking
def process_video_job(job_id):
    # Automatically tracks:
    # - jobs_in_progress (increases during execution)
    # - jobs_total with status='processing' at start
    # - jobs_total with status='completed' on success
    # - jobs_total with status='failed' on exception
    
    # Your processing logic here
    pass
```

### Cost Tracking

```python
from metrics import metrics, track_api_cost

# Manual cost tracking
metrics.track_cost('openai', 0.05, job_id='job-123')

# Using decorator with cost calculator
@track_api_cost('openai', lambda result: 0.02 * len(result['tokens']) / 1000)
def call_openai_api(job_id, prompt):
    # API call logic
    return {'tokens': [...], 'response': '...'}

# Budget tracking
metrics.set_budget_remaining('job-123', 4.50)  # $4.50 remaining
```

### Performance Tracking

```python
from metrics import metrics, track_processing_stage

# Manual timing
with metrics.track_processing_time('transcription', video_duration=300):
    # Transcription logic here
    # Automatically calculates duration and ratio
    pass

# Using decorator
@track_processing_stage('analysis')
def analyze_video(job_id, video_duration):
    # Automatically times execution and tracks errors
    return analysis_results
```

### Quality Metrics

```python
# Audio quality
metrics.track_audio_spread(1.2)  # 1.2 LU spread
metrics.track_audio_normalization(-3.5)  # -3.5 dB adjustment

# Video analysis quality
metrics.track_segments_detected(15)  # Found 15 segments
metrics.track_highlight_score(0.92)  # High-quality highlight
```

### Error Tracking

```python
try:
    # Processing logic
    pass
except APIError as e:
    metrics.track_error('transcription', 'APIError')
    raise
```

### System Resource Tracking

```python
from metrics import (
    track_ffmpeg_process_start,
    track_ffmpeg_process_end,
    update_system_metrics
)

# FFmpeg process tracking
track_ffmpeg_process_start()  # When starting FFmpeg
try:
    # Run FFmpeg
    pass
finally:
    track_ffmpeg_process_end()  # Always cleanup

# Update connection metrics
update_system_metrics(db_pool=db.pool, redis_client=redis_conn)
```

## Complete Integration Example

```python
from metrics import metrics, track_processing_stage, track_api_cost
from video_validator import perform_preflight_check
from checkpoint import SmartVideoEditorCheckpoint

class SmartVideoEditor:
    def __init__(self):
        self.checkpoint = SmartVideoEditorCheckpoint(...)
        # Start metrics server
        metrics.start_http_server()
    
    @track_processing_stage('validation')
    def validate_input(self, job_id: str, input_path: str, video_duration: float = None):
        """Validate with automatic metrics"""
        result = perform_preflight_check(job_id, input_path)
        if not result['valid']:
            metrics.track_error('validation', 'InvalidVideoError')
            raise VideoValidationError(result['error'])
        return result
    
    @track_processing_stage('analysis')
    def analyze_video(self, job_id: str, video_duration: float):
        """Analyze with segment tracking"""
        # Analysis logic
        segments = self._detect_segments(...)
        
        # Track quality metrics
        metrics.track_segments_detected(len(segments))
        for segment in segments:
            metrics.track_highlight_score(segment['score'])
        
        return {'segments': segments}
    
    @track_api_cost('whisper', lambda r: 0.006 * r['duration'] / 60)
    def transcribe_audio(self, job_id: str, audio_path: str, duration: float):
        """Transcribe with cost tracking"""
        # Whisper API call
        return {'transcript': '...', 'duration': duration}
    
    def normalize_audio(self, job_id: str, audio_data):
        """Normalize with quality tracking"""
        # Calculate loudness spread
        spread_lu = self._calculate_loudness_spread(audio_data)
        metrics.track_audio_spread(spread_lu)
        
        # Apply normalization
        adjustment_db = self._calculate_adjustment(audio_data)
        metrics.track_audio_normalization(adjustment_db)
        
        return normalized_audio
```

## Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
  - name: video_processing
    rules:
      - alert: HighErrorRate
        expr: rate(video_processing_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate in video processing"
          
      - alert: HighProcessingRatio
        expr: histogram_quantile(0.95, video_processing_ratio) > 1.8
        for: 10m
        annotations:
          summary: "Processing taking too long relative to video duration"
          
      - alert: BudgetExceeded
        expr: increase(video_processing_budget_exceeded_total[1h]) > 5
        for: 5m
        annotations:
          summary: "Multiple jobs exceeding budget"
          
      - alert: HighAPICost
        expr: rate(video_processing_cost_usd_total[1h]) > 20
        for: 5m
        annotations:
          summary: "API costs exceeding $20/hour"
```

## Grafana Dashboard

Key panels to include:

1. **Job Overview**
   - Jobs by status (stacked area chart)
   - Jobs in progress (gauge)
   - Success rate (single stat)

2. **Performance**
   - Processing time by stage (heatmap)
   - Processing ratio P50/P95/P99 (graph)
   - Current FFmpeg processes (gauge)

3. **Costs**
   - Cost per hour by API (graph)
   - Total cost today (single stat)
   - Budget remaining (gauge)

4. **Quality**
   - Audio loudness spread distribution (histogram)
   - Highlight score distribution (histogram)
   - Segments detected per video (histogram)

5. **Errors**
   - Error rate by stage (graph)
   - Error types (table)

## Best Practices

1. **Use Decorators**: Prefer decorators for automatic instrumentation
2. **Track Early**: Add metrics from the start, not as an afterthought
3. **Be Consistent**: Use consistent label names across metrics
4. **Avoid Cardinality Explosion**: Don't use high-cardinality labels (like full file paths)
5. **Set Alerts**: Configure alerts for key business metrics
6. **Regular Reviews**: Review metrics weekly to identify trends

## Testing Metrics

```python
# In tests, you can verify metrics are recorded
def test_processing_records_metrics():
    before = metrics.processing_duration.labels(stage='test')._count.get()
    
    # Run processing
    process_video()
    
    after = metrics.processing_duration.labels(stage='test')._count.get()
    assert after > before
```

## Troubleshooting

### Metrics Not Updating
- Ensure metrics server is started: `metrics.start_http_server()`
- Check the correct port: default is 9099
- Verify no firewall blocking the port

### High Memory Usage
- Metrics are kept in memory
- High-cardinality labels can cause memory growth
- Use `prometheus_client.REGISTRY.collect()` to inspect metrics

### Missing Metrics
- Check decorator placement
- Ensure exceptions aren't swallowing metric updates
- Use `try/finally` blocks for cleanup metrics