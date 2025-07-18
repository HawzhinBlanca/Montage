# Video Processing Pipeline - System Integration Guide

## Overview

The video processing pipeline integrates multiple components to deliver a resilient, observable, and production-ready system. This guide explains how all modules work together to process videos efficiently while meeting quality standards.

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Input     │     │  Database    │     │   Redis      │
│   Video     │     │ (PostgreSQL) │     │ (Checkpoint) │
└──────┬──────┘     └──────┬───────┘     └──────┬───────┘
       │                   │                     │
       ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────┐
│              SmartVideoEditor Pipeline               │
├──────────────────────────────────────────────────────┤
│  1. Validation   →  2. Analysis  →  3. Editing      │
│  (HDR Check)        (Segments)      (FIFO/Concat)   │
│                                                      │
│  4. Audio Norm  →  5. Color Conv →  6. Verify       │
│  (Two-pass)         (BT.709)         (Final QC)     │
└──────────────────────────────────────────────────────┘
       │                   │                     │
       ▼                   ▼                     ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Output    │     │  Prometheus  │     │   Grafana    │
│   Video     │     │  Metrics     │     │  Dashboard   │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Component Integration

### 1. Database Layer (PostgreSQL)

Stores persistent data:
- Job metadata and status
- Video file hashes
- Transcript cache
- Processing metrics
- API costs

```python
from db import Database

db = Database()
job_id = db.insert('video_job', {
    'src_hash': file_hash,
    'status': 'queued',
    'input_path': input_path
})
```

### 2. Checkpoint System (Redis)

Enables crash recovery:
- Saves stage completion status
- Stores intermediate results
- Allows job resumption

```python
from checkpoint import SmartVideoEditorCheckpoint

checkpoint = SmartVideoEditorCheckpoint(checkpoint_mgr)
checkpoint.save_stage_data(job_id, 'validation', duration=120.5)
```

### 3. Video Validation

Pre-flight checks including:
- File integrity
- Codec support
- **HDR rejection**
- Duration limits

```python
from video_validator import perform_preflight_check

result = perform_preflight_check(job_id, input_path)
if not result['valid']:
    raise ValueError(result['error'])
```

### 4. Video Processing

High-performance editing:
- FIFO-based pipeline
- Concat demuxer approach
- Parallel segment extraction

```python
from concat_editor import ConcatEditor, EditSegment

editor = ConcatEditor()
segments = [EditSegment(source, start, end) for ...]
editor.execute_edit(segments, output_path)
```

### 5. Audio Normalization

Two-pass loudness normalization:
- Achieves -16 LUFS target
- Ensures ≤ 1.5 LU spread

```python
from audio_normalizer import AudioNormalizer

normalizer = AudioNormalizer()
normalizer.normalize_audio(input_path, output_path)
```

### 6. Color Space Conversion

Ensures BT.709 output:
- Validates SDR input
- Converts color space
- Sets proper metadata

```python
from color_converter import ColorSpaceConverter

converter = ColorSpaceConverter()
converter.convert_to_bt709(input_path, output_path)
```

### 7. Metrics & Monitoring

Comprehensive observability:
- Processing performance
- API costs
- Audio quality
- Error rates

```python
from metrics import metrics

metrics.track_processing_time('editing', video_duration)
metrics.track_cost('openai', 0.05, job_id)
```

## Complete Processing Flow

### Job Processing Example

```python
from smart_video_editor import SmartVideoEditor

# Initialize editor (starts all subsystems)
editor = SmartVideoEditor()

# Define video segments to extract
highlights = [
    {
        'start_time': 30,
        'end_time': 60,
        'score': 0.9,
        'transition': 'fade'
    },
    {
        'start_time': 120,
        'end_time': 150,
        'score': 0.85
    }
]

# Process job (automatic checkpointing)
result = editor.process_job(
    job_id="job-123",
    input_path="/videos/input.mp4",
    highlights=highlights
)

# Result includes:
# - output_path: Final video location
# - color_space: 'bt709' (guaranteed)
# - audio_loudness: -16.0 LUFS
# - segments_processed: Number of segments
```

### Stage-by-Stage Breakdown

#### Stage 1: Validation
```python
# Automatic in process_job, or manual:
result = perform_preflight_check(job_id, input_path)
# Checks: corruption, HDR, codec support
```

#### Stage 2: Analysis
```python
# Prepare segments and track metrics
metrics.track_segments_detected(len(highlights))
for h in highlights:
    metrics.track_highlight_score(h['score'])
```

#### Stage 3: Editing
```python
# Uses FIFO pipeline for efficiency
segments = create_edit_segments(highlights, input_path)
editor.execute_edit(segments, temp_output, apply_transitions=True)
```

#### Stage 4: Audio Normalization
```python
# Two-pass loudness normalization
normalizer.normalize_audio(
    temp_output, 
    normalized_output,
    target=NormalizationTarget(-16.0, -1.0, 7.0)
)
```

#### Stage 5: Color Conversion
```python
# Ensure BT.709 output
converter.convert_to_bt709(
    normalized_output,
    final_output
)
```

#### Stage 6: Verification
```python
# Final quality checks
info = converter.analyze_color_space(final_output)
assert info.color_primaries == 'bt709'
```

## Error Handling & Recovery

### Checkpoint-Based Recovery

If processing fails at any stage:

```python
# On restart, automatically resumes
resume_info = checkpoint.get_resume_point(job_id)
if resume_info:
    # Continues from last successful stage
    print(f"Resuming from: {resume_info['resume_from_stage']}")
```

### Error Tracking

All errors are tracked in metrics:

```python
try:
    process_stage()
except Exception as e:
    metrics.track_error(stage_name, type(e).__name__)
    db.update('video_job', {
        'status': 'failed',
        'error_message': str(e)
    }, {'id': job_id})
```

## Performance Optimization

### Parallel Processing

Multiple operations run concurrently:

```python
# Segments extracted in parallel
with FFmpegPipeline() as pipeline:
    for segment in segments:
        pipeline.add_process(extract_cmd, f"seg_{i}")
    pipeline.wait_all()
```

### FIFO Pipelines

Eliminate disk I/O between processes:

```python
# Data flows through memory pipes
decoder → FIFO → filter → FIFO → encoder
```

### Metrics-Driven Optimization

Monitor performance ratios:

```python
# Alert if processing takes too long
if processing_time / video_duration > 1.2:
    logger.warning("Performance degraded")
```

## Configuration

### Environment Variables

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_DB=video_processing
POSTGRES_USER=videouser
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
CHECKPOINT_TTL=86400

# Processing
FFMPEG_PATH=/usr/local/bin/ffmpeg
TEMP_DIR=/fast/ssd/temp
PROMETHEUS_PORT=9099

# Limits
BUDGET_LIMIT=5.00
MAX_POOL_SIZE=32
```

### Quality Targets

```python
# Audio normalization
LOUDNESS_TARGET = -16.0  # LUFS
LOUDNESS_SPREAD_MAX = 1.5  # LU

# Video encoding
VIDEO_CRF = 18  # Quality factor
VIDEO_PRESET = 'fast'  # Speed/quality balance

# Color space
OUTPUT_COLOR_SPACE = 'bt709'
```

## Monitoring & Alerts

### Key Metrics

```prometheus
# Processing performance
video_processing_ratio{stage="editing"} < 1.2

# Audio quality
video_processing_audio_loudness_spread_lu < 1.5

# Error rate
rate(video_processing_errors_total[5m]) < 0.02

# Cost tracking
video_processing_cost_usd_total{api_name="openai"} < 20/hour
```

### Grafana Dashboard

Visualizes:
- Job throughput
- Processing times by stage
- Audio quality distribution
- Cost breakdown
- Error rates

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Specific components
pytest tests/test_video_validator.py -v
pytest tests/test_concat_editor.py -v
pytest tests/test_audio_normalizer.py -v
pytest tests/test_color_converter.py -v
```

### Integration Tests

```python
# Test complete pipeline
def test_full_pipeline():
    editor = SmartVideoEditor()
    result = editor.process_job(
        "test-job",
        "test_input.mp4",
        test_highlights
    )
    
    # Verify all requirements
    assert result['color_space'] == 'bt709'
    assert abs(result['audio_loudness'] - (-16.0)) < 0.5
```

### Performance Tests

```python
# Verify processing ratio requirement
def test_performance_requirement():
    # 20-minute 1080p video
    duration = 20 * 60
    
    start = time.time()
    process_video(test_file)
    elapsed = time.time() - start
    
    ratio = elapsed / duration
    assert ratio < 1.2  # Must process faster than 1.2x
```

## Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: video_processing
      POSTGRES_USER: videouser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    command: redis-server --maxmemory 2gb

  app:
    build: .
    environment:
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./videos:/videos
      - ./output:/output
    depends_on:
      - postgres
      - redis

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana:/etc/grafana/provisioning
```

### Production Checklist

- [ ] Database migrations applied
- [ ] Redis configured with persistence
- [ ] FFmpeg installed with required codecs
- [ ] Sufficient disk space for temp files
- [ ] Monitoring alerts configured
- [ ] Backup strategy in place
- [ ] API keys secured
- [ ] Rate limits configured

## Troubleshooting

### Common Issues

1. **HDR Input Rejected**
   - Error: "HDR input not supported"
   - Solution: Ensure all inputs are SDR

2. **High Audio Spread**
   - Error: Spread > 1.5 LU
   - Solution: Check source audio consistency

3. **Color Space Not BT.709**
   - Check zscale filter is applied
   - Verify encoding flags are set

4. **Processing Too Slow**
   - Check CPU/memory usage
   - Verify FIFO pipeline is working
   - Consider batch size adjustments

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Track detailed metrics
metrics.track_processing_time('debug_stage')
```

## Final Validation

The system passes the litmus test when:

✓ AI Quality: ≤ 5 highlight clips (15-60s each)  
✓ Audio Quality: LU spread ≤ 1.5 LU  
✓ Visual Quality: Proper framing/cropping  
✓ Technical: SDR BT.709 output  
✓ Cost: < $4.00 per job  
✓ Performance: ≤ 1.5x source duration  