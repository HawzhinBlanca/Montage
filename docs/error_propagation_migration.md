# Error Propagation System Migration Guide

This guide explains how to migrate existing code to use the new comprehensive error propagation system.

## Overview

The new error propagation system provides:
- **Domain-specific exceptions** for video processing operations
- **Rich error context** with job IDs, file paths, and metadata
- **Automatic retry strategies** based on error types
- **Production debugging** with detailed error traces
- **No more silent failures** - all errors are explicitly handled

## Quick Start

### 1. Import Error Handling

```python
# Import exceptions
from src.core.exceptions import (
    VideoProcessingError, VideoDecodeError, FFmpegError,
    TranscriptionError, WhisperError, DeepgramError,
    DiskSpaceError, MemoryError, ConfigurationError,
    create_error_context
)

# Import utilities
from src.utils.error_handler import (
    with_error_context, with_retry, safe_execute,
    handle_ffmpeg_error, handle_api_error
)
```

### 2. Replace Silent Failures

**Before:**
```python
def extract_audio(video_path):
    try:
        # ... extraction logic ...
        return audio_path
    except Exception as e:
        logger.error(f"Failed: {e}")
        return None  # Silent failure!
```

**After:**
```python
@with_error_context("AudioExtraction", "extract_audio")
def extract_audio(video_path):
    if not os.path.exists(video_path):
        raise VideoDecodeError(
            "Video file not found",
            video_path=video_path
        ).with_suggestion("Check file path")
    
    try:
        # ... extraction logic ...
        return audio_path
    except ffmpeg.Error as e:
        raise handle_ffmpeg_error(
            command=cmd,
            return_code=e.returncode,
            stderr=str(e),
            operation="Audio extraction"
        )
```

### 3. Add Automatic Retries

**Before:**
```python
def call_api():
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        return None
    return response.json()
```

**After:**
```python
@with_retry(max_attempts=3, base_delay=2.0)
def call_api():
    response = requests.post(url, data=payload)
    if response.status_code == 429:
        raise RateLimitError(
            "API rate limit exceeded",
            service="deepgram",
            retry_after=int(response.headers.get('Retry-After', 60))
        )
    elif response.status_code == 401:
        raise AuthenticationError(
            "Invalid API credentials",
            service="deepgram"
        )
    elif response.status_code != 200:
        raise handle_api_error(
            service="deepgram",
            error=Exception(f"HTTP {response.status_code}"),
            status_code=response.status_code,
            response_body=response.text
        )
    return response.json()
```

### 4. Batch Error Handling

**Before:**
```python
def process_segments(segments):
    results = []
    for segment in segments:
        try:
            result = process_segment(segment)
            results.append(result)
        except:
            pass  # Skip failed segments
    return results
```

**After:**
```python
def process_segments(segments):
    results = []
    collector = ErrorCollector()
    
    for segment in segments:
        with collector.collect(segment_id=segment.id):
            result = process_segment(segment)
            results.append(result)
    
    # Fail if too many errors
    if collector.has_errors():
        failed = len(collector.get_errors())
        if failed > len(segments) / 2:
            collector.raise_if_errors(
                f"Too many segments failed ({failed}/{len(segments)})"
            )
        else:
            logger.warning(f"{failed} segments failed, continuing")
    
    return results
```

## Migration Checklist

### Phase 1: Core Infrastructure
- [x] Create `src/core/exceptions.py` with exception hierarchy
- [x] Create `src/utils/error_handler.py` with utilities
- [x] Add comprehensive test suite

### Phase 2: Critical Path Updates
- [ ] Update `analyze_video.py` → `analyze_video_v2.py`
- [ ] Update `video_processor.py` → `video_processor_v2.py`
- [ ] Update `highlight_selector.py` with error handling
- [ ] Update database operations in `db.py`

### Phase 3: API Integration
- [ ] Update `api_wrappers.py` with specialized error handling
- [ ] Update `rate_limiter.py` to raise appropriate errors
- [ ] Update `resolve_mcp.py` with error context

### Phase 4: Utilities
- [ ] Update `ffmpeg_utils.py` to raise specific errors
- [ ] Update `intelligent_crop.py` with error recovery
- [ ] Update `memory_manager.py` with resource errors

## Common Patterns

### 1. File Operations
```python
@with_error_context("FileOperation", "read_video")
def read_video(path):
    if not os.path.exists(path):
        raise VideoDecodeError(
            "Video file not found",
            video_path=path
        )
    
    if os.path.getsize(path) == 0:
        raise VideoDecodeError(
            "Video file is empty",
            video_path=path
        )
    
    # ... rest of logic
```

### 2. External API Calls
```python
@with_retry(max_attempts=3, exponential_base=2.0)
def transcribe_with_api(audio_path, job_id):
    try:
        result = api_client.transcribe(audio_path)
        return result
    except APIException as e:
        raise handle_api_error(
            service="transcription_api",
            error=e,
            status_code=getattr(e, 'status_code', None)
        )
```

### 3. Resource Checks
```python
def check_resources(required_memory_mb):
    available = get_available_memory()
    if available < required_memory_mb:
        raise MemoryError(
            "Insufficient memory for operation",
            required_mb=required_memory_mb,
            available_mb=available
        ).with_suggestion("Close other applications or reduce video quality")
```

### 4. Configuration Validation
```python
def validate_config():
    if not os.getenv('DEEPGRAM_API_KEY'):
        raise ConfigurationError(
            "Deepgram API key not configured",
            config_key="DEEPGRAM_API_KEY"
        ).with_suggestion("Set DEEPGRAM_API_KEY environment variable")
```

## Testing Error Handling

### Unit Tests
```python
def test_video_decode_error():
    with pytest.raises(VideoDecodeError) as exc_info:
        process_corrupted_video("/bad/video.mp4")
    
    error = exc_info.value
    assert error.context.video_path == "/bad/video.mp4"
    assert len(error.suggestions) > 0
```

### Integration Tests
```python
def test_error_recovery():
    # Test that partial failures don't crash the pipeline
    segments = create_test_segments(10)
    
    # Corrupt some segments
    segments[3].input_file = "/nonexistent.mp4"
    segments[7].input_file = "/nonexistent.mp4"
    
    # Should process 8/10 segments successfully
    results = process_segments_with_recovery(segments)
    assert len(results) == 8
```

## Monitoring Integration

Errors automatically integrate with logging and metrics:

```python
# Errors are automatically logged with context
error = VideoProcessingError("Failed to process")
error.add_context(job_id="job_123", duration=45.2)
# Logs: ERROR - Failed to process [job_id=job_123, duration=45.2]

# Critical errors trigger alerts
error = DiskSpaceError("No space", required_gb=10, available_gb=0.1)
# Logs: CRITICAL - No space [required_gb=10, available_gb=0.1]
```

## Best Practices

1. **Always provide context**: Use job_id and video_path when available
2. **Add helpful suggestions**: Guide users on how to resolve errors
3. **Use appropriate severity**: Critical for showstoppers, High for failures
4. **Choose correct retry strategy**: No retry for user errors, exponential for APIs
5. **Preserve error chains**: Use `raise ... from e` to maintain stack traces
6. **Fail fast on critical errors**: Don't hide configuration or auth failures
7. **Batch operations should be resilient**: Use ErrorCollector for partial failures

## Gradual Migration

You can migrate incrementally:

1. Start with critical paths (video analysis, processing)
2. Add error context decorators to existing functions
3. Replace return None with explicit exceptions
4. Add retry decorators to flaky operations
5. Update tests to verify error handling

The old and new code can coexist during migration. Use the `_v2` suffix for updated modules during transition.