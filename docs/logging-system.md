# Centralized Logging System

This document describes the centralized logging system implemented for the Montage video processing pipeline.

## Overview

The logging system provides:

- **Structured logging** with JSON output for production monitoring
- **Correlation IDs** for request tracing across services
- **Performance monitoring** with timing and resource usage
- **File rotation** for production environments
- **Colored console output** for development
- **Centralized configuration** across all modules

## Configuration

### Auto-Configuration

The logging system auto-configures with sensible defaults when imported:

```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("This will work with default configuration")
```

### Manual Configuration

For custom configuration:

```python
from src.utils.logging_config import configure_logging, get_logger

# Configure logging
configure_logging(
    log_level="INFO",                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_dir="/var/log/montage",         # Log directory (None for auto-detect)
    enable_json=True,                   # Force JSON output
    enable_file_logging=True,           # Enable file logging
    max_file_size=50 * 1024 * 1024,    # 50MB max file size
    backup_count=5,                     # Keep 5 backup files
    console_format="colored"            # colored, json, or simple
)

logger = get_logger(__name__)
```

### Environment Detection

The system automatically detects the environment:

- **Production**: JSON logging, file output to `/var/log/montage`
- **Container**: JSON logging (detected via `/.dockerenv` or `CONTAINER=true`)
- **Development**: Colored console output, file output to `~/.cache/montage/logs`

## Usage

### Basic Logging

```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

### Structured Logging

Add contextual data for monitoring and analysis:

```python
logger.info(
    "Video processing started",
    extra={
        'video_file': 'sample.mp4',
        'duration_seconds': 120.5,
        'codec': 'h264',
        'resolution': '1920x1080',
        'file_size_mb': 25.6
    }
)
```

### Correlation Context

Use correlation IDs for request tracing:

```python
from src.utils.logging_config import correlation_context

with correlation_context(correlation_id="req-123", job_id="job-456"):
    logger.info("Processing request")  # Logs will include correlation IDs
    
    # Nested contexts work too
    with correlation_context(correlation_id="nested-789"):
        logger.info("Nested operation")
```

### Performance Monitoring

Monitor operation performance:

```python
from src.utils.logging_config import log_performance

with log_performance("video_transcription"):
    # Your processing code here
    transcribe_video(video_path)
    # Automatically logs start/end times and resource usage
```

### Function Call Logging

Decorate functions for automatic logging:

```python
from src.utils.logging_config import log_function_call

@log_function_call(log_args=True, log_return=True)
def process_video(video_path: str, quality: str = "high"):
    # Function execution is automatically logged
    return processed_video
```

### Exception Logging

Log exceptions with full context:

```python
try:
    risky_operation()
except Exception as e:
    logger.error(
        "Operation failed",
        extra={
            'operation': 'video_processing',
            'input_file': video_path,
            'error_type': type(e).__name__
        },
        exc_info=True  # Include full traceback
    )
    raise
```

## Log Formats

### Development (Colored Console)
```
[16:06:50.232] INFO     __main__             | test-123:job-456 | Processing video file
```

### Production (JSON)
```json
{
    "timestamp": "2024-01-20T16:06:50.232Z",
    "level": "INFO",
    "logger": "__main__",
    "message": "Processing video file",
    "module": "__main__",
    "function": "process_video",
    "line": 45,
    "correlation_id": "test-123",
    "job_id": "job-456",
    "process_id": 12345,
    "thread_id": 67890,
    "extra": {
        "video_file": "sample.mp4",
        "duration_seconds": 120.5
    }
}
```

## File Organization

### Log Files

- **Main log**: `montage.log` - All log levels
- **Error log**: `montage-error.log` - WARNING and above only
- **Rotation**: Files rotate at 50MB, keeping 5 backups

### Log Locations

- **Production**: `/var/log/montage/`
- **Development**: `~/.cache/montage/logs/`
- **Custom**: Configurable via `log_dir` parameter

## Integration with Main Pipeline

### Job Processing

The main pipeline uses correlation contexts for job tracking:

```python
def process_job(self, job_id: str):
    with correlation_context(job_id=job_id):
        logger.info("Starting job processing")
        # All subsequent logs will include job_id
```

### Stage Processing

Each pipeline stage is wrapped with performance monitoring:

```python
with log_performance(f"stage_{stage_name}"):
    result = execute_stage(stage_name, context)
```

### Error Handling

Structured error logging with context:

```python
except Exception as e:
    logger.error(
        "Pipeline stage failed",
        extra={
            'job_id': job_id,
            'stage': stage_name,
            'error_type': type(e).__name__
        },
        exc_info=True
    )
```

## Monitoring Integration

### Log Aggregation

JSON logs can be easily ingested by:

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Splunk**
- **CloudWatch Logs**
- **Prometheus + Grafana** (via log metrics)

### Alerting

Set up alerts on:

- ERROR level logs for immediate issues
- High correlation_id volume for performance issues
- Missing job completion logs for stuck jobs

### Metrics

Extract metrics from structured logs:

- Processing times by stage
- Error rates by module
- Resource usage patterns
- Job throughput

## Best Practices

### Do's

✅ Use structured logging with `extra` fields
✅ Include correlation IDs for request tracing
✅ Log at appropriate levels (INFO for business events, DEBUG for troubleshooting)
✅ Include context in error messages
✅ Use performance logging for timing operations

### Don'ts

❌ Don't log sensitive data (API keys, passwords, PII)
❌ Don't use print() statements - use logger instead
❌ Don't log at DEBUG level in production unless needed
❌ Don't include large objects in log messages
❌ Don't ignore exceptions - always log them

### Migration from Print Statements

Replace print statements with appropriate log levels:

```python
# Old
print(f"Processing {file_name}")
print(f"❌ Error: {error}")

# New
logger.info("Processing file", extra={'file_name': file_name})
logger.error("Processing failed", extra={'error': str(error)}, exc_info=True)
```

## Troubleshooting

### Common Issues

1. **Logs not appearing**: Check log level configuration
2. **File permission errors**: Ensure log directory is writable
3. **Missing correlation IDs**: Verify context managers are used
4. **JSON parsing errors**: Check for serializable data in `extra` fields

### Debug Configuration

Enable debug logging temporarily:

```python
configure_logging(log_level="DEBUG", console_format="colored")
```

### Log File Location

Find current log files:

```python
from src.utils.logging_config import _config
print(f"Log directory: {_config._log_dir}")
```

## Testing

Run the logging system test:

```bash
python test_logging.py
```

This verifies all logging functionality is working correctly.