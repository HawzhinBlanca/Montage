# FIFO-Based Video Processing Guide

## Overview

The video processor module implements a high-performance video editing pipeline using UNIX FIFOs (named pipes) to eliminate intermediate file I/O. This approach enables parallel processing and meets the performance requirement of processing 20-minute 1080p video in under 1.2x its duration.

## Key Features

- **Parallel Processing**: Extract multiple segments simultaneously
- **Zero Intermediate Files**: Use FIFOs for inter-process communication
- **Automatic Cleanup**: FIFOs cleaned up even on interruption
- **Performance Monitoring**: Integrated metrics tracking
- **Error Handling**: Comprehensive error detection and reporting

## Architecture

```
Input Video → Parallel Extraction → FIFOs → Concatenation → Transitions → Output
                    ↓                ↓            ↓              ↓
                Process 1        Pipe 1      Process N      Filter Process
                Process 2        Pipe 2
                Process N        Pipe N
```

## Usage

### Basic Video Editing

```python
from video_processor import VideoEditor, VideoSegment

editor = VideoEditor()

# Define segments to extract
segments = [
    VideoSegment(10, 40, "input.mp4"),    # 30-second segment starting at 10s
    VideoSegment(60, 90, "input.mp4"),    # 30-second segment starting at 60s
    VideoSegment(120, 180, "input.mp4"),  # 60-second segment starting at 120s
]

# Process efficiently with transitions
editor.extract_and_concatenate_efficient(
    "input.mp4",
    segments,
    "output.mp4",
    apply_transitions=True  # Smooth transitions between segments
)
```

### Manual Pipeline Control

```python
from video_processor import FFmpegPipeline, FIFOManager

# Create pipeline
with FFmpegPipeline() as pipeline:
    # Create FIFOs
    input_fifo = pipeline.fifo_manager.create_fifo("_input")
    output_fifo = pipeline.fifo_manager.create_fifo("_output")
    
    # Add processes
    decode_process = pipeline.add_process([
        'ffmpeg', '-i', 'input.mp4',
        '-f', 'mpegts', '-c:v', 'rawvideo',
        input_fifo
    ], "decoder")
    
    filter_process = pipeline.add_process([
        'ffmpeg', '-f', 'mpegts', '-i', input_fifo,
        '-vf', 'scale=1920:1080',
        '-f', 'mpegts', output_fifo
    ], "filter")
    
    encode_process = pipeline.add_process([
        'ffmpeg', '-f', 'mpegts', '-i', output_fifo,
        '-c:v', 'libx264', '-preset', 'fast',
        'output.mp4'
    ], "encoder")
    
    # Wait for completion
    results = pipeline.wait_all(timeout=300)
    
    if all(code == 0 for code in results.values()):
        print("Processing completed successfully")
```

### Custom Filtering

```python
# Apply complex filter through FIFO pipeline
editor.process_with_filter_fifo(
    "input.mp4",
    "output.mp4",
    filter_complex="[0:v]scale=1920:1080,eq=brightness=0.1:contrast=1.2[v]",
    video_codec="libx264",
    audio_codec="aac"
)
```

### Segment Extraction

```python
# Extract segments in parallel
segments = [
    VideoSegment(0, 30, "input.mp4", "intro"),
    VideoSegment(300, 360, "input.mp4", "highlight"),
    VideoSegment(600, 630, "input.mp4", "outro")
]

# Returns FIFO paths immediately while extraction runs in background
segment_fifos = editor.extract_segments_parallel("input.mp4", segments)

# Use FIFOs for further processing
editor.concatenate_segments_fifo(segment_fifos, "compilation.mp4")
```

## Performance Optimization

### 1. Parallel Extraction

Segments are extracted in parallel, dramatically reducing total processing time:

```python
# Sequential: 3 segments × 10s each = 30s
# Parallel: max(10s, 10s, 10s) = 10s
```

### 2. FIFO Streaming

Data flows directly between processes without disk I/O:

```python
# Traditional: Write → Disk → Read → Process → Write → Disk
# FIFO: Process → Memory Buffer → Process
```

### 3. Codec Selection

Use appropriate codecs for different stages:

```python
# Intermediate processing (speed priority)
intermediate_codec = "libx264 -preset ultrafast"

# Final output (quality priority)  
final_codec = "libx264 -preset fast"

# Lossless intermediate
lossless_audio = "pcm_s16le"
```

## Integration with SmartVideoEditor

```python
from video_processor import VideoEditor, VideoSegment
from metrics import track_processing_stage

class SmartVideoEditor:
    def __init__(self):
        self.video_editor = VideoEditor()
    
    @track_processing_stage('editing')
    def execute_edit(self, job_id: str, input_path: str, 
                     highlights: List[Dict], video_duration: float):
        """Execute video editing with FIFO pipeline"""
        
        # Convert highlights to segments
        segments = []
        for highlight in highlights:
            segment = VideoSegment(
                highlight['start_time'],
                highlight['end_time'],
                input_path,
                f"highlight_{highlight['id']}"
            )
            segments.append(segment)
        
        # Process with FIFO pipeline
        output_path = f"/output/{job_id}_edited.mp4"
        
        self.video_editor.extract_and_concatenate_efficient(
            input_path,
            segments,
            output_path,
            apply_transitions=True
        )
        
        return {
            'output_path': output_path,
            'segments_processed': len(segments),
            'total_duration': sum(s.duration for s in segments)
        }
```

## Error Handling

```python
try:
    editor.extract_and_concatenate_efficient(
        input_file, segments, output_file
    )
except FFmpegError as e:
    logger.error(f"FFmpeg processing failed: {e}")
    # Handle FFmpeg-specific errors
except VideoProcessingError as e:
    logger.error(f"Video processing error: {e}")
    # Handle general processing errors
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## Best Practices

### 1. Segment Batching

Process segments in reasonable batches to avoid resource exhaustion:

```python
# Process in batches of 10-20 segments
batch_size = 15
for i in range(0, len(all_segments), batch_size):
    batch = all_segments[i:i + batch_size]
    editor.extract_segments_parallel(input_file, batch)
```

### 2. Resource Management

Always use context managers for automatic cleanup:

```python
# Good - automatic cleanup
with FFmpegPipeline() as pipeline:
    # Processing here
    pass

# Also good - try/finally
pipeline = FFmpegPipeline()
try:
    # Processing here
finally:
    pipeline.cleanup()
```

### 3. Performance Monitoring

Track processing ratios to ensure performance targets:

```python
from metrics import metrics

# After processing
ratio = processing_time / video_duration
if ratio > 1.2:
    logger.warning(f"Performance degraded: {ratio:.2f}x target duration")
    
# Check metrics
p95_ratio = metrics.processing_ratio.labels(stage='editing').quantile(0.95)
```

### 4. FIFO Sizing

FIFOs have limited buffer size. For large data flows, ensure consumers keep up:

```python
# Set appropriate encoding presets
# Fast preset for intermediate processing
intermediate_preset = "ultrafast"  # Keeps up with input

# Better quality for final output
final_preset = "fast"  # Balance of speed and quality
```

## Troubleshooting

### FIFO Creation Fails

```python
# Error: OSError: [Errno 13] Permission denied
# Solution: Ensure temp directory is writable
os.makedirs(Config.TEMP_DIR, exist_ok=True, mode=0o755)
```

### Process Hangs

```python
# Hanging process detection
results = pipeline.wait_all(timeout=300)  # 5-minute timeout

# Force cleanup if needed
pipeline.cleanup()  # Terminates all processes
```

### Performance Issues

1. **Check System Resources**
   ```bash
   # Monitor during processing
   htop  # CPU usage
   iotop  # I/O usage
   ```

2. **Verify FIFO Flow**
   ```python
   # Add debug logging
   logger.debug(f"FIFO created: {fifo_path}")
   ```

3. **Profile FFmpeg Commands**
   ```bash
   # Test individual commands
   time ffmpeg -i input.mp4 -c copy -f mpegts /tmp/test_fifo
   ```

## Testing

Run the video processor tests:

```bash
# All tests
pytest tests/test_video_processor.py -v

# Performance tests only
pytest tests/test_video_processor.py::TestIntegration::test_performance_requirement -v
```

## Configuration

Key configuration options in `config.py`:

```python
# FFmpeg binary path
FFMPEG_PATH = '/usr/local/bin/ffmpeg'

# Temporary directory for FIFOs
TEMP_DIR = '/tmp/video_processing'

# Processing timeout
PROCESSING_TIMEOUT = 1800  # 30 minutes
```