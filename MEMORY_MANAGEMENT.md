# Comprehensive Memory Management for Video Processing

This document describes the comprehensive memory management system implemented for Montage to prevent Out of Memory (OOM) errors during large video file processing.

## Overview

The memory management system provides:

1. **Real-time Memory Monitoring** - Tracks system and process memory usage
2. **Memory Pressure Detection** - Automatically responds to memory constraints
3. **Adaptive Processing** - Adjusts quality and processing methods based on available memory
4. **Resource Cleanup** - Ensures proper cleanup of video processing resources
5. **Streaming Processing** - Handles large files in memory-safe chunks
6. **FFmpeg Memory Management** - Monitors and limits FFmpeg process memory usage

## Quick Start

### Basic Setup

```python
from src.utils.memory_init import setup_memory_management

# Initialize memory management (recommended at application startup)
success = setup_memory_management()
if not success:
    print("Warning: Memory management not available")
```

### Processing Videos with Memory Management

```python
from src.providers.video_processor import VideoEditor
from src.core.analyze_video import analyze_video

# Video processing automatically uses memory management
editor = VideoEditor()  # Now includes memory management

# Analyze video with automatic memory optimization
result = analyze_video("large_video.mp4")  # Automatically uses chunked processing for large files

# Process video segments with memory constraints
segments = [VideoSegment(10, 40, "input.mp4"), VideoSegment(60, 90, "input.mp4")]
editor.extract_and_concatenate_efficient("input.mp4", segments, "output.mp4")
```

## Components

### 1. Memory Monitor (`memory_manager.py`)

Provides real-time memory monitoring with automatic pressure detection.

```python
from src.utils.memory_manager import get_memory_monitor, memory_guard

# Get current memory stats
monitor = get_memory_monitor()
stats = monitor.get_current_stats()
print(f"Available memory: {stats.available_mb}MB")
print(f"Pressure level: {stats.pressure_level.value}")

# Use memory guard for operations
with memory_guard(max_memory_mb=1024) as monitor:
    # Your memory-constrained operation here
    process_video()
```

### 2. Resource Manager (`resource_manager.py`)

Manages video processing resources with automatic cleanup.

```python
from src.utils.resource_manager import managed_tempfile, managed_opencv_capture

# Automatic temp file cleanup
with managed_tempfile(suffix=".mp4") as temp_file:
    # Process video, file automatically cleaned up
    process_video_segment(input_file, temp_file)

# Memory-safe OpenCV operations
with managed_opencv_capture("video.mp4") as cap:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process frame
```

### 3. FFmpeg Memory Manager (`ffmpeg_memory_manager.py`)

Specialized memory management for FFmpeg processes.

```python
from src.utils.ffmpeg_memory_manager import memory_safe_ffmpeg, get_ffmpeg_memory_manager

# Memory-safe FFmpeg execution
cmd = ['ffmpeg', '-i', 'input.mp4', '-c:v', 'libx264', 'output.mp4']

with memory_safe_ffmpeg(cmd, max_memory_mb=512) as process:
    result = process.wait()

# Process large videos in chunks
manager = get_ffmpeg_memory_manager()
success = manager.process_video_chunks(
    "large_input.mp4", 
    "output.mp4", 
    processing_function
)
```

## Memory Pressure Levels

The system automatically detects and responds to different memory pressure levels:

### LOW (< 70% memory used)
- Full quality processing
- All features enabled
- Maximum performance

### MODERATE (70-80% memory used)  
- Gentle garbage collection
- Slightly reduced quality presets
- Balanced processing

### HIGH (80-90% memory used)
- Aggressive cleanup
- Reduced worker count
- Fast processing presets
- Automatic chunking for large files

### CRITICAL (> 90% memory used)
- Emergency measures
- Terminate non-essential processes
- Minimal quality settings
- Force chunked processing

## Adaptive Configuration

The system automatically adapts processing configuration based on available memory:

```python
from src.utils.memory_init import get_safe_processing_config

config = get_safe_processing_config("video.mp4")
print(f"Recommended workers: {config['max_workers']}")
print(f"Chunk size: {config['chunk_size_mb']}MB")
print(f"Quality preset: {config['quality_preset']}")
```

## Configuration Options

### Processing Modes

1. **FULL_QUALITY** - Best quality, high memory usage
2. **BALANCED** - Balance quality vs memory  
3. **MEMORY_OPTIMIZED** - Lower quality, minimal memory
4. **SURVIVAL** - Minimal processing to avoid OOM

### FFmpeg Memory Profiles

1. **MINIMAL** - Absolute minimal memory usage
2. **LOW** - Low memory usage
3. **BALANCED** - Balanced memory/performance  
4. **HIGH** - High memory for best performance

## Monitoring and Debugging

### Check Memory Status

```python
from src.utils.memory_init import get_memory_status

status = get_memory_status()
print("Memory Status:", status['memory'])
print("Active Resources:", status['resources'])
print("FFmpeg Processes:", status['ffmpeg'])
```

### Force Cleanup

```python
from src.utils.memory_init import force_memory_cleanup

# Force cleanup of all resources
force_memory_cleanup()
```

### Monitor Specific Processes

```python
monitor = get_memory_monitor()

# Track FFmpeg process
monitor.track_process(process.pid, "ffmpeg_encode", is_ffmpeg=True)

# Get tracked processes
processes = monitor.get_ffmpeg_processes()
for proc in processes:
    print(f"Process {proc.pid}: {proc.memory_mb:.1f}MB")
```

## Best Practices

### 1. Initialize Early
Always initialize memory management at application startup:

```python
from src.utils.memory_init import setup_memory_management

# At application startup
setup_memory_management()
```

### 2. Use Context Managers
Always use provided context managers for resource safety:

```python
# Good
with managed_tempfile() as temp_file:
    process_video(temp_file)

# Bad - no automatic cleanup
temp_file = create_temp_file()
process_video(temp_file)  # File might not get cleaned up
```

### 3. Check Memory Before Large Operations

```python
monitor = get_memory_monitor()
stats = monitor.get_current_stats()

if stats.available_mb < 1024:  # Less than 1GB available
    logger.warning("Low memory - using chunked processing")
    use_chunked_processing = True
```

### 4. Handle Large Files Appropriately

```python
file_size_mb = os.path.getsize("video.mp4") / 1024 / 1024

if file_size_mb > 500:  # > 500MB
    logger.info("Large file detected - using streaming processing")
    # System will automatically use chunked processing
```

## Error Handling

The memory management system provides graceful degradation:

```python
try:
    # Attempt memory-optimized processing
    result = analyze_video("large_video.mp4")
except MemoryError as e:
    logger.error(f"Memory constraint exceeded: {e}")
    # System will automatically retry with more conservative settings
except Exception as e:
    logger.error(f"Processing failed: {e}")
    # Force cleanup and retry
    force_memory_cleanup()
```

## Performance Impact

The memory management system is designed to have minimal performance impact:

- Monitoring overhead: < 1% CPU usage
- Memory tracking: < 10MB RAM overhead  
- Automatic optimizations improve performance for memory-constrained systems
- Prevents OOM crashes that would otherwise terminate processing

## Troubleshooting

### Memory Management Not Working

1. Check if components are available:
```python
from src.utils.memory_init import is_memory_management_available
print(f"Available: {is_memory_management_available()}")
```

2. Check system resources:
```python
status = get_memory_status()
if 'error' in status:
    print(f"Error: {status['error']}")
```

### High Memory Usage

1. Check active processes:
```python
manager = get_ffmpeg_memory_manager()
print(f"FFmpeg processes: {manager.get_active_process_count()}")
print(f"Total memory: {manager.get_total_memory_usage():.1f}MB")
```

2. Force cleanup:
```python
force_memory_cleanup()
```

### Processing Failures

1. Check memory pressure:
```python
stats = monitor.get_current_stats()
if stats.pressure_level.value in ["high", "critical"]:
    print("High memory pressure detected")
```

2. Reduce processing load:
```python
config = get_safe_processing_config()
# Use config['max_workers'] and config['chunk_size_mb'] for processing
```

## Integration Examples

### With Existing Video Processor

```python
from src.providers.video_processor import VideoEditor
from src.utils.memory_init import setup_memory_management

# Initialize memory management
setup_memory_management()

# Use existing VideoEditor - now automatically memory-optimized
editor = VideoEditor()
segments = [VideoSegment(10, 40, "input.mp4")]
editor.extract_and_concatenate_efficient("input.mp4", segments, "output.mp4")
```

### Custom Processing Function

```python
from src.utils.resource_manager import memory_constrained_operation
from src.utils.ffmpeg_memory_manager import build_memory_safe_ffmpeg_command

def custom_video_processor(input_path, output_path):
    # Estimate memory needs
    estimated_mb = estimate_processing_memory(input_path, "custom")
    
    with memory_constrained_operation(estimated_mb, "custom_processing"):
        # Build memory-safe FFmpeg command
        base_cmd = ['ffmpeg', '-i', input_path, '-c:v', 'libx264', output_path]
        optimized_cmd = build_memory_safe_ffmpeg_command(base_cmd, input_path)
        
        # Execute with memory monitoring
        with memory_safe_ffmpeg(optimized_cmd) as process:
            return process.wait() == 0
```

## System Requirements

- **Minimum RAM**: 4GB (8GB recommended)
- **Python packages**: `psutil`, `opencv-python` (optional for enhanced features)
- **FFmpeg**: Required for video processing
- **Operating System**: Linux, macOS, Windows

The memory management system automatically adapts to available resources and gracefully degrades on systems with limited memory.