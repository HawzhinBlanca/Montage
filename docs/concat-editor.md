# Concat Demuxer Video Editor Guide

## Overview

The concat editor implements an efficient video editing approach using FFmpeg's concat demuxer. This method avoids complex filter graphs, enabling processing of videos with many segments while keeping filter strings under 300 characters.

## Key Advantages

- **Scalable**: Handles 50+ segments efficiently
- **Simple Filters**: Filter strings stay under 300 chars
- **Parallel Extraction**: Segments extracted simultaneously  
- **Minimal Re-encoding**: Copy streams when possible
- **Batch Transitions**: Efficient transition handling for many segments

## Architecture

```
Segments → Parallel Extract → Temp Files → Concat List → Concat Demuxer → Transitions → Output
              ↓                    ↓            ↓              ↓               ↓
          Process 1            File 1      concat.txt      FFmpeg         Filter/Post
          Process 2            File 2
          Process N            File N
```

## Usage

### Basic Editing

```python
from concat_editor import ConcatEditor, EditSegment

editor = ConcatEditor()

# Define edit segments
segments = [
    EditSegment("source.mp4", 10, 40),    # 30s segment
    EditSegment("source.mp4", 60, 90),    # 30s segment
    EditSegment("source.mp4", 120, 150),  # 30s segment
]

# Execute edit with transitions
result = editor.execute_edit(
    segments,
    "output.mp4",
    apply_transitions=True
)

print(f"Processed {result['segments_processed']} segments")
print(f"Processing ratio: {result['processing_ratio']:.2f}x")
```

### Custom Transitions

```python
# Segments with custom transitions
segments = [
    EditSegment(
        "video.mp4", 
        start_time=0,
        end_time=30,
        transition_type="fade",
        transition_duration=0.5
    ),
    EditSegment(
        "video.mp4",
        start_time=60, 
        end_time=90,
        transition_type="dissolve",
        transition_duration=1.0
    )
]

editor.execute_edit(segments, "edited.mp4")
```

### Filter Length Verification

```python
# Verify filter complexity before processing
segments = [EditSegment("input.mp4", i*10, (i+1)*10) for i in range(50)]

is_valid, length = editor.verify_filter_length(segments)
print(f"Filter valid: {is_valid}")
print(f"Filter length: {length} chars (limit: 300)")
```

### Integration with Highlights

```python
from concat_editor import create_edit_segments

# Convert highlight data to edit segments
highlights = [
    {
        'start_time': 30,
        'end_time': 60,
        'score': 0.9,
        'transition': 'fade',
        'transition_duration': 0.5
    },
    {
        'start_time': 120,
        'end_time': 150,
        'score': 0.85
    }
]

segments = create_edit_segments(highlights, "source.mp4")
result = editor.execute_edit(segments, "highlights.mp4")
```

## How It Works

### 1. Segment Extraction

Segments are extracted in parallel to temporary MPEG-TS files:

```python
def _extract_segments_to_files(self, segments):
    # Each segment extracted with:
    ffmpeg -ss START -t DURATION -i INPUT -c copy -f mpegts segment_N.ts
```

### 2. Concat List Creation

A text file lists all segments for the concat demuxer:

```
file '/tmp/segment_001.ts'
file '/tmp/segment_002.ts'
file '/tmp/segment_003.ts'
```

### 3. Concatenation

For simple concatenation without transitions:

```bash
ffmpeg -f concat -safe 0 -i concat.txt -c:v libx264 -c:a aac output.mp4
```

### 4. Transition Application

For small segment counts (< 10), uses xfadeall filter:

```
[0:v]xfadeall=transitions=2:d=0.5:d=0.5:offsets=29.75:59.75[v]
```

For many segments, applies transitions in post-processing to avoid filter complexity.

## Performance Optimization

### Parallel Extraction

All segments are extracted simultaneously:

```python
# Sequential: N segments × extraction_time each
# Parallel: max(extraction_time) across all segments
```

### MPEG-TS Format

Using MPEG-TS for intermediate files enables:
- Stream copying (no re-encoding)
- Timestamp correction
- Seamless concatenation

### Batch Transitions

For 50+ segments, transitions are applied efficiently:

1. First pass: Simple concatenation
2. Second pass: Apply fades at cut points

This keeps filter complexity manageable.

## SmartVideoEditor Integration

```python
from concat_editor import ConcatEditor, create_edit_segments
from metrics import track_processing_stage

class SmartVideoEditor:
    def __init__(self):
        self.editor = ConcatEditor()
    
    @track_processing_stage('editing')
    def execute_edit(self, job_id: str, input_path: str,
                     highlights: List[Dict], video_duration: float):
        """Execute edit using concat demuxer approach"""
        
        # Convert highlights to segments
        segments = create_edit_segments(highlights, input_path)
        
        # Verify filter complexity
        is_valid, filter_length = self.editor.verify_filter_length(segments)
        if not is_valid:
            logger.warning(f"Filter too complex ({filter_length} chars)")
        
        # Execute edit
        output_path = f"/output/{job_id}_edited.mp4"
        result = self.editor.execute_edit(
            segments,
            output_path,
            apply_transitions=True
        )
        
        # Track metrics
        metrics.processing_ratio.labels(stage='editing').observe(
            result['processing_ratio']
        )
        
        return {
            'output_path': output_path,
            'segments': len(segments),
            'duration': result['total_duration'],
            'ratio': result['processing_ratio']
        }
```

## Configuration

### Codec Settings

```python
# Speed-optimized for concatenation
video_codec = "libx264 -preset fast"
audio_codec = "aac"

# Quality-optimized
video_codec = "libx264 -preset slow -crf 18"
audio_codec = "aac -b:a 192k"
```

### Transition Types

Currently supported:
- `fade`: Fade to/from black
- `dissolve`: Cross-dissolve (with xfade)
- Custom transitions can be added

### Temp Directory

```python
# Configure temp directory for segment files
editor.temp_dir = "/fast/ssd/temp"  # Use fast storage
```

## Error Handling

```python
try:
    result = editor.execute_edit(segments, output_file)
except FFmpegError as e:
    # Handle FFmpeg-specific errors
    logger.error(f"FFmpeg error: {e}")
except Exception as e:
    # Handle general errors
    logger.error(f"Edit failed: {e}")
```

## Best Practices

### 1. Segment Batching

For very long videos, process in batches:

```python
BATCH_SIZE = 50

for i in range(0, len(all_segments), BATCH_SIZE):
    batch = all_segments[i:i + BATCH_SIZE]
    editor.execute_edit(batch, f"output_part_{i}.mp4")
```

### 2. Transition Duration

Keep transitions short for better performance:

```python
# Good - short transitions
transition_duration = 0.5  # 0.5 seconds

# Avoid - long transitions
transition_duration = 2.0  # Requires more processing
```

### 3. Source File Optimization

When possible, use the same source file:

```python
# Efficient - single source
segments = [
    EditSegment("video.mp4", 10, 40),
    EditSegment("video.mp4", 60, 90)
]

# Less efficient - multiple sources
segments = [
    EditSegment("video1.mp4", 10, 40),
    EditSegment("video2.mp4", 0, 30)
]
```

## Testing

### Unit Tests

```bash
# Run concat editor tests
pytest tests/test_concat_editor.py -v

# Test filter length verification
pytest tests/test_concat_editor.py::TestConcatEditor::test_filter_length_constraint -v
```

### Performance Testing

```python
# Test with many segments
segments = [EditSegment("input.mp4", i*10, (i+1)*10) for i in range(100)]

start = time.time()
result = editor.execute_edit(segments, "output.mp4")
elapsed = time.time() - start

print(f"Processed {len(segments)} segments in {elapsed:.1f}s")
print(f"Average per segment: {elapsed/len(segments):.2f}s")
```

## Troubleshooting

### Filter Too Long

If filter exceeds 300 chars:
- Editor automatically falls back to batch mode
- No manual intervention needed

### Temp File Errors

```python
# Ensure temp directory exists and is writable
os.makedirs(editor.temp_dir, exist_ok=True, mode=0o755)

# Check available space
import shutil
free_space = shutil.disk_usage(editor.temp_dir).free
print(f"Free space: {free_space / 1e9:.1f} GB")
```

### Concatenation Failures

Common issues:
- Mismatched codecs between segments
- Corrupted segment files
- Insufficient disk space

Debug with:
```bash
# Test concat list manually
ffmpeg -f concat -safe 0 -i concat.txt -c copy test.mp4
```