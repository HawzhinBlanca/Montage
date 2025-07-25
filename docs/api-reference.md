# Montage API Reference

Complete reference for all production-ready functionality.

## Core Modules

### Video Analysis (`src/core/analyze_video.py`)

Main video analysis with real ASR transcription.

#### `analyze_video(video_path: str) -> Dict[str, Any]`

Analyzes video using real Whisper and Deepgram transcription.

**Parameters:**
- `video_path`: Path to video file

**Returns:**
```python
{
    "sha": "file_hash",
    "words": [
        {
            "word": "hello",
            "start": 1.2,
            "end": 1.5,
            "confidence": 0.95,
            "speaker": "SPEAKER_00"
        }
    ],
    "transcript": "full text transcript",
    "speaker_turns": [
        {
            "speaker": "SPEAKER_00", 
            "start": 0.0,
            "end": 10.5
        }
    ]
}
```

#### `transcribe_whisper(wav_path: str) -> List[Dict]`

Real faster-whisper transcription with word timestamps.

#### `transcribe_deepgram(wav_path: str, job_id: str) -> List[Dict]`

Real Deepgram API transcription with cost tracking.

#### `rover_merge(fw_words: List, dg_words: List) -> List[Dict]`

ROVER algorithm for combining multiple ASR outputs.

### Highlight Selection (`src/core/highlight_selector.py`)

Content analysis and segment selection with real functionality only.

#### `select_highlights(transcript_segments: List, audio_energy: List, job_id: str) -> List[Dict]`

Select highlight segments using real content analysis.

**Parameters:**
- `transcript_segments`: Word-level transcript data
- `audio_energy`: Audio RMS energy levels
- `job_id`: Job identifier for tracking

**Returns:**
```python
[
    {
        "start_ms": 5000,
        "end_ms": 25000,
        "text": "segment text",
        "title": "Auto-generated title",
        "score": 0.85,
        "sentence_complete": True,
        "keyword_count": 3
    }
]
```

#### `local_rule_scoring(segments: List, audio_energy: List, job_id: str) -> List[Dict]`

Real local scoring based on:
- Complete sentence detection
- Keyword matching
- Audio energy analysis
- Duration optimization

### Video Processing (`src/providers/video_processor.py`)

High-performance FIFO-based video processing pipeline.

#### `VideoEditor.extract_and_concatenate_efficient(input_file: str, segments: List, output_file: str, apply_transitions: bool) -> None`

Process video segments with parallel extraction and professional encoding.

**Features:**
- FIFO-based streaming (zero intermediate files)
- Parallel segment extraction
- Professional transitions and encoding
- Performance ratio tracking

#### `VideoSegment` class

```python
@dataclass
class VideoSegment:
    start_time: float
    end_time: float
    input_file: str
    segment_id: str = None
    
    @property
    def duration(self) -> float
```

### Intelligent Cropping (`src/utils/intelligent_crop.py`)

OpenCV-based face detection and content-aware cropping.

#### `IntelligentCropper.analyze_video_content(video_path: str, start_ms: int, end_ms: int) -> Dict`

Analyze video segment for optimal cropping.

**Returns:**
```python
{
    "crop_center": (0.6, 0.4),  # Optimal crop center (x, y)
    "confidence": 0.85,         # Face detection confidence
    "face_count": 2,           # Number of faces detected
    "motion_detected": True,    # Motion analysis result
    "frames_analyzed": 15      # Frames processed
}
```

#### Face Detection Features
- Multi-method detection (Haar cascades + DNN)
- Motion analysis with optical flow
- Rule of thirds positioning
- Confidence scoring

### Cost Management (`src/core/cost.py`)

Budget enforcement with hard caps.

#### `@priced(service: str, per_unit_usd: Decimal)` decorator

Enforces budget limits before API calls.

```python
@priced("deepgram.nova-2", Decimal("0.0003"))
def transcribe_audio(audio_path: str, job_id: str = "default"):
    # API call with automatic cost tracking
```

#### Functions
- `get_current_cost() -> float`: Current spend
- `reset_cost()`: Reset for new job
- `check_budget(additional_cost: float) -> Tuple[bool, float]`: Budget check

### Metrics & Monitoring (`src/core/metrics.py`)

Comprehensive Prometheus-based monitoring.

#### `MetricsManager` class

Singleton metrics manager with 15+ metric types.

#### Key Metrics
- Job lifecycle tracking
- Processing duration and ratios
- API cost monitoring
- Error tracking by stage and type
- Resource usage (FFmpeg processes, DB connections)

#### Decorators
```python
@track_processing_stage("transcription")
def process_audio():
    # Automatically tracked with duration and error handling

@track_api_cost("openai", lambda result: result["cost"])
def call_openai_api():
    # Cost tracking with success/failure rates
```

### Performance Optimization (`src/core/performance.py`)

Advanced performance monitoring and optimization.

#### `PerformanceMonitor` class

Real-time resource monitoring and optimization.

```python
monitor = PerformanceMonitor()
usage = monitor.get_resource_usage()
# Returns: ResourceUsage(cpu_percent, memory_mb, memory_percent, disk_io_mb)
```

#### `PerformanceConfig` presets
```python
# Speed optimized
config = PerformanceConfig.fast_preset()

# Quality optimized  
config = PerformanceConfig.quality_preset()
```

#### Optimization Functions
- `get_optimal_worker_count() -> int`: System-based worker calculation
- `get_ffmpeg_optimization_flags(preset: str) -> List`: Optimized FFmpeg parameters
- `optimize_for_video_processing()`: Apply system-wide optimizations

## Utility Modules

### FFmpeg Utils (`src/utils/ffmpeg_utils.py`)

Professional video processing utilities.

#### `concatenate_video_segments(clips: List, source_video: str, output_path: str, vertical_format: bool, professional: bool) -> bool`

High-level video concatenation with:
- Professional encoding settings
- Intelligent cropping integration
- Subtitle support
- Error handling

#### `create_subtitle_file(words: List, start_ms: int, end_ms: int, output_path: str) -> bool`

Generate SRT subtitles from word-level timestamps.

### Video Validation (`src/utils/video_validator.py`)

Comprehensive video file validation.

#### `VideoValidator.validate_file(file_path: str) -> Tuple[bool, VideoMetadata, Optional[str]]`

Validates video files with:
- Corruption detection
- Format verification
- Metadata extraction
- Error reporting

## Configuration & Setup

### Secret Management (`src/utils/secret_loader.py`)

Secure secret loading with AWS Secrets Manager fallback.

#### `get(name: str, default: str = None) -> Optional[str]`

Load secrets with priority:
1. LRU cache
2. AWS Secrets Manager
3. Environment variables
4. Default values

### Database (`src/core/db.py`)

PostgreSQL database operations with connection pooling.

#### `Database` class

Professional database operations with:
- Connection pooling
- Transaction management
- Error handling
- Query optimization

## Command Line Interface

### Main CLI (`run_montage.py`)

User-friendly entry point:

```bash
# Basic processing
python run_montage.py video.mp4

# Vertical format for social media
python run_montage.py video.mp4 --vertical

# Premium quality mode
python run_montage.py video.mp4 --mode premium

# Video info only
python run_montage.py video.mp4 --info
```

### Advanced CLI (`main.py`)

Professional pipeline orchestrator:

```bash
python main.py input.mp4 output.mp4 \
  --smart-crop \
  --aspect-ratio 9:16 \
  --edit-plan plan.json
```

## Error Handling

All modules include comprehensive error handling:

- Custom exception classes
- Graceful degradation
- Detailed logging
- Resource cleanup
- Recovery mechanisms

### Common Exception Types

- `VideoValidationError`: File validation issues
- `VideoProcessingError`: Processing failures  
- `FFmpegError`: FFmpeg command failures
- `RuntimeError`: Budget cap violations

## Performance Characteristics

- **Processing Ratio**: < 1.2x for 1080p video
- **Memory Usage**: < 8GB for typical workloads
- **Cost Tracking**: Real-time with hard $5 cap
- **Parallel Processing**: Up to 6 workers automatically tuned
- **Zero-Copy**: FIFO-based streaming eliminates temp files

## Production Deployment

See `/infrastructure` for:
- Docker containers
- Kubernetes manifests  
- Terraform modules
- Monitoring setup

## Testing

Comprehensive test suite in `/tests` with:
- Unit tests for all components
- Integration tests with real containers
- Performance benchmarks
- Error scenario coverage