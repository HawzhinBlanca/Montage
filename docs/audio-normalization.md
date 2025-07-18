# Audio Normalization Guide

## Overview

The audio normalizer implements two-pass loudness normalization using FFmpeg's loudnorm filter, ensuring consistent audio levels across video segments. It meets the requirement of maintaining RMS spread ≤ 1.5 LU between concatenated segments.

## Two-Pass Normalization Process

### Pass 1: Analysis
Measures the input audio characteristics:
- **Integrated Loudness** (LUFS): Overall loudness
- **True Peak** (dBTP): Maximum peak level
- **Loudness Range** (LU): Dynamic range
- **Threshold**: Gating threshold

### Pass 2: Normalization
Applies precise adjustments using measured values to achieve:
- Target integrated loudness: -16 LUFS (streaming standard)
- Maximum true peak: -1 dBTP (prevent clipping)
- Target loudness range: 7 LU (good dynamics)

## Usage

### Basic Normalization

```python
from audio_normalizer import AudioNormalizer, NormalizationTarget

normalizer = AudioNormalizer()

# Use default streaming targets
result = normalizer.normalize_audio("input.mp4", "output.mp4")

print(f"Input: {result['input_loudness']:.1f} LUFS")
print(f"Output: {result['output_loudness']:.1f} LUFS")
print(f"Adjusted by: {result['adjustment_db']:.1f} dB")
```

### Custom Targets

```python
# Broadcast target
broadcast_target = NormalizationTarget(
    integrated=-23.0,  # EBU R128 broadcast
    true_peak=-1.0,
    lra=15.0          # Wider range allowed
)

result = normalizer.normalize_audio(
    "input.mp4", 
    "output.mp4",
    target=broadcast_target
)

# Podcast target
podcast_target = NormalizationTarget(
    integrated=-19.0,  # Louder for mobile
    true_peak=-1.0,
    lra=7.0           # Consistent levels
)
```

### Normalizing Multiple Segments

```python
# Ensure consistent loudness across segments
segments = ["segment1.mp4", "segment2.mp4", "segment3.mp4"]
outputs = ["norm_seg1.mp4", "norm_seg2.mp4", "norm_seg3.mp4"]

result = normalizer.normalize_segments(segments, outputs)

print(f"Initial spread: {result['initial_spread']:.1f} LU")
print(f"Final spread: {result['final_spread']:.1f} LU")
print(f"Meets target: {result['meets_target']}")  # True if ≤ 1.5 LU
```

### EBU R128 Analysis

```python
# Detailed loudness analysis
measurements = normalizer.apply_ebur128_analysis("video.mp4")

print(f"Integrated: {measurements['integrated_lufs']:.1f} LUFS")
print(f"LRA: {measurements['lra_lu']:.1f} LU")
print(f"True Peak: {measurements['peak_dbfs']:.1f} dBFS")
```

## Integration with Video Processing

### SmartVideoEditor Integration

```python
from audio_normalizer import AudioNormalizer, NormalizationTarget
from concat_editor import ConcatEditor
from metrics import track_processing_stage

class SmartVideoEditor:
    def __init__(self):
        self.normalizer = AudioNormalizer()
        self.editor = ConcatEditor()
    
    @track_processing_stage('full_processing')
    def process_video(self, job_id: str, input_path: str, 
                      segments: List[EditSegment]) -> Dict[str, Any]:
        """Process video with editing and normalization"""
        
        # Step 1: Edit video
        temp_edited = f"/tmp/{job_id}_edited.mp4"
        edit_result = self.editor.execute_edit(
            segments,
            temp_edited,
            apply_transitions=True
        )
        
        # Step 2: Normalize audio
        final_output = f"/output/{job_id}_final.mp4"
        norm_result = self.normalizer.normalize_audio(
            temp_edited,
            final_output,
            target=NormalizationTarget()  # -16 LUFS streaming
        )
        
        # Track quality metrics
        metrics.track_audio_spread(norm_result.get('spread_lu', 0))
        metrics.track_audio_normalization(norm_result['adjustment_db'])
        
        return {
            'output_path': final_output,
            'segments': len(segments),
            'loudness': norm_result['output_loudness'],
            'spread': norm_result.get('spread_lu', 0)
        }
```

### Convenience Function

```python
from audio_normalizer import normalize_video_audio

# Simple one-line normalization
result = normalize_video_audio("input.mp4", "normalized.mp4")
```

## Technical Details

### Loudness Standards

| Standard | Integrated | True Peak | LRA | Use Case |
|----------|------------|-----------|-----|----------|
| Streaming | -16 LUFS | -1 dBTP | 7 LU | YouTube, Netflix |
| Broadcast | -23 LUFS | -1 dBTP | 15 LU | TV (EBU R128) |
| Podcast | -19 LUFS | -1 dBTP | 7 LU | Spoken content |
| Music | -14 LUFS | -1 dBTP | 10 LU | Music streaming |

### FFmpeg Filter Syntax

Pass 1 - Analysis:
```bash
ffmpeg -i input.mp4 -af loudnorm=print_format=json -f null -
```

Pass 2 - Normalization:
```bash
ffmpeg -i input.mp4 -af loudnorm=I=-16:TP=-1:LRA=7:measured_I=-23.5:measured_TP=-3.2:measured_LRA=8.7 -c:a aac output.mp4
```

### Understanding Measurements

- **LUFS** (Loudness Units Full Scale): Perceptual loudness measurement
- **dBTP** (dB True Peak): Actual peak level including inter-sample peaks
- **LU** (Loudness Units): Relative loudness difference
- **LRA** (Loudness Range): Difference between soft and loud parts

## Best Practices

### 1. Target Selection

```python
# For dialogue-heavy content
dialogue_target = NormalizationTarget(
    integrated=-19.0,  # Slightly louder
    true_peak=-3.0,    # More headroom
    lra=10.0          # Allow natural dynamics
)

# For action sequences
action_target = NormalizationTarget(
    integrated=-16.0,  # Standard level
    true_peak=-1.0,    # Less headroom needed
    lra=7.0           # Tighter control
)
```

### 2. Segment Consistency

When processing multiple segments:

```python
# Process all segments with same target
target = NormalizationTarget()  # Use consistent target

for segment in segments:
    normalizer.normalize_audio(segment, output, target=target)
```

### 3. Quality Settings

```python
# In _apply_normalization method
'-c:a', 'aac',      # Audio codec
'-b:a', '192k',     # Bitrate (192k for good quality)
'-ar', '48000',     # Sample rate (optional)
```

## Troubleshooting

### High Spread Between Segments

If segments have > 1.5 LU spread after normalization:

1. Check source material consistency
2. Consider pre-processing very quiet/loud segments
3. Use segment normalization method:

```python
# This ensures all segments match target
result = normalizer.normalize_segments(segments, outputs)
```

### Distortion After Normalization

If audio sounds distorted:

1. Check true peak isn't too high:
```python
target = NormalizationTarget(true_peak=-3.0)  # More headroom
```

2. Verify input isn't already clipping
3. Consider wider LRA for dynamic content

### Processing Performance

For faster processing:

```python
# Copy video stream (no re-encoding)
'-c:v', 'copy',

# Use faster audio encoder
'-c:a', 'aac', '-aac_coder', 'fast',
```

## Verification

### Check Normalization Results

```python
# Verify output meets requirements
stats = normalizer._analyze_loudness("output.mp4")

print(f"Output loudness: {stats.input_i:.1f} LUFS")
print(f"Within target: {abs(stats.input_i - (-16.0)) < 0.5}")
```

### Automated Testing

```python
def test_normalization_quality(output_file: str) -> bool:
    """Test if normalization meets quality standards"""
    normalizer = AudioNormalizer()
    stats = normalizer._analyze_loudness(output_file)
    
    # Check loudness is within tolerance
    target_loudness = -16.0
    tolerance = 0.5
    
    return abs(stats.input_i - target_loudness) <= tolerance
```

## Metrics Integration

The normalizer automatically tracks:

- `audio_normalization_adjustments`: Histogram of dB adjustments
- `audio_loudness_spread`: Histogram of LU spread between segments

Monitor these metrics to ensure consistent quality:

```prometheus
# Alert if spread too high
alert: HighAudioSpread
expr: histogram_quantile(0.95, audio_loudness_spread_lu_bucket) > 1.5
for: 5m
annotations:
  summary: "Audio spread exceeding 1.5 LU target"
```