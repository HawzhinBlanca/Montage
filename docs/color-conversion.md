# Color Space Conversion Guide

## Overview

The color converter module ensures all video outputs are in BT.709 color space, meeting broadcast and streaming standards. It validates SDR input (rejecting HDR) and applies proper color space conversion throughout the pipeline.

## Key Features

- **HDR Detection & Rejection**: Fails fast on HDR input as per requirements
- **BT.709 Conversion**: Ensures all outputs use BT.709 primaries
- **Pipeline Integration**: Works seamlessly with editing and encoding
- **Verification**: Confirms output color space is correct

## Color Space Standards

### Supported (SDR)
- **BT.709**: HD video standard (most common)
- **BT.601/BT.470**: SD video standards
- **sRGB**: Computer graphics (treated as BT.709)

### Rejected (HDR)
- **BT.2020**: Wide color gamut
- **SMPTE 2084**: HDR10 PQ curve
- **ARIB STD-B67**: HLG (Hybrid Log-Gamma)
- **SMPTE 428**: Digital cinema

## Usage

### Basic Color Space Analysis

```python
from color_converter import ColorSpaceConverter

converter = ColorSpaceConverter()

# Analyze video color space
info = converter.analyze_color_space("input.mp4")

print(f"Color space: {info.color_space}")
print(f"Primaries: {info.color_primaries}")
print(f"Transfer: {info.color_transfer}")
print(f"Is HDR: {info.is_hdr}")
```

### SDR Validation

```python
# Validate input is SDR (not HDR)
is_valid, error_msg = converter.validate_sdr_input("video.mp4")

if not is_valid:
    print(f"Invalid input: {error_msg}")
    # Will print: "HDR input not supported"
```

### BT.709 Conversion

```python
# Convert any SDR input to BT.709
result = converter.convert_to_bt709(
    "input.mp4",
    "output_bt709.mp4",
    video_codec="libx264",
    audio_codec="copy"  # Don't re-encode audio
)

print(f"Source: {result['source_primaries']}")
print(f"Output: {result['output_primaries']}")  # Should be 'bt709'
print(f"Success: {result['conversion_successful']}")
```

### With Additional Filters

```python
# Apply other filters while ensuring BT.709 output
result = converter.convert_to_bt709(
    "input.mp4",
    "output.mp4",
    additional_filters="scale=1920:1080,unsharp=5:5:1.0"
)
```

## Integration with Video Pipeline

### SmartVideoEditor Integration

The SmartVideoEditor automatically ensures BT.709 output:

```python
from smart_video_editor import SmartVideoEditor

editor = SmartVideoEditor()

# Process job - color conversion is automatic
result = editor.process_job(
    job_id="job-123",
    input_path="input.mp4",
    highlights=[...]
)

# Output guaranteed to be BT.709
print(f"Color space: {result['color_space']}")  # 'bt709'
```

### Manual Pipeline Integration

```python
from concat_editor import ConcatEditor
from color_converter import ColorSpaceConverter

# Edit video
editor = ConcatEditor()
editor.execute_edit(segments, "temp_edited.mp4")

# Ensure BT.709 output
converter = ColorSpaceConverter()
converter.convert_to_bt709(
    "temp_edited.mp4",
    "final_output.mp4"
)
```

### Safe Encoding Commands

```python
# Build FFmpeg command with color safety
cmd = converter.build_safe_encoding_command(
    "input.mp4",
    "output.mp4",
    video_filters="denoise,sharpen",
    video_codec="libx264"
)

# Command includes:
# - User filters + color conversion filter
# - Color space metadata flags
# - Proper encoding parameters
```

## Technical Details

### ZScale Filter Chain

The converter uses a sophisticated filter chain:

```
zscale=t=linear:npl=100,         # Convert to linear light
format=gbrpf32le,                # Float precision RGB
zscale=p=bt709:t=bt709:m=bt709:r=tv,  # Convert to BT.709
format=yuv420p                   # Back to YUV for encoding
```

### FFmpeg Color Parameters

Ensures metadata is correct:

```bash
-colorspace bt709      # Color space flag
-color_primaries bt709 # Color primaries
-color_trc bt709      # Transfer characteristics  
-color_range tv       # Broadcast safe range
```

### Verification

After conversion, ffprobe should show:

```
color_space=bt709
color_primaries=bt709
color_transfer=bt709
color_range=tv
```

## Best Practices

### 1. Always Validate First

```python
# Good practice
is_valid, error = converter.validate_sdr_input(input_file)
if not is_valid:
    raise ValueError(f"Invalid input: {error}")

# Then process...
```

### 2. Preserve Quality

```python
# Use high-quality encoding
converter.convert_to_bt709(
    input_file,
    output_file,
    video_codec="libx264",
    audio_codec="copy"  # Avoid audio re-encoding
)
```

### 3. Chain Filters Efficiently

```python
# Combine all video filters in one pass
all_filters = f"{edit_filters},{color_filters}"
```

## Error Handling

### HDR Input

```python
try:
    converter.convert_to_bt709("hdr_video.mp4", "output.mp4")
except ColorConversionError as e:
    print(f"Error: {e}")  # "HDR input not supported"
```

### Missing Video Stream

```python
try:
    info = converter.analyze_color_space("audio_only.mp3")
except ColorConversionError as e:
    print(f"Error: {e}")  # "No video stream found"
```

## Testing Color Space

### Verify Output

```python
def verify_bt709_output(video_file: str) -> bool:
    """Verify video is BT.709"""
    converter = ColorSpaceConverter()
    info = converter.analyze_color_space(video_file)
    
    return (
        info.color_primaries == 'bt709' and
        info.color_space == 'bt709' and
        not info.is_hdr
    )

# Test
assert verify_bt709_output("output.mp4")
```

### Command Line Verification

```bash
# Check with ffprobe
ffprobe -v error -select_streams v:0 -show_entries stream=color_primaries,color_space,color_transfer -of json output.mp4

# Should show:
# "color_space": "bt709"
# "color_primaries": "bt709"  
# "color_transfer": "bt709"
```

## Performance Considerations

### Filter Complexity

The zscale filter chain adds processing time:
- ~10-20% overhead for color conversion
- Use hardware acceleration when available
- Consider resolution before conversion

### Optimization

```python
# For faster processing, combine with scaling
additional_filters = "scale=1920:1080"  # Do scaling first
converter.convert_to_bt709(input, output, additional_filters=additional_filters)
```

## Troubleshooting

### Output Not BT.709

1. Check source color space
2. Verify filter chain is applied
3. Ensure encoding flags are set
4. Test with simple input first

### Performance Issues

1. Reduce video resolution first
2. Use faster preset: `-preset faster`
3. Skip conversion for already-BT.709 content
4. Use hardware encoding if available

### Quality Loss

1. Use higher CRF value: `-crf 17`
2. Avoid multiple conversions
3. Keep intermediate files lossless
4. Verify source quality first