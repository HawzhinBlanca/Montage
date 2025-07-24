#!/usr/bin/env python3
"""Test to verify if audio normalization is actually integrated into the pipeline"""

import os
import sys
import subprocess
import tempfile

from montage.providers.audio_normalizer import AudioNormalizer, NormalizationTarget

def create_test_video_with_quiet_audio(output_path, duration=5):
    """Create a test video with very quiet audio (-40 LUFS)"""
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size=640x480:rate=30",
        "-f", "lavfi", 
        "-i", f"sine=frequency=440:duration={duration}:sample_rate=44100",
        "-af", "volume=-40dB",  # Make it very quiet
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def measure_loudness(video_path):
    """Measure loudness of a video file"""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-af", "loudnorm=print_format=json",
        "-f", "null",
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract LUFS value from output
    import re
    match = re.search(r'"input_i"\s*:\s*"(-?\d+\.?\d*)"', result.stderr)
    if match:
        return float(match.group(1))
    return None

def test_normalization():
    """Test if normalization actually changes audio levels"""
    print("🧪 Testing Audio Normalization Integration\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test videos with different loudness levels
        quiet_video = os.path.join(temp_dir, "quiet_test.mp4")
        loud_video = os.path.join(temp_dir, "loud_test.mp4")
        
        print("1. Creating test videos...")
        # Very quiet video
        create_test_video_with_quiet_audio(quiet_video)
        
        # Loud video
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", "testsrc=duration=5:size=640x480:rate=30",
            "-f", "lavfi", 
            "-i", "sine=frequency=880:duration=5:sample_rate=44100",
            "-af", "volume=0dB",  # Normal loudness
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            loud_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Measure original loudness
        print("\n2. Measuring original loudness...")
        quiet_loudness = measure_loudness(quiet_video)
        loud_loudness = measure_loudness(loud_video)
        
        print(f"   Quiet video: {quiet_loudness:.1f} LUFS")
        print(f"   Loud video: {loud_loudness:.1f} LUFS")
        print(f"   Difference: {abs(loud_loudness - quiet_loudness):.1f} LU")
        
        # Test normalization
        print("\n3. Testing AudioNormalizer...")
        normalizer = AudioNormalizer()
        
        quiet_normalized = os.path.join(temp_dir, "quiet_normalized.mp4")
        loud_normalized = os.path.join(temp_dir, "loud_normalized.mp4")
        
        # Normalize both videos to -16 LUFS
        target = NormalizationTarget(integrated=-16.0, true_peak=-1.0, lra=7.0)
        
        quiet_result = normalizer.normalize_audio(quiet_video, quiet_normalized, target)
        loud_result = normalizer.normalize_audio(loud_video, loud_normalized, target)
        
        print("\n4. Results:")
        print(f"   Quiet video adjustment: {quiet_result['adjustment_db']:.1f} dB")
        print(f"   Loud video adjustment: {loud_result['adjustment_db']:.1f} dB")
        
        # Measure normalized loudness
        quiet_norm_loudness = measure_loudness(quiet_normalized)
        loud_norm_loudness = measure_loudness(loud_normalized)
        
        print(f"\n5. Final loudness:")
        print(f"   Quiet normalized: {quiet_norm_loudness:.1f} LUFS")
        print(f"   Loud normalized: {loud_norm_loudness:.1f} LUFS")
        print(f"   Spread: {abs(loud_norm_loudness - quiet_norm_loudness):.1f} LU")
        
        # Check if normalization worked
        print("\n6. Verification:")
        if abs(quiet_norm_loudness - (-16.0)) < 2.0 and abs(loud_norm_loudness - (-16.0)) < 2.0:
            print("   ✅ Normalization is working correctly!")
            print("   ✅ Both videos normalized to within 2 LU of target")
        else:
            print("   ❌ Normalization may not be working as expected")
            
        # Test segment normalization
        print("\n7. Testing segment normalization...")
        segments = [quiet_video, loud_video]
        output_segments = [
            os.path.join(temp_dir, "seg1_norm.mp4"),
            os.path.join(temp_dir, "seg2_norm.mp4")
        ]
        
        seg_result = normalizer.normalize_segments(segments, output_segments, target)
        print(f"   Initial spread: {seg_result['initial_spread']:.1f} LU")
        print(f"   Final spread: {seg_result['final_spread']:.1f} LU")
        print(f"   Meets target (≤1.5 LU): {'✅' if seg_result['meets_target'] else '❌'}")
        
        # Check if it's integrated in the pipeline
        print("\n8. Checking pipeline integration...")
        print("   Searching for AudioNormalizer usage in main pipeline...")
        
        # Search for usage in actual pipeline files
        pipeline_files = [
            "src/providers/video_processor.py",
            "src/providers/concat_editor.py", 
            "src/cli/run_pipeline.py",
            "main.py"
        ]
        
        integration_found = False
        for file in pipeline_files:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    content = f.read()
                    if any(term in content for term in ['AudioNormalizer', 'normalize_audio', 'loudnorm']):
                        print(f"   ✅ Found in {file}")
                        integration_found = True
                        
        if not integration_found:
            print("   ⚠️  AudioNormalizer not found in main pipeline files")
            print("   ⚠️  The normalization code exists but may not be integrated!")

if __name__ == "__main__":
    test_normalization()