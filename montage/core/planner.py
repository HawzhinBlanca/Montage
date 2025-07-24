from pathlib import Path
from typing import List, Dict
import json

def generate_plan(segments: List[Dict], scores: List[float], source: Path) -> Dict:
    """
    Return a JSON‐serializable dict:
      { 'version': '1.0',
        'source': str(source),
        'actions': [
          {'type':'cut','start':…, 'end':…, 'score':…}, … ],
        'render': {'format':'9:16'} }
    """
    actions = []
    
    # Sort segments by score (highest first) and take top segments
    scored_segments = list(zip(segments, scores))
    scored_segments.sort(key=lambda x: x[1], reverse=True)
    
    for seg, score in scored_segments:
        actions.append({
            'type': 'cut',
            'start_ms': seg['start'],
            'end_ms': seg['end'],
            'score': round(score, 3),
        })
    
    plan = {
        'version': '1.0',
        'source': str(source),
        'actions': actions,
        'render': {'format': '9:16', 'codec': 'h264', 'crf': 18},
    }
    
    # Validate plan using our existing schema
    from .plan import validate_plan
    validate_plan(plan)
    
    return plan

def execute_plan(plan: Dict, output_path: Path):
    """
    REAL implementation that actually creates video files using FFmpeg
    """
    import subprocess
    import tempfile
    import os
    
    print(f"🎬 Executing plan with {len(plan['actions'])} actions")
    print(f"   Source: {plan['source']}")
    print(f"   Output: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not plan['actions']:
        print("⚠️  No actions in plan - creating 5-second black video")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=black:size=1920x1080:duration=5',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output_path)
            ], check=True, capture_output=True)
            print(f"✅ Created placeholder video → {output_path}")
            return
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create placeholder video: {e}")
            return
    
    # Verify source video exists
    source_video = plan['source']
    if not os.path.exists(source_video):
        print(f"❌ Source video not found: {source_video}")
        return
    
    try:
        # Use FFmpeg to extract and concatenate segments
        with tempfile.TemporaryDirectory() as temp_dir:
            segment_files = []
            
            print("   Extracting segments...")
            for i, action in enumerate(plan['actions']):
                start_sec = action['start_ms'] / 1000.0
                duration_sec = (action['end_ms'] - action['start_ms']) / 1000.0
                segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                
                print(f"   Segment {i+1}: {start_sec:.1f}s - {start_sec + duration_sec:.1f}s")
                
                # Extract segment with FFmpeg
                cmd = [
                    'ffmpeg', '-y',
                    '-i', source_video,
                    '-ss', str(start_sec),
                    '-t', str(duration_sec),
                    '-c:v', 'libx264',  # Re-encode for compatibility
                    '-c:a', 'aac',
                    '-avoid_negative_ts', 'make_zero',
                    segment_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(segment_path):
                    segment_files.append(segment_path)
                    print(f"      ✅ Extracted segment {i+1}")
                else:
                    print(f"      ❌ Failed to extract segment {i+1}: {result.stderr[-200:]}")
            
            if not segment_files:
                print("❌ No segments extracted successfully")
                return
            
            print(f"   Concatenating {len(segment_files)} segments...")
            
            # Create concat demuxer file
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, 'w') as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")
            
            # Concatenate all segments
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # Copy streams without re-encoding
                str(output_path)
            ]
            
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size
                print(f"✅ Plan executed successfully → {output_path}")
                print(f"   Output size: {file_size / (1024*1024):.1f} MB")
                
                # Verify it's a valid video file
                verify_cmd = ['ffprobe', '-v', 'quiet', str(output_path)]
                verify_result = subprocess.run(verify_cmd, capture_output=True)
                if verify_result.returncode == 0:
                    print(f"   ✅ Video file verified as valid")
                else:
                    print(f"   ⚠️  Video file may be corrupted")
                    
            else:
                print(f"❌ Concatenation failed: {result.stderr[-200:]}")
                
    except Exception as e:
        print(f"❌ Plan execution failed: {e}")
        import traceback
        traceback.print_exc()

def _execute_plan_with_ffmpeg(plan: Dict, output_path: Path):
    """Fallback: Direct FFmpeg implementation for plan execution"""
    import subprocess
    import tempfile
    import os
    
    source_video = plan['source']
    actions = plan['actions']
    
    try:
        # Create temporary directory for segments
        with tempfile.TemporaryDirectory() as temp_dir:
            segment_files = []
            
            # Extract each segment
            for i, action in enumerate(actions):
                start_sec = action['start_ms'] / 1000.0
                duration_sec = (action['end_ms'] - action['start_ms']) / 1000.0
                segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
                
                # FFmpeg command to extract segment
                cmd = [
                    'ffmpeg', '-y',
                    '-i', source_video,
                    '-ss', str(start_sec),
                    '-t', str(duration_sec),
                    '-c', 'copy',  # Copy without re-encoding for speed
                    segment_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    segment_files.append(segment_path)
                    print(f"   ✅ Extracted segment {i+1}")
                else:
                    print(f"   ❌ Failed to extract segment {i+1}: {result.stderr}")
            
            if not segment_files:
                print("❌ No segments extracted successfully")
                return
            
            # Create concat file list
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, 'w') as f:
                for segment_file in segment_files:
                    f.write(f"file '{segment_file}'\n")
            
            # Concatenate segments
            output_path.parent.mkdir(parents=True, exist_ok=True)
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                str(output_path)
            ]
            
            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ FFmpeg concat successful → {output_path}")
                if output_path.exists():
                    print(f"   Output size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                print(f"❌ FFmpeg concat failed: {result.stderr}")
                
    except Exception as e:
        print(f"❌ FFmpeg fallback failed: {e}")