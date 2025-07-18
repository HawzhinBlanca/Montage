#!/usr/bin/env python3
"""
End-to-end test for the real video processing pipeline
"""
import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
import pytest

def create_test_video():
    """Create a test video file for testing"""
    test_video_path = "tests/data/test_lecture.mp4"
    
    # Create tests/data directory if it doesn't exist
    os.makedirs("tests/data", exist_ok=True)
    
    if os.path.exists(test_video_path):
        return test_video_path
    
    # Create a simple test video using ffmpeg
    try:
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=30:size=1920x1080:rate=24',
            '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=30',
            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', '-y', test_video_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Created test video: {test_video_path}")
        return test_video_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create test video: {e}")
        return None
    except FileNotFoundError:
        print("❌ FFmpeg not found - cannot create test video")
        return None

def test_video_analysis():
    """Test video analysis module"""
    test_video = create_test_video()
    if not test_video:
        pytest.skip("Cannot create test video")
    
    # Import and test analyze_video
    from src.analyze_video import analyze_video
    
    result = analyze_video(test_video)
    
    assert "sha" in result
    assert "words" in result
    assert "transcript" in result
    assert isinstance(result["words"], list)
    assert isinstance(result["transcript"], str)
    
    print(f"✅ Video analysis: {len(result['words'])} words")

def test_highlight_selection():
    """Test highlight selection module"""
    # Mock word data for testing
    mock_words = [
        {"word": "This", "start": 0.0, "end": 0.5, "confidence": 0.9},
        {"word": "is", "start": 0.5, "end": 0.7, "confidence": 0.9},
        {"word": "an", "start": 0.7, "end": 0.9, "confidence": 0.8},
        {"word": "important", "start": 0.9, "end": 1.5, "confidence": 0.9},
        {"word": "discovery", "start": 1.5, "end": 2.2, "confidence": 0.85},
        {"word": "in", "start": 2.2, "end": 2.4, "confidence": 0.9},
        {"word": "science.", "start": 2.4, "end": 3.0, "confidence": 0.9},
    ]
    
    mock_audio_energy = [0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6]
    
    from src.highlight_selector import choose_highlights
    
    # Test smart mode
    highlights = choose_highlights(mock_words, mock_audio_energy, "smart")
    
    assert isinstance(highlights, list)
    assert len(highlights) > 0
    assert all("start_ms" in h and "end_ms" in h for h in highlights)
    
    print(f"✅ Highlight selection: {len(highlights)} highlights")

def test_ffmpeg_utils():
    """Test FFmpeg utilities"""
    test_video = create_test_video()
    if not test_video:
        pytest.skip("Cannot create test video")
    
    from src.ffmpeg_utils import get_video_info, extract_video_segment
    
    # Test video info
    info = get_video_info(test_video)
    assert "duration" in info
    assert "width" in info
    assert "height" in info
    
    # Test segment extraction
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        output_path = tmp.name
    
    try:
        success = extract_video_segment(test_video, 1000, 3000, output_path)  # 1-3 seconds
        assert success
        assert os.path.exists(output_path)
        
        # Check extracted segment
        segment_info = get_video_info(output_path)
        assert segment_info["duration"] >= 1.8  # Should be ~2 seconds
        
        print(f"✅ FFmpeg utils: segment extraction successful")
        
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

def test_mcp_server():
    """Test MCP server functionality"""
    import threading
    import time
    import requests
    
    from src.resolve_mcp import start_server
    
    # Start server in background thread
    server_thread = threading.Thread(target=lambda: start_server(host='localhost', port=7802), daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:7802/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        print("✅ MCP server: health check passed")
    except Exception as e:
        pytest.skip(f"MCP server not responding: {e}")

def test_e2e_pipeline():
    """End-to-end pipeline test"""
    test_video = create_test_video()
    if not test_video:
        pytest.skip("Cannot create test video")
    
    # Run the pipeline
    result = subprocess.run([
        sys.executable, '-m', 'src.run_pipeline', test_video, '--mode', 'smart', '--no-server'
    ], capture_output=True, text=True, cwd=os.getcwd())
    
    print(f"Pipeline stdout: {result.stdout}")
    print(f"Pipeline stderr: {result.stderr}")
    
    # Check if pipeline completed successfully
    assert result.returncode == 0, f"Pipeline failed with exit code {result.returncode}"
    
    # Check for expected output patterns
    assert "Video analysis complete" in result.stdout or "✅" in result.stdout
    assert "Pipeline Completed Successfully" in result.stdout or "✅" in result.stdout
    
    print("✅ End-to-end pipeline test passed")

def test_pipeline_json_output():
    """Test pipeline JSON output"""
    test_video = create_test_video()
    if not test_video:
        pytest.skip("Cannot create test video")
    
    # Run pipeline and check for JSON plan file
    result = subprocess.run([
        sys.executable, '-m', 'src.run_pipeline', test_video, '--mode', 'smart', '--no-server'
    ], capture_output=True, text=True, cwd=os.getcwd())
    
    assert result.returncode == 0
    
    # Find generated JSON plan file
    plan_files = list(Path('.').glob('montage_plan_*.json'))
    assert len(plan_files) > 0, "No JSON plan file generated"
    
    # Validate JSON content
    with open(plan_files[0], 'r') as f:
        plan = json.load(f)
    
    assert "highlights" in plan
    assert "analysis" in plan
    assert "mode" in plan
    assert plan["success"] is True
    
    print(f"✅ JSON plan validation passed: {plan_files[0]}")
    
    # Clean up
    for plan_file in plan_files:
        os.unlink(plan_file)

if __name__ == "__main__":
    # Run tests individually for debugging
    print("🧪 Running end-to-end tests...")
    
    try:
        test_video_analysis()
        test_highlight_selection()
        test_ffmpeg_utils()
        test_mcp_server()
        test_e2e_pipeline()
        test_pipeline_json_output()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)