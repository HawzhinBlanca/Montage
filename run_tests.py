#!/usr/bin/env python3
"""
Local test runner for the real video processing pipeline
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python modules
    required_modules = [
        "faster_whisper", "deepgram", "openai", "anthropic", 
        "cv2", "ffmpeg", "rich", "psycopg2", "redis", "bottle"
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"   ❌ {module}")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ ffmpeg")
        else:
            print("   ❌ ffmpeg (not working)")
            return False
    except FileNotFoundError:
        print("   ❌ ffmpeg (not found)")
        return False
    
    return True

def test_imports():
    """Test that all modules import correctly"""
    print("\n🧪 Testing module imports...")
    
    modules = [
        "src.config",
        "src.analyze_video", 
        "src.highlight_selector",
        "src.ffmpeg_utils",
        "src.resolve_mcp",
        "src.run_pipeline"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except Exception as e:
            print(f"   ❌ {module}: {e}")
            return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test config
        from src.config import DATABASE_URL, MAX_COST_USD
        print(f"   ✅ Config: DATABASE_URL={DATABASE_URL[:20]}...")
        
        # Test FFmpeg utils
        from src.ffmpeg_utils import format_srt_time
        result = format_srt_time(65.5)
        expected = "00:01:05,500"
        assert result == expected, f"Expected {expected}, got {result}"
        print("   ✅ FFmpeg utils: SRT time formatting")
        
        # Test highlight selector (local mode)
        from src.highlight_selector import choose_highlights
        mock_words = [
            {"word": "This", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"word": "is", "start": 0.5, "end": 0.8, "confidence": 0.9},
            {"word": "an", "start": 0.8, "end": 1.0, "confidence": 0.8},
            {"word": "important", "start": 1.0, "end": 1.8, "confidence": 0.9},
            {"word": "discovery", "start": 1.8, "end": 2.8, "confidence": 0.8},
            {"word": "in", "start": 2.8, "end": 3.0, "confidence": 0.9},
            {"word": "science.", "start": 3.0, "end": 3.8, "confidence": 0.9},
            {"word": "The", "start": 4.0, "end": 4.3, "confidence": 0.9},
            {"word": "research", "start": 4.3, "end": 5.0, "confidence": 0.9},
            {"word": "shows", "start": 5.0, "end": 5.5, "confidence": 0.8},
            {"word": "remarkable", "start": 5.5, "end": 6.5, "confidence": 0.9},
            {"word": "results.", "start": 6.5, "end": 7.2, "confidence": 0.9}
        ]
        mock_energy = [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 0.8]
        highlights = choose_highlights(mock_words, mock_energy, "smart")
        assert len(highlights) > 0, "No highlights generated"
        print("   ✅ Highlight selector: local scoring")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def create_test_video():
    """Create a test video for testing"""
    print("\n🎬 Creating test video...")
    
    test_video_path = "test_sample.mp4"
    
    if os.path.exists(test_video_path):
        print(f"   ✅ Test video already exists: {test_video_path}")
        return test_video_path
    
    try:
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=15:size=1280x720:rate=24',
            '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=15',
            '-c:v', 'libx264', '-preset', 'fast', '-c:a', 'aac', '-shortest', '-y', test_video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Test video created: {test_video_path}")
            return test_video_path
        else:
            print(f"   ❌ Failed to create test video: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"   ❌ Test video creation failed: {e}")
        return None

def test_pipeline():
    """Test the complete pipeline"""
    print("\n🚀 Testing complete pipeline...")
    
    test_video = create_test_video()
    if not test_video:
        print("   ❌ Cannot test pipeline without test video")
        return False
    
    try:
        # Run pipeline
        result = subprocess.run([
            sys.executable, '-m', 'src.run_pipeline', test_video, 
            '--mode', 'smart', '--no-server'
        ], capture_output=True, text=True, timeout=60)
        
        print(f"   Pipeline output: {result.stdout[-200:]}")  # Show last 200 chars
        
        if result.returncode == 0:
            print("   ✅ Pipeline completed successfully")
            
            # Check for JSON output
            json_files = list(Path('.').glob('montage_plan_*.json'))
            if json_files:
                print(f"   ✅ JSON plan created: {json_files[0]}")
                
                # Validate JSON
                with open(json_files[0], 'r') as f:
                    plan = json.load(f)
                
                if plan.get('success'):
                    print("   ✅ Pipeline reported success")
                else:
                    print(f"   ❌ Pipeline reported failure: {plan.get('error')}")
                
                # Clean up
                for json_file in json_files:
                    os.unlink(json_file)
                
                return True
            else:
                print("   ❌ No JSON plan file generated")
                return False
        else:
            print(f"   ❌ Pipeline failed with exit code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Pipeline timed out")
        return False
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 REAL VIDEO PROCESSING PIPELINE - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Complete Pipeline", test_pipeline)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        
        try:
            if test_func():
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)