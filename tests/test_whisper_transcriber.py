import os
import pytest
from montage.core.real_whisper_transcriber import RealWhisperTranscriber

@pytest.mark.skipif("WHISPER_MODEL_PATH" not in os.environ, reason="model not cached")
def test_whisper_transcription():
    """Test real Whisper transcription"""
    transcriber = RealWhisperTranscriber(model_size="tiny")
    
    # Create a test audio file
    test_file = "test_audio.wav"
    os.system(f'ffmpeg -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 {test_file} -y 2>/dev/null')
    
    try:
        result = transcriber.transcribe(test_file)
        
        assert "segments" in result
        assert "text" in result
        assert "language" in result
        assert "duration" in result
        assert result["duration"] > 0
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)
            
def test_whisper_init():
    """Test Whisper initialization"""
    transcriber = RealWhisperTranscriber()
    assert transcriber.model_size == "base"
    assert transcriber.device in ["cpu", "cuda"]
    
def test_whisper_file_not_found():
    """Test error handling for missing file"""
    transcriber = RealWhisperTranscriber()
    
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("nonexistent.mp4")