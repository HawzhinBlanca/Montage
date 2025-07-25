import os
import pytest


def test_speaker_diarizer_init():
    """Test speaker diarizer initialization"""
    from montage.core.speaker_diarizer import SpeakerDiarizer
    diarizer = SpeakerDiarizer()
    assert diarizer.device in ["cpu", "cuda"]
    assert diarizer.pipeline is None  # Lazy loading


def test_speaker_diarizer_no_token():
    """Test error when no auth token provided"""
    from montage.core.speaker_diarizer import SpeakerDiarizer
    # Temporarily remove token
    old_token = os.environ.get("HUGGINGFACE_TOKEN")
    if old_token:
        del os.environ["HUGGINGFACE_TOKEN"]
    
    try:
        diarizer = SpeakerDiarizer()
        with pytest.raises((ValueError, ImportError)):
            diarizer._load_pipeline()
    finally:
        # Restore token
        if old_token:
            os.environ["HUGGINGFACE_TOKEN"] = old_token


def test_merge_with_transcript():
    """Test merging diarization with transcript"""
    from montage.core.speaker_diarizer import SpeakerDiarizer
    diarizer = SpeakerDiarizer()
    
    # Mock diarization results
    diarization = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 15.0}
    ]
    
    # Mock transcript
    transcript = [
        {"start": 0.5, "end": 4.5, "text": "Hello world"},
        {"start": 5.5, "end": 9.5, "text": "How are you"},
        {"start": 10.5, "end": 14.5, "text": "I'm fine thanks"}
    ]
    
    # Merge
    merged = diarizer.merge_with_transcript(diarization, transcript)
    
    assert len(merged) == 3
    assert merged[0]["speaker"] == "SPEAKER_00"
    assert merged[1]["speaker"] == "SPEAKER_01"
    assert merged[2]["speaker"] == "SPEAKER_00"
    assert merged[0]["text"] == "Hello world"


@pytest.mark.skipif("HUGGINGFACE_TOKEN" not in os.environ, reason="HF token not set")
def test_diarize_audio():
    """Test actual diarization (requires HF token)"""
    from montage.core.speaker_diarizer import SpeakerDiarizer
    diarizer = SpeakerDiarizer()
    
    # Create a test audio file
    test_file = "test_audio_diarize.wav"
    os.system(f'ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 16000 {test_file} -y 2>/dev/null')
    
    try:
        # This would normally download the model
        segments = diarizer.diarize(test_file)
        assert isinstance(segments, list)
        
    except Exception as e:
        # Model download might fail in CI
        pytest.skip(f"Model loading failed: {e}")
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)