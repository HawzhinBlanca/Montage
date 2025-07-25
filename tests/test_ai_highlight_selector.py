import os
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock


def test_ai_highlight_init():
    """Test AI highlight selector initialization"""
    from montage.core.ai_highlight_selector import AIHighlightSelector
    
    # Mock the API clients
    with patch('montage.core.ai_highlight_selector.anthropic.Anthropic') as mock_anthropic:
        with patch('montage.core.ai_highlight_selector.genai.configure') as mock_genai:
            # Set env vars
            os.environ["ANTHROPIC_API_KEY"] = "test-key"
            os.environ["GEMINI_API_KEY"] = "test-key"
            
            selector = AIHighlightSelector()
            
            # Check initialization
            assert selector.claude_client is not None
            assert selector.gemini_model is not None


def test_story_beat_detector():
    """Test story beat detection"""
    from montage.core.ai_highlight_selector import StoryBeatDetector
    
    detector = StoryBeatDetector()
    
    # Test transcript
    transcript = {
        "segments": [
            {"text": "Let me tell you about something amazing", "start": 0, "end": 5},
            {"text": "It started when I was young", "start": 5, "end": 10},
            {"text": "Then everything changed suddenly", "start": 10, "end": 15},
            {"text": "In the end, I learned a valuable lesson", "start": 15, "end": 20}
        ]
    }
    
    beats = detector.detect_beats(transcript)
    
    assert len(beats) == 4
    assert beats[0][0] == "hook"
    assert beats[1][0] == "context"
    assert beats[2][0] == "climax"
    assert beats[3][0] == "resolution"


def test_highlight_segment_dataclass():
    """Test HighlightSegment dataclass"""
    from montage.core.ai_highlight_selector import HighlightSegment
    
    segment = HighlightSegment(
        start=10.0,
        end=40.0,
        score=0.9,
        title="Amazing moment",
        reason="High engagement",
        story_beat="climax",
        emotions=["joy", "surprise"],
        keywords=["amazing", "moment"]
    )
    
    assert segment.start == 10.0
    assert segment.end == 40.0
    assert segment.score == 0.9
    assert segment.story_beat == "climax"
    assert "joy" in segment.emotions


@pytest.mark.asyncio
async def test_ai_selection_mock():
    """Test AI highlight selection with mocked responses"""
    from montage.core.ai_highlight_selector import AIHighlightSelector
    
    # Mock the API clients
    with patch('montage.core.ai_highlight_selector.anthropic.Anthropic') as mock_anthropic:
        with patch('montage.core.ai_highlight_selector.genai') as mock_genai:
            # Set env vars
            os.environ["ANTHROPIC_API_KEY"] = "test-key"
            os.environ["GEMINI_API_KEY"] = "test-key"
            
            # Mock Claude response
            mock_claude_response = MagicMock()
            mock_claude_response.content = [MagicMock(text='{"highlights": [{"start": 10, "end": 30, "title": "Test", "viral_potential": 0.8}]}')]
            mock_anthropic.return_value.messages.create.return_value = mock_claude_response
            
            # Mock Gemini response
            mock_gemini_response = MagicMock()
            mock_gemini_response.text = '{"segments": [{"start_time": 40, "end_time": 60, "title": "Test2", "viral_score": 0.7}]}'
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_gemini_response
            
            selector = AIHighlightSelector()
            
            # Test transcript
            transcript = {
                "segments": [
                    {"text": "This is a test", "start": 0, "end": 5},
                    {"text": "Another segment", "start": 5, "end": 10}
                ]
            }
            
            highlights = await selector.select_highlights(transcript, 120.0, 2)
            
            assert len(highlights) > 0
            assert highlights[0].start >= 0


def test_keyword_extraction():
    """Test keyword extraction from text"""
    from montage.core.ai_highlight_selector import AIHighlightSelector
    
    selector = AIHighlightSelector()
    keywords = selector._extract_keywords("This is an amazing test with important keywords and wonderful phrases")
    
    assert "amazing" in keywords
    assert "important" in keywords
    assert len(keywords) <= 5  # Should return top 5
    assert all(len(k) > 3 for k in keywords)  # All keywords should be > 3 chars
    assert "the" not in keywords  # Stopword should be filtered