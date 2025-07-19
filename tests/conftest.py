"""Test configuration and fixtures for Montage pipeline tests."""

import os
import pytest
import tempfile
from unittest.mock import patch


@pytest.fixture(autouse=True, scope="function")
def patch_database_for_tests(monkeypatch):
    """Patch DATABASE_URL to use in-memory SQLite for tests."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("MAX_COST_USD", "1.00")
    monkeypatch.setenv("DEBUG", "true")


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Mock API keys for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture
def sample_video_path():
    """Path to test video file."""
    test_data_dir = os.path.join(os.path.dirname(__file__), "data")
    video_path = os.path.join(test_data_dir, "test_lecture.mp4")
    
    if not os.path.exists(video_path):
        # Create a dummy video file if it doesn't exist
        test_video = os.path.join(os.path.dirname(__file__), "..", "test_video.mp4")
        if os.path.exists(test_video):
            return test_video
    
    return video_path