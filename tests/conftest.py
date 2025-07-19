"""Test configuration and fixtures for Montage pipeline tests."""

import os
import sys
import pytest
import tempfile
import pathlib
import psycopg2
from testcontainers.postgres import PostgresContainer

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def pg_url():
    """Provide a real PostgreSQL container for tests."""
    with PostgresContainer("postgres:15") as pg:
        # Load schema
        conn = psycopg2.connect(pg.get_connection_url())
        schema_path = pathlib.Path(__file__).parent.parent / "schema.sql"
        conn.cursor().execute(schema_path.read_text())
        conn.commit()
        conn.close()
        yield pg.get_connection_url()


@pytest.fixture(autouse=True, scope="function")
def _patch_env(pg_url, monkeypatch):
    """Set default environment variables for all tests with real Postgres."""
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("MAX_COST_USD", "0.01")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SENTRY_DSN", "")  # Disable Sentry in tests
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # Reduce log noise in tests
    monkeypatch.setenv("ENVIRONMENT", "test")


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
