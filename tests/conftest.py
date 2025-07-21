"""Comprehensive test configuration and fixtures for Montage pipeline tests."""

import os
import sys
import json
import uuid
import pytest
import logging
import tempfile
import pathlib
import shutil
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
from typing import Generator, Dict, Any, Optional

import psycopg2
from psycopg2.pool import SimpleConnectionPool
import redis
import numpy as np
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
from faker import Faker

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules after path setup
from src.core.db import DatabasePool
from src.utils.logging_config import get_logger, correlation_context
from src.core.metrics import metrics
from src.core.checkpoint import CheckpointManager
from src.core.rate_limiter import RateLimiter
from src.utils.memory_manager import get_memory_monitor
from src.utils.resource_manager import get_video_resource_manager

# Disable verbose logging during tests
logging.getLogger().setLevel(logging.CRITICAL)

# Initialize faker for test data
fake = Faker()

# ===== Session-scoped fixtures (shared across all tests) =====


@pytest.fixture(scope="session")
def docker_services():
    """Manage Docker containers for the entire test session."""
    containers = {}

    # Start PostgreSQL container
    pg_container = PostgresContainer("postgres:15-alpine")
    pg_container.start()
    containers["postgres"] = pg_container

    # Start Redis container
    redis_container = RedisContainer("redis:7-alpine")
    redis_container.start()
    containers["redis"] = redis_container

    yield containers

    # Cleanup
    for container in containers.values():
        container.stop()


@pytest.fixture(scope="session")
def pg_url(docker_services):
    """Provide a real PostgreSQL container URL for tests."""
    pg_container = docker_services["postgres"]
    url = pg_container.get_connection_url()

    # Initialize database schema
    conn = psycopg2.connect(url)
    schema_path = pathlib.Path(__file__).parent / "data" / "schema.sql"

    if not schema_path.exists():
        # Use the main schema file
        schema_path = pathlib.Path(__file__).parent.parent / "schema.sql"

    if schema_path.exists():
        with conn.cursor() as cur:
            cur.execute(schema_path.read_text())
        conn.commit()
    conn.close()

    return url


@pytest.fixture(scope="session")
def redis_url(docker_services):
    """Provide a real Redis container URL for tests."""
    redis_container = docker_services["redis"]
    return redis_container.get_connection_url()


@pytest.fixture(scope="session")
def test_videos_dir():
    """Directory containing test video files."""
    return pathlib.Path(__file__).parent / "data"


# ===== Function-scoped fixtures (fresh for each test) =====


@pytest.fixture(autouse=True)
def setup_test_env(pg_url, redis_url, monkeypatch):
    """Set up test environment variables for each test."""
    # Core settings
    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    # API keys (mocked)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "test-pyannote-token")

    # Limits and thresholds
    monkeypatch.setenv("MAX_COST_USD", "0.01")
    monkeypatch.setenv("MEMORY_LIMIT_MB", "512")
    monkeypatch.setenv("MAX_WORKERS", "2")

    # Disable external services
    monkeypatch.setenv("SENTRY_DSN", "")
    monkeypatch.setenv("DISABLE_RATE_LIMITING", "true")


@pytest.fixture
def db_pool(pg_url):
    """Provide a fresh database connection pool."""
    pool = DatabasePool(connection_url=pg_url)
    yield pool
    pool.close()


@pytest.fixture
def db_connection(db_pool):
    """Provide a database connection with automatic cleanup."""
    with db_pool.get_connection() as conn:
        yield conn


@pytest.fixture
def redis_client(redis_url):
    """Provide a Redis client."""
    client = redis.from_url(redis_url)
    yield client
    client.flushdb()  # Clean up after test
    client.close()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's automatically cleaned up."""
    temp_path = tempfile.mkdtemp(prefix="montage_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def correlation_id():
    """Generate a unique correlation ID for the test."""
    return str(uuid.uuid4())


@pytest.fixture
def test_context(correlation_id):
    """Set up test correlation context."""
    with correlation_context(
        correlation_id, user_id="test_user", session_id="test_session"
    ):
        yield correlation_id


# ===== Mock fixtures =====


@pytest.fixture
def mock_whisper_model():
    """Mock faster-whisper model."""
    with patch("faster_whisper.WhisperModel") as mock:
        model_instance = Mock()
        model_instance.transcribe.return_value = (
            [
                Mock(text="This is a test transcript.", start=0.0, end=5.0),
                Mock(text="With multiple segments.", start=5.0, end=8.0),
            ],
            Mock(language="en", duration=8.0),
        )
        mock.return_value = model_instance
        yield model_instance


@pytest.fixture
def mock_deepgram_client():
    """Mock Deepgram client."""
    with patch("deepgram.DeepgramClient") as mock:
        client_instance = Mock()
        response = Mock()
        response.results.channels[0].alternatives[0].transcript = "Deepgram transcript"
        response.results.channels[0].alternatives[0].words = [
            Mock(word="Deepgram", start=0.0, end=0.5),
            Mock(word="transcript", start=0.5, end=1.0),
        ]
        client_instance.listen.prerecorded.v.return_value = response
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    with patch("anthropic.Anthropic") as mock:
        client_instance = Mock()
        response = Mock()
        response.content = [Mock(text="Mocked Claude response")]
        response.usage = Mock(input_tokens=100, output_tokens=50)
        client_instance.messages.create.return_value = response
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    with patch("openai.OpenAI") as mock:
        client_instance = Mock()
        response = Mock()
        response.choices = [Mock(message=Mock(content="Mocked GPT response"))]
        response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        client_instance.chat.completions.create.return_value = response
        mock.return_value = client_instance
        yield client_instance


# ===== Test data fixtures =====


@pytest.fixture
def sample_video_path(test_videos_dir):
    """Path to a sample test video."""
    video_path = test_videos_dir / "short.mp4"
    if not video_path.exists():
        # Fallback to minimal video
        video_path = test_videos_dir / "minimal.mp4"
    return str(video_path)


@pytest.fixture
def sample_transcript():
    """Sample transcript data."""
    return {
        "segments": [
            {
                "text": "Welcome to the lecture.",
                "start": 0.0,
                "end": 2.5,
                "speaker": "SPEAKER_00",
            },
            {
                "text": "Today we'll discuss testing.",
                "start": 2.5,
                "end": 5.0,
                "speaker": "SPEAKER_00",
            },
            {
                "text": "Testing is very important.",
                "start": 5.0,
                "end": 7.5,
                "speaker": "SPEAKER_00",
            },
        ],
        "language": "en",
        "duration": 7.5,
    }


@pytest.fixture
def sample_highlights():
    """Sample highlight data."""
    return [
        {
            "slug": "intro-welcome",
            "title": "Welcome Introduction",
            "start_ms": 0,
            "end_ms": 2500,
            "score": 0.85,
        },
        {
            "slug": "testing-importance",
            "title": "Testing Importance",
            "start_ms": 5000,
            "end_ms": 7500,
            "score": 0.92,
        },
    ]


@pytest.fixture
def video_metadata():
    """Sample video metadata."""
    return {
        "duration": 7.5,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "codec": "h264",
        "bitrate": 5000000,
        "size_bytes": 4687500,
    }


# ===== Helper fixtures =====


@pytest.fixture
def create_test_job(db_connection):
    """Factory to create test jobs in the database."""

    def _create_job(**kwargs):
        job_data = {
            "video_path": kwargs.get("video_path", "/test/video.mp4"),
            "status": kwargs.get("status", "pending"),
            "priority": kwargs.get("priority", 5),
            "metadata": json.dumps(kwargs.get("metadata", {})),
            "created_at": kwargs.get("created_at", datetime.utcnow()),
        }

        with db_connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_job (video_path, status, priority, metadata, created_at)
                VALUES (%(video_path)s, %(status)s, %(priority)s, %(metadata)s, %(created_at)s)
                RETURNING id
            """,
                job_data,
            )
            job_id = cur.fetchone()[0]
            db_connection.commit()

        return job_id

    return _create_job


@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg subprocess calls."""
    with patch("subprocess.Popen") as mock_popen:
        process_mock = Mock()
        process_mock.poll.return_value = None
        process_mock.returncode = 0
        process_mock.communicate.return_value = (b"", b"")
        mock_popen.return_value = process_mock
        yield mock_popen


@pytest.fixture
def rate_limiter(redis_client):
    """Provide a rate limiter instance."""
    return RateLimiter(redis_client=redis_client)


@pytest.fixture
def checkpoint_manager(db_pool, redis_client):
    """Provide a checkpoint manager instance."""
    return CheckpointManager(db_pool=db_pool, redis_client=redis_client)


@pytest.fixture
def memory_monitor():
    """Provide memory monitor for tests."""
    monitor = get_memory_monitor()
    if monitor:
        monitor.start_monitoring()
        yield monitor
        monitor.stop_monitoring()
    else:
        yield None


@pytest.fixture
def resource_manager():
    """Provide resource manager for tests."""
    return get_video_resource_manager()


# ===== Performance and stress test fixtures =====


@pytest.fixture
def large_video_path(test_videos_dir):
    """Path to a large test video for stress testing."""
    return str(test_videos_dir / "long.mp4")


@pytest.fixture
def benchmark_video_set(test_videos_dir):
    """Set of videos for benchmarking."""
    return {
        "minimal": str(test_videos_dir / "minimal.mp4"),
        "short": str(test_videos_dir / "short.mp4"),
        "medium": str(test_videos_dir / "medium.mp4"),
        "long": str(test_videos_dir / "long.mp4"),
        "4k": str(test_videos_dir / "4k.mp4"),
    }


# ===== Cleanup and monitoring =====


@pytest.fixture(autouse=True)
def cleanup_metrics():
    """Reset metrics after each test."""
    yield
    # Reset Prometheus metrics if needed
    try:
        metrics._reset_metrics()
    except:
        pass


@pytest.fixture(autouse=True)
def monitor_test_resources():
    """Monitor resource usage during tests."""
    import psutil

    process = psutil.Process()

    # Record initial state
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_handles = len(process.open_files())

    yield

    # Check for leaks
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    final_handles = len(process.open_files())

    # Warn if significant leak detected
    memory_increase = final_memory - initial_memory
    handle_increase = final_handles - initial_handles

    if memory_increase > 100:  # 100MB threshold
        pytest.warning(f"Possible memory leak: {memory_increase:.1f}MB increase")

    if handle_increase > 10:
        pytest.warning(f"Possible handle leak: {handle_increase} handles not closed")


# ===== Test markers and skips =====


def pytest_configure(config):
    """Register custom markers."""
    markers = [
        "unit: Unit tests (fast, isolated)",
        "integration: Integration tests (may use real services)",
        "e2e: End-to-end tests (full pipeline)",
        "slow: Tests that take > 5 seconds",
        "memory_intensive: Tests requiring significant memory",
        "requires_gpu: Tests requiring GPU acceleration",
        "requires_api_keys: Tests requiring real API keys",
        "stress: Stress/performance tests",
        "flaky: Tests known to be flaky (retry enabled)",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


# ===== Pytest hooks =====


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.nodeid or "stress" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Auto-mark integration tests
        if "integration" in item.nodeid or "test_e2e" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Skip GPU tests if no GPU available
        if "gpu" in item.nodeid:
            try:
                import torch

                if not torch.cuda.is_available():
                    item.add_marker(pytest.mark.skip(reason="GPU not available"))
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))


# ===== Async support =====


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
