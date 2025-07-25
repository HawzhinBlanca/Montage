import os
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock

@pytest.fixture(autouse=True, scope="session")
def _bootstrap_env():
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret")


@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """Mock database to avoid import-time connections"""
    class _FakeDB:
        def execute(self, *args, **kwargs):
            return 1
        
        def insert(self, *args, **kwargs):
            return {"job_id": "test-job-id"}
        
        def find_one(self, collection, query):
            if "job_id" in query:
                return {
                    "job_id": query["job_id"],
                    "status": "completed",
                    "created_at": "2025-01-01T00:00:00Z",
                    "completed_at": "2025-01-01T00:01:00Z",
                    "output_path": "/outputs/test.mp4",
                    "metadata": {}
                }
            return None
        
        def get_connection(self):
            return self
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def cursor(self):
            return self
        
        def execute(self, query):
            return self
        
        def fetchall(self):
            return [(100, 200, 300)]
    
    fake_db = _FakeDB()
    monkeypatch.setattr("montage.api.web_server.get_db", lambda: fake_db)


@pytest.fixture(autouse=True)
def mock_celery(monkeypatch):
    """Mock Celery to avoid import-time broker connections"""
    class _FakeCeleryTask:
        def delay(self, *args, **kwargs):
            class _AsyncResult:
                id = "test-task-id"
            return _AsyncResult()
        
        def apply_async(self, *args, **kwargs):
            class _AsyncResult:
                id = "test-task-id"
            return _AsyncResult()
    
    fake_task = _FakeCeleryTask()
    monkeypatch.setattr("montage.api.web_server.get_celery", lambda: fake_task)


@pytest.fixture(autouse=True)
def mock_auth(monkeypatch):
    """Mock authentication to avoid secrets loading"""
    async def mock_require_api_key(*args, **kwargs):
        return "test-api-key"
    
    async def mock_verify_key(request, x_api_key=None):
        if request.url.path == "/health":
            return None
        if x_api_key == "test-api-key":
            return x_api_key
        return None
    
    monkeypatch.setattr("montage.api.auth.require_api_key", mock_require_api_key)
    monkeypatch.setattr("montage.api.auth.validate_api_key", lambda x: x == "test-api-key")
    monkeypatch.setattr("montage.api.auth._TASK_ALLOWED_KEYS", {"test-api-key"})
    monkeypatch.setattr("montage.api.auth.verify_key", mock_verify_key)


@pytest.fixture(autouse=True)
def mock_upload_validator(monkeypatch):
    """Mock upload validator to avoid file system operations"""
    class _FakeValidator:
        class limits:
            MAX_FILE_SIZE_BYTES = 1024 * 1024 * 100  # 100MB
        
        async def validate_upload(self, file, api_key, temp_path):
            return file.filename, "video/mp4"
        
        def get_upload_stats(self, api_key):
            return {
                "uploads_last_hour": 5,
                "uploads_last_day": 20,
                "hourly_limit": 10,
                "daily_limit": 50
            }
    
    monkeypatch.setattr("montage.api.web_server.upload_validator", _FakeValidator())


@pytest.fixture(autouse=True)
def mock_file_operations(monkeypatch, tmp_path):
    """Mock file operations to use temp directory"""
    monkeypatch.setattr("montage.api.web_server.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("montage.api.web_server.OUTPUT_DIR", tmp_path / "outputs")
    
    # Create directories
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    
    # Create a test output file
    test_output = tmp_path / "outputs" / "test.mp4"
    test_output.write_bytes(b"fake video content")


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Mock settings to avoid import-time configuration loading"""
    class _FakeSettings:
        environment = "test"
        cors_origins = ["http://localhost:3000"]
        api_prefix = "/api/v1"
        
        class features:
            enable_caching = False
    
    monkeypatch.setattr("montage.api.web_server.settings", _FakeSettings())


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()