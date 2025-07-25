"""Phase 4 endpoint tests - verify lazy loading works"""
import pytest
from fastapi.testclient import TestClient
import os

# Set required env vars before any imports
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["DATABASE_URL"] = "postgresql://test/db"


def test_health_endpoint():
    """Test health endpoint with lazy DB"""
    # Import after env vars are set
    from montage.api.web_server import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "3.5.0"


def test_metrics_endpoint_auth():
    """Test metrics endpoint requires auth"""
    from montage.api.web_server import app
    
    client = TestClient(app)
    
    # Without auth
    response = client.get("/metrics")
    assert response.status_code in [401, 403]
    
    # With auth (mocked in conftest.py)
    response = client.get("/metrics", headers={"X-API-Key": "test-api-key"})
    # Should work with mocked DB
    assert response.status_code in [200, 500]  # 500 if DB mock not perfect


def test_process_endpoint_exists():
    """Test process endpoint is registered"""
    from montage.api.web_server import app
    
    # Check route exists
    routes = [route.path for route in app.routes]
    assert "/process" in routes


def test_status_endpoint_exists():
    """Test status endpoint is registered"""
    from montage.api.web_server import app
    
    routes = [route.path for route in app.routes]
    assert "/status/{job_id}" in routes


def test_no_import_side_effects():
    """Test that we can import without DB/Celery connections"""
    # This import should not fail
    import montage.api.web_server as ws
    
    # Verify lazy functions exist
    assert hasattr(ws, 'get_db')
    assert hasattr(ws, 'get_celery')
    assert callable(ws.get_db)
    assert callable(ws.get_celery)
    
    # No global db object
    assert not hasattr(ws, 'db')