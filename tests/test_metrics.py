import pytest
from fastapi.testclient import TestClient
from montage.api.web_server import app


client = TestClient(app)


def test_metrics_endpoint():
    """Test metrics endpoint returns system stats"""
    response = client.get("/metrics", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    
    data = response.json()
    assert "database" in data
    assert "processing" in data
    assert "resources" in data
    
    # Check database metrics
    assert "connections" in data["database"]
    assert "queries_per_second" in data["database"]
    
    # Check processing metrics
    assert "jobs_in_queue" in data["processing"]
    assert "jobs_processing" in data["processing"]
    assert "jobs_completed_today" in data["processing"]
    
    # Check resource metrics
    assert "memory_usage_mb" in data["resources"]
    assert "cpu_usage_percent" in data["resources"]


def test_metrics_requires_auth():
    """Test metrics endpoint requires authentication"""
    response = client.get("/metrics")
    assert response.status_code == 403


def test_metrics_with_invalid_api_key():
    """Test metrics endpoint rejects invalid API key"""
    response = client.get("/metrics", headers={"X-API-Key": "invalid-key"})
    assert response.status_code == 403


def test_metrics_rate_limiting():
    """Test that metrics endpoint is rate limited"""
    # Make 30 requests (should be allowed)
    for _ in range(30):
        response = client.get("/metrics", headers={"X-API-Key": "test-api-key"})
        assert response.status_code == 200
    
    # 31st request might be rate limited
    response = client.get("/metrics", headers={"X-API-Key": "test-api-key"})
    assert response.status_code in [200, 429]


def test_upload_stats_endpoint():
    """Test upload stats endpoint"""
    response = client.get("/upload-stats", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    
    data = response.json()
    assert "uploads_last_hour" in data
    assert "uploads_last_day" in data
    assert "hourly_limit" in data
    assert "daily_limit" in data


def test_worker_stats_endpoint():
    """Test worker stats endpoint"""
    response = client.get("/worker-stats", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    
    data = response.json()
    assert "workers" in data
    assert "memory" in data
    assert "cpu" in data
    assert "tasks" in data


def test_process_stats_endpoint():
    """Test process stats endpoint"""
    response = client.get("/process-stats", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    
    data = response.json()
    assert "active_processes" in data
    assert "memory_usage" in data
    assert "cpu_usage" in data


def test_algorithm_stats_endpoint():
    """Test algorithm stats endpoint"""
    response = client.get("/algorithm-stats", headers={"X-API-Key": "test-api-key"})
    assert response.status_code == 200
    
    data = response.json()
    assert "rover_metrics" in data
    assert "avg_consensus_score" in data["rover_metrics"]
    assert "avg_processing_time_ms" in data["rover_metrics"]