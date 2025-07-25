import pytest
import io
from fastapi.testclient import TestClient
from fastapi import UploadFile
from montage.api.web_server import app


client = TestClient(app)


class TestProcessEndpoint:
    """Test /process endpoint"""
    
    def test_process_video_success(self):
        """Test successful video processing request"""
        # Create a fake video file
        video_content = b"fake video content"
        video_file = io.BytesIO(video_content)
        
        response = client.post(
            "/process",
            files={"file": ("test.mp4", video_file, "video/mp4")},
            data={"mode": "smart", "vertical": "false"},
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"
        assert "created_at" in data
    
    def test_process_requires_auth(self):
        """Test process endpoint requires authentication"""
        video_file = io.BytesIO(b"fake video")
        
        response = client.post(
            "/process",
            files={"file": ("test.mp4", video_file, "video/mp4")}
        )
        
        assert response.status_code == 403
    
    def test_process_invalid_mode(self):
        """Test process endpoint with invalid mode"""
        video_file = io.BytesIO(b"fake video")
        
        response = client.post(
            "/process",
            files={"file": ("test.mp4", video_file, "video/mp4")},
            data={"mode": "invalid_mode"},
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 422


class TestStatusEndpoint:
    """Test /status endpoint"""
    
    def test_get_job_status_success(self):
        """Test successful job status retrieval"""
        response = client.get(
            "/status/test-job-id",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "completed"
    
    def test_get_job_status_not_found(self, monkeypatch):
        """Test job status when job doesn't exist"""
        def mock_db():
            class _DB:
                def find_one(self, *args):
                    return None
            return _DB()
        
        monkeypatch.setattr("montage.api.web_server.get_db", mock_db)
        
        response = client.get(
            "/status/nonexistent-job",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 404
    
    def test_status_requires_auth(self):
        """Test status endpoint requires authentication"""
        response = client.get("/status/test-job-id")
        assert response.status_code == 403


class TestDownloadEndpoint:
    """Test /download endpoint"""
    
    def test_download_success(self):
        """Test successful file download"""
        response = client.get(
            "/download/test-job-id",
            headers={"X-API-Key": "test-api-key"}
        )
        
        # Should return file response
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
    
    def test_download_job_not_found(self, monkeypatch):
        """Test download when job doesn't exist"""
        def mock_db():
            class _DB:
                def find_one(self, *args):
                    return None
            return _DB()
        
        monkeypatch.setattr("montage.api.web_server.get_db", mock_db)
        
        response = client.get(
            "/download/nonexistent-job",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 404
    
    def test_download_requires_auth(self):
        """Test download endpoint requires authentication"""
        response = client.get("/download/test-job-id")
        assert response.status_code == 403


class TestSecretValidation:
    """Test secret validation endpoint"""
    
    def test_validate_secrets_endpoint(self):
        """Test secrets validation endpoint"""
        response = client.get(
            "/validate-secrets",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "validation_results" in data
        assert "sources" in data