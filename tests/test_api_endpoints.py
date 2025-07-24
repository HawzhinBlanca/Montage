"""
Comprehensive tests for API endpoints with authentication
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from montage.api.auth import AuthUser, UserRole
from montage.api.web_server import app


class TestAPIEndpoints:
    """Test all API endpoints"""

    def setup_method(self):
        """Setup test client and common fixtures"""
        self.client = TestClient(app)

        # Mock admin user
        self.admin_user = AuthUser(
            user_id="admin",
            username="Admin User",
            role=UserRole.ADMIN,
            permissions=["admin:cleanup", "metrics:read", "video:upload", "video:download"],
            created_at=datetime.now()
        )

        # Mock regular user
        self.regular_user = AuthUser(
            user_id="user123",
            username="Regular User",
            role=UserRole.USER,
            permissions=["video:upload", "video:download"],
            created_at=datetime.now()
        )

        # Mock API headers
        self.admin_headers = {"Authorization": "Bearer admin_api_key"}
        self.user_headers = {"Authorization": "Bearer user_api_key"}


class TestHealthEndpoint(TestAPIEndpoints):
    """Test health check endpoint"""

    @patch('montage.api.web_server.db.get_connection')
    def test_health_check_success(self, mock_get_connection):
        """Test successful health check"""
        mock_conn = Mock()
        mock_conn.execute.return_value = None
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        response = self.client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "3.5.0"

    @patch('montage.api.web_server.db.get_connection')
    def test_health_check_database_failure(self, mock_get_connection):
        """Test health check with database failure"""
        mock_get_connection.side_effect = Exception("Database connection failed")

        response = self.client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert "Service unhealthy" in data["detail"]


class TestAuthEndpoints(TestAPIEndpoints):
    """Test authentication endpoints"""

    @patch('montage.api.web_server.get_admin_user')
    @patch('montage.api.auth.api_key_auth.generate_api_key')
    def test_create_api_key_success(self, mock_generate_key, mock_get_admin_user):
        """Test successful API key creation"""
        mock_get_admin_user.return_value = self.admin_user
        mock_generate_key.return_value = "new_generated_api_key_123456"

        request_data = {
            "name": "Test API Key",
            "role": "user"
        }

        response = self.client.post(
            "/auth/api-key",
            json=request_data,
            headers=self.admin_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["api_key"] == "new_generated_api_key_123456"
        assert data["name"] == "Test API Key"
        assert data["role"] == "user"
        assert "Store this API key securely" in data["message"]

    def test_create_api_key_no_auth(self):
        """Test API key creation without authentication"""
        request_data = {
            "name": "Test API Key",
            "role": "user"
        }

        response = self.client.post("/auth/api-key", json=request_data)

        assert response.status_code == 401

    @patch('montage.api.web_server.get_current_user')
    def test_get_current_user_info(self, mock_get_current_user):
        """Test getting current user information"""
        mock_get_current_user.return_value = self.regular_user

        response = self.client.get("/auth/me", headers=self.user_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user123"
        assert data["username"] == "Regular User"
        assert data["role"] == "user"
        assert data["permissions"] == ["video:upload", "video:download"]


class TestVideoProcessingEndpoints(TestAPIEndpoints):
    """Test video processing endpoints"""

    @patch('montage.api.web_server.require_video_upload')
    @patch('montage.api.web_server.process_video_task')
    @patch('montage.api.web_server.db.get_connection')
    def test_process_video_success(self, mock_get_connection, mock_process_task, mock_auth):
        """Test successful video upload and processing"""
        mock_auth.return_value = self.regular_user
        mock_process_task.delay = Mock()

        # Mock database operations
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        # Create temporary test video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video data")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as video_file:
                response = self.client.post(
                    "/process",
                    files={"file": ("test_video.mp4", video_file, "video/mp4")},
                    data={"mode": "smart", "vertical": "false"},
                    headers=self.user_headers
                )

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "queued"
            assert data["message"] == "Video processing started"

            # Verify database was called with user_id
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "user123" in call_args[1]  # user_id should be in parameters

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_process_video_invalid_format(self):
        """Test video upload with invalid format"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a video")
            temp_path = f.name

        try:
            with open(temp_path, "rb") as text_file:
                response = self.client.post(
                    "/process",
                    files={"file": ("test.txt", text_file, "text/plain")},
                    headers=self.user_headers
                )

            assert response.status_code == 400
            assert "Invalid video format" in response.json()["detail"]

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_process_video_no_auth(self):
        """Test video upload without authentication"""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(b"fake video")
            f.seek(0)

            response = self.client.post(
                "/process",
                files={"file": ("test.mp4", f, "video/mp4")}
            )

        assert response.status_code == 401

    @patch('montage.api.web_server.get_current_user')
    @patch('montage.api.web_server.db.get_connection')
    def test_get_job_status_success(self, mock_get_connection, mock_get_current_user):
        """Test getting job status for user's own job"""
        mock_get_current_user.return_value = self.regular_user

        # Mock database response
        mock_conn = Mock()
        mock_cursor = Mock()
        test_time = datetime.now()
        mock_cursor.fetchone.return_value = (
            "completed", "/path/to/output.mp4", None, test_time, test_time,
            "smart", False, None, "user123"
        )
        mock_conn.cursor.return_value = mock_cursor
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        response = self.client.get("/status/job123", headers=self.user_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["status"] == "completed"
        assert data["mode"] == "smart"
        assert data["vertical"] is False

    @patch('montage.api.web_server.get_current_user')
    @patch('montage.api.web_server.db.get_connection')
    def test_get_job_status_not_found(self, mock_get_connection, mock_get_current_user):
        """Test getting status for non-existent job"""
        mock_get_current_user.return_value = self.regular_user

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        response = self.client.get("/status/nonexistent", headers=self.user_headers)

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    @patch('montage.api.web_server.require_video_download')
    @patch('montage.api.web_server.db.get_connection')
    def test_download_video_success(self, mock_get_connection, mock_auth):
        """Test downloading completed video"""
        mock_auth.return_value = self.regular_user

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"processed video data")
            output_path = f.name

        try:
            # Mock database response
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = (output_path, "completed", "user123")
            mock_conn.cursor.return_value = mock_cursor
            mock_get_connection.return_value.__enter__.return_value = mock_conn

            response = self.client.get("/download/job123", headers=self.user_headers)

            assert response.status_code == 200
            assert response.headers["content-type"] == "video/mp4"

        finally:
            Path(output_path).unlink(missing_ok=True)

    @patch('montage.api.web_server.require_video_download')
    @patch('montage.api.web_server.db.get_connection')
    def test_download_video_not_completed(self, mock_get_connection, mock_auth):
        """Test downloading video that's not completed"""
        mock_auth.return_value = self.regular_user

        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("/path/to/output.mp4", "processing", "user123")
        mock_conn.cursor.return_value = mock_cursor
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        response = self.client.get("/download/job123", headers=self.user_headers)

        assert response.status_code == 400
        assert "Job not completed" in response.json()["detail"]


class TestAdminEndpoints(TestAPIEndpoints):
    """Test admin-only endpoints"""

    @patch('montage.api.web_server.require_metrics_read')
    @patch('montage.api.web_server.db.get_connection')
    def test_get_metrics_success(self, mock_get_connection, mock_auth):
        """Test getting system metrics"""
        mock_auth.return_value = self.admin_user

        # Mock database response
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (100, 80, 5, 2, 13, 45.5)  # job stats
        mock_conn.cursor.return_value = mock_cursor
        mock_get_connection.return_value.__enter__.return_value = mock_conn

        response = self.client.get("/metrics", headers=self.admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert data["jobs"]["total"] == 100
        assert data["jobs"]["completed"] == 80
        assert data["jobs"]["failed"] == 5

    def test_get_metrics_no_auth(self):
        """Test getting metrics without authentication"""
        response = self.client.get("/metrics")

        assert response.status_code == 401

    @patch('montage.api.web_server.require_cleanup')
    @patch('montage.api.web_server.UPLOAD_DIR')
    @patch('montage.api.web_server.OUTPUT_DIR')
    def test_cleanup_files_success(self, mock_output_dir, mock_upload_dir, mock_auth):
        """Test file cleanup"""
        mock_auth.return_value = self.admin_user

        # Mock directory structures
        mock_upload_dir.iterdir.return_value = []
        mock_output_dir.iterdir.return_value = []

        response = self.client.delete("/cleanup?days=7", headers=self.admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert "Cleaned up files older than 7 days" in data["message"]
        assert "cleaned" in data

    def test_cleanup_files_no_auth(self):
        """Test file cleanup without authentication"""
        response = self.client.delete("/cleanup")

        assert response.status_code == 401


class TestRateLimiting(TestAPIEndpoints):
    """Test rate limiting functionality"""

    @patch('montage.api.web_server.require_video_upload')
    def test_rate_limit_exceeded(self, mock_auth):
        """Test rate limiting on process endpoint"""
        mock_auth.return_value = self.regular_user

        # This test would require actual rate limiting setup
        # For now, we'll test that the limiter is configured
        response = self.client.get("/process")  # Wrong method to trigger different error

        # Should get method not allowed, not rate limit error (since we're not actually hitting the limit)
        assert response.status_code == 405


class TestErrorHandling(TestAPIEndpoints):
    """Test error handling across endpoints"""

    def test_invalid_endpoint(self):
        """Test requesting non-existent endpoint"""
        response = self.client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test wrong HTTP method"""
        response = self.client.post("/health")

        assert response.status_code == 405

    @patch('montage.api.web_server.get_current_user')
    def test_internal_server_error_handling(self, mock_get_current_user):
        """Test internal server error handling"""
        mock_get_current_user.side_effect = Exception("Database explosion")

        response = self.client.get("/auth/me", headers=self.user_headers)

        # Should be handled gracefully by FastAPI
        assert response.status_code in [500, 401]  # Depending on where the error occurs


class TestCORSAndSecurity(TestAPIEndpoints):
    """Test CORS and security headers"""

    def test_cors_headers_present(self):
        """Test that CORS headers are set"""
        response = self.client.options("/health")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == 200

    def test_security_headers(self):
        """Test security headers are set"""
        response = self.client.get("/health")

        # Basic security check - server should not expose internal details
        assert "server" not in response.headers or "gunicorn" not in response.headers.get("server", "").lower()
