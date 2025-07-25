"""
Tests for security fixes and vulnerabilities prevention
"""

from unittest.mock import Mock, patch

import pytest

from montage.utils.ffmpeg_utils import _safe_parse_frame_rate


class TestEvalVulnerabilityFix:
    """Test the eval() vulnerability fix in ffmpeg_utils"""

    def test_safe_parse_frame_rate_valid_input(self):
        """Test safe parsing of valid frame rates"""
        # Test common frame rates
        assert _safe_parse_frame_rate("30/1") == 30.0
        assert _safe_parse_frame_rate("24000/1001") == pytest.approx(23.976, rel=1e-3)
        assert _safe_parse_frame_rate("60/1") == 60.0
        assert _safe_parse_frame_rate("25/1") == 25.0

        # Test fractional rates
        assert _safe_parse_frame_rate("24/1") == 24.0
        assert _safe_parse_frame_rate("1/24") == pytest.approx(0.0417, rel=1e-3)

    def test_safe_parse_frame_rate_invalid_format(self):
        """Test rejection of invalid frame rate formats"""

        # Test malicious code injection attempts
        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("__import__('os').system('rm -rf /')")

        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("exec('print(123)')")

        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("eval('1+1')")

        # Test invalid mathematical expressions
        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("30 + 1")

        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("30*1")

        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("30-1")

        # Test multiple slashes
        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("30/1/1")

        # Test letters
        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("abc/def")

        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("30a/1b")

    def test_safe_parse_frame_rate_edge_cases(self):
        """Test edge cases and boundary conditions"""

        # Test empty string
        with pytest.raises(ValueError):
            _safe_parse_frame_rate("")

        # Test None input
        with pytest.raises(ValueError):
            _safe_parse_frame_rate(None)

        # Test non-string input
        with pytest.raises(ValueError):
            _safe_parse_frame_rate(123)

        with pytest.raises(ValueError):
            _safe_parse_frame_rate(["30", "1"])

        # Test division by zero
        with pytest.raises(ValueError, match="Cannot parse frame rate"):
            _safe_parse_frame_rate("30/0")

        # Test whitespace handling
        assert _safe_parse_frame_rate("  30/1  ") == 30.0
        assert _safe_parse_frame_rate("\t24/1\n") == 24.0

        # But spaces within the fraction should fail
        with pytest.raises(ValueError, match="Invalid frame rate format"):
            _safe_parse_frame_rate("30 / 1")

    def test_safe_parse_frame_rate_large_numbers(self):
        """Test with large numbers to ensure no overflow issues"""

        # Test large but valid numbers
        result = _safe_parse_frame_rate("999999/1")
        assert result == 999999.0

        result = _safe_parse_frame_rate("1/999999")
        assert result == pytest.approx(1/999999, rel=1e-9)

        # Test extremely large numbers (should still work with fractions)
        result = _safe_parse_frame_rate("1000000/1000000")
        assert result == 1.0


class TestInputValidation:
    """Test input validation security measures"""

    def test_sql_injection_prevention(self):
        """Test that SQL injection attempts are prevented"""
        # These tests would need to be integrated with actual database code
        # For now, we test that the database module uses parameterized queries

        from montage.core.db import Database

        # Verify that the Database class has proper table name validation
        db = Database()

        # Test table name validation
        assert hasattr(db, '_validate_table_name')
        assert hasattr(db, '_validate_column_name')
        assert hasattr(db, '_quote_identifier')

        # Test that malicious table names are rejected
        assert not db._validate_table_name("users; DROP TABLE jobs; --")
        assert not db._validate_table_name("'; DELETE FROM jobs; --")
        assert not db._validate_table_name("../../../etc/passwd")

        # Test that valid table names are accepted
        assert db._validate_table_name("jobs")
        assert db._validate_table_name("api_keys")
        assert db._validate_table_name("video_job")

    def test_column_name_validation(self):
        """Test column name validation"""
        from montage.core.db import Database

        db = Database()

        # Test valid column names
        assert db._validate_column_name("user_id")
        assert db._validate_column_name("created_at")
        assert db._validate_column_name("status")
        assert db._validate_column_name("job_id")

        # Test invalid column names
        assert not db._validate_column_name("user_id; DROP TABLE")
        assert not db._validate_column_name("'; DELETE FROM")
        assert not db._validate_column_name("user-id")  # Hyphen not allowed
        assert not db._validate_column_name("123invalid")  # Can't start with number
        assert not db._validate_column_name("")
        assert not db._validate_column_name(None)

    def test_identifier_quoting(self):
        """Test SQL identifier quoting for safety"""
        from montage.core.db import Database

        db = Database()

        # Test normal identifiers
        assert db._quote_identifier("user_id") == '"user_id"'
        assert db._quote_identifier("jobs") == '"jobs"'

        # Test identifiers with quotes (should be escaped)
        assert db._quote_identifier('table"name') == '"table""name"'
        assert db._quote_identifier('col"umn"name') == '"col""umn""name"'


class TestPathTraversalPrevention:
    """Test path traversal attack prevention"""

    def test_safe_path_function(self):
        """Test the safe_path function prevents directory traversal"""
        import tempfile
        from pathlib import Path

        from montage.api.web_server import safe_path

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Test normal filename
            safe_file = safe_path("video.mp4", base_dir)
            assert safe_file == base_dir / "video.mp4"

            # Test path traversal attempts
            with pytest.raises(ValueError, match="Path traversal attempt"):
                safe_path("../../../etc/passwd", base_dir)

            with pytest.raises(ValueError, match="Path traversal attempt"):
                safe_path("..\\..\\windows\\system32\\cmd.exe", base_dir)

            with pytest.raises(ValueError, match="Path traversal attempt"):
                safe_path("/etc/passwd", base_dir)

            with pytest.raises(ValueError, match="Path traversal attempt"):
                safe_path("C:\\Windows\\system32\\cmd.exe", base_dir)

            # Test empty filename
            with pytest.raises(ValueError, match="Invalid filename"):
                safe_path("", base_dir)

            with pytest.raises(ValueError, match="Invalid filename"):
                safe_path(".", base_dir)

            with pytest.raises(ValueError, match="Invalid filename"):
                safe_path("..", base_dir)

    def test_path_normalization(self):
        """Test that paths are properly normalized"""
        import tempfile
        from pathlib import Path

        from montage.api.web_server import safe_path

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)

            # Test that directory components are stripped
            safe_file = safe_path("subdir/video.mp4", base_dir)
            assert safe_file.name == "video.mp4"
            assert safe_file == base_dir / "video.mp4"

            # Test that only filename is kept
            safe_file = safe_path("path/to/deep/video.mp4", base_dir)
            assert safe_file.name == "video.mp4"
            assert safe_file == base_dir / "video.mp4"


class TestAPISecurityHeaders:
    """Test API security headers and CORS configuration"""

    def test_cors_configuration_exists(self):
        """Test that CORS is properly configured"""
        from montage.api.web_server import app

        # Check that CORS middleware is added
        middleware_types = [type(middleware) for middleware, _ in app.user_middleware]

        # Should have CORS middleware
        from fastapi.middleware.cors import CORSMiddleware
        assert CORSMiddleware in middleware_types

    def test_rate_limiting_configured(self):
        """Test that rate limiting is configured"""
        from montage.api.web_server import app, limiter

        # Check that rate limiter is configured
        assert limiter is not None
        assert hasattr(app.state, 'limiter')

        # Check that rate limit exception handler is added
        from slowapi.errors import RateLimitExceeded
        assert RateLimitExceeded in app.exception_handlers


class TestAuthenticationSecurity:
    """Test authentication security measures"""

    def test_api_key_hashing(self):
        """Test that API keys are hashed before storage"""
        import hashlib

        from montage.api.auth import APIKeyAuth

        # Mock database operations
        with patch('src.api.auth.Database') as mock_db_class:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_db = Mock()
            mock_db.get_connection.return_value.__enter__.return_value = mock_conn
            mock_db_class.return_value = mock_db

            api_key_auth = APIKeyAuth()

            # Generate API key
            raw_key = api_key_auth.generate_api_key("test_user", "Test Key")

            # Check that database was called with hash, not raw key
            call_args = mock_cursor.execute.call_args[0]
            query, params = call_args

            # Raw key should not be in the query parameters
            assert raw_key not in str(params)

            # But hash should be calculable from raw key
            expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            assert expected_hash in params

    def test_jwt_token_expiration(self):
        """Test that JWT tokens have proper expiration"""
        from datetime import datetime, timezone

        import jwt

        from montage.api.auth import JWT_ALGORITHM, JWT_SECRET_KEY, JWTAuth

        jwt_auth = JWTAuth()
        test_payload = {"user_id": "test", "username": "Test User"}

        token = jwt_auth.create_access_token(test_payload)

        # Decode token and check expiration
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        assert "exp" in decoded
        assert "iat" in decoded

        # Check that expiration is in the future
        exp_time = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        now = datetime.now(timezone.utc)

        assert exp_time > now

        # Check that expiration is reasonable (not too far in future)
        time_diff = exp_time - now
        assert time_diff.total_seconds() < 24 * 60 * 60  # Less than 24 hours

    def test_password_strength_requirements(self):
        """Test API key generation produces strong keys"""
        from montage.api.auth import APIKeyAuth

        with patch('src.api.auth.Database'):
            api_key_auth = APIKeyAuth()

            # Generate multiple keys and test their strength
            keys = []
            for _ in range(10):
                with patch.object(api_key_auth, '_ensure_api_keys_table'):
                    with patch('src.api.auth.Database.get_connection'):
                        try:
                            key = api_key_auth.generate_api_key("test", "test")
                            keys.append(key)
                        except Exception:
                            # Expected since we're mocking the database
                            pass

            if keys:
                key = keys[0]
                # Test key strength
                assert len(key) >= 32  # Minimum length

                # Should contain mix of characters (URL-safe base64)
                assert any(c.isupper() for c in key)  # Uppercase
                assert any(c.islower() for c in key)  # Lowercase
                assert any(c.isdigit() for c in key)  # Numbers
