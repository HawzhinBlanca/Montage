"""
Comprehensive tests for API authentication system
"""

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException

from montage.api.auth import (
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    APIKeyAuth,
    AuthToken,
    AuthUser,
    JWTAuth,
    UserRole,
    get_admin_user,
    get_current_user,
    require_permission,
)


class TestAPIKeyAuth:
    """Test API key authentication"""

    def setup_method(self):
        """Setup test fixtures"""
        self.api_key_auth = APIKeyAuth()

    @patch('src.api.auth.Database')
    def test_generate_api_key_success(self, mock_db_class):
        """Test successful API key generation"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db = Mock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db_class.return_value = mock_db

        api_key_auth = APIKeyAuth()

        # Test key generation
        key = api_key_auth.generate_api_key("test_user", "Test Key", UserRole.USER)

        # Verify key format
        assert isinstance(key, str)
        assert len(key) > 20  # Should be reasonably long

        # Verify database calls
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        # Check that key hash was stored, not raw key
        call_args = mock_cursor.execute.call_args[0]
        assert "test_user" in call_args[1]
        assert "Test Key" in call_args[1]
        assert key not in call_args[1]  # Raw key should not be in DB

    @patch('src.api.auth.Database')
    def test_validate_api_key_success(self, mock_db_class):
        """Test successful API key validation"""
        # Mock database response
        mock_conn = Mock()
        mock_cursor = Mock()
        test_time = datetime.now()
        mock_cursor.fetchone.return_value = (
            "test_user", "Test Key", "user", ["video:upload"], test_time, None
        )
        mock_conn.cursor.return_value = mock_cursor
        mock_db = Mock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db_class.return_value = mock_db

        api_key_auth = APIKeyAuth()

        # Test validation
        test_key = "test_api_key_123456789"
        user = api_key_auth.validate_api_key(test_key)

        # Verify result
        assert user is not None
        assert user.user_id == "test_user"
        assert user.username == "Test Key"
        assert user.role == UserRole.USER
        assert user.permissions == ["video:upload"]
        assert user.created_at == test_time

        # Verify hash was used for lookup
        expected_hash = hashlib.sha256(test_key.encode()).hexdigest()
        call_args = mock_cursor.execute.call_args_list[0][0]
        assert expected_hash in call_args[1]

    @patch('src.api.auth.Database')
    def test_validate_api_key_not_found(self, mock_db_class):
        """Test API key validation when key not found"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db = Mock()
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db_class.return_value = mock_db

        api_key_auth = APIKeyAuth()

        user = api_key_auth.validate_api_key("invalid_key")
        assert user is None

    @patch('src.api.auth.Database')
    def test_validate_empty_api_key(self, mock_db_class):
        """Test validation of empty API key"""
        api_key_auth = APIKeyAuth()

        assert api_key_auth.validate_api_key("") is None
        assert api_key_auth.validate_api_key(None) is None


class TestJWTAuth:
    """Test JWT token authentication"""

    def setup_method(self):
        """Setup test fixtures"""
        self.jwt_auth = JWTAuth()
        self.test_payload = {
            "user_id": "test_user",
            "username": "Test User",
            "role": "user",
            "permissions": ["video:upload"]
        }

    def test_create_access_token(self):
        """Test JWT token creation"""
        token = self.jwt_auth.create_access_token(self.test_payload)

        # Verify token format
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # JWT has 3 parts

        # Decode and verify contents
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        assert decoded["user_id"] == "test_user"
        assert decoded["username"] == "Test User"
        assert "exp" in decoded  # Expiration set
        assert "iat" in decoded  # Issued at set

    def test_verify_valid_token(self):
        """Test verification of valid token"""
        token = self.jwt_auth.create_access_token(self.test_payload)

        decoded = self.jwt_auth.verify_token(token)

        assert decoded is not None
        assert decoded["user_id"] == "test_user"
        assert decoded["username"] == "Test User"

    def test_verify_expired_token(self):
        """Test verification of expired token"""
        # Create token with past expiration
        past_payload = self.test_payload.copy()
        past_payload["exp"] = datetime.now(timezone.utc) - timedelta(hours=1)

        expired_token = jwt.encode(past_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        result = self.jwt_auth.verify_token(expired_token)
        assert result is None

    def test_verify_invalid_token(self):
        """Test verification of malformed token"""
        result = self.jwt_auth.verify_token("invalid.token.here")
        assert result is None

        result = self.jwt_auth.verify_token("")
        assert result is None


class TestAuthDependencies:
    """Test FastAPI authentication dependencies"""

    @pytest.mark.asyncio
    async def test_get_current_user_with_api_key_header(self):
        """Test authentication with X-API-Key header"""
        mock_request = Mock()
        mock_request.headers = {"X-API-Key": "test_api_key"}

        test_user = AuthUser(
            user_id="test_user",
            username="Test User",
            role=UserRole.USER,
            permissions=["video:upload"],
            created_at=datetime.now()
        )

        with patch('src.api.auth.api_key_auth.validate_api_key', return_value=test_user):
            result = await get_current_user(mock_request, None)

        assert result == test_user

    @pytest.mark.asyncio
    async def test_get_current_user_with_bearer_jwt(self):
        """Test authentication with Bearer JWT token"""
        mock_request = Mock()
        mock_request.headers = {}

        mock_credentials = Mock()
        mock_credentials.credentials = "jwt_token_here"

        jwt_payload = {
            "user_id": "test_user",
            "username": "Test User",
            "role": "user",
            "permissions": ["video:upload"],
            "created_at": datetime.now().isoformat()
        }

        with patch('src.api.auth.jwt_auth.verify_token', return_value=jwt_payload):
            result = await get_current_user(mock_request, mock_credentials)

        assert result.user_id == "test_user"
        assert result.username == "Test User"
        assert result.role == UserRole.USER

    @pytest.mark.asyncio
    async def test_get_current_user_with_bearer_api_key(self):
        """Test authentication with Bearer API key"""
        mock_request = Mock()
        mock_request.headers = {}

        mock_credentials = Mock()
        mock_credentials.credentials = "api_key_here"

        test_user = AuthUser(
            user_id="test_user",
            username="Test User",
            role=UserRole.USER,
            permissions=["video:upload"],
            created_at=datetime.now()
        )

        with patch('src.api.auth.jwt_auth.verify_token', return_value=None), \
             patch('src.api.auth.api_key_auth.validate_api_key', return_value=test_user):
            result = await get_current_user(mock_request, mock_credentials)

        assert result == test_user

    @pytest.mark.asyncio
    async def test_get_current_user_no_auth(self):
        """Test authentication failure with no credentials"""
        mock_request = Mock()
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_request, None)

        assert exc_info.value.status_code == 401
        assert "Invalid or missing authentication" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_admin_user_success(self):
        """Test admin user requirement with admin user"""
        admin_user = AuthUser(
            user_id="admin",
            username="Admin User",
            role=UserRole.ADMIN,
            permissions=[],
            created_at=datetime.now()
        )

        result = await get_admin_user(admin_user)
        assert result == admin_user

    @pytest.mark.asyncio
    async def test_get_admin_user_forbidden(self):
        """Test admin user requirement with regular user"""
        regular_user = AuthUser(
            user_id="user",
            username="Regular User",
            role=UserRole.USER,
            permissions=[],
            created_at=datetime.now()
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user(regular_user)

        assert exc_info.value.status_code == 403
        assert "Admin access required" in str(exc_info.value.detail)

    def test_require_permission_success(self):
        """Test permission requirement with authorized user"""
        user_with_permission = AuthUser(
            user_id="user",
            username="User",
            role=UserRole.USER,
            permissions=["video:upload"],
            created_at=datetime.now()
        )

        permission_check = require_permission("video:upload")
        result = permission_check(user_with_permission)

        assert result == user_with_permission

    def test_require_permission_admin_bypass(self):
        """Test permission requirement with admin user (should bypass)"""
        admin_user = AuthUser(
            user_id="admin",
            username="Admin",
            role=UserRole.ADMIN,
            permissions=[],
            created_at=datetime.now()
        )

        permission_check = require_permission("any:permission")
        result = permission_check(admin_user)

        assert result == admin_user

    def test_require_permission_forbidden(self):
        """Test permission requirement with unauthorized user"""
        user_without_permission = AuthUser(
            user_id="user",
            username="User",
            role=UserRole.USER,
            permissions=["other:permission"],
            created_at=datetime.now()
        )

        permission_check = require_permission("video:upload")

        with pytest.raises(HTTPException) as exc_info:
            permission_check(user_without_permission)

        assert exc_info.value.status_code == 403
        assert "Permission 'video:upload' required" in str(exc_info.value.detail)


class TestUserRole:
    """Test UserRole enum"""

    def test_user_roles(self):
        """Test all user roles are properly defined"""
        assert UserRole.ADMIN == "admin"
        assert UserRole.USER == "user"
        assert UserRole.READONLY == "readonly"

    def test_role_comparison(self):
        """Test role comparison works correctly"""
        admin_user = AuthUser(
            user_id="admin", username="Admin", role=UserRole.ADMIN,
            permissions=[], created_at=datetime.now()
        )

        regular_user = AuthUser(
            user_id="user", username="User", role=UserRole.USER,
            permissions=[], created_at=datetime.now()
        )

        assert admin_user.role == UserRole.ADMIN
        assert regular_user.role == UserRole.USER
        assert admin_user.role != regular_user.role


class TestAuthModels:
    """Test authentication data models"""

    def test_auth_user_model(self):
        """Test AuthUser model"""
        created_at = datetime.now()
        last_login = datetime.now() - timedelta(hours=1)

        user = AuthUser(
            user_id="test_user",
            username="Test User",
            role=UserRole.USER,
            permissions=["video:upload", "video:download"],
            created_at=created_at,
            last_login=last_login
        )

        assert user.user_id == "test_user"
        assert user.username == "Test User"
        assert user.role == UserRole.USER
        assert len(user.permissions) == 2
        assert user.created_at == created_at
        assert user.last_login == last_login

    def test_auth_token_model(self):
        """Test AuthToken model"""
        user = AuthUser(
            user_id="test_user",
            username="Test User",
            role=UserRole.USER,
            permissions=[],
            created_at=datetime.now()
        )

        token = AuthToken(
            access_token="jwt_token_here",
            token_type="bearer",
            expires_in=3600,
            user=user
        )

        assert token.access_token == "jwt_token_here"
        assert token.token_type == "bearer"
        assert token.expires_in == 3600
        assert token.user == user
