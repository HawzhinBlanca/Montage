"""
Secure API authentication system with JWT and API key support
"""

import os

from fastapi import Depends, Header, HTTPException, Request

# Task T2: Load ALLOWED_KEYS environment variable into existing auth system
_TASK_ALLOWED_KEYS = {*filter(bool, os.getenv("ALLOWED_KEYS", "").split(","))}

async def verify_key(request: Request, x_api_key: str = Header(None)):
    """T2: Simple API key verification that works with existing auth system"""
    if request.url.path == "/health":      # monitoring stays open
        return
    if x_api_key and x_api_key in _TASK_ALLOWED_KEYS:
        return x_api_key
    # Fall through to existing auth system
    return None

# Keep rest of the file for backward compatibility
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import jwt
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..core.db import Database
from ..utils.logging_config import get_logger
from ..utils.secret_loader import get

logger = get_logger(__name__)

# Import unified settings
from ..settings import settings

# Security configuration from unified settings
JWT_SECRET_KEY = settings.security.jwt_secret_key.get_secret_value()
JWT_ALGORITHM = settings.security.jwt_algorithm
JWT_EXPIRY_HOURS = settings.security.jwt_expiration_hours
API_KEY_LENGTH = 32

def _load_allowed_api_keys() -> List[str]:
    """Load allowed API keys from Vault or environment variables"""
    try:
        # Try to get from Vault first
        from ..utils.secret_loader import get

        # Check for comma-separated list in single secret
        allowed_keys_str = get("ALLOWED_API_KEYS")
        if allowed_keys_str:
            keys = [key.strip() for key in allowed_keys_str.split(',') if key.strip()]
            if keys:
                logger.info(f"✅ Loaded {len(keys)} API keys from secret store")
                return keys

        # Fallback: Check for individual numbered keys
        keys = []
        for i in range(1, 11):  # Support up to 10 API keys
            key = get(f"API_KEY_{i}")
            if key and key != "your-api-key-here":
                keys.append(key)

        if keys:
            logger.info(f"✅ Loaded {len(keys)} API keys from numbered secrets")
            return keys

        # No hardcoded keys - must be configured via environment
        logger.error("❌ No API keys configured - API will reject all requests")
        return []

    except Exception as e:
        logger.error(f"Failed to load API keys: {e}")
        return []


# P0-05: Load API keys after function definition
ALLOWED_API_KEYS = _load_allowed_api_keys()

security = HTTPBearer(auto_error=False)
db = Database()


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key against allowed keys from Vault/Environment
    
    P0-05 Requirement: Header X-API-Key validated against ALLOWED_API_KEYS in Vault
    """
    if not api_key:
        return False

    # T2: Check against ALLOWED_KEYS from Tasks.md first
    if api_key in _TASK_ALLOWED_KEYS:
        return True

    # Check against loaded API keys
    if api_key in ALLOWED_API_KEYS:
        return True

    # Hash-based comparison for additional security (optional)
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    for allowed_key in ALLOWED_API_KEYS:
        if allowed_key.startswith("hash:"):
            allowed_hash = allowed_key[5:]  # Remove "hash:" prefix
            if api_key_hash == allowed_hash:
                return True

    return False


async def get_api_key_from_header(request: Request) -> Optional[str]:
    """Extract API key from X-API-Key header"""
    return request.headers.get("X-API-Key") or request.headers.get("x-api-key")


async def require_api_key(request: Request) -> str:
    """
    P0-05: Require valid API key in X-API-Key header
    
    Raises HTTPException if API key is missing or invalid
    Returns the valid API key for rate limiting purposes
    """
    api_key = await get_api_key_from_header(request)

    if not api_key:
        logger.warning(f"API key missing from request from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if not validate_api_key(api_key):
        logger.warning(f"Invalid API key attempted from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    logger.debug(f"Valid API key used from {request.client.host if request.client else 'unknown'}")
    return api_key


class UserRole(str, Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


class AuthUser(BaseModel):
    """Authenticated user model"""
    user_id: str
    username: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None


class AuthToken(BaseModel):
    """Authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUser


class APIKeyAuth:
    """API Key authentication handler"""

    def __init__(self):
        self.db = Database()
        self._ensure_api_keys_table()

    def _ensure_api_keys_table(self):
        """Ensure API keys table exists"""
        try:
            with self.db.get_connection() as conn:
                if conn is None:
                    return

                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id SERIAL PRIMARY KEY,
                        key_hash VARCHAR(64) NOT NULL UNIQUE,
                        user_id VARCHAR(100) NOT NULL,
                        name VARCHAR(200),
                        role VARCHAR(20) DEFAULT 'user',
                        permissions TEXT[],
                        created_at TIMESTAMP DEFAULT NOW(),
                        last_used TIMESTAMP,
                        is_active BOOLEAN DEFAULT true,
                        rate_limit_per_hour INTEGER DEFAULT 1000
                    )
                """)

                # Create index for fast lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_keys_hash 
                    ON api_keys(key_hash) WHERE is_active = true
                """)

                conn.commit()
                logger.info("API keys table ensured")

        except Exception as e:
            logger.error(f"Failed to create API keys table: {e}")

    def generate_api_key(self, user_id: str, name: str, role: UserRole = UserRole.USER) -> str:
        """Generate new API key for user"""
        try:
            # Generate secure random key
            raw_key = secrets.token_urlsafe(API_KEY_LENGTH)
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

            # Store in database
            with self.db.get_connection() as conn:
                if conn is None:
                    raise HTTPException(status_code=500, detail="Database unavailable")

                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO api_keys (key_hash, user_id, name, role, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (key_hash, user_id, name, role.value))

                conn.commit()

            logger.info(f"Generated API key for user {user_id}")
            return raw_key

        except Exception as e:
            logger.error(f"Failed to generate API key: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate API key")

    def validate_api_key(self, api_key: str) -> Optional[AuthUser]:
        """Validate API key and return user info"""
        try:
            if not api_key:
                return None

            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            with self.db.get_connection() as conn:
                if conn is None:
                    return None

                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, name, role, permissions, created_at, last_used
                    FROM api_keys
                    WHERE key_hash = %s AND is_active = true
                """, (key_hash,))

                row = cursor.fetchone()
                if not row:
                    return None

                user_id, name, role, permissions, created_at, last_used = row

                # Update last used timestamp
                cursor.execute("""
                    UPDATE api_keys SET last_used = NOW() WHERE key_hash = %s
                """, (key_hash,))
                conn.commit()

                return AuthUser(
                    user_id=user_id,
                    username=name or user_id,
                    role=UserRole(role),
                    permissions=permissions or [],
                    created_at=created_at,
                    last_login=last_used
                )

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None


class JWTAuth:
    """JWT token authentication handler"""

    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        try:
            expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
            to_encode = user_data.copy()
            to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

            encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            return encoded_jwt

        except Exception as e:
            logger.error(f"Failed to create JWT token: {e}")
            raise HTTPException(status_code=500, detail="Failed to create token")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None


# Global auth handlers
api_key_auth = APIKeyAuth()
jwt_auth = JWTAuth()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthUser:
    """
    Get current authenticated user from JWT token or API key
    
    Supports:
    - Authorization: Bearer <jwt_token>
    - Authorization: Bearer <api_key>
    - X-API-Key: <api_key>
    """

    # Try API key from header first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user = api_key_auth.validate_api_key(api_key)
        if user:
            return user

    # Try bearer token (JWT or API key)
    if credentials:
        token = credentials.credentials

        # Try as JWT first
        jwt_payload = jwt_auth.verify_token(token)
        if jwt_payload:
            return AuthUser(
                user_id=jwt_payload.get("user_id"),
                username=jwt_payload.get("username"),
                role=UserRole(jwt_payload.get("role", "user")),
                permissions=jwt_payload.get("permissions", []),
                created_at=datetime.fromisoformat(jwt_payload.get("created_at")),
                last_login=None
            )

        # Try as API key
        user = api_key_auth.validate_api_key(token)
        if user:
            return user

    # No valid authentication found
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_admin_user(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Require admin role"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_dependency(current_user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if permission not in current_user.permissions and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return current_user

    return permission_dependency


# Pre-defined permission dependencies
require_video_upload = require_permission("video:upload")
require_video_download = require_permission("video:download")
require_metrics_read = require_permission("metrics:read")
require_cleanup = require_permission("admin:cleanup")
