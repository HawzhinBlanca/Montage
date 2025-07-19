#!/usr/bin/env python3
"""
Secret management module - Load secrets from AWS Secrets Manager or environment
Falls back to environment variables if AWS is not configured
"""
import os
import json
import logging
from typing import Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import AWS SDK
try:
    import boto3
    from botocore.exceptions import ClientError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("AWS SDK not available, falling back to environment variables")

# Cache for secrets
_secret_cache: Dict[str, str] = {}


def _get_aws_secret(secret_name: str) -> Optional[str]:
    """Retrieve secret from AWS Secrets Manager"""
    if not AWS_AVAILABLE:
        return None

    try:
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name="secretsmanager",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )

        # Retrieve the secret
        response = client.get_secret_value(SecretId=f"montage/{secret_name}")

        # Parse the secret
        if "SecretString" in response:
            secret = response["SecretString"]
            # Try to parse as JSON first
            try:
                secret_dict = json.loads(secret)
                return secret_dict.get("value", secret)
            except json.JSONDecodeError:
                return secret

        return None

    except ClientError as e:
        logger.debug(f"AWS Secrets Manager error for {secret_name}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error retrieving {secret_name} from AWS: {e}")
        return None


@lru_cache(maxsize=128)
def get(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get secret value by name

    Priority:
    1. Cached value
    2. AWS Secrets Manager
    3. Environment variable
    4. Default value

    Args:
        name: Secret name (e.g., 'OPENAI_API_KEY')
        default: Default value if secret not found

    Returns:
        Secret value or default
    """
    # Check cache first
    if name in _secret_cache:
        return _secret_cache[name]

    # Try AWS Secrets Manager
    aws_secret = _get_aws_secret(name.lower().replace("_", "-"))
    if aws_secret:
        _secret_cache[name] = aws_secret
        logger.info(f"Loaded {name} from AWS Secrets Manager")
        return aws_secret

    # Fall back to environment variable
    env_value = os.getenv(name)
    if env_value:
        _secret_cache[name] = env_value
        logger.debug(f"Loaded {name} from environment")
        return env_value

    # Use default
    if default:
        logger.warning(f"Using default value for {name}")
        return default

    logger.warning(f"Secret {name} not found in AWS or environment")
    return None


def clear_cache():
    """Clear the secret cache"""
    _secret_cache.clear()
    get.cache_clear()


# Convenience functions for common secrets
def get_openai_key() -> Optional[str]:
    """Get OpenAI API key"""
    return get("OPENAI_API_KEY")


def get_anthropic_key() -> Optional[str]:
    """Get Anthropic API key"""
    return get("ANTHROPIC_API_KEY")


def get_deepgram_key() -> Optional[str]:
    """Get Deepgram API key"""
    return get("DEEPGRAM_API_KEY")


def get_gemini_key() -> Optional[str]:
    """Get Gemini API key"""
    return get("GEMINI_API_KEY")


def get_database_url() -> str:
    """Get database URL with fallback"""
    return get(
        "DATABASE_URL", "postgresql://postgres:pass@localhost:5432/postgres"
    )  # pragma: allowlist secret


def get_redis_url() -> str:
    """Get Redis URL with fallback"""
    return get("REDIS_URL", "redis://localhost:6379")
