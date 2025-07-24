#!/usr/bin/env python3
"""
Secret management module - Load secrets from AWS Secrets Manager or environment
Falls back to environment variables if AWS is not configured
"""
import json
import logging
import os
from functools import lru_cache
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try to import cloud SDKs
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.debug("AWS SDK not available")

try:
    import hvac  # HashiCorp Vault client
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logger.debug("Vault client not available")

# Cache for secrets
_secret_cache: Dict[str, str] = {}


def _get_vault_secret(secret_name: str) -> Optional[str]:
    """Retrieve secret from HashiCorp Vault"""
    if not VAULT_AVAILABLE:
        return None

    try:
        # Get Vault configuration from environment
        vault_url = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
        vault_token = os.getenv("VAULT_TOKEN")

        if not vault_token:
            # Try to read token from Kubernetes service account
            token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(token_path):
                with open(token_path) as f:
                    vault_token = f.read().strip()
            else:
                logger.debug("No Vault token available")
                return None

        # Create Vault client
        client = hvac.Client(url=vault_url, token=vault_token)

        # Check if client is authenticated
        if not client.is_authenticated():
            logger.warning("Vault client not authenticated")
            return None

        # Read secret from KV v2 engine
        secret_path = f"secret/data/montage/{secret_name.lower()}"
        response = client.secrets.kv.v2.read_secret_version(
            path=f"montage/{secret_name.lower()}"
        )

        if response and 'data' in response and 'data' in response['data']:
            secret_data = response['data']['data']
            # Return the secret value - try multiple key formats
            for key in [secret_name.upper(), secret_name, 'value']:
                if key in secret_data:
                    return secret_data[key]

        logger.debug(f"Secret {secret_name} not found in Vault")
        return None

    except Exception as e:
        logger.debug(f"Failed to retrieve {secret_name} from Vault: {e}")
        return None


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
    2. HashiCorp Vault (if available)
    3. AWS Secrets Manager (if available)
    4. Environment variable
    5. Default value

    Args:
        name: Secret name (e.g., 'OPENAI_API_KEY')
        default: Default value if secret not found

    Returns:
        Secret value or default
    """
    # Check cache first
    if name in _secret_cache:
        return _secret_cache[name]

    secret_value = None

    # Try Vault first (highest security)
    if VAULT_AVAILABLE:
        secret_value = _get_vault_secret(name)
        if secret_value:
            logger.info(f"Loaded {name} from Vault")
            _secret_cache[name] = secret_value
            return secret_value

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

    logger.warning(f"Secret {name} not found in Vault, AWS, or environment")
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


def validate_required_secrets() -> Dict[str, bool]:
    """
    Validate that all required API secrets are available
    Returns dict with validation status for each required secret
    
    This function is called by FastAPI on startup to ensure
    all necessary secrets are properly configured.
    """
    required_secrets = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPGRAM_API_KEY",
        "GEMINI_API_KEY"
    ]

    validation_results = {}
    all_valid = True

    for secret_name in required_secrets:
        secret_value = get(secret_name)
        is_valid = bool(secret_value and not secret_value.startswith("PLACEHOLDER"))
        validation_results[secret_name] = is_valid

        if is_valid:
            logger.info(f"✅ {secret_name}: Available")
        else:
            logger.error(f"❌ {secret_name}: Missing or placeholder")
            all_valid = False

    if all_valid:
        logger.info("🔑 All required secrets validated successfully")
    else:
        logger.error("🚨 Some required secrets are missing - check configuration")

    validation_results["all_valid"] = all_valid
    return validation_results


def get_secret_sources_status() -> Dict[str, bool]:
    """Get status of available secret sources for debugging"""
    return {
        "vault_available": VAULT_AVAILABLE,
        "aws_available": AWS_AVAILABLE,
        "environment_fallback": True,  # Always available
        "vault_authenticated": _is_vault_authenticated() if VAULT_AVAILABLE else False
    }


def _is_vault_authenticated() -> bool:
    """Check if Vault client can authenticate"""
    if not VAULT_AVAILABLE:
        return False

    try:
        vault_url = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
        vault_token = os.getenv("VAULT_TOKEN")

        if not vault_token:
            token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(token_path):
                with open(token_path) as f:
                    vault_token = f.read().strip()

        if not vault_token:
            return False

        client = hvac.Client(url=vault_url, token=vault_token)
        return client.is_authenticated()
    except Exception:
        return False


def get_deepgram_key() -> Optional[str]:
    """Get Deepgram API key"""
    return get("DEEPGRAM_API_KEY")


def get_gemini_key() -> Optional[str]:
    """Get Gemini API key"""
    return get("GEMINI_API_KEY")


def get_database_url() -> str:
    """Get database URL - REQUIRES environment variable for security"""
    db_url = get("DATABASE_URL", None)

    if db_url is None:
        # SECURITY: Never provide default credentials
        raise ValueError(
            "DATABASE_URL environment variable is required. "
            "Set it to your database connection string. "
            "Example: DATABASE_URL='postgresql://username:password@host:port/database'"
        )

    # SECURITY: Basic validation of database URL format - PostgreSQL only
    if not db_url.startswith(("postgresql://", "postgres://")):
        raise ValueError(
            f"Invalid DATABASE_URL format: {db_url}. "
            "Must start with postgresql:// or postgres:// (PostgreSQL only)"
        )

    # SECURITY: Warn about common insecure patterns
    if "password" in db_url.lower() or "pass@" in db_url.lower():
        import logging

        logging.warning(
            "WARNING: DATABASE_URL contains common weak passwords. "
            "Use strong, unique passwords in production."
        )

    return db_url


def get_redis_url() -> str:
    """Get Redis URL with fallback"""
    return get("REDIS_URL", "redis://localhost:6379")
