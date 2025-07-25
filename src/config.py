#!/usr/bin/env python3
"""
Configuration module - Load configuration using secure secret management
"""
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import secret loader
try:
    from .utils.secret_loader import (
        get,
        get_database_url,
        get_redis_url,
        get_openai_key,
        get_anthropic_key,
        get_deepgram_key,
    )
except ImportError:
    # Fallback if running from different context
    import sys

    sys.path.append(str(Path(__file__).parent))
    from utils.secret_loader import (
        get,
        get_database_url,
        get_redis_url,
        get_openai_key,
        get_anthropic_key,
        get_deepgram_key,
    )

# Load .env for local development (secrets should come from AWS in production)
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)  # Don't override AWS secrets

# Database configuration
DATABASE_URL = get_database_url()
REDIS_URL = get_redis_url()

# API keys - loaded securely
OPENAI_API_KEY = get_openai_key()
ANTHROPIC_API_KEY = get_anthropic_key()
DEEPGRAM_API_KEY = get_deepgram_key()

# Cost limits
MAX_COST_USD = float(get("MAX_COST_USD", "5.00"))

# Set up logger
logger = logging.getLogger(__name__)

# Log configuration status (without exposing secrets)

if OPENAI_API_KEY:
    logger.info("✅ OpenAI API key configured")
else:
    logger.warning("⚠️  OPENAI_API_KEY not set - premium features will be disabled")

if ANTHROPIC_API_KEY:
    logger.info("✅ Anthropic API key configured")
else:
    logger.warning("⚠️  ANTHROPIC_API_KEY not set - premium features will be disabled")

if DEEPGRAM_API_KEY:
    logger.info("✅ Deepgram API key configured")
else:
    logger.warning("⚠️  DEEPGRAM_API_KEY not set - using only local ASR")

logger.info(f"✅ Config loaded - MAX_COST_USD: ${MAX_COST_USD}")
