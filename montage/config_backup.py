#!/usr/bin/env python3
"""
Configuration module - Legacy compatibility layer
Provides backward compatibility while migrating to new settings system
"""
import logging
from pathlib import Path

from dotenv import load_dotenv

# Import new settings system
from .settings import (
    settings,
    api_keys,
    cost_limits,
    db_settings,
    redis_settings,
)

# Load .env for local development
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)

# Legacy exports for backward compatibility
DATABASE_URL = db_settings.url.get_secret_value()
REDIS_URL = redis_settings.url.get_secret_value()

# API keys - maintain legacy interface
OPENAI_API_KEY = api_keys.openai.get_secret_value() if api_keys.openai else None
ANTHROPIC_API_KEY = api_keys.anthropic.get_secret_value() if api_keys.anthropic else None
DEEPGRAM_API_KEY = api_keys.deepgram.get_secret_value() if api_keys.deepgram else None

# Cost limits
MAX_COST_USD = cost_limits.max_cost_usd

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

# Export settings for new code
__all__ = [
    "DATABASE_URL",
    "REDIS_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPGRAM_API_KEY",
    "MAX_COST_USD",
    "settings",  # New unified settings
]