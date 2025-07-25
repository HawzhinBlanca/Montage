"""
Legacy adapter for backward compatibility with old configuration system
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Try to import from utils
try:
    from .utils.secret_loader import get
except ImportError:
    # Fallback for different import contexts
    import sys
    sys.path.append(str(Path(__file__).parent))
    from utils.secret_loader import get


class LegacySettings:
    """Adapter to make legacy code work with new settings structure"""

    def __init__(self):
        # Load .env for local development
        project_root = Path(__file__).parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)

        # Load settings with env fallback
        self.database_url = os.getenv("DATABASE_URL") or get("DATABASE_URL", "")
        self.redis_url = os.getenv("REDIS_URL") or get("REDIS_URL", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or get("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or get("ANTHROPIC_API_KEY", "")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY") or get("DEEPGRAM_API_KEY", "")
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY") or get("JWT_SECRET_KEY", "")
        self.max_cost_usd = float(os.getenv("MAX_COST_USD") or get("MAX_COST_USD", "5.0"))

        # Additional settings
        self.use_gpu = (os.getenv("USE_GPU") or get("USE_GPU", "false")).lower() == "true"
        self.max_workers = int(os.getenv("MAX_WORKERS") or get("MAX_WORKERS", "4"))
        self.cache_ttl = int(os.getenv("CACHE_TTL") or get("CACHE_TTL", "3600"))

    @classmethod
    def load(cls):
        """Factory method to create settings instance"""
        return cls()

