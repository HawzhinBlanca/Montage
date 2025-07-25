"""
Settings V2 - New configuration system
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """V2 settings with structured configuration"""
    database_url: str
    redis_url: str
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    jwt_secret_key: str = ""
    max_cost_usd: float = 5.0

    # Add other settings as needed
    use_gpu: bool = False
    max_workers: int = 4
    cache_ttl: int = 3600


def get_settings() -> Settings:
    """Load settings from environment variables"""
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://localhost/montage"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
        jwt_secret_key=os.getenv("JWT_SECRET_KEY", "default-secret-key"),
        max_cost_usd=float(os.getenv("MAX_COST_USD", "5.0")),
        use_gpu=os.getenv("USE_GPU", "false").lower() == "true",
        max_workers=int(os.getenv("MAX_WORKERS", "4")),
        cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
    )

