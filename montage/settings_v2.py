"""
Settings V2 - New configuration system using Pydantic
"""
import os
from typing import Optional
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: SecretStr = Field(
        default_factory=lambda: SecretStr(
            os.getenv("DATABASE_URL", "postgresql://localhost/montage")
        )
    )
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=40, ge=0, le=200)


class RedisConfig(BaseModel):
    """Redis configuration"""
    url: SecretStr = Field(
        default_factory=lambda: SecretStr(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
    )
    ttl: int = Field(default=3600, ge=0)


class APIKeysConfig(BaseModel):
    """API keys configuration"""
    openai: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("OPENAI_API_KEY")) else None
    )
    anthropic: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("ANTHROPIC_API_KEY")) else None
    )
    deepgram: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(val) if (val := os.getenv("DEEPGRAM_API_KEY")) else None
    )


class SecurityConfig(BaseModel):
    """Security configuration"""
    jwt_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(
            os.getenv("JWT_SECRET_KEY", "default-secret-key")
        )
    )


class ProcessingConfig(BaseModel):
    """Processing configuration"""
    use_gpu: bool = Field(default=False)
    max_workers: int = Field(default=4, ge=1, le=32)
    max_cost_usd: float = Field(default=5.0, ge=0.0)


class Settings(BaseSettings):
    """V2 settings with structured configuration using Pydantic"""

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Legacy compatibility properties
    @property
    def database_url(self) -> str:
        return self.database.url.get_secret_value()

    @property
    def redis_url(self) -> str:
        return self.redis.url.get_secret_value()

    @property
    def openai_api_key(self) -> Optional[str]:
        return self.api_keys.openai.get_secret_value() if self.api_keys.openai else None

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self.api_keys.anthropic.get_secret_value() if self.api_keys.anthropic else None

    @property
    def deepgram_api_key(self) -> Optional[str]:
        return self.api_keys.deepgram.get_secret_value() if self.api_keys.deepgram else None

    @property
    def jwt_secret_key(self) -> str:
        return self.security.jwt_secret_key.get_secret_value()

    @property
    def max_cost_usd(self) -> float:
        return self.processing.max_cost_usd

    @property
    def use_gpu(self) -> bool:
        return self.processing.use_gpu

    @property
    def max_workers(self) -> int:
        return self.processing.max_workers

    @property
    def cache_ttl(self) -> int:
        return self.redis.ttl

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Parse nested fields from env vars like PROCESSING__USE_GPU
        env_nested_delimiter="__",
        # Allow extra fields for forward compatibility
        extra="allow",
    )


@lru_cache()
def get_settings() -> Settings:
    """Load settings from environment variables (cached)"""
    # Override from env vars if needed
    overrides = {}

    # Handle legacy env var names
    if os.getenv("USE_GPU"):
        overrides.setdefault("processing", {})
        overrides["processing"]["use_gpu"] = os.getenv("USE_GPU", "false").lower() == "true"

    if os.getenv("MAX_WORKERS"):
        overrides.setdefault("processing", {})
        overrides["processing"]["max_workers"] = int(os.getenv("MAX_WORKERS", "4"))

    if os.getenv("MAX_COST_USD"):
        overrides.setdefault("processing", {})
        overrides["processing"]["max_cost_usd"] = float(os.getenv("MAX_COST_USD", "5.0"))

    if os.getenv("CACHE_TTL"):
        overrides.setdefault("redis", {})
        overrides["redis"]["ttl"] = int(os.getenv("CACHE_TTL", "3600"))

    return Settings(**overrides)
