#!/usr/bin/env python3
"""
Unified configuration system using Pydantic v2
Provides type-safe, validated settings with environment variable support
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Legacy secret_loader import removed - Phase 3-5


class DatabaseSettings(BaseModel):
    """Database configuration"""
    url: SecretStr = Field(
        default_factory=lambda: SecretStr(url) if (url := os.getenv("DATABASE_URL", "")) else SecretStr("postgresql://localhost/montage")
    )
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=40, ge=0, le=200)
    pool_timeout: float = Field(default=30.0, gt=0)
    pool_recycle: int = Field(default=3600, gt=0)
    echo_sql: bool = Field(default=False)
    
    @property
    def sync_url(self) -> str:
        """Get synchronous database URL"""
        url = self.url.get_secret_value()
        if url.startswith("postgresql+asyncpg://"):
            return url.replace("postgresql+asyncpg://", "postgresql://")
        return url
    
    @property
    def async_url(self) -> str:
        """Get asynchronous database URL"""
        url = self.url.get_secret_value()
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://")
        return url


class RedisSettings(BaseModel):
    """Redis configuration"""
    url: SecretStr = Field(
        default_factory=lambda: SecretStr(url) if (url := os.getenv("REDIS_URL", "")) else SecretStr("redis://localhost:6379/0")
    )
    max_connections: int = Field(default=50, ge=1)
    socket_timeout: int = Field(default=5, ge=1)
    socket_connect_timeout: int = Field(default=5, ge=1)
    retry_on_timeout: bool = Field(default=True)
    decode_responses: bool = Field(default=True)


class APIKeysSettings(BaseModel):
    """API keys configuration"""
    openai: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(key) if (key := os.getenv("OPENAI_API_KEY")) else None
    )
    anthropic: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(key) if (key := os.getenv("ANTHROPIC_API_KEY")) else None
    )
    deepgram: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(key) if (key := os.getenv("DEEPGRAM_API_KEY")) else None
    )
    gemini: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(key) if (key := os.getenv("GEMINI_API_KEY", "")) else None
    )
    huggingface_token: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(key) if (key := os.getenv("HUGGINGFACE_TOKEN", "")) else None
    )
    
    @property
    def has_openai(self) -> bool:
        return self.openai is not None
    
    @property
    def has_anthropic(self) -> bool:
        return self.anthropic is not None
    
    @property
    def has_deepgram(self) -> bool:
        return self.deepgram is not None
    
    @property
    def has_gemini(self) -> bool:
        return self.gemini is not None


class CostLimitsSettings(BaseModel):
    """Cost control settings"""
    max_cost_usd: float = Field(default=5.0, gt=0, le=100)
    max_tokens_per_request: int = Field(default=4096, gt=0)
    max_api_retries: int = Field(default=3, ge=0)
    api_timeout_seconds: int = Field(default=30, gt=0)
    
    # Per-provider limits
    openai_daily_limit: float = Field(default=10.0, gt=0)
    anthropic_daily_limit: float = Field(default=10.0, gt=0)
    deepgram_daily_limit: float = Field(default=5.0, gt=0)
    gemini_daily_limit: float = Field(default=2.0, gt=0)


class VideoProcessingSettings(BaseModel):
    """Video processing configuration"""
    max_file_size_mb: int = Field(default=5000, gt=0)
    allowed_formats: List[str] = Field(
        default=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
    )
    output_format: str = Field(default="mp4")
    output_codec: str = Field(default="h264")
    output_crf: int = Field(default=23, ge=0, le=51)
    
    # Processing limits
    max_duration_seconds: int = Field(default=3600, gt=0)
    chunk_duration_seconds: int = Field(default=300, gt=0)
    max_concurrent_jobs: int = Field(default=4, ge=1)
    worker_concurrency: int = Field(default=2, ge=1, description="Celery worker concurrency")
    
    # Quality settings
    enable_vmaf: bool = Field(default=True)
    vmaf_min_score: float = Field(default=80.0, ge=0, le=100)
    enable_audio_normalization: bool = Field(default=True)
    
    # Highlight selection
    highlight_keywords: List[str] = Field(
        default=["important", "amazing", "incredible", "revealed", "exclusive", "breaking"],
        description="Keywords for highlight scoring"
    )
    target_loudness_lufs: float = Field(default=-16.0)
    
    @field_validator("allowed_formats")
    def validate_formats(cls, v):
        """Ensure all formats start with a dot"""
        return [fmt if fmt.startswith(".") else f".{fmt}" for fmt in v]


class SecuritySettings(BaseModel):
    """Security configuration"""
    allowed_upload_paths: List[Path] = Field(
        default_factory=lambda: [
            Path.home() / "Videos",
            Path.home() / "Downloads",
            Path("/tmp"),
            Path("/var/tmp"),
        ]
    )
    max_path_depth: int = Field(default=10, gt=0)
    enable_path_sanitization: bool = Field(default=True)
    enable_sql_injection_protection: bool = Field(default=True)
    
    # Authentication
    jwt_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("JWT_SECRET_KEY", ""))
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24, gt=0)
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests_per_minute: int = Field(default=60, gt=0)
    rate_limit_burst_size: int = Field(default=10, gt=0)
    
    @model_validator(mode="after")
    def validate_security(self):
        """Ensure critical security settings"""
        if not self.jwt_secret_key.get_secret_value():
            raise ValueError(
                "JWT_SECRET_KEY must be set in environment variables. "
                "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        return self


class LoggingSettings(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_path: Optional[Path] = Field(default=None)
    max_file_size_mb: int = Field(default=100, gt=0)
    backup_count: int = Field(default=5, gt=0)
    
    # Sensitive data masking
    mask_secrets: bool = Field(default=True)
    mask_patterns: List[str] = Field(
        default=[
            r"api[_-]?key",
            r"secret",
            r"password",
            r"token",
            r"authorization",
        ]
    )
    
    # Structured logging
    use_json_format: bool = Field(default=False)
    include_process_info: bool = Field(default=True)
    include_thread_info: bool = Field(default=False)


class MonitoringSettings(BaseModel):
    """Monitoring and observability configuration"""
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    enable_health_checks: bool = Field(default=True)
    
    # Sentry
    sentry_dsn: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(dsn) if (dsn := os.getenv("SENTRY_DSN", "")) else None
    )
    sentry_environment: str = Field(default="development")
    sentry_traces_sample_rate: float = Field(default=0.1, ge=0, le=1)
    
    # Prometheus
    prometheus_enabled: bool = Field(default=False)
    prometheus_port: int = Field(default=9090, gt=0, le=65535)
    
    # Health check endpoints
    health_check_path: str = Field(default="/health")
    ready_check_path: str = Field(default="/ready")


class FeatureFlags(BaseModel):
    """Feature flags for gradual rollout"""
    enable_speaker_diarization: bool = Field(default=True)
    enable_smart_crop: bool = Field(default=True)
    enable_ab_testing: bool = Field(default=True)
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    
    # Local AI preferences
    prefer_local_models: bool = Field(default=True)


class Settings(BaseSettings):
    """Main settings class combining all configuration sections"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Application metadata
    app_name: str = Field(default="Montage AI Video Pipeline")
    app_version: str = Field(default="3.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api_keys: APIKeysSettings = Field(default_factory=APIKeysSettings)
    costs: CostLimitsSettings = Field(default_factory=CostLimitsSettings)
    video: VideoProcessingSettings = Field(default_factory=VideoProcessingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "output")
    temp_dir: Path = Field(default_factory=lambda: Path("/tmp") / "montage")
    
    @model_validator(mode="after")
    def create_directories(self):
        """Ensure required directories exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        return self
    
    @model_validator(mode="after")
    def validate_environment(self):
        """Validate environment-specific settings"""
        if self.environment == "production":
            # Enforce production requirements
            if not self.security.enable_path_sanitization:
                raise ValueError("Path sanitization must be enabled in production")
            if not self.security.enable_sql_injection_protection:
                raise ValueError("SQL injection protection must be enabled in production")
            if self.debug:
                raise ValueError("Debug mode must be disabled in production")
            if not self.monitoring.sentry_dsn:
                raise ValueError("Sentry DSN must be configured in production")
        return self
    
    def get_safe_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary with secrets masked"""
        data = self.model_dump()
        
        def mask_secrets(obj):
            if isinstance(obj, dict):
                return {
                    k: mask_secrets(v) if not isinstance(v, SecretStr) else "***MASKED***"
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [mask_secrets(item) for item in obj]
            else:
                return obj
        
        return mask_secrets(data)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience exports
settings = get_settings()
db_settings = settings.database
redis_settings = settings.redis
api_keys = settings.api_keys
cost_limits = settings.costs
video_settings = settings.video
security_settings = settings.security
logging_settings = settings.logging
monitoring_settings = settings.monitoring
feature_flags = settings.features