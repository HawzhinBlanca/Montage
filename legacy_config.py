import os
import multiprocessing
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Database
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "video_processing")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "videouser")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

    # Connection pool sizing - 2x CPU cores as per requirements
    CPU_COUNT = multiprocessing.cpu_count()
    MAX_POOL_SIZE = int(os.getenv("MAX_POOL_SIZE", CPU_COUNT * 2))
    MIN_POOL_SIZE = int(os.getenv("MIN_POOL_SIZE", 2))

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    CHECKPOINT_TTL = int(os.getenv("CHECKPOINT_TTL", 86400))  # 24 hours

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Budget
    BUDGET_LIMIT = float(os.getenv("BUDGET_LIMIT", 5.00))

    # Monitoring
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 9099))
    METRICS_PORT = int(os.getenv("METRICS_PORT", 9099))
    ALERTS_PORT = int(os.getenv("ALERTS_PORT", 9098))

    # Webhook configuration
    WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

    # Redis URL (alternative format)
    REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

    # Processing
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
    FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")
    TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/video_processing")

    # FFmpeg quality settings
    VIDEO_CRF = int(os.getenv("VIDEO_CRF", 23))
    VIDEO_PRESET = os.getenv("VIDEO_PRESET", "medium")
    AUDIO_BITRATE = os.getenv("AUDIO_BITRATE", "128k")

    # Processing limits
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", 4))
    MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", 10800))  # 3 hours

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def get_database_url(cls):
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []

        if not cls.POSTGRES_PASSWORD:
            errors.append("POSTGRES_PASSWORD is required")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")

        if cls.MAX_POOL_SIZE < cls.MIN_POOL_SIZE:
            errors.append(
                f"MAX_POOL_SIZE ({cls.MAX_POOL_SIZE}) must be >= MIN_POOL_SIZE ({cls.MIN_POOL_SIZE})"
            )

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        return True
