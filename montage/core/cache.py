"""
Redis caching layer for API results and transcription data
"""

import hashlib
import json
import pickle
from typing import Any, Dict, List, Optional

import redis

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Cache configuration
from ..settings import settings
REDIS_URL = settings.redis.url.get_secret_value()
DEFAULT_TTL = settings.features.cache_ttl_seconds if settings.features.enable_caching else 3600
TRANSCRIPTION_TTL = 86400  # 24 hours for transcriptions
API_RESULT_TTL = 1800  # 30 minutes for API results


class CacheError(Exception):
    """Cache operation error"""
    def __init__(self, message="Cache operation failed", key=None):
        super().__init__(message)
        self.message = message
        self.key = key


class RedisCache:
    """Redis-based caching with smart serialization and key management"""

    def __init__(self, redis_url: str = REDIS_URL):
        """Initialize Redis cache client"""
        self.redis_url = redis_url
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Redis with error handling"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis cache at {self.redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis cache: {e}")
            self.client = None

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Use pickle for complex objects, JSON for simple ones
            if isinstance(value, (dict, list, str, int, float, bool)) and value is not None:
                return json.dumps(value, default=str).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise CacheError(f"Serialization failed: {e}")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first (more common)
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise CacheError(f"Deserialization failed: {e}")

    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key"""
        # Hash long keys to avoid Redis key length limits
        if len(key) > 200:
            key = hashlib.sha256(key.encode()).hexdigest()

        return f"montage_cache:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            return None

        cache_key = self._make_key(namespace, key)

        try:
            data = self.client.get(cache_key)
            if data is None:
                logger.debug(f"Cache miss: {cache_key}")
                return None

            value = self._deserialize(data)
            logger.debug(f"Cache hit: {cache_key}")
            return value

        except Exception as e:
            logger.error(f"Cache get error for {cache_key}: {e}")
            return None

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with TTL"""
        if not self.client:
            return False

        cache_key = self._make_key(namespace, key)
        ttl = ttl or DEFAULT_TTL

        try:
            data = self._serialize(value)

            result = self.client.setex(cache_key, ttl, data)
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s)")
            return result

        except Exception as e:
            logger.error(f"Cache set error for {cache_key}: {e}")
            return False

    def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache"""
        if not self.client:
            return False

        cache_key = self._make_key(namespace, key)

        try:
            result = self.client.delete(cache_key)
            logger.debug(f"Cache delete: {cache_key}")
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete error for {cache_key}: {e}")
            return False

    def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.client:
            return False

        cache_key = self._make_key(namespace, key)

        try:
            return self.client.exists(cache_key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for {cache_key}: {e}")
            return False

    def flush_namespace(self, namespace: str) -> int:
        """Delete all keys in namespace"""
        if not self.client:
            return 0

        pattern = f"montage_cache:{namespace}:*"

        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Flushed {deleted} keys from namespace '{namespace}'")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Cache flush error for namespace {namespace}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.client:
            return {"status": "disconnected"}

        try:
            info = self.client.info()

            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


class TranscriptionCache:
    """Specialized cache for transcription results"""

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.namespace = "transcription"

    def get_transcript(self, file_hash: str, engine: str) -> Optional[List[Dict]]:
        """Get cached transcription result"""
        key = f"{engine}:{file_hash}"
        return self.cache.get(self.namespace, key)

    def set_transcript(
        self,
        file_hash: str,
        engine: str,
        transcript: List[Dict]
    ) -> bool:
        """Cache transcription result"""
        key = f"{engine}:{file_hash}"
        return self.cache.set(self.namespace, key, transcript, TRANSCRIPTION_TTL)

    def get_merged_transcript(self, file_hash: str) -> Optional[List[Dict]]:
        """Get cached merged transcription result"""
        key = f"merged:{file_hash}"
        return self.cache.get(self.namespace, key)

    def set_merged_transcript(
        self,
        file_hash: str,
        transcript: List[Dict]
    ) -> bool:
        """Cache merged transcription result"""
        key = f"merged:{file_hash}"
        return self.cache.set(self.namespace, key, transcript, TRANSCRIPTION_TTL)


class APIResultCache:
    """Specialized cache for API results"""

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.namespace = "api_results"

    def get_job_result(self, job_id: str) -> Optional[Dict]:
        """Get cached job processing result"""
        return self.cache.get(self.namespace, f"job:{job_id}")

    def set_job_result(self, job_id: str, result: Dict) -> bool:
        """Cache job processing result"""
        return self.cache.set(
            self.namespace,
            f"job:{job_id}",
            result,
            API_RESULT_TTL
        )

    def get_video_analysis(self, video_hash: str) -> Optional[Dict]:
        """Get cached video analysis result"""
        return self.cache.get(self.namespace, f"analysis:{video_hash}")

    def set_video_analysis(self, video_hash: str, analysis: Dict) -> bool:
        """Cache video analysis result"""
        return self.cache.set(
            self.namespace,
            f"analysis:{video_hash}",
            analysis,
            TRANSCRIPTION_TTL  # Longer TTL for expensive analysis
        )

    def get_highlights(self, video_hash: str, mode: str) -> Optional[List[Dict]]:
        """Get cached highlight selection result"""
        key = f"highlights:{mode}:{video_hash}"
        return self.cache.get(self.namespace, key)

    def set_highlights(
        self,
        video_hash: str,
        mode: str,
        highlights: List[Dict]
    ) -> bool:
        """Cache highlight selection result"""
        key = f"highlights:{mode}:{video_hash}"
        return self.cache.set(self.namespace, key, highlights, TRANSCRIPTION_TTL)


class CacheDecorator:
    """Decorator for automatic function result caching"""

    def __init__(
        self,
        cache: RedisCache,
        namespace: str,
        ttl: Optional[int] = None,
        key_func: Optional[callable] = None
    ):
        self.cache = cache
        self.namespace = namespace
        self.ttl = ttl or DEFAULT_TTL
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        # Create deterministic key from arguments
        key_parts = [func_name]

        # Add positional args
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # Hash complex objects
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])

        # Add keyword args
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={hashlib.md5(str(v).encode()).hexdigest()[:8]}")

        return ":".join(key_parts)

    def __call__(self, func):
        """Decorator implementation"""
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self.key_func(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = self.cache.get(self.namespace, cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function
            logger.debug(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)

            # Cache result
            self.cache.set(self.namespace, cache_key, result, self.ttl)

            return result

        return wrapper


# Global cache instances
_cache_instance = None
_transcription_cache = None
_api_result_cache = None


def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


def get_transcription_cache() -> TranscriptionCache:
    """Get global transcription cache instance"""
    global _transcription_cache
    if _transcription_cache is None:
        _transcription_cache = TranscriptionCache(get_cache())
    return _transcription_cache


def get_api_result_cache() -> APIResultCache:
    """Get global API result cache instance"""
    global _api_result_cache
    if _api_result_cache is None:
        _api_result_cache = APIResultCache(get_cache())
    return _api_result_cache


def cache_result(namespace: str, ttl: Optional[int] = None, key_func: Optional[callable] = None):
    """Decorator for caching function results"""
    return CacheDecorator(get_cache(), namespace, ttl, key_func)
