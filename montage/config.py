"""
Unified configuration entry-point.
`settings` is instantiated **on first access** to avoid import-time crashes.
"""
from functools import lru_cache

@lru_cache
def _load_settings():
    # Always use the main settings module
    from .settings import get_settings
    return get_settings()

# public handle
class _SettingsProxy:
    def __getattr__(self, item):
        return getattr(_load_settings(), item)

settings = _SettingsProxy()   # lazy proxy

def reload_settings():
    """Hot-reload runtime settings (e.g., after env change)."""
    _load_settings.cache_clear()        # reset lru_cache
    return _load_settings()

__all__ = ["settings", "reload_settings"]
