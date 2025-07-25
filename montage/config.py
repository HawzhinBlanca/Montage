"""
Unified configuration entry-point.
`settings` is instantiated **on first access** to avoid import-time crashes.
"""
import os
from functools import lru_cache

USE_V2 = os.getenv("USE_SETTINGS_V2", "false").lower() == "true"

@lru_cache
def _load_settings():
    if USE_V2:
        from .settings_v2 import get_settings
        return get_settings()
    else:
        from .legacy_adapter import LegacySettings
        return LegacySettings.load()

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
