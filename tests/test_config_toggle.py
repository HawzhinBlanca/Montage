# tests/test_config_toggle.py
import importlib
import os

import pytest


# Set required env vars for tests
@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test/db")

def _reload():
    import montage.config as cfg
    importlib.reload(cfg)
    return cfg.settings

def test_flag_toggle(monkeypatch):
    monkeypatch.setenv("USE_SETTINGS_V2", "true")
    assert _reload().database_url is not None

    monkeypatch.setenv("USE_SETTINGS_V2", "false")
    assert _reload().database_url is not None

def test_reload_runtime(monkeypatch):
    monkeypatch.setenv("USE_SETTINGS_V2", "true")
    from montage.config import reload_settings
    new_cfg = reload_settings()
    assert new_cfg.database_url

def test_v2_pydantic_features(monkeypatch):
    """Test that V2 settings use Pydantic features properly"""
    # This test specifically tests V2 features, so skip if not using V2
    if os.getenv("USE_SETTINGS_V2", "false").lower() != "true":
        # Set it for this test
        monkeypatch.setenv("USE_SETTINGS_V2", "true")

    monkeypatch.setenv("MAX_WORKERS", "8")
    monkeypatch.setenv("USE_GPU", "true")

    # Clear cache to ensure fresh load
    from montage.config import _load_settings
    _load_settings.cache_clear()

    # Import V2 settings directly
    from montage.settings_v2 import get_settings
    settings = get_settings()

    # Test legacy compatibility properties
    assert settings.database_url is not None
    assert settings.max_workers == 8
    assert settings.use_gpu is True

    # Test structured access
    assert hasattr(settings, 'database')
    assert hasattr(settings, 'processing')
    assert settings.processing.max_workers == 8
    assert settings.processing.use_gpu is True
