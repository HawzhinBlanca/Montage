# tests/test_config_toggle.py
import importlib

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

