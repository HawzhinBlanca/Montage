"""Phase 3 Config Unification Tests"""
import os
import pytest


def test_v2_settings_structure():
    """Test V2 settings have proper Pydantic structure"""
    os.environ["USE_SETTINGS_V2"] = "true"
    os.environ["DATABASE_URL"] = "postgresql://test/db"
    os.environ["JWT_SECRET_KEY"] = "test-secret"
    os.environ["MAX_WORKERS"] = "8"
    os.environ["USE_GPU"] = "true"

    # Import fresh to get V2 settings
    from montage.settings_v2 import get_settings
    settings = get_settings()

    # Test structure
    assert hasattr(settings, 'database')
    assert hasattr(settings, 'redis')
    assert hasattr(settings, 'api_keys')
    assert hasattr(settings, 'security')
    assert hasattr(settings, 'processing')

    # Test values
    assert settings.database.url.get_secret_value() == "postgresql://test/db"
    assert settings.security.jwt_secret_key.get_secret_value() == "test-secret"
    assert settings.processing.max_workers == 8
    assert settings.processing.use_gpu is True

    # Test legacy properties
    assert settings.database_url == "postgresql://test/db"
    assert settings.jwt_secret_key == "test-secret"
    assert settings.max_workers == 8
    assert settings.use_gpu is True


def test_v2_settings_validation():
    """Test V2 settings validation works"""
    os.environ["USE_SETTINGS_V2"] = "true"
    os.environ["DATABASE_URL"] = "postgresql://test/db"
    os.environ["JWT_SECRET_KEY"] = "test-secret"

    from montage.settings_v2 import ProcessingConfig

    # Test validation on individual config
    config = ProcessingConfig(max_workers=20)
    assert config.max_workers == 20

    # Test validation limits
    with pytest.raises(ValueError):
        ProcessingConfig(max_workers=100)  # Above max of 32

    with pytest.raises(ValueError):
        ProcessingConfig(max_cost_usd=-1.0)  # Below min of 0.0


def test_config_lazy_loading():
    """Test that config uses lazy loading"""
    os.environ["USE_SETTINGS_V2"] = "true"
    os.environ["DATABASE_URL"] = "postgresql://test/db"
    os.environ["JWT_SECRET_KEY"] = "test-secret"

    from montage.config import settings, _SettingsProxy

    # Should be proxy before access
    assert isinstance(settings, _SettingsProxy)

    # Accessing attribute triggers load
    db_url = settings.database_url
    assert db_url == "postgresql://test/db"


def test_config_reload(monkeypatch):
    """Test config reload functionality"""
    monkeypatch.setenv("USE_SETTINGS_V2", "true")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test/db")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("MAX_WORKERS", "4")

    # Clear any cached settings first
    from montage.config import _load_settings
    _load_settings.cache_clear()

    from montage.config import settings, reload_settings

    # Initial value
    assert settings.max_workers == 4

    # Change env var
    monkeypatch.setenv("MAX_WORKERS", "8")

    # Reload
    new_settings = reload_settings()
    assert new_settings.max_workers == 8
