# Phase 3: Config Unification - IN PROGRESS

## Completed Tasks

### 1. ✅ Phase 3-1 Adjustments (100% Complete)
- **Lazy-load proxy**: Implemented in `montage/config.py` to prevent import-time crashes
- **Secret fallback**: Legacy adapter now checks env vars before secret loader
- **Stable tests**: Created `tests/test_config_toggle.py` with 2 passing tests
- **Runtime reload**: Added `reload_settings()` function for hot-reload capability

### 2. ✅ Enhanced settings_v2.py with Pydantic BaseSettings
- Replaced dataclass with proper Pydantic BaseSettings
- Created structured configuration models:
  - `DatabaseConfig`: Database settings with validation
  - `RedisConfig`: Redis settings with TTL
  - `APIKeysConfig`: API keys as SecretStr
  - `SecurityConfig`: JWT and security settings
  - `ProcessingConfig`: GPU, workers, cost limits with validation
- Added legacy compatibility properties for backward compatibility
- Implemented proper env var parsing with `env_nested_delimiter="__"`

### 3. ✅ Testing with USE_SETTINGS_V2=true
```bash
$ DATABASE_URL=postgresql://test/db JWT_SECRET_KEY=test-secret USE_SETTINGS_V2=true python -c "from montage.config import settings; print(f'V2 Settings loaded: {settings.database_url}')"
V2 Settings loaded: postgresql://test/db
```

## Current Status

### Configuration System Architecture:
1. **montage/config.py**: Lazy-load proxy that routes to V2 or legacy based on USE_SETTINGS_V2
2. **montage/settings_v2.py**: New Pydantic-based settings with validation and structure
3. **montage/legacy_adapter.py**: Backward compatibility for legacy code
4. **montage/settings.py**: Existing Pydantic settings (will be replaced by settings_v2.py)

### Key Features Implemented:
- ✅ Type-safe configuration with Pydantic v2
- ✅ Environment variable support with fallbacks
- ✅ Validation with min/max constraints
- ✅ SecretStr for sensitive values
- ✅ Nested configuration structure
- ✅ Legacy compatibility properties
- ✅ Hot-reload capability
- ✅ Lazy loading to prevent import-time issues

## Next Steps

### 1. Deploy 24-hour Canary (Phase 3 requirement)
- Deploy with USE_SETTINGS_V2=true to staging
- Monitor for 24 hours
- Collect metrics:
  - Startup time
  - Memory usage
  - Configuration errors
  - Application functionality

### 2. Success Criteria for Phase 3 Completion
- ✅ settings_v2.py implemented with Pydantic
- ✅ USE_SETTINGS_V2 flag working
- ⏳ 24-hour canary deployment successful
- ⏳ All SLOs green during canary
- ⏳ Startup log shows "Config source=settings_v2"
- ⏳ Delete legacy loaders after validation

## Technical Details

### V2 Settings Structure:
```python
Settings(
    database=DatabaseConfig(...),
    redis=RedisConfig(...),
    api_keys=APIKeysConfig(...),
    security=SecurityConfig(...),
    processing=ProcessingConfig(...)
)
```

### Environment Variable Mapping:
- Legacy: `MAX_WORKERS=8`
- Nested: `PROCESSING__MAX_WORKERS=8` (also supported)

### Validation Examples:
- `max_workers`: Range 1-32
- `max_cost_usd`: Minimum 0.0
- `pool_size`: Range 1-100

## Rollback Plan
If issues arise during 24-hour canary:
1. Set `USE_SETTINGS_V2=false` immediately
2. Redeploy with legacy settings
3. No code changes required - just environment variable