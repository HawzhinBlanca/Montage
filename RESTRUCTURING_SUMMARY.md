# Project Restructuring Summary

## Files Moved to src/core/
- checkpoint.py → src/core/checkpoint.py
- db_secure.py → src/core/db.py (renamed)
- metrics.py → src/core/metrics.py
- user_success_metrics.py → src/core/user_success_metrics.py
- adaptive_quality_pipeline_master.py → src/core/adaptive_quality_pipeline_master.py
- smart_video_editor.py → src/core/smart_video_editor.py

## Files Moved to src/utils/
- video_validator.py → src/utils/video_validator.py
- retry_utils.py → src/utils/retry_utils.py
- cleanup_manager.py → src/utils/cleanup_manager.py
- monitoring_integration.py → src/utils/monitoring_integration.py
- budget_guard.py → src/utils/budget_guard.py

## Files Moved to src/providers/
- video_processor.py → src/providers/video_processor.py
- video_probe.py → src/providers/video_probe.py
- audio_normalizer.py → src/providers/audio_normalizer.py
- concat_editor.py → src/providers/concat_editor.py
- progressive_renderer.py → src/providers/progressive_renderer.py
- selective_enhancer.py → src/providers/selective_enhancer.py
- smart_crop.py → src/providers/smart_crop.py
- smart_track.py → src/providers/smart_track.py
- speaker_diarizer.py → src/providers/speaker_diarizer.py
- transcript_analyzer.py → src/providers/transcript_analyzer.py

## Files Kept in Root
- main.py (main entry point)
- run_montage.py (CLI wrapper)
- migrate.py (database migration tool)
- color_converter.py (to be decided - has imports from both src.core.metrics and legacy_config)
- legacy_config.py (renamed from config.py to avoid conflicts with src/config.py)

## Import Updates Made
- Updated all test files to use new import paths (e.g., `from src.core.checkpoint import ...`)
- Updated internal imports in moved files to use relative imports
- Updated main.py to use new import paths
- Updated example files to use new import paths
- Added sys.path configuration to tests/conftest.py for proper module resolution
- Renamed root config.py to legacy_config.py to avoid conflicts with src/config.py

## Notes
- The database connection errors in tests are expected since PostgreSQL is not running
- All imports have been updated to reflect the new structure
- The project now has a cleaner organization with business logic in core/, utilities in utils/, and external integrations in providers/