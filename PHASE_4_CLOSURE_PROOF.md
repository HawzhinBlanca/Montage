# Phase 4 Closure Proof Bundle

Generated: 2025-07-25

## Required Changes for Phase 4 Closure

As per the context, Phase 4 cannot close until we decouple import-time DB & Celery connections.

### ✅ 1. Decoupled Import-time DB & Celery

**Before:**
```python
from ..core.db import Database
from .celery_app import process_video_task

# Initialize database
db = Database()  # ← triggers connect at import
```

**After:**
```python
# Lazy-load dependencies to avoid import-time side effects
def get_db():
    """Get database connection lazily"""
    from ..core.db import Database
    return Database()

def get_celery():
    """Get Celery task lazily"""
    from .celery_app import process_video_task
    return process_video_task
```

### ✅ 2. Updated Handlers with Dependency Injection

All handlers now use `Depends(get_db)` and `Depends(get_celery)`:

- `health_check`: `db = Depends(get_db)`
- `process_video`: `db = Depends(get_db), process_video_task = Depends(get_celery)`
- `get_job_status`: `db = Depends(get_db)`
- `download_result`: `db = Depends(get_db)`
- `get_metrics`: `db = Depends(get_db)`

### ✅ 3. Test Results

**tests/test_lazy_load.py**: 5 tests passed
- ✅ `test_import_web_server`: Import without side effects
- ✅ `test_get_db_lazy`: DB function is lazy
- ✅ `test_get_celery_lazy`: Celery function is lazy
- ✅ `test_health_endpoint_exists`: Endpoint registered
- ✅ `test_no_import_side_effects`: No global db object

### ✅ 4. Verification

```bash
grep -R "get_pool(" montage/api | wc -l
# Result: 0 (no direct DB pool calls)
```

## Files in Proof Bundle

1. **diff_lazy_db.patch** - Shows all code changes
2. **pytest_summary.txt** - 25 tests passed
3. **stub_scan.out** - 0 direct DB calls
4. **coverage_report.txt** - Unchanged from Phase 3

## Conclusion

Phase 4 is now ready to close:
- ✅ Import-time side effects removed
- ✅ All endpoints use dependency injection
- ✅ Tests verify lazy loading works
- ✅ ADR decision documented (Keep Separate)
- ✅ No code changes needed for Keep Separate decision

The failing test harness issue has been resolved by implementing proper lazy loading patterns.