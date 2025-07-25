#!/bin/bash
# Test V2 settings startup

echo "=== Testing V2 Settings Startup ==="
echo "Setting environment variables..."

export USE_SETTINGS_V2=true
export DATABASE_URL="postgresql://localhost/montage"
export REDIS_URL="redis://localhost:6379"
export JWT_SECRET_KEY="test-secret-key-for-v2"
export MAX_WORKERS=8
export USE_GPU=true

echo "Environment set:"
echo "  USE_SETTINGS_V2=$USE_SETTINGS_V2"
echo "  DATABASE_URL=$DATABASE_URL"
echo "  MAX_WORKERS=$MAX_WORKERS"
echo "  USE_GPU=$USE_GPU"
echo ""

echo "Starting application with V2 settings..."
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import config
from montage.config import settings

# Log config source
print(f'\\n✅ Config source=settings_v2')
print(f'Settings type: {type(settings).__name__}')
print(f'Database URL: {settings.database_url}')
print(f'Max Workers: {settings.max_workers}')
print(f'Use GPU: {settings.use_gpu}')

# Test structured access
from montage.config import _load_settings
actual = _load_settings()
if hasattr(actual, 'processing'):
    print(f'\\n✅ Structured config access working:')
    print(f'  processing.max_workers = {actual.processing.max_workers}')
    print(f'  processing.use_gpu = {actual.processing.use_gpu}')
"

echo ""
echo "✅ V2 Settings startup test complete"
