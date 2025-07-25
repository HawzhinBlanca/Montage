#!/bin/bash
# Rollback database changes and dispose of connections

set -euo pipefail

echo "Rolling back database changes..."

# Disable async pool if enabled
export USE_ASYNC_POOL=false

# Check if alembic is available
if command -v alembic &> /dev/null; then
    echo "Rolling back last migration..."
    alembic downgrade -1
else
    echo "Alembic not found, skipping migration rollback"
fi

# Dispose of any active connections
echo "Disposing database connections..."
python - <<'PY'
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from montage.core.db import engine

    async def dispose_connections():
        """Dispose of all database connections"""
        if hasattr(engine, 'dispose'):
            await engine.dispose()
            print("Database connections disposed successfully")
        else:
            print("No async engine found to dispose")

    # Run disposal
    asyncio.run(dispose_connections())

except ImportError as e:
    print(f"Could not import database engine: {e}")
    print("Skipping connection disposal")
except Exception as e:
    print(f"Error disposing connections: {e}")
    sys.exit(1)
PY

echo "Rollback complete"
