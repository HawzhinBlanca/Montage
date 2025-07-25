"""Database pool metrics collection for monitoring"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def get_pool_stats(engine) -> Dict[str, Any]:
    """
    Get current pool statistics from SQLAlchemy engine

    Returns:
        Dict with pool metrics: size, checked_in, checked_out, overflow
    """
    try:
        pool = engine.pool
        return {
            "size": pool.size(),                          # total slots
            "checked_in": pool.checkedin(),              # available connections
            "checked_out": pool.checkedout(),            # in-use connections
            "overflow": pool.overflow(),                  # connections beyond pool_size
            "total": pool.size() + pool.overflow(),      # total possible connections
            "utilization": pool.checkedout() / pool.size() if pool.size() > 0 else 0
        }
    except AttributeError:
        # For async engines, pool might be accessed differently
        logger.warning("Pool metrics not available for this engine type")
        return {
            "size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
            "total": 0,
            "utilization": 0
        }
    except Exception as e:
        logger.error(f"Error collecting pool stats: {e}")
        return {
            "size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
            "total": 0,
            "utilization": 0,
            "error": str(e)
        }


def get_connection_stats(engine) -> Dict[str, Any]:
    """
    Get connection statistics including query performance

    Returns:
        Dict with connection and performance metrics
    """
    pool_stats = get_pool_stats(engine)

    return {
        "pool": pool_stats,
        "config": {
            "pool_size": getattr(engine.pool, '_pool_size', 0),
            "max_overflow": getattr(engine.pool, '_max_overflow', 0),
            "timeout": getattr(engine.pool, '_timeout', 30),
            "recycle": getattr(engine.pool, '_recycle', 3600)
        }
    }
