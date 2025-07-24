"""
Cost Guard - Tasks.md Step 0
Hard cap MAX_COST_VIDEO_USD=0.05 decorator with Redis tracking
"""
import functools
import logging
import time
from decimal import Decimal
from typing import Any, Callable, Dict

import redis

logger = logging.getLogger(__name__)

class CostGuard:
    """Cost guard with hard cap enforcement"""

    MAX_COST_VIDEO_USD = 0.05  # Tasks.md specification: $0.05 hard cap

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def enforce_cost_cap(self, service: str) -> Callable:
        """
        Decorator to enforce cost cap on API calls
        Tracks spending per job_id and enforces MAX_COST_VIDEO_USD limit
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Extract job_id from function arguments
                job_id = self._extract_job_id(*args, **kwargs)
                if not job_id:
                    logger.warning(f"No job_id found for cost-tracked function {func.__name__}")
                    return func(*args, **kwargs)

                # Check current spending for this job
                current_cost = self._get_job_cost(job_id)

                if current_cost >= self.MAX_COST_VIDEO_USD:
                    error_msg = f"Cost cap exceeded for job {job_id}: ${current_cost:.4f} >= ${self.MAX_COST_VIDEO_USD}"
                    logger.error(error_msg)
                    raise CostCapExceededError(error_msg, job_id, current_cost)

                # Execute function and track cost
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)

                    # If function returned cost info, track it
                    if isinstance(result, dict) and 'cost_usd' in result:
                        cost = Decimal(str(result['cost_usd']))
                        self._add_job_cost(job_id, cost, service)

                        # Check if we exceeded cap after this call
                        new_total = self._get_job_cost(job_id)
                        if new_total > self.MAX_COST_VIDEO_USD:
                            logger.warning(f"Job {job_id} exceeded cost cap: ${new_total:.4f}")

                    return result

                except Exception as e:
                    # Track failed API call costs if available
                    elapsed = time.time() - start_time
                    logger.error(f"Cost-tracked function {func.__name__} failed after {elapsed:.1f}s: {e}")
                    raise

            return wrapper
        return decorator

    def _extract_job_id(self, *args, **kwargs) -> str:
        """Extract job_id from function arguments"""
        # Check kwargs first
        if 'job_id' in kwargs:
            return kwargs['job_id']

        # Check positional args - common patterns
        for arg in args:
            if isinstance(arg, str) and (len(arg) == 36 or len(arg) == 32):  # UUID patterns
                return arg
            if isinstance(arg, dict) and 'job_id' in arg:
                return arg['job_id']

        return None

    def _get_job_cost(self, job_id: str) -> float:
        """Get current cost for job_id"""
        cost_key = f"cost:{job_id}"
        cost_str = self.redis.get(cost_key)

        if cost_str:
            return float(cost_str)
        return 0.0

    def _add_job_cost(self, job_id: str, cost: Decimal, service: str) -> None:
        """Add cost to job tracking"""
        cost_key = f"cost:{job_id}"

        # Atomic increment with expiry
        pipe = self.redis.pipeline()
        pipe.incrbyfloat(cost_key, float(cost))
        pipe.expire(cost_key, 86400)  # 24 hour expiry
        pipe.execute()

        # Track service breakdown
        service_key = f"cost:{job_id}:{service}"
        pipe = self.redis.pipeline()
        pipe.incrbyfloat(service_key, float(cost))
        pipe.expire(service_key, 86400)
        pipe.execute()

        # Log cost tracking
        total_cost = self._get_job_cost(job_id)
        logger.info(f"Job {job_id}: +${cost:.4f} ({service}) = ${total_cost:.4f} total")

    def get_job_cost_breakdown(self, job_id: str) -> Dict[str, float]:
        """Get detailed cost breakdown for job"""
        pattern = f"cost:{job_id}:*"
        service_keys = self.redis.keys(pattern)

        breakdown = {
            'total': self._get_job_cost(job_id),
            'services': {}
        }

        for key in service_keys:
            service = key.decode().split(':')[-1]  # Last part after cost:job_id:
            cost = float(self.redis.get(key) or 0)
            breakdown['services'][service] = cost

        return breakdown

    def reset_job_cost(self, job_id: str) -> None:
        """Reset cost tracking for job (admin function)"""
        pattern = f"cost:{job_id}*"
        keys = self.redis.keys(pattern)

        if keys:
            self.redis.delete(*keys)
            logger.info(f"Reset cost tracking for job {job_id}")

    def track_cost(self, service: str, cost_usd: float, job_id: str) -> None:
        """Track cost for a service call - used by test suite"""
        cost_decimal = Decimal(str(cost_usd))
        self._add_job_cost(job_id, cost_decimal, service)

    def get_current_cost(self, service_or_job_id: str) -> float:
        """Get current cost for service or job - used by test suite"""
        # If it looks like a job ID, get job cost
        if len(service_or_job_id) > 10:  # Assume job IDs are longer
            return self._get_job_cost(service_or_job_id)
        else:
            # For service, we'd need a specific job_id, return 0 for now
            return 0.0

    def check_budget(self, service_or_job_id: str, proposed_cost: float) -> bool:
        """Check if proposed cost would exceed budget - used by test suite"""
        current_cost = self.get_current_cost(service_or_job_id)
        return (current_cost + proposed_cost) <= self.MAX_COST_VIDEO_USD


class CostCapExceededError(Exception):
    """Exception raised when cost cap is exceeded"""

    def __init__(self, message: str, job_id: str, current_cost: float):
        super().__init__(message)
        self.job_id = job_id
        self.current_cost = current_cost


# Convenience decorator for common usage
def cost_capped(service: str, redis_client: redis.Redis = None):
    """Convenience decorator using default CostGuard instance"""
    if redis_client is None:
        redis_client = redis.Redis(decode_responses=True)

    guard = CostGuard(redis_client)
    return guard.enforce_cost_cap(service)
