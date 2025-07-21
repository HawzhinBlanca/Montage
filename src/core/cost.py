#!/usr/bin/env python3
"""
Budget tracking that ACTUALLY blocks spending
"""
from decimal import Decimal
from functools import wraps
from prometheus_client import Counter
import logging

logger = logging.getLogger(__name__)

# Global cost tracker - reset per job
_API_COST = Decimal("0")
_BUDGET_HARD_CAP = Decimal("5.00")  # USD, per-job

# Prometheus metric
api_cost_total = Counter("api_cost_usd_total", "Cumulative $ spent", ["job", "service"])


def priced(service: str, per_unit_usd: Decimal):
    """Decorator that enforces budget cap before API calls"""

    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            global _API_COST

            # Extract job_id and units from kwargs or use defaults
            job_id = kwargs.pop("job_id", "default")
            units = kwargs.pop("units", 1)

            cost = per_unit_usd * units

            # HARD STOP if budget exceeded
            if _API_COST + cost > _BUDGET_HARD_CAP:
                logger.error(
                    f"BUDGET CAP HIT: Current ${_API_COST} + ${cost} > ${_BUDGET_HARD_CAP}"
                )
                raise RuntimeError(
                    f"Budget cap ${_BUDGET_HARD_CAP} hit - blocking {service} call"
                )

            # Make the actual API call
            try:
                result = fn(*args, **kwargs)

                # Only charge if successful
                _API_COST += cost
                api_cost_total.labels(job=job_id, service=service).inc(float(cost))
                logger.info(f"API call to {service}: ${cost} (total: ${_API_COST})")

                return result

            except Exception as e:
                logger.error(f"API call to {service} failed: {e}")
                raise

        return wrapper

    return deco


def get_current_cost() -> float:
    """Get current API spend"""
    return float(_API_COST)


def reset_cost():
    """Reset cost tracker for new job"""
    global _API_COST
    _API_COST = Decimal("0")
    logger.info("Cost tracker reset")


def check_budget(additional_cost: float = 0) -> tuple[bool, float]:
    """Check if budget allows additional spend"""
    current = float(_API_COST)
    remaining = float(_BUDGET_HARD_CAP) - current - additional_cost
    within_budget = remaining >= 0
    return within_budget, remaining


# Missing classes and functions for compatibility
class BudgetExceededError(Exception):
    """Raised when budget cap is exceeded"""

    pass


class BudgetManager:
    """Budget management singleton"""

    @staticmethod
    def get_current_cost() -> float:
        return get_current_cost()

    @staticmethod
    def reset_cost():
        return reset_cost()

    @staticmethod
    def check_budget(additional_cost: float = 0) -> tuple[bool, float]:
        return check_budget(additional_cost)

    @staticmethod
    def enforce_budget(cost: float):
        """Enforce budget before spending"""
        if _API_COST + Decimal(str(cost)) > _BUDGET_HARD_CAP:
            raise BudgetExceededError(f"Budget cap ${_BUDGET_HARD_CAP} exceeded")


# Create singleton instance
budget_manager = BudgetManager()

# Alias for compatibility
with_budget_tracking = priced


# Simple retry decorator for missing retry_utils
def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """Simple retry decorator with exponential backoff"""
    import time
    import random

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e

                    # Exponential backoff with jitter
                    time.sleep(delay + random.uniform(0, 0.1))
                    delay *= 2
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                    )

        return wrapper

    return decorator
