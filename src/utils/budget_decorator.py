#!/usr/bin/env python3
"""
Budget tracking decorator for API calls - enforces MAX_COST_USD hard limit
"""
import functools
import time
import threading
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field

from ..config import MAX_COST_USD


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded"""

    pass


@dataclass
class BudgetTracker:
    """Thread-safe budget tracking"""

    max_budget: float = MAX_COST_USD
    total_spent: float = 0.0
    call_history: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_cost(self, amount: float, api_name: str, operation: str) -> None:
        """Add cost to budget tracker"""
        with self._lock:
            if self.total_spent + amount > self.max_budget:
                raise BudgetExceededError(
                    f"Budget exceeded: ${self.total_spent + amount:.3f} > ${self.max_budget:.3f}"
                )

            self.total_spent += amount
            self.call_history.append(
                {
                    "timestamp": time.time(),
                    "amount": amount,
                    "api_name": api_name,
                    "operation": operation,
                    "total_spent": self.total_spent,
                }
            )

    def get_remaining_budget(self) -> float:
        """Get remaining budget"""
        with self._lock:
            return self.max_budget - self.total_spent

    def get_usage_report(self) -> Dict[str, Any]:
        """Get budget usage report"""
        with self._lock:
            return {
                "max_budget": self.max_budget,
                "total_spent": self.total_spent,
                "remaining": self.max_budget - self.total_spent,
                "call_count": len(self.call_history),
                "last_call": self.call_history[-1] if self.call_history else None,
            }


# Global budget tracker instance
budget_tracker = BudgetTracker()


def priced(api_name: str, operation: str, cost_calculator: Optional[Callable] = None):
    """
    Decorator to track API costs and enforce budget limits

    Args:
        api_name: Name of the API (e.g., 'openai', 'anthropic', 'deepgram')
        operation: Operation being performed (e.g., 'transcribe', 'analyze')
        cost_calculator: Optional function to calculate cost from result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-check budget
            if budget_tracker.total_spent >= budget_tracker.max_budget:
                raise BudgetExceededError(
                    f"Budget already exceeded: ${budget_tracker.total_spent:.3f} >= ${budget_tracker.max_budget:.3f}"
                )

            # Execute function
            try:
                result = func(*args, **kwargs)

                # Calculate cost
                if cost_calculator:
                    cost = cost_calculator(result, *args, **kwargs)
                elif hasattr(result, "usage") and hasattr(result.usage, "total_tokens"):
                    # OpenAI-style usage tracking
                    tokens = result.usage.total_tokens
                    if api_name == "openai":
                        cost = tokens * 0.00015 / 1000  # GPT-4o-mini pricing
                    else:
                        cost = tokens * 0.001 / 1000  # Generic pricing
                elif isinstance(result, dict) and "cost" in result:
                    # Result contains cost field
                    cost = result["cost"]
                else:
                    # Default cost estimation
                    cost = 0.0

                # Track cost
                if cost > 0:
                    budget_tracker.add_cost(cost, api_name, operation)

                return result

            except BudgetExceededError:
                # Re-raise budget errors
                raise
            except Exception as e:
                # Log error but don't track cost for failed calls
                print(f"❌ API call failed ({api_name}/{operation}): {e}")
                raise

        return wrapper

    return decorator


def get_budget_status() -> Dict[str, Any]:
    """Get current budget status"""
    return budget_tracker.get_usage_report()


def reset_budget() -> None:
    """Reset budget tracker (for testing)"""
    global budget_tracker
    budget_tracker = BudgetTracker()


# Cost calculation functions for different APIs
def calculate_deepgram_cost(result: Any, audio_path: str, *args, **kwargs) -> float:
    """Calculate Deepgram cost based on audio duration"""
    try:
        import ffmpeg

        probe = ffmpeg.probe(audio_path)
        duration = float(probe["format"]["duration"])
        return duration * 0.0125 / 60  # $0.0125 per minute
    except Exception:
        return 0.05  # Default cost


def calculate_anthropic_cost(result: Any, *args, **kwargs) -> float:
    """Calculate Anthropic cost based on usage"""
    try:
        if hasattr(result, "usage"):
            input_tokens = result.usage.input_tokens
            output_tokens = result.usage.output_tokens
            # Claude pricing: $0.003 per 1K input tokens, $0.015 per 1K output tokens
            return (input_tokens * 0.003 + output_tokens * 0.015) / 1000
        return 0.0
    except Exception:
        return 0.01  # Default cost


def calculate_openai_cost(result: Any, *args, **kwargs) -> float:
    """Calculate OpenAI cost based on usage"""
    try:
        if hasattr(result, "usage"):
            prompt_tokens = result.usage.prompt_tokens
            completion_tokens = result.usage.completion_tokens
            # GPT-4o-mini pricing: $0.00015 per 1K input, $0.0006 per 1K output
            return (prompt_tokens * 0.00015 + completion_tokens * 0.0006) / 1000
        return 0.0
    except Exception:
        return 0.01  # Default cost
