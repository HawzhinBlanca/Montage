"""Budget control decorator and utilities for API cost management"""

import logging
import functools
import time
from typing import Callable, Any, Dict, Optional
from threading import Lock
from datetime import datetime, timedelta

from config import Config
from db import Database
from metrics import metrics

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when API costs exceed budget limit"""
    pass


class BudgetManager:
    """Manages API cost tracking and budget enforcement"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.db = Database()
        self.budget_limit = Config.BUDGET_LIMIT
        self._job_costs = {}  # Cache for job costs
        self._cost_lock = Lock()
        self._initialized = True
        
        logger.info(f"Budget manager initialized with limit: ${self.budget_limit}")
    
    def track_cost(self, job_id: str, api_name: str, cost_usd: float) -> None:
        """Track API cost for a job"""
        with self._cost_lock:
            # Update cache
            if job_id not in self._job_costs:
                self._job_costs[job_id] = 0.0
            self._job_costs[job_id] += cost_usd
            
            # Store in database
            self.db.insert('api_cost_log', {
                'job_id': job_id,
                'api_name': api_name,
                'cost_usd': cost_usd
            })
            
            # Update job total
            self.db.execute(
                """
                UPDATE video_job 
                SET total_cost = COALESCE(total_cost, 0) + %s 
                WHERE id = %s
                """,
                (cost_usd, job_id)
            )
            
            # Track in metrics
            metrics.track_cost(api_name, cost_usd, job_id)
            
            logger.info(f"API cost tracked: {api_name} = ${cost_usd:.4f} for job {job_id}")
    
    def check_budget(self, job_id: str, additional_cost: float = 0) -> Tuple[bool, float]:
        """
        Check if job is within budget.
        
        Returns:
            (is_within_budget, current_total_cost)
        """
        with self._cost_lock:
            # Get current cost from cache or database
            if job_id in self._job_costs:
                current_cost = self._job_costs[job_id]
            else:
                # Load from database
                job = self.db.find_one('video_job', {'id': job_id})
                current_cost = float(job.get('total_cost', 0)) if job else 0.0
                self._job_costs[job_id] = current_cost
            
            projected_cost = current_cost + additional_cost
            is_within_budget = projected_cost <= self.budget_limit
            
            # Update metrics
            remaining = max(0, self.budget_limit - current_cost)
            metrics.set_budget_remaining(job_id, remaining)
            
            return is_within_budget, current_cost
    
    def get_job_cost_breakdown(self, job_id: str) -> Dict[str, float]:
        """Get detailed cost breakdown for a job"""
        costs = self.db.execute(
            """
            SELECT api_name, SUM(cost_usd) as total
            FROM api_cost_log
            WHERE job_id = %s
            GROUP BY api_name
            ORDER BY total DESC
            """,
            (job_id,)
        )
        
        breakdown = {row['api_name']: float(row['total']) for row in costs}
        breakdown['total'] = sum(breakdown.values())
        
        return breakdown
    
    def get_hourly_cost_rate(self) -> float:
        """Get current cost rate per hour"""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        result = self.db.execute(
            """
            SELECT SUM(cost_usd) as total
            FROM api_cost_log
            WHERE created_at >= %s
            """,
            (one_hour_ago,)
        )
        
        return float(result[0]['total']) if result and result[0]['total'] else 0.0
    
    def reset_job_cost_cache(self, job_id: str) -> None:
        """Reset cached cost for a job"""
        with self._cost_lock:
            if job_id in self._job_costs:
                del self._job_costs[job_id]


# Singleton instance
budget_manager = BudgetManager()


def priced(api_name: str, cost_calculator: Callable[[Any], float]):
    """
    Decorator that tracks API costs and enforces budget limits.
    
    Args:
        api_name: Name of the API being called
        cost_calculator: Function that calculates cost from the result
        
    Example:
        @priced('openai', lambda result: 0.02 * len(result['tokens']) / 1000)
        def call_gpt(prompt):
            return openai.complete(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract job_id from arguments
            job_id = kwargs.get('job_id', 'unknown')
            
            # For methods, check if first arg has job_id
            if not job_id or job_id == 'unknown':
                if args and hasattr(args[0], 'job_id'):
                    job_id = args[0].job_id
                elif len(args) > 1 and isinstance(args[1], str):
                    # Assume second arg might be job_id
                    job_id = args[1]
            
            # Pre-check: Ensure we're not already over budget
            is_within_budget, current_cost = budget_manager.check_budget(job_id)
            if not is_within_budget:
                error_msg = (
                    f"Budget exceeded for job {job_id}: "
                    f"${current_cost:.2f} >= ${budget_manager.budget_limit:.2f}"
                )
                logger.error(error_msg)
                metrics.budget_exceeded_total.inc()
                raise BudgetExceededError(error_msg)
            
            # Track API call
            metrics.track_api_call(api_name, 'attempt')
            
            start_time = time.time()
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate cost
                try:
                    cost = cost_calculator(result)
                except Exception as e:
                    logger.error(f"Failed to calculate cost for {api_name}: {e}")
                    cost = 0.0
                
                # Track cost
                if cost > 0:
                    budget_manager.track_cost(job_id, api_name, cost)
                    
                    # Post-check: Ensure we haven't exceeded budget
                    is_within_budget, new_total = budget_manager.check_budget(job_id)
                    if not is_within_budget:
                        logger.warning(
                            f"Job {job_id} exceeded budget after {api_name} call: "
                            f"${new_total:.2f} > ${budget_manager.budget_limit:.2f}"
                        )
                        metrics.budget_exceeded_total.inc()
                
                # Track successful API call
                metrics.track_api_call(api_name, 'success')
                
                return result
                
            except BudgetExceededError:
                # Re-raise budget errors
                metrics.track_api_call(api_name, 'budget_exceeded')
                raise
                
            except Exception as e:
                # Track failed API call
                elapsed = time.time() - start_time
                
                if 'rate limit' in str(e).lower():
                    metrics.track_api_call(api_name, 'rate_limited')
                    metrics.track_rate_limit_backoff(api_name, elapsed)
                else:
                    metrics.track_api_call(api_name, 'error')
                
                raise
        
        return wrapper
    return decorator


# Utility functions

def estimate_cost(api_name: str, **params) -> float:
    """Estimate cost for an API call before making it"""
    
    estimators = {
        'openai': lambda: params.get('tokens', 1000) * 0.02 / 1000,
        'whisper': lambda: params.get('duration', 60) * 0.006 / 60,
        'gpt-4': lambda: params.get('tokens', 1000) * 0.06 / 1000,
        'dalle': lambda: params.get('size', '1024x1024') == '1024x1024' and 0.02 or 0.018
    }
    
    estimator = estimators.get(api_name)
    if estimator:
        return estimator()
    
    return 0.0


def check_budget_before_call(job_id: str, api_name: str, **params) -> None:
    """
    Pre-flight budget check before making an API call.
    
    Raises:
        BudgetExceededError: If estimated cost would exceed budget
    """
    estimated_cost = estimate_cost(api_name, **params)
    is_within_budget, current_cost = budget_manager.check_budget(job_id, estimated_cost)
    
    if not is_within_budget:
        projected_total = current_cost + estimated_cost
        raise BudgetExceededError(
            f"Estimated cost would exceed budget for job {job_id}: "
            f"${projected_total:.2f} > ${budget_manager.budget_limit:.2f}"
        )


# Example cost calculators for common APIs

def openai_completion_cost(response: Dict) -> float:
    """Calculate cost for OpenAI completion API"""
    usage = response.get('usage', {})
    total_tokens = usage.get('total_tokens', 0)
    
    # GPT-3.5-turbo pricing
    cost_per_1k = 0.002  # $0.002 per 1K tokens
    
    return (total_tokens / 1000) * cost_per_1k


def whisper_transcription_cost(response: Dict) -> float:
    """Calculate cost for Whisper transcription"""
    duration = response.get('duration', 0)  # in seconds
    
    # Whisper API pricing
    cost_per_minute = 0.006
    
    return (duration / 60) * cost_per_minute


def gpt4_completion_cost(response: Dict) -> float:
    """Calculate cost for GPT-4 API"""
    usage = response.get('usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    
    # GPT-4 pricing (8K context)
    prompt_cost_per_1k = 0.03
    completion_cost_per_1k = 0.06
    
    prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (completion_tokens / 1000) * completion_cost_per_1k
    
    return prompt_cost + completion_cost


# Integration with job processing

def with_budget_tracking(job_id: str):
    """Context manager for budget tracking within a job"""
    
    class BudgetContext:
        def __init__(self, job_id):
            self.job_id = job_id
            self.start_cost = 0
            
        def __enter__(self):
            _, self.start_cost = budget_manager.check_budget(self.job_id)
            logger.info(f"Starting job {self.job_id} with cost: ${self.start_cost:.2f}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            _, end_cost = budget_manager.check_budget(self.job_id)
            job_cost = end_cost - self.start_cost
            
            logger.info(f"Job {self.job_id} cost: ${job_cost:.2f} (total: ${end_cost:.2f})")
            
            if exc_type is BudgetExceededError:
                logger.error(f"Job {self.job_id} terminated due to budget exceeded")
                # Don't suppress the exception
                return False
    
    return BudgetContext(job_id)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example API call with budget tracking
    @priced('openai', openai_completion_cost)
    def call_gpt(prompt: str, job_id: str = 'test-job'):
        # Simulate API response
        return {
            'choices': [{'text': 'AI response here'}],
            'usage': {'total_tokens': 150}
        }
    
    # Use the decorated function
    try:
        with with_budget_tracking('test-job-001'):
            response = call_gpt("Hello AI", job_id='test-job-001')
            print(f"Response received")
            
            # Check remaining budget
            _, current_cost = budget_manager.check_budget('test-job-001')
            remaining = Config.BUDGET_LIMIT - current_cost
            print(f"Remaining budget: ${remaining:.2f}")
            
    except BudgetExceededError as e:
        print(f"Budget exceeded: {e}")