#!/usr/bin/env python3
"""
Comprehensive rate limiting system to prevent API cost explosions in production.

Features:
1. Token Bucket Rate Limiter with burst capability
2. Per-API Service Limits with different configurations
3. Cost-Aware Rate Limiting based on budget constraints
4. Circuit Breaker pattern for API failure resilience
5. Request Queuing with priority levels
6. Adaptive Limits based on success rates
7. Production-ready monitoring and metrics
"""

import asyncio
import time
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Callable, Any, List, Tuple
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge
import queue

# Import cost tracking
try:
    from .cost import get_current_cost, check_budget, BudgetExceededError
except ImportError:
    # Fallback for testing
    def get_current_cost() -> float:
        return 0.0

    def check_budget(additional_cost: float = 0) -> tuple[bool, float]:
        return True, 100.0

    class BudgetExceededError(Exception):
        pass


logger = logging.getLogger(__name__)

# Prometheus Metrics
rate_limit_requests = Counter(
    "rate_limit_requests_total", "Total rate limit requests", ["service", "status"]
)
rate_limit_queue_size = Gauge(
    "rate_limit_queue_size", "Current queue size", ["service", "priority"]
)
rate_limit_wait_time = Histogram(
    "rate_limit_wait_seconds", "Time spent waiting for rate limit", ["service"]
)
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"],
)
api_success_rate = Gauge(
    "api_success_rate", "API success rate over time window", ["service"]
)


class Priority(Enum):
    """Request priority levels"""

    CRITICAL = 0  # Critical requests - highest priority
    HIGH = 1  # High priority requests
    NORMAL = 2  # Normal priority requests
    LOW = 3  # Low priority requests - can be delayed/dropped


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = 0  # Normal operation
    OPEN = 1  # Circuit is open - requests fail fast
    HALF_OPEN = 2  # Testing if service is recovered


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting per service"""

    # Token bucket parameters
    max_tokens: int = 10  # Maximum tokens in bucket
    refill_rate: float = 1.0  # Tokens added per second
    burst_capacity: int = 5  # Additional burst tokens

    # Cost-based limiting
    max_cost_per_minute: Decimal = Decimal("1.00")  # Max spend per minute
    cost_per_request: Decimal = Decimal("0.01")  # Estimated cost per request

    # Circuit breaker parameters
    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: int = 30  # Seconds before attempting recovery
    success_threshold: int = 3  # Successes needed to close circuit

    # Queue parameters
    max_queue_size: int = 100  # Maximum queued requests
    request_timeout: int = 300  # Request timeout in seconds

    # Adaptive parameters
    min_interval: float = 0.1  # Minimum time between requests
    max_interval: float = 5.0  # Maximum time between requests
    backoff_multiplier: float = 1.5  # Exponential backoff multiplier


@dataclass
class RequestMetrics:
    """Track request success/failure rates"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_success_time: float = 0
    last_failure_time: float = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate


@dataclass
class QueuedRequest:
    """A queued API request"""

    func: Callable
    args: tuple
    kwargs: dict
    priority: Priority
    cost_estimate: Decimal
    future: asyncio.Future
    queued_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        # Lower priority value = higher priority
        return self.priority.value < other.priority.value


class TokenBucket:
    """Thread-safe token bucket for rate limiting"""

    def __init__(self, max_tokens: int, refill_rate: float, burst_capacity: int = 0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.burst_capacity = burst_capacity
        self._tokens = float(max_tokens)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        time_passed = now - self._last_refill
        self._last_refill = now

        # Add tokens based on refill rate
        tokens_to_add = time_passed * self.refill_rate
        max_capacity = self.max_tokens + self.burst_capacity
        self._tokens = min(max_capacity, self._tokens + tokens_to_add)

    @property
    def available_tokens(self) -> float:
        """Get current available tokens"""
        with self._lock:
            self._refill()
            return self._tokens

    def time_until_available(self, tokens: int = 1) -> float:
        """Time in seconds until tokens are available"""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self._tokens
            return tokens_needed / self.refill_rate


class CircuitBreaker:
    """Circuit breaker for API failure protection"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker moving to HALF_OPEN state")
                    return True
                return False

            # HALF_OPEN state - allow limited requests
            return True

    def record_success(self):
        """Record a successful request"""
        with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    logger.info(f"Circuit breaker CLOSED - service recovered")

    def record_failure(self):
        """Record a failed request"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker OPEN - too many failures ({self.failure_count})"
                    )

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


class PriorityQueue:
    """Thread-safe priority queue for requests"""

    def __init__(self, maxsize: int = 0):
        self._queue = queue.PriorityQueue(maxsize)

    def put(self, item: QueuedRequest, timeout: Optional[float] = None):
        """Add item to queue"""
        return self._queue.put(item, timeout=timeout)

    def get(self, timeout: Optional[float] = None) -> QueuedRequest:
        """Get item from queue"""
        return self._queue.get(timeout=timeout)

    def qsize(self) -> int:
        """Get queue size"""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()


class RateLimiter:
    """Comprehensive rate limiter with all features"""

    def __init__(self, service_name: str, config: RateLimitConfig):
        self.service_name = service_name
        self.config = config

        # Core components
        self.token_bucket = TokenBucket(
            config.max_tokens, config.refill_rate, config.burst_capacity
        )
        self.circuit_breaker = CircuitBreaker(config)
        self.request_queue = PriorityQueue(config.max_queue_size)

        # Metrics and tracking
        self.metrics = RequestMetrics()
        self.cost_window = deque(maxlen=60)  # Track costs over 60 seconds
        self.adaptive_interval = config.min_interval

        # Background processing
        self._stop_processing = False
        self._processor_thread = threading.Thread(
            target=self._process_queue, daemon=True
        )
        self._processor_thread.start()

        logger.info(f"Rate limiter initialized for service: {service_name}")

    def _process_queue(self):
        """Background thread to process queued requests"""
        while not self._stop_processing:
            try:
                # Get request with timeout to allow periodic cleanup
                request = self.request_queue.get(timeout=1.0)

                # Check if request has timed out
                if time.time() - request.queued_at > self.config.request_timeout:
                    request.future.set_exception(
                        TimeoutError(
                            f"Request timed out after {self.config.request_timeout}s in queue"
                        )
                    )
                    continue

                # Process the request
                self._execute_request(request)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")

    def _execute_request(self, request: QueuedRequest):
        """Execute a queued request with all protections"""
        try:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                request.future.set_exception(
                    RuntimeError(f"Circuit breaker is OPEN for {self.service_name}")
                )
                return

            # Wait for token bucket
            start_wait = time.time()
            while not self.token_bucket.consume():
                time.sleep(0.1)  # Small sleep to avoid busy waiting

                # Check for timeout
                if time.time() - start_wait > 30:  # 30 second max wait
                    request.future.set_exception(
                        TimeoutError("Rate limit wait timeout")
                    )
                    return

            wait_time = time.time() - start_wait
            rate_limit_wait_time.labels(service=self.service_name).observe(wait_time)

            # Apply adaptive interval
            if self.adaptive_interval > 0:
                time.sleep(self.adaptive_interval)

            # Execute the actual request
            start_time = time.time()
            try:
                result = request.func(*request.args, **request.kwargs)

                # Record success
                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                self.metrics.consecutive_successes += 1
                self.metrics.consecutive_failures = 0
                self.metrics.last_success_time = time.time()

                self.circuit_breaker.record_success()
                rate_limit_requests.labels(
                    service=self.service_name, status="success"
                ).inc()

                # Adjust adaptive interval (decrease on success)
                self.adaptive_interval = max(
                    self.config.min_interval,
                    self.adaptive_interval / self.config.backoff_multiplier,
                )

                # Track cost
                self._track_cost(request.cost_estimate)

                request.future.set_result(result)

            except Exception as e:
                # Record failure
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.metrics.consecutive_failures += 1
                self.metrics.consecutive_successes = 0
                self.metrics.last_failure_time = time.time()

                self.circuit_breaker.record_failure()
                rate_limit_requests.labels(
                    service=self.service_name, status="failure"
                ).inc()

                # Adjust adaptive interval (increase on failure)
                self.adaptive_interval = min(
                    self.config.max_interval,
                    self.adaptive_interval * self.config.backoff_multiplier,
                )

                request.future.set_exception(e)

            # Update metrics
            api_success_rate.labels(service=self.service_name).set(
                self.metrics.success_rate
            )
            circuit_breaker_state.labels(service=self.service_name).set(
                self.circuit_breaker.state.value
            )

        except Exception as e:
            logger.error(f"Error executing request for {self.service_name}: {e}")
            request.future.set_exception(e)

    def _track_cost(self, cost: Decimal):
        """Track API costs in rolling window"""
        now = time.time()
        self.cost_window.append((now, cost))

        # Clean old entries (older than 1 minute)
        while self.cost_window and now - self.cost_window[0][0] > 60:
            self.cost_window.popleft()

    def _get_current_cost_rate(self) -> Decimal:
        """Get current cost per minute"""
        return sum(cost for _, cost in self.cost_window)

    async def execute_async(
        self,
        func: Callable,
        *args,
        priority: Priority = Priority.NORMAL,
        cost_estimate: Optional[Decimal] = None,
        **kwargs,
    ) -> Any:
        """Execute function with rate limiting (async version)"""

        if cost_estimate is None:
            cost_estimate = self.config.cost_per_request

        # Check budget constraints
        within_budget, remaining = check_budget(float(cost_estimate))
        if not within_budget:
            raise BudgetExceededError(
                f"Insufficient budget for {self.service_name} request"
            )

        # Check cost-based rate limiting
        current_cost_rate = self._get_current_cost_rate()
        if current_cost_rate + cost_estimate > self.config.max_cost_per_minute:
            raise RuntimeError(
                f"Cost rate limit exceeded for {self.service_name}: "
                f"${current_cost_rate + cost_estimate}/min > ${self.config.max_cost_per_minute}/min"
            )

        # Create request
        future = asyncio.Future()
        request = QueuedRequest(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            cost_estimate=cost_estimate,
            future=future,
        )

        # Queue the request
        try:
            self.request_queue.put(request, timeout=1.0)
            rate_limit_queue_size.labels(
                service=self.service_name, priority=priority.name
            ).set(self.request_queue.qsize())

        except queue.Full:
            raise RuntimeError(f"Rate limit queue is full for {self.service_name}")

        # Wait for result
        return await future

    def execute_sync(
        self,
        func: Callable,
        *args,
        priority: Priority = Priority.NORMAL,
        cost_estimate: Optional[Decimal] = None,
        **kwargs,
    ) -> Any:
        """Execute function with rate limiting (sync version)"""

        # Create and run async version
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.execute_async(
                    func,
                    *args,
                    priority=priority,
                    cost_estimate=cost_estimate,
                    **kwargs,
                )
            )
        finally:
            loop.close()

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        return {
            "service_name": self.service_name,
            "available_tokens": self.token_bucket.available_tokens,
            "circuit_breaker_state": self.circuit_breaker.state.name,
            "queue_size": self.request_queue.qsize(),
            "success_rate": self.metrics.success_rate,
            "adaptive_interval": self.adaptive_interval,
            "current_cost_rate": float(self._get_current_cost_rate()),
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
        }

    def shutdown(self):
        """Shutdown the rate limiter"""
        self._stop_processing = True
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5.0)


class RateLimitManager:
    """Global manager for all rate limiters"""

    _instance = None
    _lock = threading.Lock()

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

        self.limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

        # Default configurations for different services
        self.default_configs = {
            "deepgram": RateLimitConfig(
                max_tokens=5,
                refill_rate=1.0,
                burst_capacity=2,
                max_cost_per_minute=Decimal("2.00"),
                cost_per_request=Decimal("0.003"),
                failure_threshold=3,
                recovery_timeout=60,
            ),
            "openai": RateLimitConfig(
                max_tokens=3,
                refill_rate=0.5,
                burst_capacity=1,
                max_cost_per_minute=Decimal("1.00"),
                cost_per_request=Decimal("0.01"),
                failure_threshold=3,
                recovery_timeout=30,
            ),
            "anthropic": RateLimitConfig(
                max_tokens=2,
                refill_rate=0.3,
                burst_capacity=1,
                max_cost_per_minute=Decimal("1.50"),
                cost_per_request=Decimal("0.05"),
                failure_threshold=2,
                recovery_timeout=45,
            ),
            "gemini": RateLimitConfig(
                max_tokens=10,
                refill_rate=2.0,
                burst_capacity=5,
                max_cost_per_minute=Decimal("0.50"),
                cost_per_request=Decimal("0.0001"),
                failure_threshold=5,
                recovery_timeout=20,
            ),
            "default": RateLimitConfig(),
        }

        self._initialized = True
        logger.info("Rate limit manager initialized")

    def get_limiter(
        self, service_name: str, config: Optional[RateLimitConfig] = None
    ) -> RateLimiter:
        """Get or create rate limiter for service"""
        with self._lock:
            if service_name not in self.limiters:
                if config is None:
                    config = self.default_configs.get(
                        service_name, self.default_configs["default"]
                    )

                self.limiters[service_name] = RateLimiter(service_name, config)
                logger.info(f"Created rate limiter for service: {service_name}")

            return self.limiters[service_name]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all rate limiters"""
        return {
            service: limiter.get_status() for service, limiter in self.limiters.items()
        }

    def shutdown_all(self):
        """Shutdown all rate limiters"""
        with self._lock:
            for limiter in self.limiters.values():
                limiter.shutdown()
            self.limiters.clear()
        logger.info("All rate limiters shut down")


# Global manager instance
rate_limit_manager = RateLimitManager()


def rate_limited(
    service_name: str,
    priority: Priority = Priority.NORMAL,
    cost_estimate: Optional[Decimal] = None,
    config: Optional[RateLimitConfig] = None,
):
    """Decorator for rate limiting function calls"""

    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            limiter = rate_limit_manager.get_limiter(service_name, config)
            return limiter.execute_sync(
                func, *args, priority=priority, cost_estimate=cost_estimate, **kwargs
            )

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            limiter = rate_limit_manager.get_limiter(service_name, config)
            return await limiter.execute_async(
                func, *args, priority=priority, cost_estimate=cost_estimate, **kwargs
            )

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience functions
def get_rate_limiter(service_name: str) -> RateLimiter:
    """Get rate limiter for a service"""
    return rate_limit_manager.get_limiter(service_name)


def get_rate_limit_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all rate limiters"""
    return rate_limit_manager.get_all_status()


def shutdown_rate_limiters():
    """Shutdown all rate limiters"""
    rate_limit_manager.shutdown_all()
